import asyncio
import base64
import json
import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import websockets

try:
    from local_whisper import LocalWhisperStream, forward_audio_from_client_to_whisper
    from text_translation import (
        get_correction_text_model,
        get_summary_text_model,
        get_translation_text_model,
        request_chunk_correction,
        request_text_summary,
        request_text_translation,
    )
    from ws_utils import safe_receive_message, safe_send_envelope
except ModuleNotFoundError:
    from face_server.local_whisper import LocalWhisperStream, forward_audio_from_client_to_whisper
    from face_server.text_translation import (
        get_correction_text_model,
        get_summary_text_model,
        get_translation_text_model,
        request_chunk_correction,
        request_text_summary,
        request_text_translation,
    )
    from face_server.ws_utils import safe_receive_message, safe_send_envelope

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-realtime")
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-transcribe")
STT_BACKEND = os.getenv("STT_BACKEND", "realtime").strip().lower()
SUBTITLE_SUMMARY_TRIGGER_CHARS = int(os.getenv("SUBTITLE_SUMMARY_TRIGGER_CHARS", "120"))
REALTIME_WS_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

app = FastAPI()

SUBTITLE_SESSION_INSTRUCTIONS = (
    "Subtitle mode. Focus on stable and low-latency transcription. "
    "Do not generate suggestions unless explicitly asked."
)

SUBTITLE_TURN_DETECTION = {
    "type": "server_vad",
    "threshold": 0.50,
    "silence_duration_ms": 60,
    "prefix_padding_ms": 120,
}

LANGUAGE_LABELS = {
    "auto": "auto-detect",
    "ja": "Japanese",
    "en": "English",
    "zh": "Chinese",
    "zh-Hans": "Simplified Chinese",
    "ko": "Korean",
}


async def openai_realtime_connect():
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }
    return await websockets.connect(
        REALTIME_WS_URL,
        additional_headers=headers,
        max_size=2**24,
    )


def language_label(code: str) -> str:
    return LANGUAGE_LABELS.get(code, code)


def summary_source_label(transcription_language: str) -> str:
    if transcription_language == "auto":
        return "the source language of the summary"
    return language_label(transcription_language)


def correction_source_label(transcription_language: str) -> str:
    if transcription_language == "auto":
        return "the same language as the transcript chunk"
    return language_label(transcription_language)


def is_english_source(transcription_language: str) -> bool:
    return transcription_language == "en"


def is_simplified_chinese_source(transcription_language: str) -> bool:
    return transcription_language in {"zh", "zh-Hans"}


def build_transcribe_session_update(transcription_language: str) -> dict:
    source_label = language_label(transcription_language)
    instructions = SUBTITLE_SESSION_INSTRUCTIONS
    if transcription_language != "auto":
        instructions += (
            f" The expected spoken language is {source_label}. "
            f" Transcribe in {source_label} only unless the audio is clearly in another language. "
            "When audio is ambiguous, strongly prefer the expected language and script. "
        )

    return {
        "type": "session.update",
        "session": {
            "modalities": ["text"],
            "instructions": instructions,
            "turn_detection": SUBTITLE_TURN_DETECTION,
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": TRANSCRIBE_MODEL,
            },
        },
    }


async def send_ows_event(ows, payload: dict):
    await ows.send(json.dumps(payload, ensure_ascii=False))


async def forward_audio_from_client(ws: WebSocket, ows):
    while True:
        msg = await safe_receive_message(ws)
        if msg is None:
            return
        if "bytes" in msg and msg["bytes"] is not None:
            payload = {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(msg["bytes"]).decode("utf-8"),
            }
            await ows.send(json.dumps(payload))
        elif "text" in msg and msg["text"] is not None:
            try:
                body = json.loads(msg["text"])
            except Exception:
                continue

            if body.get("type") == "reset":
                await ows.send(json.dumps({"type": "input_audio_buffer.clear"}))


async def subtitle_translation_worker(
    ws: WebSocket,
    queue: asyncio.Queue[tuple[str, str]],
    transcription_language: str,
    translation_language: str,
):
    source_label = language_label(transcription_language)
    target_label = language_label(translation_language)
    print("Using text translation model:", get_translation_text_model(), f"for {source_label} -> {target_label}")
    while True:
        segment_id, transcript = await queue.get()
        try:
            translation = await request_text_translation(
                transcript,
                source_label,
                target_label,
            )
            if translation:
                sent = await safe_send_envelope(ws, {
                    "type": "translation",
                    "segment_id": segment_id,
                    "transcript": transcript,
                    "translation": translation,
                })
                if not sent:
                    return
        finally:
            queue.task_done()


async def subtitle_summary_worker(
    ws: WebSocket,
    queue: asyncio.Queue[tuple[str, list[str]]],
    transcription_language: str,
):
    print("Using subtitle summary model:", get_summary_text_model())
    print("Using subtitle correction model:", get_correction_text_model())
    source_label = summary_source_label(transcription_language)
    correction_label = correction_source_label(transcription_language)
    while True:
        summary_id, transcripts = await queue.get()
        try:
            chunk_text = "\n".join(item.strip() for item in transcripts if item.strip()).strip()
            if not chunk_text:
                continue
            note_source = (await request_chunk_correction(chunk_text, correction_label)).strip() or chunk_text
            source_summary = (await request_text_summary(note_source)).strip()
            if not source_summary:
                continue
            english_task = None
            chinese_task = None
            english_summary = source_summary if is_english_source(transcription_language) else ""
            chinese_summary = source_summary if is_simplified_chinese_source(transcription_language) else ""

            if not english_summary:
                english_task = asyncio.create_task(
                    request_text_translation(source_summary, source_label, "English")
                )
            if not chinese_summary:
                chinese_task = asyncio.create_task(
                    request_text_translation(source_summary, source_label, "Simplified Chinese")
                )

            if english_task is not None:
                english_summary = (await english_task).strip()
            if chinese_task is not None:
                chinese_summary = (await chinese_task).strip()
            summaries = {
                "source": source_summary,
                "en": english_summary.strip(),
                "zh-Hans": chinese_summary.strip(),
            }
            if summaries.get("source") or summaries.get("en") or summaries.get("zh-Hans"):
                sent = await safe_send_envelope(ws, {
                    "type": "summary",
                    "summary_id": summary_id,
                    "summary_type": "chunk",
                    "note_source": note_source,
                    "summary": summaries.get("source") or summaries.get("zh-Hans") or summaries.get("en") or "",
                    "summaries": summaries,
                })
                if not sent:
                    return
        finally:
            queue.task_done()


async def relay_subtitle_events_to_client(ws: WebSocket, transcribe_ows,
                                          translation_queue: asyncio.Queue[tuple[str, str]],
                                          on_final_transcript):
    current_transcript = ""

    async for raw in transcribe_ows:
        try:
            event = json.loads(raw)
        except Exception:
            continue

        event_type = event.get("type", "")

        if event_type == "conversation.item.input_audio_transcription.delta":
            delta = event.get("delta", "")
            if delta:
                current_transcript += delta
                sent = await safe_send_envelope(ws, {
                    "type": "transcript_delta",
                    "delta": delta,
                })
                if not sent:
                    return
            continue

        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = (event.get("transcript") or current_transcript).strip()
            current_transcript = ""
            segment_id = f"seg_{uuid.uuid4().hex[:12]}"

            sent = await safe_send_envelope(ws, {
                "type": "transcript_final",
                "segment_id": segment_id,
                "transcript": transcript,
                "translation": "",
                "next_say": [],
                "intent": "",
            })
            if not sent:
                return

            if transcript:
                await translation_queue.put((segment_id, transcript))
                await on_final_transcript(transcript)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    transcribe_ows = None
    transcriber = None
    try:
        if STT_BACKEND == "realtime":
            transcribe_ows = await openai_realtime_connect()
        elif STT_BACKEND != "local_whisper":
            print("Unsupported STT_BACKEND:", STT_BACKEND)
            await ws.close()
            return
    except Exception as exc:
        print("OpenAI realtime connect failed:", repr(exc))
        await ws.close()
        return

    translation_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    summary_queue: asyncio.Queue[tuple[str, list[str]]] = asyncio.Queue()
    translation_task = None
    summary_task = None
    transcribe_closed = False
    stt_error_sent = False
    pending_summary_transcripts: list[str] = []
    pending_summary_chars = 0
    summary_lock = asyncio.Lock()

    try:
        try:
            first = await ws.receive_text()
            body = json.loads(first)
            if body.get("type") != "config":
                pass
        except Exception:
            body = {}

        transcription_language = str(
            body.get("transcription_language") or "auto"
        ).strip()
        translation_language = str(
            body.get("translation_language") or "zh-Hans"
        ).strip()

        if transcribe_ows is not None:
            await send_ows_event(
                transcribe_ows,
                build_transcribe_session_update(transcription_language),
            )

        async def flush_pending_summary(force: bool = False):
            nonlocal pending_summary_chars
            async with summary_lock:
                if not pending_summary_transcripts:
                    return
                if not force and pending_summary_chars < SUBTITLE_SUMMARY_TRIGGER_CHARS:
                    return
                summary_id = f"sum_{uuid.uuid4().hex[:12]}"
                batch = list(pending_summary_transcripts)
                pending_summary_transcripts.clear()
                pending_summary_chars = 0
            await summary_queue.put((summary_id, batch))

        async def handle_local_delta(delta: str):
            await safe_send_envelope(ws, {
                "type": "transcript_delta",
                "delta": delta,
            })

        async def handle_summary_transcript(transcript: str):
            nonlocal pending_summary_chars
            if not transcript:
                return
            async with summary_lock:
                pending_summary_transcripts.append(transcript)
                pending_summary_chars += len(transcript)
            await flush_pending_summary(force=False)

        async def handle_local_final(transcript: str):
            transcript = transcript.strip()
            segment_id = f"seg_{uuid.uuid4().hex[:12]}"
            sent = await safe_send_envelope(ws, {
                "type": "transcript_final",
                "segment_id": segment_id,
                "transcript": transcript,
                "translation": "",
                "next_say": [],
                "intent": "",
            })
            if not sent:
                return

            if transcript:
                await translation_queue.put((segment_id, transcript))
                await handle_summary_transcript(transcript)
        translation_task = asyncio.create_task(
            subtitle_translation_worker(
                ws,
                translation_queue,
                transcription_language,
                translation_language,
            )
        )
        summary_task = asyncio.create_task(
            subtitle_summary_worker(
                ws,
                summary_queue,
                transcription_language,
            )
        )

        async def notify_stt_connection_closed(reason: str):
            nonlocal transcribe_closed, stt_error_sent
            transcribe_closed = True
            if stt_error_sent:
                return
            stt_error_sent = True
            await safe_send_envelope(ws, {
                "type": "error",
                "stage": "stt_connection",
                "message": "transcribe websocket closed",
                "reason": reason,
            })

        async def handle_client_messages():
            while True:
                msg = await safe_receive_message(ws)
                if msg is None:
                    return
                if "bytes" in msg and msg["bytes"] is not None:
                    if transcribe_ows is not None and not transcribe_closed:
                        payload = {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(msg["bytes"]).decode("utf-8"),
                        }
                        try:
                            await transcribe_ows.send(json.dumps(payload))
                        except websockets.exceptions.ConnectionClosed as exc:
                            await notify_stt_connection_closed(repr(exc))
                    elif transcriber is not None:
                        await transcriber.add_audio(msg["bytes"])
                    continue

                if "text" not in msg or msg["text"] is None:
                    continue

                try:
                    body = json.loads(msg["text"])
                except Exception:
                    continue

                if body.get("type") == "reset":
                    if transcribe_ows is not None and not transcribe_closed:
                        try:
                            await transcribe_ows.send(json.dumps({"type": "input_audio_buffer.clear"}))
                        except websockets.exceptions.ConnectionClosed as exc:
                            await notify_stt_connection_closed(repr(exc))
                    elif transcriber is not None:
                        await transcriber.reset()

        if STT_BACKEND == "realtime":
            await asyncio.gather(
                handle_client_messages(),
                relay_subtitle_events_to_client(
                    ws,
                    transcribe_ows,
                    translation_queue,
                    handle_summary_transcript,
                ),
            )
        else:
            transcriber = LocalWhisperStream(
                transcription_language,
                handle_local_delta,
                handle_local_final,
            )

            try:
                await transcriber.start()
                await asyncio.gather(
                    handle_client_messages(),
                    transcriber.run(),
                )
            finally:
                await transcriber.close()
    except WebSocketDisconnect:
        pass
    finally:
        if "flush_pending_summary" in locals():
            await flush_pending_summary(force=True)
        if summary_task is not None:
            await summary_queue.join()
            summary_task.cancel()
            await asyncio.gather(summary_task, return_exceptions=True)
        if translation_task is not None:
            await translation_queue.join()
            translation_task.cancel()
            await asyncio.gather(translation_task, return_exceptions=True)
        try:
            if transcribe_ows is not None:
                await transcribe_ows.close()
        except Exception:
            pass
