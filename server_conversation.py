import asyncio
import base64
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import websockets

try:
    from local_whisper import LocalWhisperStream, forward_audio_from_client_to_whisper
    from soniox_stt import SonioxRealtimeStream, soniox_target_language
    from text_translation import get_translation_text_model, request_text_translation
    from ws_utils import safe_receive_message, safe_send_envelope
except ModuleNotFoundError:
    from face_server.local_whisper import LocalWhisperStream, forward_audio_from_client_to_whisper
    from face_server.soniox_stt import SonioxRealtimeStream, soniox_target_language
    from face_server.text_translation import get_translation_text_model, request_text_translation
    from face_server.ws_utils import safe_receive_message, safe_send_envelope

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-realtime")
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-transcribe")
STT_BACKEND = os.getenv("STT_BACKEND", "realtime").strip().lower()
CONVERSATION_MEMORY_TRIGGER_CHARS = int(os.getenv("CONVERSATION_MEMORY_TRIGGER_CHARS", "80"))
REALTIME_WS_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
DATA_DIR = Path(__file__).resolve().parent / "data" / "conversation_sessions"

app = FastAPI()

TRANSCRIBE_TURN_DETECTION = {
    "type": "server_vad",
    "threshold": 0.5,
    "silence_duration_ms": 200,
    "prefix_padding_ms": 400,
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


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def language_label(code: str) -> str:
    return LANGUAGE_LABELS.get(code, code)


def sanitize_session_id(session_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", session_id.strip())
    return cleaned or f"session_{uuid.uuid4().hex}"


def session_dir(client_session_id: str) -> Path:
    path = DATA_DIR / sanitize_session_id(client_session_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def state_path(client_session_id: str) -> Path:
    return session_dir(client_session_id) / "session_state.json"


def events_path(client_session_id: str) -> Path:
    return session_dir(client_session_id) / "events.jsonl"


def load_session_state(client_session_id: str) -> dict | None:
    path = state_path(client_session_id)
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_session_state(state: dict):
    path = state_path(state["client_session_id"])
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_event(client_session_id: str, payload: dict):
    path = events_path(client_session_id)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def build_default_goal(outline: str) -> str:
    outline = outline.strip()
    if outline:
        first_line = next((line.strip() for line in outline.splitlines() if line.strip()), "")
        if first_line:
            return first_line[:180]
    return "Help the user complete the current conversation task."


def build_initial_session_state(
    client_session_id: str,
    openai_session_id: str,
    goal: str,
) -> dict:
    timestamp = now_iso()
    return {
        "client_session_id": client_session_id,
        "openai_session_id": openai_session_id,
        "goal": goal,
        "status": "active",
        "summary": "This is a new conversation. The task has not been developed yet.",
        "known_info": [],
        "open_loops": [],
        "next_actions": [],
        "last_user_message": "",
        "updated_at": timestamp,
    }


def maybe_extract_openai_session_id(event: dict) -> str:
    session = event.get("session")
    if isinstance(session, dict):
        session_id = session.get("id")
        if isinstance(session_id, str) and session_id:
            return session_id
    return ""


def build_user_message(client_session_id: str, texts: list[str]) -> dict:
    text = "\n".join(item.strip() for item in texts if item.strip()).strip()
    return {
        "session_id": client_session_id,
        "message_id": f"m_{uuid.uuid4().hex[:12]}",
        "role": "unknown",
        "text": text,
        "timestamp": now_iso(),
    }


def build_workflow_result(session_state: dict, user_message: dict) -> dict:
    message_text = (user_message.get("text") or "").strip()
    goal = (session_state.get("goal") or "").strip()
    open_loops = session_state.get("open_loops") or []

    doc_keywords = [
        "document", "documents", "material", "materials", "summary", "summarize",
        "faq", "sop", "manual", "guide",
        "资料", "文档", "文件", "总结", "整理",
    ]
    needs_docs = any(keyword in message_text.lower() for keyword in doc_keywords)
    needs_clarification = not goal or goal.lower() in {
        "help the user complete the current conversation task.",
        "help me",
    }
    goal_clear = not needs_clarification

    if open_loops:
        recommended_focus = open_loops[0]
    elif needs_clarification:
        recommended_focus = "Clarify the user's concrete goal before moving on."
    elif needs_docs:
        recommended_focus = "Extract the most relevant factual points from supporting documents."
    else:
        recommended_focus = "Move the conversation toward the stated goal with the next best reply."

    return {
        "goal_clear": goal_clear,
        "needs_docs": needs_docs,
        "needs_clarification": needs_clarification,
        "recommended_focus": recommended_focus,
    }


def merge_session_state(session_state: dict, memory_patch: dict, user_message: dict) -> dict:
    merged = dict(session_state)
    merged["summary"] = (memory_patch.get("summary") or merged.get("summary") or "").strip()

    known_info = list(merged.get("known_info") or [])
    known_info.extend(memory_patch.get("known_info_add") or [])
    merged["known_info"] = dedupe_preserve_order([str(item) for item in known_info])

    open_loops = list(merged.get("open_loops") or [])
    open_loops.extend(memory_patch.get("open_loops_add") or [])
    open_loops = dedupe_preserve_order([str(item) for item in open_loops])
    remove_set = {
        str(item).strip()
        for item in (memory_patch.get("open_loops_remove") or [])
        if str(item).strip()
    }
    if remove_set:
        open_loops = [item for item in open_loops if item not in remove_set]
    merged["open_loops"] = open_loops

    if "next_actions_replace" in memory_patch:
        merged["next_actions"] = dedupe_preserve_order(
            [str(item) for item in (memory_patch.get("next_actions_replace") or [])]
        )

    merged["last_user_message"] = user_message["text"]
    merged["updated_at"] = now_iso()
    return merged


def build_transcribe_session_update(transcription_language: str) -> dict:
    instructions = "Conversation mode transcription only. Transcribe the spoken audio accurately."
    if transcription_language != "auto":
        instructions += (
            f" The expected spoken language is {language_label(transcription_language)}. "
            f" Transcribe in {language_label(transcription_language)} only unless the audio is clearly in another language. "
            "When audio is ambiguous, strongly prefer the expected language and script. "
        )

    return {
        "type": "session.update",
        "session": {
            "modalities": ["text"],
            "instructions": instructions,
            "turn_detection": TRANSCRIBE_TURN_DETECTION,
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": TRANSCRIBE_MODEL,
            },
        },
    }


def build_assistant_session_update() -> dict:
    return {
        "type": "session.update",
        "session": {
            "modalities": ["text"],
            "instructions": (
                "You are a conversation-state and reply-planning assistant. "
                "Always follow the requested JSON schema exactly and return plain JSON only."
            ),
        },
    }


async def send_ows_event(ows, payload: dict):
    await ows.send(json.dumps(payload, ensure_ascii=False))


async def read_response_text(ows) -> str:
    text_buf = []

    async for raw in ows:
        try:
            event = json.loads(raw)
        except Exception:
            continue

        event_type = event.get("type", "")

        if event_type in ("response.output_text.delta", "response.text.delta"):
            delta = event.get("delta", "")
            if delta:
                text_buf.append(delta)
            continue

        if event_type in ("response.output_text.done", "response.text.done"):
            return (event.get("text") or "".join(text_buf)).strip()

    return ""


def parse_json_object(text: str) -> dict:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(text[start:end + 1])
        if isinstance(obj, dict):
            return obj

    raise ValueError("Model did not return a JSON object")


async def request_json_response(ows, instructions: str) -> dict:
    await send_ows_event(ows, {
        "type": "response.create",
        "response": {
            "modalities": ["text"],
            "instructions": instructions,
        },
    })
    final_text = await read_response_text(ows)
    return parse_json_object(final_text)


async def derive_goal_from_outline(assistant_ows, outline: str) -> str:
    if not outline.strip():
        return build_default_goal("")

    prompt = (
        "Summarize the user's outline into one concise actionable conversation goal.\n"
        "Return JSON only:\n"
        '{"goal":"..."}\n\n'
        f"Outline:\n{outline}"
    )
    try:
        result = await request_json_response(assistant_ows, prompt)
        goal = str(result.get("goal") or "").strip()
        if goal:
            return goal
    except Exception:
        pass

    return build_default_goal(outline)


async def generate_opening_suggestion(assistant_ows, session_state: dict, outline: str) -> dict:
    prompt = (
        "Create an opening conversation suggestion for the user based on the goal and outline.\n"
        "Return JSON only in this schema:\n"
        '{"type":"suggestion","stage":"opening","zh_translation":"","next_say":[{"ja":"...","romaji":"...","zh":"..."}],"intent":"opening"}\n\n'
        f"Goal:\n{session_state.get('goal', '')}\n\n"
        f"Outline:\n{outline}"
    )
    suggestion = await request_json_response(assistant_ows, prompt)
    suggestion["type"] = "suggestion"
    suggestion["stage"] = suggestion.get("stage") or "opening"
    suggestion["intent"] = suggestion.get("intent") or "opening"
    suggestion["zh_translation"] = str(suggestion.get("zh_translation") or "")
    suggestion["next_say"] = suggestion.get("next_say") or []
    return suggestion


async def extract_memory_patch(
    assistant_ows,
    session_state: dict,
    user_message: dict,
    last_assistant_reply: list[str],
) -> dict:
    prompt = (
        "Update the conversation memory state from the previous session state and the current user message.\n"
        "Return JSON only in this schema:\n"
        '{'
        '"summary":"...",'
        '"known_info_add":["..."],'
        '"open_loops_add":["..."],'
        '"open_loops_remove":["..."],'
        '"next_actions_replace":["..."]'
        '}\n\n'
        f"Session state:\n{json.dumps(session_state, ensure_ascii=False)}\n\n"
        f"Current user message:\n{json.dumps(user_message, ensure_ascii=False)}\n\n"
        f"Last assistant suggestions:\n{json.dumps(last_assistant_reply, ensure_ascii=False)}"
    )
    patch = await request_json_response(assistant_ows, prompt)
    return {
        "summary": str(patch.get("summary") or session_state.get("summary") or "").strip(),
        "known_info_add": [str(item) for item in (patch.get("known_info_add") or [])],
        "open_loops_add": [str(item) for item in (patch.get("open_loops_add") or [])],
        "open_loops_remove": [str(item) for item in (patch.get("open_loops_remove") or [])],
        "next_actions_replace": [str(item) for item in (patch.get("next_actions_replace") or [])],
    }


async def generate_assistant_reply(
    assistant_ows,
    session_state: dict,
    user_message: dict,
    workflow_result: dict,
) -> dict:
    prompt = (
        "Generate the next conversation suggestions for the user.\n"
        "Return JSON only in this schema:\n"
        '{"type":"suggestion","stage":"followup","zh_translation":"","next_say":[{"ja":"...","romaji":"...","zh":"..."}],"intent":"..."}\n\n'
        "Rules:\n"
        "- Return 1 to 3 helpful next utterances.\n"
        "- Keep them aligned with the current goal and open loops.\n"
        "- Use concise, natural Japanese in ja.\n"
        "- Provide romaji and concise Chinese gloss for each line.\n"
        "- Do not add markdown.\n\n"
        f"Session state:\n{json.dumps(session_state, ensure_ascii=False)}\n\n"
        f"Workflow result:\n{json.dumps(workflow_result, ensure_ascii=False)}\n\n"
        f"Current user message:\n{json.dumps(user_message, ensure_ascii=False)}"
    )
    suggestion = await request_json_response(assistant_ows, prompt)
    suggestion["type"] = "suggestion"
    suggestion["stage"] = suggestion.get("stage") or "followup"
    suggestion["intent"] = suggestion.get("intent") or "followup"
    suggestion["zh_translation"] = str(suggestion.get("zh_translation") or "")
    suggestion["next_say"] = suggestion.get("next_say") or []
    return suggestion


def build_assistant_reply_event(client_session_id: str, suggestion: dict) -> dict:
    next_say = suggestion.get("next_say") or []
    text = [str(item.get("ja") or "").strip() for item in next_say if isinstance(item, dict)]
    return {
        "type": "assistant_reply",
        "session_id": client_session_id,
        "message_id": f"a_{uuid.uuid4().hex[:12]}",
        "text": [item for item in text if item],
        "timestamp": now_iso(),
    }


def build_memory_patch_event(client_session_id: str, memory_patch: dict) -> dict:
    return {
        "type": "memory_patch",
        "session_id": client_session_id,
        "patch": memory_patch,
        "timestamp": now_iso(),
    }


def build_error_event(client_session_id: str, stage: str, message: str) -> dict:
    return {
        "type": "error",
        "session_id": client_session_id,
        "stage": stage,
        "message": message,
        "timestamp": now_iso(),
    }


def make_segment_id() -> str:
    return f"seg_{uuid.uuid4().hex[:12]}"


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


async def translation_worker(
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


async def send_transcript_and_translation(
    ws: WebSocket,
    transcript: str,
    on_final_transcript,
    translation_queue: asyncio.Queue[tuple[str, str]] | None = None,
    direct_translation: str = "",
):
    transcript = transcript.strip()
    segment_id = make_segment_id()

    sent = await safe_send_envelope(ws, {
        "type": "transcript_final",
        "segment_id": segment_id,
        "transcript": transcript,
        "translation": "",
        "next_say": [],
        "intent": "",
    })
    if not sent or not transcript:
        return

    if direct_translation.strip():
        sent = await safe_send_envelope(ws, {
            "type": "translation",
            "segment_id": segment_id,
            "transcript": transcript,
            "translation": direct_translation.strip(),
        })
        if not sent:
            return
    elif translation_queue is not None:
        await translation_queue.put((segment_id, transcript))

    await on_final_transcript(transcript)


async def conversation_worker(
    ws: WebSocket,
    assistant_ows,
    client_session_id: str,
    outline: str,
    session_state: dict,
    pending_transcripts: list[str],
    pending_lock: asyncio.Lock,
):
    last_assistant_reply = list(session_state.get("next_actions") or [])

    while True:
        async with pending_lock:
            if not pending_transcripts:
                return
            batch = list(pending_transcripts)
            pending_transcripts.clear()

        user_message = build_user_message(client_session_id, batch)
        append_event(client_session_id, user_message)

        try:
            memory_patch = await extract_memory_patch(
                assistant_ows,
                session_state,
                user_message,
                last_assistant_reply,
            )
            append_event(client_session_id, build_memory_patch_event(client_session_id, memory_patch))

            merged_state = merge_session_state(session_state, memory_patch, user_message)
            if outline and not merged_state.get("goal"):
                merged_state["goal"] = build_default_goal(outline)
            save_session_state(merged_state)
            session_state.clear()
            session_state.update(merged_state)

            workflow_result = build_workflow_result(session_state, user_message)
            suggestion = await generate_assistant_reply(
                assistant_ows,
                session_state,
                user_message,
                workflow_result,
            )

            append_event(
                client_session_id,
                build_assistant_reply_event(client_session_id, suggestion),
            )

            last_assistant_reply = [
                str(item.get("ja") or "").strip()
                for item in (suggestion.get("next_say") or [])
                if isinstance(item, dict) and str(item.get("ja") or "").strip()
            ]

            sent = await safe_send_envelope(ws, suggestion)
            if not sent:
                return
        except Exception as exc:
            append_event(
                client_session_id,
                build_error_event(client_session_id, "conversation_worker", repr(exc)),
            )


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    transcribe_ows = None
    assistant_ows = None
    transcriber = None
    try:
        if STT_BACKEND == "realtime":
            transcribe_ows = await openai_realtime_connect()
        elif STT_BACKEND not in {"local_whisper", "soniox"}:
            print("Unsupported STT_BACKEND:", STT_BACKEND)
            await ws.close()
            return
        assistant_ows = await openai_realtime_connect()
    except Exception as exc:
        print("OpenAI realtime connect failed:", repr(exc))
        await ws.close()
        return

    translation_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    translation_task = None
    pending_memory_transcripts: list[str] = []
    pending_memory_chars = 0
    memory_lock = asyncio.Lock()
    suggestion_lock = asyncio.Lock()
    suggestion_task = None
    last_assistant_reply: list[str] = []
    flush_pending_memory = None
    transcribe_closed = False
    stt_error_sent = False

    try:
        try:
            first = await ws.receive_text()
            body = json.loads(first)
            if body.get("type") != "config":
                body = {}
        except Exception:
            body = {}

        client_session_id = str(body.get("session_id") or f"legacy_{uuid.uuid4().hex}").strip()
        outline = str(body.get("outline") or "").strip()
        transcription_language = str(body.get("transcription_language") or "auto").strip()
        translation_language = str(body.get("translation_language") or "zh-Hans").strip()
        source_language_code = soniox_target_language(transcription_language)
        target_language_code = soniox_target_language(translation_language)
        use_soniox_direct_translation = (
            STT_BACKEND == "soniox"
            and target_language_code is not None
            and (source_language_code is None or target_language_code != source_language_code)
        )

        if transcribe_ows is not None:
            await send_ows_event(
                transcribe_ows,
                build_transcribe_session_update(transcription_language),
            )
        await send_ows_event(
            assistant_ows,
            build_assistant_session_update(),
        )

        session_state = load_session_state(client_session_id)
        is_new_session = session_state is None
        if is_new_session:
            goal = await derive_goal_from_outline(assistant_ows, outline)
            session_state = build_initial_session_state(
                client_session_id,
                "",
                goal,
            )
            save_session_state(session_state)
        last_assistant_reply = list(session_state.get("next_actions") or [])

        if not use_soniox_direct_translation:
            translation_task = asyncio.create_task(
                translation_worker(
                    ws,
                    translation_queue,
                    transcription_language,
                    translation_language,
                )
            )

        if outline and is_new_session:
            try:
                opening_suggestion = await generate_opening_suggestion(
                    assistant_ows,
                    session_state,
                    outline,
                )
                append_event(
                    client_session_id,
                    build_assistant_reply_event(client_session_id, opening_suggestion),
                )
                await safe_send_envelope(ws, opening_suggestion)
            except Exception as exc:
                append_event(
                    client_session_id,
                    build_error_event(client_session_id, "opening_suggestion", repr(exc)),
                )

        async def flush_pending_memory(force: bool = False) -> bool:
            nonlocal pending_memory_chars
            async with memory_lock:
                if not pending_memory_transcripts:
                    return False
                if not force and pending_memory_chars < CONVERSATION_MEMORY_TRIGGER_CHARS:
                    return False

                batch = list(pending_memory_transcripts)
                pending_memory_transcripts.clear()
                pending_memory_chars = 0

            user_message = build_user_message(client_session_id, batch)
            append_event(client_session_id, user_message)

            try:
                memory_patch = await extract_memory_patch(
                    assistant_ows,
                    session_state,
                    user_message,
                    last_assistant_reply,
                )
                append_event(
                    client_session_id,
                    build_memory_patch_event(client_session_id, memory_patch),
                )

                merged_state = merge_session_state(session_state, memory_patch, user_message)
                if outline and not merged_state.get("goal"):
                    merged_state["goal"] = build_default_goal(outline)
                save_session_state(merged_state)
                session_state.clear()
                session_state.update(merged_state)
                return True
            except Exception as exc:
                append_event(
                    client_session_id,
                    build_error_event(client_session_id, "memory_flush", repr(exc)),
                )
                return False

        async def generate_manual_suggestion():
            nonlocal last_assistant_reply
            async with suggestion_lock:
                async with memory_lock:
                    pending_batch = list(pending_memory_transcripts)
                    pending_chars = pending_memory_chars

                if pending_batch and pending_chars >= CONVERSATION_MEMORY_TRIGGER_CHARS:
                    await flush_pending_memory(force=True)
                    pending_batch = []

                if pending_batch:
                    base_text = str(session_state.get("last_user_message") or "").strip()
                    extra_text = "\n".join(item.strip() for item in pending_batch if item.strip()).strip()
                    merged_text = "\n".join(item for item in [base_text, extra_text] if item).strip()
                    user_message = {
                        "session_id": client_session_id,
                        "message_id": f"m_{uuid.uuid4().hex[:12]}",
                        "role": "unknown",
                        "text": merged_text,
                        "timestamp": now_iso(),
                    }
                else:
                    user_message = build_user_message(
                        client_session_id,
                        [str(session_state.get("last_user_message") or "").strip()],
                    )

                try:
                    workflow_result = build_workflow_result(session_state, user_message)
                    suggestion = await generate_assistant_reply(
                        assistant_ows,
                        session_state,
                        user_message,
                        workflow_result,
                    )
                    append_event(
                        client_session_id,
                        build_assistant_reply_event(client_session_id, suggestion),
                    )
                    last_assistant_reply = [
                        str(item.get("ja") or "").strip()
                        for item in (suggestion.get("next_say") or [])
                        if isinstance(item, dict) and str(item.get("ja") or "").strip()
                    ]
                    await safe_send_envelope(ws, suggestion)
                except Exception as exc:
                    append_event(
                        client_session_id,
                        build_error_event(client_session_id, "manual_suggestion", repr(exc)),
                    )

        async def handle_transcript_final(transcript: str):
            nonlocal pending_memory_chars
            async def record_transcript(text: str):
                nonlocal pending_memory_chars
                async with memory_lock:
                    pending_memory_transcripts.append(text)
                    pending_memory_chars += len(text)
                await flush_pending_memory(force=False)

            await send_transcript_and_translation(
                ws,
                transcript,
                record_transcript,
                translation_queue=translation_queue,
            )

        async def handle_soniox_final(transcript: str, translation: str):
            async def record_transcript(text: str):
                nonlocal pending_memory_chars
                async with memory_lock:
                    pending_memory_transcripts.append(text)
                    pending_memory_chars += len(text)
                await flush_pending_memory(force=False)

            await send_transcript_and_translation(
                ws,
                transcript,
                record_transcript,
                translation_queue=None if use_soniox_direct_translation else translation_queue,
                direct_translation=translation if use_soniox_direct_translation else "",
            )

        async def relay_events_to_client():
            current_transcript = ""

            async for raw in transcribe_ows:
                try:
                    event = json.loads(raw)
                except Exception:
                    continue

                openai_session_id = maybe_extract_openai_session_id(event)
                if openai_session_id and session_state.get("openai_session_id") != openai_session_id:
                    session_state["openai_session_id"] = openai_session_id
                    session_state["updated_at"] = now_iso()
                    save_session_state(session_state)

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
                    await handle_transcript_final(transcript)

        async def handle_local_delta(delta: str):
            await safe_send_envelope(ws, {
                "type": "transcript_delta",
                "delta": delta,
            })

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
            nonlocal suggestion_task
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

                body_type = body.get("type")
                if body_type == "reset":
                    if transcribe_ows is not None and not transcribe_closed:
                        try:
                            await transcribe_ows.send(json.dumps({"type": "input_audio_buffer.clear"}))
                        except websockets.exceptions.ConnectionClosed as exc:
                            await notify_stt_connection_closed(repr(exc))
                    elif transcriber is not None:
                        await transcriber.reset()
                    continue

                if body_type == "request_suggestion":
                    if suggestion_task is None or suggestion_task.done():
                        suggestion_task = asyncio.create_task(generate_manual_suggestion())

        if STT_BACKEND == "realtime":
            await asyncio.gather(
                handle_client_messages(),
                relay_events_to_client(),
            )
        elif STT_BACKEND == "local_whisper":
            transcriber = LocalWhisperStream(
                transcription_language,
                handle_local_delta,
                handle_transcript_final,
            )

            try:
                await transcriber.start()
                await asyncio.gather(
                    handle_client_messages(),
                    transcriber.run(),
                )
            finally:
                await transcriber.close()
        else:
            transcriber = SonioxRealtimeStream(
                transcription_language,
                handle_local_delta,
                handle_soniox_final,
                translation_language=translation_language,
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
        if suggestion_task is not None:
            await asyncio.gather(suggestion_task, return_exceptions=True)
        if flush_pending_memory is not None:
            await flush_pending_memory(force=True)
        if translation_task is not None:
            await translation_queue.join()
            translation_task.cancel()
            await asyncio.gather(translation_task, return_exceptions=True)
        try:
            if transcribe_ows is not None:
                await transcribe_ows.close()
        except Exception:
            pass
        try:
            await assistant_ows.close()
        except Exception:
            pass
