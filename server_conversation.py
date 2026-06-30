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
    from conversation_reasoning import (
        MEMORY_PATCH_RESPONSE_SCHEMA,
        SUGGESTION_RESPONSE_SCHEMA,
        get_conversation_reasoning_effort,
        get_conversation_reasoning_model,
        get_conversation_reasoning_temperature,
    )
    from local_whisper import LocalWhisperStream, forward_audio_from_client_to_whisper
    from openai_responses import request_json_schema_response
    from soniox_stt import SonioxRealtimeStream, soniox_target_language
    from text_translation import get_translation_text_model, request_text_translation
    from ws_utils import safe_receive_message, safe_send_envelope
except ModuleNotFoundError:
    from face_server.conversation_reasoning import (
        MEMORY_PATCH_RESPONSE_SCHEMA,
        SUGGESTION_RESPONSE_SCHEMA,
        get_conversation_reasoning_effort,
        get_conversation_reasoning_model,
        get_conversation_reasoning_temperature,
    )
    from face_server.local_whisper import LocalWhisperStream, forward_audio_from_client_to_whisper
    from face_server.openai_responses import request_json_schema_response
    from face_server.soniox_stt import SonioxRealtimeStream, soniox_target_language
    from face_server.text_translation import get_translation_text_model, request_text_translation
    from face_server.ws_utils import safe_receive_message, safe_send_envelope

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-realtime")
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-transcribe")
STT_BACKEND = os.getenv("STT_BACKEND", "realtime").strip().lower()
SONIOX_TRANSLATION_BACKEND = os.getenv("SONIOX_TRANSLATION_BACKEND", "text").strip().lower()
TRANSLATION_CONTEXT_SEGMENTS = int(os.getenv("TRANSLATION_CONTEXT_SEGMENTS", "3"))
CONVERSATION_MEMORY_TRIGGER_CHARS = int(os.getenv("CONVERSATION_MEMORY_TRIGGER_CHARS", "80"))
CONVERSATION_RECENT_WINDOW_CHARS = int(os.getenv("CONVERSATION_RECENT_WINDOW_CHARS", "3000"))
CONVERSATION_DEBUG_TEXT_ENABLED = os.getenv("CONVERSATION_DEBUG_TEXT_ENABLED", "false").strip().lower() == "true"
CONVERSATION_DEBUG_PROMPTS_ENABLED = os.getenv("CONVERSATION_DEBUG_PROMPTS_ENABLED", "false").strip().lower() == "true"
CONVERSATION_DEBUG_AUDIO_ENABLED = os.getenv("CONVERSATION_DEBUG_AUDIO_ENABLED", "false").strip().lower() == "true"
REALTIME_WS_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
DATA_DIR = Path(__file__).resolve().parent / "data" / "conversation_sessions"

app = FastAPI()
ACTIVE_CLIENT_SESSION_IDS: set[str] = set()
ACTIVE_CLIENT_SESSION_LOCK = asyncio.Lock()

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
        state = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(state, dict):
            migrate_session_state(state)
            return state
        return None
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


def migrate_session_state(state: dict):
    if "latest_conversation" not in state and "last_user_message" in state:
        state["latest_conversation"] = str(state.get("last_user_message") or "")
    state.pop("last_user_message", None)
    if "user_request" not in state and "goal" in state:
        state["user_request"] = str(state.get("goal") or "")
    state.pop("goal", None)
    state["recent_conversation"] = normalize_recent_conversation(state.get("recent_conversation") or [])
    state["recent_conversation_text"] = build_recent_conversation_text(state["recent_conversation"])


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


def build_default_user_request(outline: str) -> str:
    outline = outline.strip()
    return outline or "Help the user with the current conversation."


def build_initial_session_state(
    client_session_id: str,
    openai_session_id: str,
    user_request: str,
) -> dict:
    timestamp = now_iso()
    return {
        "client_session_id": client_session_id,
        "openai_session_id": openai_session_id,
        "user_request": user_request,
        "status": "active",
        "summary": "This is a new conversation. The task has not been developed yet.",
        "known_info": [],
        "open_loops": [],
        "next_actions": [],
        "latest_conversation": "",
        "recent_conversation": [],
        "recent_conversation_text": "",
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


def format_source_transcript(text: str, source: str = "") -> str:
    text = text.strip()
    if not text:
        return ""
    return f"[{normalize_audio_source(source)}] {text}"


def format_transcript_items(items: list[dict] | list[str]) -> list[str]:
    formatted: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            source = str(item.get("source") or "")
            formatted_text = format_source_transcript(text, source)
        else:
            formatted_text = str(item or "").strip()
        if formatted_text:
            formatted.append(formatted_text)
    return formatted


def normalize_recent_conversation(items) -> list[dict]:
    if not isinstance(items, list):
        return []
    normalized: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        normalized.append({
            "source": normalize_audio_source(str(item.get("source") or "")),
            "text": text,
            "timestamp": str(item.get("timestamp") or now_iso()),
        })
    return trim_recent_conversation(normalized)


def build_recent_conversation_text(items: list[dict]) -> str:
    lines: list[str] = []
    for item in items:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        source = normalize_audio_source(str(item.get("source") or ""))
        lines.append(f"[{source}] {text}")
    return "\n".join(lines).strip()


def trim_recent_conversation(items: list[dict]) -> list[dict]:
    kept: list[dict] = []
    total_chars = 0
    for item in reversed(items):
        text = str(item.get("text") or "")
        item_chars = len(text)
        if kept and total_chars + item_chars > CONVERSATION_RECENT_WINDOW_CHARS:
            break
        kept.append(item)
        total_chars += item_chars
    kept.reverse()
    return kept


def append_recent_conversation(session_state: dict, text: str, source: str = ""):
    text = text.strip()
    if not text:
        return
    recent = normalize_recent_conversation(session_state.get("recent_conversation") or [])
    recent.append({
        "source": normalize_audio_source(source),
        "text": text,
        "timestamp": now_iso(),
    })
    recent = trim_recent_conversation(recent)
    session_state["recent_conversation"] = recent
    session_state["recent_conversation_text"] = build_recent_conversation_text(recent)
    session_state["updated_at"] = now_iso()


def latest_conversation_text(session_state: dict) -> str:
    return str(
        session_state.get("recent_conversation_text")
        or session_state.get("latest_conversation")
        or session_state.get("last_user_message")
        or ""
    ).strip()


def build_workflow_result(session_state: dict, user_message: dict) -> dict:
    message_text = (user_message.get("text") or "").strip()
    user_request = (session_state.get("user_request") or "").strip()
    open_loops = session_state.get("open_loops") or []

    doc_keywords = [
        "document", "documents", "material", "materials", "summary", "summarize",
        "faq", "sop", "manual", "guide",
        "资料", "文档", "文件", "总结", "整理",
    ]
    needs_docs = any(keyword in message_text.lower() for keyword in doc_keywords)
    needs_clarification = not user_request or user_request.lower() in {
        "help the user with the current conversation.",
        "help me",
    }

    if open_loops:
        recommended_focus = open_loops[0]
    elif needs_clarification:
        recommended_focus = "Clarify the user's request before moving on."
    elif needs_docs:
        recommended_focus = "Extract the most relevant factual points from supporting documents."
    else:
        recommended_focus = "Help with the user's request by suggesting the next useful reply."

    return {
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

    merged["latest_conversation"] = user_message["text"]
    merged.pop("last_user_message", None)
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
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

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

    preview = text[:500]
    raise ValueError(f"Model did not return a JSON object. output_preview={preview!r}")


async def request_json_response(ows, instructions: str, raw_text_hook=None) -> dict:
    await send_ows_event(ows, {
        "type": "response.create",
        "response": {
            "modalities": ["text"],
            "instructions": instructions,
        },
    })
    final_text = await read_response_text(ows)
    if raw_text_hook is not None:
        await raw_text_hook(final_text)
    return parse_json_object(final_text)


async def generate_opening_suggestion(
    assistant_ows,
    session_state: dict,
    outline: str,
    debug_hook=None,
    result_debug_hook=None,
    raw_text_hook=None,
) -> dict:
    prompt = (
        "Create an opening conversation suggestion that helps with the user's request and outline.\n"
        "Return exactly one valid JSON object only. Do not include markdown, code fences, commentary, or any text outside JSON.\n"
        "Use this schema:\n"
        '{"type":"suggestion","stage":"opening","next_say":[{"ja":"...","romaji":"...","zh":"..."}],"intent":"opening"}\n\n'
        f"User request:\n{session_state.get('user_request', '')}\n\n"
        f"Outline:\n{outline}"
    )
    if debug_hook is not None:
        await debug_hook("generate_opening_suggestion", prompt)

    async def emit_raw_text(text: str):
        if raw_text_hook is not None:
            await raw_text_hook("generate_opening_suggestion", text)

    suggestion = await request_json_schema_response(
        prompt,
        "conversation_opening_suggestion",
        SUGGESTION_RESPONSE_SCHEMA,
        model=get_conversation_reasoning_model(),
        reasoning_effort=get_conversation_reasoning_effort(),
        temperature=get_conversation_reasoning_temperature(),
        raw_text_hook=emit_raw_text,
    )
    if result_debug_hook is not None:
        await result_debug_hook("generate_opening_suggestion", suggestion)
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
    debug_hook=None,
    result_debug_hook=None,
    raw_text_hook=None,
) -> dict:
    prompt = (
        "Update the conversation memory state from the previous session state and the current conversation transcript.\n"
        "This is a memory-update task only. Do not give advice to the user, do not write suggested utterances, and do not answer the conversation directly.\n"
        "If there is an obvious next reply, store it as a short item in next_actions_replace instead of writing it as prose.\n"
        "Return exactly one valid JSON object only. Do not include markdown, code fences, commentary, or any text outside JSON.\n"
        "Use empty arrays when there are no items.\n"
        "Use this schema:\n"
        '{'
        '"summary":"...",'
        '"known_info_add":["..."],'
        '"open_loops_add":["..."],'
        '"open_loops_remove":["..."],'
        '"next_actions_replace":["..."]'
        '}\n\n'
        f"Session state:\n{json.dumps(session_state, ensure_ascii=False)}\n\n"
        f"Current conversation transcript:\n{json.dumps(user_message, ensure_ascii=False)}\n\n"
        f"Last assistant suggestions:\n{json.dumps(last_assistant_reply, ensure_ascii=False)}"
    )
    if debug_hook is not None:
        await debug_hook("extract_memory_patch", prompt)

    async def emit_raw_text(text: str):
        if raw_text_hook is not None:
            await raw_text_hook("extract_memory_patch", text)

    patch = await request_json_schema_response(
        prompt,
        "conversation_memory_patch",
        MEMORY_PATCH_RESPONSE_SCHEMA,
        model=get_conversation_reasoning_model(),
        reasoning_effort=get_conversation_reasoning_effort(),
        temperature=get_conversation_reasoning_temperature(),
        raw_text_hook=emit_raw_text,
    )
    if result_debug_hook is not None:
        await result_debug_hook("extract_memory_patch", patch)
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
    debug_hook=None,
    result_debug_hook=None,
    raw_text_hook=None,
) -> dict:
    prompt = (
        "Generate the next conversation suggestions for the user.\n"
        "This is a suggestion-generation task. Even if the answer is simple, wrap it in the JSON schema below.\n"
        "Return exactly one valid JSON object only. Do not include markdown, code fences, commentary, or any text outside JSON.\n"
        "Use this schema:\n"
        '{"type":"suggestion","stage":"followup","next_say":[{"ja":"...","romaji":"...","zh":"..."}],"intent":"..."}\n\n'
        "Rules:\n"
        "- Return 1 to 3 helpful next utterances.\n"
        "- Give suggestions from the user's perspective. The user is the speaker/source labeled [user_mic].\n"
        "- Treat [peer_audio] as the other person or counterpart. Suggest what [user_mic] should say next to respond.\n"
        "- If the transcript has no source label, treat it as mixed/unknown conversation context and still advise the user.\n"
        "- Keep them aligned with the user's request, current transcript, open loops, and next_actions.\n"
        "- If next_actions already contains a clear action, prioritize that action.\n"
        "- Use concise, natural Japanese in ja.\n"
        "- Provide romaji and a concise Chinese meaning for each suggested utterance in next_say[].zh.\n"
        "- Do not add markdown.\n\n"
        f"Session state:\n{json.dumps(session_state, ensure_ascii=False)}\n\n"
        f"Current conversation transcript:\n{json.dumps(user_message, ensure_ascii=False)}"
    )
    if debug_hook is not None:
        await debug_hook("generate_assistant_reply", prompt)

    async def emit_raw_text(text: str):
        if raw_text_hook is not None:
            await raw_text_hook("generate_assistant_reply", text)

    suggestion = await request_json_schema_response(
        prompt,
        "conversation_followup_suggestion",
        SUGGESTION_RESPONSE_SCHEMA,
        model=get_conversation_reasoning_model(),
        reasoning_effort=get_conversation_reasoning_effort(),
        temperature=get_conversation_reasoning_temperature(),
        raw_text_hook=emit_raw_text,
    )
    if result_debug_hook is not None:
        await result_debug_hook("generate_assistant_reply", suggestion)
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


def normalize_audio_source(source: str | None) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]", "_", (source or "").strip())
    return normalized or "mixed"


def decode_audio_chunk_message(body: dict) -> tuple[str, bytes] | None:
    if body.get("type") != "audio_chunk":
        return None
    if str(body.get("format") or "pcm16").lower() != "pcm16":
        return None
    data = str(body.get("data") or "")
    if not data:
        return None
    try:
        chunk = base64.b64decode(data, validate=True)
    except Exception:
        return None
    return normalize_audio_source(str(body.get("source") or "")), chunk


def build_prompt_debug_payload(stage: str, prompt: str, trace_id: str = "") -> dict:
    normalized = prompt.strip()
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    headline = lines[0] if lines else ""
    preview = normalized[:1200]
    return {
        "type": "debug_prompt",
        "stage": stage,
        "trace_id": trace_id,
        "headline": headline,
        "prompt_chars": len(normalized),
        "prompt_lines": len(lines),
        "preview": preview,
        "timestamp": now_iso(),
    }


def build_response_debug_payload(stage: str, payload: dict, trace_id: str = "") -> dict:
    return {
        "type": "debug_response",
        "stage": stage,
        "trace_id": trace_id,
        "payload": payload,
        "timestamp": now_iso(),
    }


def build_raw_response_debug_payload(stage: str, text: str, trace_id: str = "") -> dict:
    normalized = text.strip()
    return {
        "type": "debug_raw_response",
        "stage": stage,
        "trace_id": trace_id,
        "raw_chars": len(normalized),
        "preview": normalized[:1200],
        "timestamp": now_iso(),
    }


def build_state_debug_payload(
    stage: str,
    session_state: dict,
    user_message: dict | None = None,
    trace_id: str = "",
) -> dict:
    payload = {
        "session_state": session_state,
    }
    if user_message is not None:
        payload["user_message"] = user_message
    return {
        "type": "debug_state",
        "stage": stage,
        "trace_id": trace_id,
        "payload": payload,
        "timestamp": now_iso(),
    }


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
    queue: asyncio.Queue[tuple[str, str, str, list[str]]],
    transcription_language: str,
    translation_language: str,
):
    source_label = language_label(transcription_language)
    target_label = language_label(translation_language)
    print("Using text translation model:", get_translation_text_model(), f"for {source_label} -> {target_label}")
    while True:
        segment_id, transcript, source, context = await queue.get()
        try:
            payload = {
                "type": "translation",
                "segment_id": segment_id,
                "transcript": transcript,
                "translation": "",
            }
            if source:
                payload["source"] = source
            try:
                payload["translation"] = await request_text_translation(
                    transcript,
                    source_label,
                    target_label,
                    context=context,
                )
            except Exception as exc:
                payload["error"] = repr(exc)

            sent = await safe_send_envelope(ws, payload)
            if not sent:
                return
        finally:
            queue.task_done()


async def send_transcript_and_translation(
    ws: WebSocket,
    transcript: str,
    on_final_transcript,
    translation_queue: asyncio.Queue[tuple[str, str, str, list[str]]] | None = None,
    direct_translation: str = "",
    source: str = "",
    translation_context: list[str] | None = None,
    segment_id: str = "",
):
    transcript = transcript.strip()
    if not transcript:
        return
    segment_id = segment_id.strip() or make_segment_id()

    payload = {
        "type": "transcript_final",
        "segment_id": segment_id,
        "transcript": transcript,
        "translation": "",
        "next_say": [],
        "intent": "",
    }
    if source:
        payload["source"] = source
    sent = await safe_send_envelope(ws, payload)
    if not sent or not transcript:
        return

    if direct_translation.strip():
        translation_payload = {
            "type": "translation",
            "segment_id": segment_id,
            "transcript": transcript,
            "translation": direct_translation.strip(),
        }
        if source:
            translation_payload["source"] = source
        sent = await safe_send_envelope(ws, translation_payload)
        if not sent:
            return
    elif translation_queue is not None:
        context_snapshot = (
            list((translation_context or [])[-TRANSLATION_CONTEXT_SEGMENTS:])
            if TRANSLATION_CONTEXT_SEGMENTS > 0
            else []
        )
        await translation_queue.put((segment_id, transcript, source, context_snapshot))

    if translation_context is not None and TRANSLATION_CONTEXT_SEGMENTS > 0:
        context_text = f"[{source}] {transcript}" if source else transcript
        translation_context.append(context_text)
        del translation_context[:-TRANSLATION_CONTEXT_SEGMENTS]

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
            if outline and not merged_state.get("user_request"):
                merged_state["user_request"] = build_default_user_request(outline)
            save_session_state(merged_state)
            session_state.clear()
            session_state.update(merged_state)

            suggestion = await generate_assistant_reply(
                assistant_ows,
                session_state,
                user_message,
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
    client_session_id = ""
    session_registered = False
    try:
        if STT_BACKEND == "realtime":
            transcribe_ows = await openai_realtime_connect()
        elif STT_BACKEND not in {"local_whisper", "soniox"}:
            print("Unsupported STT_BACKEND:", STT_BACKEND)
            await ws.close()
            return
    except Exception as exc:
        print("OpenAI realtime connect failed:", repr(exc))
        await ws.close()
        return

    translation_queue: asyncio.Queue[tuple[str, str, str, list[str]]] = asyncio.Queue()
    translation_context: list[str] = []
    translation_task = None
    source_transcribers = {}
    source_transcriber_tasks: list[asyncio.Task] = []
    pending_memory_transcripts: list[dict] = []
    pending_memory_chars = 0
    memory_lock = asyncio.Lock()
    memory_update_lock = asyncio.Lock()
    assistant_request_lock = asyncio.Lock()
    suggestion_lock = asyncio.Lock()
    finalize_lock = asyncio.Lock()
    suggestion_task = None
    memory_flush_tasks: set[asyncio.Task] = set()
    last_assistant_reply: list[str] = []
    flush_pending_memory = None
    full_finalize_requested = False
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

        client_session_id = sanitize_session_id(str(body.get("session_id") or f"legacy_{uuid.uuid4().hex}"))
        async with ACTIVE_CLIENT_SESSION_LOCK:
            if client_session_id in ACTIVE_CLIENT_SESSION_IDS:
                await safe_send_envelope(ws, {
                    "type": "error",
                    "stage": "session",
                    "code": "session_already_active",
                    "message": "session_id already has an active connection",
                    "session_id": client_session_id,
                })
                await ws.close(code=1008)
                return
            ACTIVE_CLIENT_SESSION_IDS.add(client_session_id)
            session_registered = True

        outline = str(body.get("outline") or "").strip()
        transcription_language = str(body.get("transcription_language") or "auto").strip()
        translation_language = str(body.get("translation_language") or "zh-Hans").strip()
        source_language_code = soniox_target_language(transcription_language)
        target_language_code = soniox_target_language(translation_language)
        use_soniox_direct_translation = (
            STT_BACKEND == "soniox"
            and SONIOX_TRANSLATION_BACKEND == "soniox"
            and target_language_code is not None
            and (source_language_code is None or target_language_code != source_language_code)
        )

        if transcribe_ows is not None:
            await send_ows_event(
                transcribe_ows,
                build_transcribe_session_update(transcription_language),
            )
        async def emit_prompt_debug(stage: str, prompt: str, trace_id: str = ""):
            if not CONVERSATION_DEBUG_PROMPTS_ENABLED:
                return
            await safe_send_envelope(ws, build_prompt_debug_payload(stage, prompt, trace_id))

        async def emit_response_debug(stage: str, payload: dict, trace_id: str = ""):
            if not CONVERSATION_DEBUG_PROMPTS_ENABLED:
                return
            await safe_send_envelope(ws, build_response_debug_payload(stage, payload, trace_id))

        async def emit_raw_response_debug(stage: str, text: str, trace_id: str = ""):
            if not CONVERSATION_DEBUG_PROMPTS_ENABLED:
                return
            await safe_send_envelope(ws, build_raw_response_debug_payload(stage, text, trace_id))

        async def emit_audio_debug(stage: str, source: str = "", payload: dict | None = None):
            if not CONVERSATION_DEBUG_AUDIO_ENABLED:
                return
            body = {
                "type": "debug_audio",
                "stage": stage,
                "source": source,
                "payload": payload or {},
                "timestamp": now_iso(),
            }
            await safe_send_envelope(ws, body)

        session_state = load_session_state(client_session_id)
        is_new_session = session_state is None
        if is_new_session:
            session_state = build_initial_session_state(
                client_session_id,
                "",
                build_default_user_request(outline),
            )
            save_session_state(session_state)
        last_assistant_reply = list(session_state.get("next_actions") or [])

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
                opening_trace_id = f"trace_{uuid.uuid4().hex[:12]}"
                async with assistant_request_lock:
                    opening_suggestion = await generate_opening_suggestion(
                        assistant_ows,
                        session_state,
                        outline,
                        debug_hook=lambda stage, prompt: emit_prompt_debug(stage, prompt, opening_trace_id),
                        result_debug_hook=lambda stage, payload: emit_response_debug(stage, payload, opening_trace_id),
                        raw_text_hook=lambda stage, text: emit_raw_response_debug(stage, text, opening_trace_id),
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

        async def flush_memory_batch(batch: list[dict], restore_on_failure: bool = True) -> bool:
            nonlocal pending_memory_chars
            batch = [
                item for item in batch
                if isinstance(item, dict) and str(item.get("text") or "").strip()
            ]
            if not batch:
                return False

            user_message = build_user_message(client_session_id, format_transcript_items(batch))
            append_event(client_session_id, user_message)
            memory_trace_id = f"trace_{uuid.uuid4().hex[:12]}"

            try:
                async with memory_update_lock:
                    async with assistant_request_lock:
                        memory_patch = await extract_memory_patch(
                            assistant_ows,
                            session_state,
                            user_message,
                            last_assistant_reply,
                            debug_hook=lambda stage, prompt: emit_prompt_debug(stage, prompt, memory_trace_id),
                            result_debug_hook=lambda stage, payload: emit_response_debug(stage, payload, memory_trace_id),
                            raw_text_hook=lambda stage, text: emit_raw_response_debug(stage, text, memory_trace_id),
                        )
                    append_event(
                        client_session_id,
                        build_memory_patch_event(client_session_id, memory_patch),
                    )

                    merged_state = merge_session_state(session_state, memory_patch, user_message)
                    if outline and not merged_state.get("user_request"):
                        merged_state["user_request"] = build_default_user_request(outline)
                    save_session_state(merged_state)
                    session_state.clear()
                    session_state.update(merged_state)
                    if CONVERSATION_DEBUG_PROMPTS_ENABLED:
                        await safe_send_envelope(
                            ws,
                            build_state_debug_payload("memory_flush", dict(session_state), user_message, memory_trace_id),
                        )
                return True
            except Exception as exc:
                append_event(
                    client_session_id,
                    build_error_event(client_session_id, "memory_flush", repr(exc)),
                )
                if restore_on_failure:
                    async with memory_lock:
                        pending_memory_transcripts[:0] = batch
                        pending_memory_chars += sum(len(str(item.get("text") or "")) for item in batch)
                await safe_send_envelope(ws, {
                    "type": "error",
                    "stage": "memory_flush",
                    "message": "memory flush failed",
                    "reason": repr(exc),
                    "trace_id": memory_trace_id,
                })
                return False

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

            return await flush_memory_batch(batch, restore_on_failure=True)

        async def generate_manual_suggestion():
            nonlocal last_assistant_reply, pending_memory_chars
            async with suggestion_lock:
                async with memory_update_lock:
                    async with memory_lock:
                        pending_batch = list(pending_memory_transcripts)
                        if pending_batch:
                            pending_memory_transcripts.clear()
                            pending_memory_chars = 0

                    if pending_batch:
                        base_text = latest_conversation_text(session_state)
                        extra_text = "\n".join(format_transcript_items(pending_batch)).strip()
                        user_message = build_user_message(
                            client_session_id,
                            [item for item in [base_text, extra_text] if item],
                        )
                    else:
                        user_message = build_user_message(
                            client_session_id,
                            [latest_conversation_text(session_state)],
                        )

                try:
                    suggestion_trace_id = f"trace_{uuid.uuid4().hex[:12]}"
                    if CONVERSATION_DEBUG_PROMPTS_ENABLED:
                        await safe_send_envelope(
                            ws,
                            build_state_debug_payload("pre_suggestion", dict(session_state), user_message, suggestion_trace_id),
                        )
                    async with assistant_request_lock:
                        suggestion = await generate_assistant_reply(
                            assistant_ows,
                            session_state,
                            user_message,
                            debug_hook=lambda stage, prompt: emit_prompt_debug(stage, prompt, suggestion_trace_id),
                            result_debug_hook=lambda stage, payload: emit_response_debug(stage, payload, suggestion_trace_id),
                            raw_text_hook=lambda stage, text: emit_raw_response_debug(stage, text, suggestion_trace_id),
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
                    if pending_batch:
                        task = asyncio.create_task(flush_memory_batch(pending_batch, restore_on_failure=True))
                        memory_flush_tasks.add(task)
                        task.add_done_callback(memory_flush_tasks.discard)
                except Exception as exc:
                    append_event(
                        client_session_id,
                        build_error_event(client_session_id, "manual_suggestion", repr(exc)),
                    )
                    if pending_batch:
                        async with memory_lock:
                            pending_memory_transcripts[:0] = pending_batch
                            pending_memory_chars += sum(len(str(item.get("text") or "")) for item in pending_batch)

        async def handle_transcript_final(
            transcript: str,
            source: str = "",
            segment_id: str = "",
        ):
            nonlocal pending_memory_chars
            append_recent_conversation(session_state, transcript, source)
            save_session_state(session_state)

            async def record_transcript(text: str):
                nonlocal pending_memory_chars
                async with memory_lock:
                    pending_memory_transcripts.append({
                        "source": normalize_audio_source(source),
                        "text": text,
                        "timestamp": now_iso(),
                    })
                    pending_memory_chars += len(text)
                await flush_pending_memory(force=False)

            await send_transcript_and_translation(
                ws,
                transcript,
                record_transcript,
                translation_queue=translation_queue,
                source=source,
                translation_context=translation_context,
                segment_id=segment_id,
            )

        async def handle_soniox_final(transcript: str, translation: str, source: str = ""):
            append_recent_conversation(session_state, transcript, source)
            save_session_state(session_state)

            async def record_transcript(text: str):
                nonlocal pending_memory_chars
                async with memory_lock:
                    pending_memory_transcripts.append({
                        "source": normalize_audio_source(source),
                        "text": text,
                        "timestamp": now_iso(),
                    })
                    pending_memory_chars += len(text)
                await flush_pending_memory(force=False)

            await send_transcript_and_translation(
                ws,
                transcript,
                record_transcript,
                translation_queue=None if use_soniox_direct_translation else translation_queue,
                direct_translation=translation if use_soniox_direct_translation else "",
                source=source,
                translation_context=translation_context,
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

        async def handle_local_delta(delta: str, source: str = ""):
            payload = {
                "type": "transcript_delta",
                "delta": delta,
            }
            if source:
                payload["source"] = source
            await safe_send_envelope(ws, payload)

        def make_source_delta_handler(source: str):
            async def on_delta(delta: str):
                await handle_local_delta(delta, source)
            return on_delta

        def make_source_final_handler(source: str):
            async def on_final(transcript: str):
                await handle_transcript_final(transcript, source)
            return on_final

        def make_source_soniox_final_handler(source: str):
            async def on_final(transcript: str, translation: str):
                await handle_soniox_final(transcript, translation, source)
            return on_final

        async def get_source_transcriber(source: str):
            source = normalize_audio_source(source)
            existing_transcriber = source_transcribers.get(source)
            existing_is_closed = bool(getattr(existing_transcriber, "is_closed", False))
            if existing_transcriber is not None and not existing_is_closed:
                return existing_transcriber
            if existing_transcriber is not None and existing_is_closed:
                source_transcribers.pop(source, None)
            if STT_BACKEND == "local_whisper":
                source_transcriber = LocalWhisperStream(
                    transcription_language,
                    make_source_delta_handler(source),
                    make_source_final_handler(source),
                )
            elif STT_BACKEND == "soniox":
                source_transcriber = SonioxRealtimeStream(
                    transcription_language,
                    make_source_delta_handler(source),
                    make_source_soniox_final_handler(source),
                    translation_language=translation_language if use_soniox_direct_translation else "",
                )
            else:
                return None

            await source_transcriber.start()
            source_transcribers[source] = source_transcriber
            await emit_audio_debug("transcriber_start", source, {"backend": STT_BACKEND})
            task = asyncio.create_task(source_transcriber.run())
            source_transcriber_tasks.append(task)

            def report_transcriber_error(done_task: asyncio.Task):
                if done_task.cancelled():
                    return
                exc = done_task.exception()
                if exc is None:
                    return
                print(f"{STT_BACKEND} transcriber failed for source={source}:", repr(exc))
                asyncio.create_task(safe_send_envelope(ws, {
                    "type": "error",
                    "stage": "stt_transcriber",
                    "source": source,
                    "message": "transcriber task failed",
                    "reason": repr(exc),
                }))

            task.add_done_callback(report_transcriber_error)
            return source_transcriber

        async def route_audio_bytes(source: str, chunk: bytes):
            if not chunk:
                return
            source = normalize_audio_source(source)
            await emit_audio_debug("audio_chunk_received", source, {
                "backend": STT_BACKEND,
                "bytes": len(chunk),
            })
            if transcribe_ows is not None and not transcribe_closed:
                payload = {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("utf-8"),
                }
                try:
                    await transcribe_ows.send(json.dumps(payload))
                except websockets.exceptions.ConnectionClosed as exc:
                    await notify_stt_connection_closed(repr(exc))
                return

            source_transcriber = await get_source_transcriber(source)
            if source_transcriber is not None:
                await source_transcriber.add_audio(chunk)

        async def reset_source_transcribers(source: str = ""):
            if transcribe_ows is not None and not transcribe_closed:
                try:
                    await transcribe_ows.send(json.dumps({"type": "input_audio_buffer.clear"}))
                except websockets.exceptions.ConnectionClosed as exc:
                    await notify_stt_connection_closed(repr(exc))
                return

            if source:
                source_transcriber = source_transcribers.get(normalize_audio_source(source))
                if source_transcriber is not None:
                    await source_transcriber.reset()
                return

            for source_transcriber in list(source_transcribers.values()):
                await source_transcriber.reset()

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

        async def finalize_session(
            reason: str = "",
            pending_transcript: str = "",
            pending_segment_id: str = "",
        ):
            nonlocal full_finalize_requested
            async with finalize_lock:
                is_full_finalize = reason in {"page_exit", "finalize_exit", "exit_finalize"}
                if is_full_finalize:
                    full_finalize_requested = True

                pending_transcript = pending_transcript.strip()
                if pending_transcript:
                    await handle_transcript_final(
                        pending_transcript,
                        "mixed",
                        pending_segment_id,
                    )

                if is_full_finalize and suggestion_task is not None:
                    await asyncio.gather(suggestion_task, return_exceptions=True)
                if is_full_finalize and memory_flush_tasks:
                    await asyncio.wait(memory_flush_tasks, timeout=3)
                if is_full_finalize and flush_pending_memory is not None:
                    await flush_pending_memory(force=True)
                if translation_task is not None:
                    await translation_queue.join()
                await safe_send_envelope(ws, {
                    "type": "finalize_complete",
                    "reason": reason,
                })

        async def handle_client_messages():
            nonlocal suggestion_task
            while True:
                msg = await safe_receive_message(ws)
                if msg is None:
                    return
                if "bytes" in msg and msg["bytes"] is not None:
                    await route_audio_bytes("mixed", msg["bytes"])
                    continue

                if "text" not in msg or msg["text"] is None:
                    continue

                try:
                    body = json.loads(msg["text"])
                except Exception:
                    continue

                body_type = body.get("type")
                if body_type == "audio_chunk":
                    decoded = decode_audio_chunk_message(body)
                    if decoded is None:
                        await safe_send_envelope(ws, {
                            "type": "error",
                            "stage": "audio_chunk",
                            "message": "invalid audio_chunk",
                        })
                        continue
                    source, chunk = decoded
                    await route_audio_bytes(source, chunk)
                    continue

                if body_type == "reset":
                    await reset_source_transcribers(str(body.get("source") or ""))
                    continue

                if body_type == "finalize":
                    await finalize_session(
                        str(body.get("reason") or ""),
                        str(body.get("pending_transcript") or ""),
                        str(body.get("pending_segment_id") or ""),
                    )
                    continue

                if body_type == "request_suggestion":
                    if suggestion_task is None or suggestion_task.done():
                        suggestion_task = asyncio.create_task(generate_manual_suggestion())
                    continue

                if body_type == "debug_text":
                    if not CONVERSATION_DEBUG_TEXT_ENABLED:
                        await safe_send_envelope(ws, {
                            "type": "error",
                            "stage": "debug_text",
                            "message": "debug_text is disabled",
                        })
                        continue

                    debug_text = str(body.get("text") or "").strip()
                    if not debug_text:
                        await safe_send_envelope(ws, {
                            "type": "error",
                            "stage": "debug_text",
                            "message": "text is required",
                        })
                        continue

                    await handle_transcript_final(debug_text)

        if STT_BACKEND == "realtime":
            await asyncio.gather(
                handle_client_messages(),
                relay_events_to_client(),
            )
        elif STT_BACKEND == "local_whisper":
            try:
                await handle_client_messages()
            finally:
                for source_transcriber in list(source_transcribers.values()):
                    await source_transcriber.close()
                if source_transcriber_tasks:
                    await asyncio.gather(*source_transcriber_tasks, return_exceptions=True)
        else:
            try:
                await handle_client_messages()
            finally:
                for source_transcriber in list(source_transcribers.values()):
                    await source_transcriber.close()
                if source_transcriber_tasks:
                    await asyncio.gather(*source_transcriber_tasks, return_exceptions=True)
    except WebSocketDisconnect:
        pass
    finally:
        if session_registered:
            async with ACTIVE_CLIENT_SESSION_LOCK:
                ACTIVE_CLIENT_SESSION_IDS.discard(client_session_id)
        if full_finalize_requested and suggestion_task is not None:
            await asyncio.gather(suggestion_task, return_exceptions=True)
        if full_finalize_requested and memory_flush_tasks:
            await asyncio.wait(memory_flush_tasks, timeout=3)
        if full_finalize_requested and flush_pending_memory is not None:
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
