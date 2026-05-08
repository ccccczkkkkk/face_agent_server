import asyncio
import json
import os
import inspect
import time
from typing import Awaitable, Callable

import websockets


TranscriptCallback = Callable[[str], Awaitable[None] | None]
UtteranceFinalCallback = Callable[[str, str], Awaitable[None] | None]

SONIOX_WS_URL = os.getenv("SONIOX_WS_URL", "wss://stt-rt.soniox.com/transcribe-websocket")
SONIOX_MODEL = os.getenv("SONIOX_MODEL", "stt-rt-v4")
SONIOX_AUDIO_FORMAT = os.getenv("SONIOX_AUDIO_FORMAT", "pcm_s16le")
SONIOX_SAMPLE_RATE = int(os.getenv("SONIOX_SAMPLE_RATE", "24000"))
SONIOX_NUM_CHANNELS = int(os.getenv("SONIOX_NUM_CHANNELS", "1"))
SONIOX_ENABLE_ENDPOINT_DETECTION = os.getenv("SONIOX_ENABLE_ENDPOINT_DETECTION", "true").strip().lower() == "true"
SONIOX_MAX_ENDPOINT_DELAY_MS = int(os.getenv("SONIOX_MAX_ENDPOINT_DELAY_MS", "1200"))
SONIOX_KEEPALIVE_INTERVAL_MS = int(os.getenv("SONIOX_KEEPALIVE_INTERVAL_MS", "8000"))
SONIOX_TRAILING_SILENCE_MS = int(os.getenv("SONIOX_TRAILING_SILENCE_MS", "300"))
SONIOX_FORCE_FINALIZE_AFTER_CHARS = int(os.getenv("SONIOX_FORCE_FINALIZE_AFTER_CHARS", "120"))

_LANGUAGE_HINTS = {
    "ja": "ja",
    "en": "en",
    "zh": "zh",
    "zh-Hans": "zh",
    "ko": "ko",
}


def soniox_language_hint(language: str | None) -> str | None:
    return _LANGUAGE_HINTS.get((language or "").strip())


def soniox_target_language(language: str | None) -> str | None:
    return _LANGUAGE_HINTS.get((language or "").strip())


class SonioxRealtimeStream:
    def __init__(
        self,
        transcription_language: str | None,
        on_delta: TranscriptCallback,
        on_final: UtteranceFinalCallback,
        translation_language: str | None = None,
    ):
        self.transcription_language = (transcription_language or "auto").strip()
        self.translation_language = (translation_language or "").strip()
        self.on_delta = on_delta
        self.on_final = on_final
        self._ws = None
        self._receiver_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._closed = False
        self._started = False
        self._last_audio_or_keepalive = time.monotonic()
        self._final_text = ""
        self._final_translation_text = ""
        self._partial_text = ""
        self._finalize_requested = False

    async def start(self) -> None:
        if self._started:
            return

        api_key = os.environ["SONIOX_API_KEY"]
        self._ws = await websockets.connect(
            SONIOX_WS_URL,
            max_size=2**24,
        )
        await self._ws.send(json.dumps(self._build_start_message(api_key)))
        self._receiver_task = asyncio.create_task(self._receiver_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        self._started = True

    async def add_audio(self, chunk: bytes) -> None:
        if self._closed or self._ws is None or not chunk:
            return
        self._last_audio_or_keepalive = time.monotonic()
        await self._ws.send(chunk)

    async def reset(self) -> None:
        if self._closed or self._ws is None:
            return
        await self._send_finalize()

    async def run(self) -> None:
        if self._receiver_task is not None:
            await self._receiver_task

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            await asyncio.gather(self._keepalive_task, return_exceptions=True)
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._receiver_task is not None:
            await asyncio.gather(self._receiver_task, return_exceptions=True)

    def _build_start_message(self, api_key: str) -> dict:
        payload = {
            "api_key": api_key,
            "model": SONIOX_MODEL,
            "audio_format": SONIOX_AUDIO_FORMAT,
            "sample_rate": SONIOX_SAMPLE_RATE,
            "num_channels": SONIOX_NUM_CHANNELS,
        }
        if SONIOX_ENABLE_ENDPOINT_DETECTION:
            payload["enable_endpoint_detection"] = True
            payload["max_endpoint_delay_ms"] = SONIOX_MAX_ENDPOINT_DELAY_MS

        language_hint = soniox_language_hint(self.transcription_language)
        if language_hint:
            payload["language_hints"] = [language_hint]
            payload["language_hints_strict"] = True

        target_language = soniox_target_language(self.translation_language)
        if target_language and target_language != language_hint:
            payload["translation"] = {
                "type": "one_way",
                "target_language": target_language,
            }

        return payload

    async def _send_finalize(self) -> None:
        if self._ws is None:
            return
        self._finalize_requested = True
        self._last_audio_or_keepalive = time.monotonic()
        await self._ws.send(json.dumps({
            "type": "finalize",
            "trailing_silence_ms": SONIOX_TRAILING_SILENCE_MS,
        }))

    async def _keepalive_loop(self) -> None:
        try:
            interval_s = max(1.0, SONIOX_KEEPALIVE_INTERVAL_MS / 1000.0)
            while not self._closed and self._ws is not None:
                await asyncio.sleep(interval_s)
                if time.monotonic() - self._last_audio_or_keepalive < interval_s:
                    continue
                await self._ws.send(json.dumps({"type": "keepalive"}))
                self._last_audio_or_keepalive = time.monotonic()
        except asyncio.CancelledError:
            pass

    async def _receiver_loop(self) -> None:
        if self._ws is None:
            return

        async for raw in self._ws:
            if isinstance(raw, bytes):
                continue
            event = json.loads(raw)
            error_message = str(event.get("error_message") or "").strip()
            if error_message:
                raise RuntimeError(error_message)

            tokens = event.get("tokens") or []
            if tokens:
                await self._process_tokens(tokens)

            if event.get("finished"):
                break

    async def _process_tokens(self, tokens: list[dict]) -> None:
        current_nonfinal = ""
        should_emit_final = False

        for token in tokens:
            text = str(token.get("text") or "")
            if not text:
                continue
            is_final = bool(token.get("is_final"))
            translation_status = str(token.get("translation_status") or "none")

            if is_final:
                if text in {"<end>", "<fin>"}:
                    should_emit_final = True
                    continue
                if translation_status == "translation":
                    self._final_translation_text += text
                else:
                    self._final_text += text
            else:
                if translation_status != "translation":
                    current_nonfinal += text

        candidate_partial = (self._final_text + current_nonfinal).strip()
        if candidate_partial:
            previous = self._partial_text
            if candidate_partial.startswith(previous):
                delta = candidate_partial[len(previous):]
                if delta:
                    await self._call(self.on_delta, delta)
            self._partial_text = candidate_partial
            if (
                SONIOX_FORCE_FINALIZE_AFTER_CHARS > 0
                and len(candidate_partial) >= SONIOX_FORCE_FINALIZE_AFTER_CHARS
                and not self._finalize_requested
            ):
                await self._send_finalize()

        if should_emit_final:
            final_text = self._final_text.strip()
            final_translation_text = self._final_translation_text.strip()
            self._final_text = ""
            self._final_translation_text = ""
            self._partial_text = ""
            self._finalize_requested = False
            if final_text:
                await self._call_final(self.on_final, final_text, final_translation_text)

    async def _call(self, callback: TranscriptCallback, text: str) -> None:
        result = callback(text)
        if inspect.isawaitable(result):
            await result

    async def _call_final(self, callback: UtteranceFinalCallback, transcript: str, translation: str) -> None:
        result = callback(transcript, translation)
        if inspect.isawaitable(result):
            await result
