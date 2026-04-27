import asyncio
import gc
import inspect
import os
from typing import Awaitable, Callable

try:
    from ws_utils import safe_receive_message
except ModuleNotFoundError:
    from face_server.ws_utils import safe_receive_message


WHISPER_TARGET_SAMPLE_RATE = 16000
WHISPER_IDLE_UNLOAD_SECONDS = 60.0

TranscriptCallback = Callable[[str], Awaitable[None] | None]

_WHISPER_MODEL = None
_WHISPER_ACTIVE_STREAMS = 0
_WHISPER_MODEL_LOCK = asyncio.Lock()
_WHISPER_UNLOAD_TASK: asyncio.Task | None = None


def whisper_language_code(language: str | None) -> str | None:
    normalized = (language or "").strip()
    if normalized in ("", "auto"):
        return None
    if normalized in ("zh", "zh-Hans"):
        return "zh"
    if normalized in ("ja", "en", "ko"):
        return normalized
    return None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _build_whisper_model():
    from faster_whisper import WhisperModel

    model_size = os.getenv("WHISPER_MODEL_SIZE", "medium")
    device = os.getenv("WHISPER_DEVICE", "cuda")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    print(
        "Loading faster-whisper model:",
        f"size={model_size}",
        f"device={device}",
        f"compute_type={compute_type}",
    )
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def _get_loaded_whisper_model():
    if _WHISPER_MODEL is None:
        raise RuntimeError("Whisper model is not loaded")
    return _WHISPER_MODEL


async def acquire_whisper_model():
    global _WHISPER_MODEL, _WHISPER_ACTIVE_STREAMS, _WHISPER_UNLOAD_TASK
    async with _WHISPER_MODEL_LOCK:
        _WHISPER_ACTIVE_STREAMS += 1
        if _WHISPER_UNLOAD_TASK is not None:
            _WHISPER_UNLOAD_TASK.cancel()
            _WHISPER_UNLOAD_TASK = None
        if _WHISPER_MODEL is None:
            _WHISPER_MODEL = await asyncio.to_thread(_build_whisper_model)
        return _WHISPER_MODEL


async def release_whisper_model():
    global _WHISPER_ACTIVE_STREAMS, _WHISPER_UNLOAD_TASK
    async with _WHISPER_MODEL_LOCK:
        if _WHISPER_ACTIVE_STREAMS > 0:
            _WHISPER_ACTIVE_STREAMS -= 1
        if _WHISPER_ACTIVE_STREAMS == 0 and _WHISPER_UNLOAD_TASK is None:
            _WHISPER_UNLOAD_TASK = asyncio.create_task(_delayed_unload_whisper_model())


async def _delayed_unload_whisper_model():
    global _WHISPER_MODEL, _WHISPER_UNLOAD_TASK
    try:
        await asyncio.sleep(_env_float("WHISPER_IDLE_UNLOAD_SECONDS", WHISPER_IDLE_UNLOAD_SECONDS))
        async with _WHISPER_MODEL_LOCK:
            if _WHISPER_ACTIVE_STREAMS != 0 or _WHISPER_MODEL is None:
                return
            print("Unloading faster-whisper model after idle timeout")
            _WHISPER_MODEL = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except asyncio.CancelledError:
        pass
    finally:
        async with _WHISPER_MODEL_LOCK:
            if _WHISPER_ACTIVE_STREAMS == 0:
                _WHISPER_UNLOAD_TASK = None


def pcm16_rms(pcm_bytes: bytes) -> float:
    import numpy as np

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    samples /= 32768.0
    return float(np.sqrt(np.mean(samples * samples)))


def pcm16_to_float32_audio(pcm_bytes: bytes, sample_rate: int):
    import numpy as np

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return samples
    samples /= 32768.0
    if sample_rate == WHISPER_TARGET_SAMPLE_RATE:
        return samples

    duration = samples.size / float(sample_rate)
    target_size = max(1, int(duration * WHISPER_TARGET_SAMPLE_RATE))
    old_positions = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    new_positions = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.interp(new_positions, old_positions, samples).astype(np.float32)


class LocalWhisperStream:
    def __init__(
        self,
        transcription_language: str | None,
        on_delta: TranscriptCallback,
        on_final: TranscriptCallback,
    ):
        self.transcription_language = whisper_language_code(transcription_language)
        self.on_delta = on_delta
        self.on_final = on_final
        self.sample_rate = _env_int("WHISPER_SAMPLE_RATE", 24000)
        self.window_seconds = _env_float("WHISPER_WINDOW_SECONDS", 6.0)
        self.step_seconds = _env_float("WHISPER_STEP_SECONDS", 1.5)
        self.overlap_seconds = _env_float("WHISPER_OVERLAP_SECONDS", 1.0)
        self.final_silence_ms = _env_int("WHISPER_FINAL_SILENCE_MS", 700)
        self.silence_threshold = _env_float("WHISPER_SILENCE_THRESHOLD", 0.01)
        self._buffer = bytearray()
        self._lock = asyncio.Lock()
        self._transcribe_lock = asyncio.Lock()
        self._last_partial = ""
        self._silence_ms = 0.0
        self._has_voice = False
        self._closed = False
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        await acquire_whisper_model()
        self._started = True

    async def add_audio(self, chunk: bytes) -> None:
        if self._closed or not chunk:
            return

        chunk_duration_ms = len(chunk) / 2 / float(self.sample_rate) * 1000.0
        rms = pcm16_rms(chunk)

        should_finalize = False
        async with self._lock:
            self._buffer.extend(chunk)
            if rms < self.silence_threshold:
                self._silence_ms += chunk_duration_ms
            else:
                self._silence_ms = 0.0
                self._has_voice = True

            if self._has_voice and self._silence_ms >= self.final_silence_ms:
                should_finalize = True

        if should_finalize:
            await self._emit_final()

    async def reset(self) -> None:
        async with self._lock:
            self._buffer.clear()
            self._last_partial = ""
            self._silence_ms = 0.0
            self._has_voice = False

    async def close(self) -> None:
        self._closed = True
        async with self._lock:
            has_voice = self._has_voice and bool(self._buffer)
        if has_voice:
            await self._emit_final()
        if self._started:
            self._started = False
            await release_whisper_model()

    async def run(self) -> None:
        while not self._closed:
            await asyncio.sleep(self.step_seconds)
            audio = await self._recent_audio()
            if not audio:
                continue

            partial_text = await self._transcribe(audio)
            if not partial_text:
                continue

            async with self._lock:
                previous = self._last_partial
                if partial_text.startswith(previous):
                    delta = partial_text[len(previous):]
                    self._last_partial = partial_text
                else:
                    delta = ""

            if delta:
                await self._call(self.on_delta, delta)

    async def _recent_audio(self) -> bytes:
        bytes_per_second = self.sample_rate * 2
        min_bytes = int(max(0.5, self.overlap_seconds) * bytes_per_second)
        max_bytes = int(self.window_seconds * bytes_per_second)
        async with self._lock:
            if not self._has_voice or len(self._buffer) < min_bytes:
                return b""
            return bytes(self._buffer[-max_bytes:])

    async def _emit_final(self) -> None:
        async with self._transcribe_lock:
            async with self._lock:
                audio = bytes(self._buffer)
                self._buffer.clear()
                self._last_partial = ""
                self._silence_ms = 0.0
                self._has_voice = False

            if not audio:
                return

            final_text = await self._transcribe(audio)
            if final_text:
                await self._call(self.on_final, final_text)

    async def _transcribe(self, pcm_bytes: bytes) -> str:
        audio = pcm16_to_float32_audio(pcm_bytes, self.sample_rate)
        if audio.size == 0:
            return ""

        return await asyncio.to_thread(self._transcribe_sync, audio)

    def _transcribe_sync(self, audio) -> str:
        segments, _ = _get_loaded_whisper_model().transcribe(
            audio,
            language=self.transcription_language,
            vad_filter=False,
            beam_size=1,
        )
        return "".join(segment.text for segment in segments).strip()

    async def _call(self, callback: TranscriptCallback, text: str) -> None:
        result = callback(text)
        if inspect.isawaitable(result):
            await result


async def forward_audio_from_client_to_whisper(ws, transcriber: LocalWhisperStream):
    while True:
        msg = await safe_receive_message(ws)
        if msg is None:
            return
        if "bytes" in msg and msg["bytes"] is not None:
            await transcriber.add_audio(msg["bytes"])
        elif "text" in msg and msg["text"] is not None:
            try:
                import json

                body = json.loads(msg["text"])
            except Exception:
                continue

            if body.get("type") == "reset":
                await transcriber.reset()
