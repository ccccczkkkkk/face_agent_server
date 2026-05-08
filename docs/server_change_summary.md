# Server Change Summary

## Overview

This server update includes six main areas:

1. Optional local Whisper STT backend
2. Optional Soniox realtime STT backend
3. Text-model translation path
3. Protocol updates for transcript/translation alignment
4. Manual conversation suggestion flow
5. Subtitle chunk summary and note output
6. Soniox direct translation path

## 1. STT Backend

The server now supports three STT backends:

```env
STT_BACKEND=realtime
STT_BACKEND=local_whisper
STT_BACKEND=soniox
```

Default behavior remains:

```env
STT_BACKEND=realtime
```

New modules:

- `local_whisper.py`
- `soniox_stt.py`

Both routes support the new backend:

- `/conversation/ws`
- `/subtitle/ws`

### Soniox Notes

`soniox` is intended for cloud-friendly realtime STT without self-hosted GPU requirements.

Current behavior:

- `subtitle` supports Soniox realtime transcription
- `conversation` supports Soniox realtime transcription
- language hints are applied when `transcription_language` is explicitly set
- optional direct translation is available from Soniox in supported target languages
- forced segmentation is supported through a character threshold

## 2. Translation Path

Translation no longer uses the Realtime websocket path.

It now uses a normal text model request through:

- `text_translation.py`

Default translation model:

```env
TRANSLATION_TEXT_MODEL=gpt-4.1-mini
```

This text-model path is still used:

- for non-Soniox STT backends
- as a fallback when Soniox direct translation is not available for the requested target language

## 3. Protocol Updates

### Added `segment_id`

Both `conversation` and `subtitle` now include `segment_id` in:

- `transcript_final`
- `translation`

This allows the client to align transcript and translation reliably.

### Event Examples

`transcript_final`

```json
{
  "type": "transcript_final",
  "segment_id": "seg_xxx",
  "transcript": "はじめまして。",
  "translation": "",
  "next_say": [],
  "intent": ""
}
```

`translation`

```json
{
  "type": "translation",
  "segment_id": "seg_xxx",
  "transcript": "はじめまして。",
  "translation": "初次见面。"
}
```

Recommended client behavior:

1. On `transcript_final`, create a transcript row keyed by `segment_id`
2. On `translation`, fill the matching row by `segment_id`

### Added `summary_id`

`subtitle` chunk summaries now include `summary_id`.

This allows the client to append or manage summary blocks reliably.

`summary`

```json
{
  "type": "summary",
  "summary_id": "sum_xxx",
  "summary_type": "chunk",
  "note_source": "Corrected chunk transcript used for notes",
  "summary": "Source-language summary text",
  "summaries": {
    "source": "Source-language summary text",
    "en": "English summary text",
    "zh-Hans": "Simplified Chinese summary text"
  }
}
```

Recommended client behavior:

1. Use `summary_id` as the key for each summary block
2. Use `note_source` as the corrected full note text for that chunk
3. Use `summaries.source` / `summaries.en` / `summaries.zh-Hans` for language-specific display
4. Append each chunk summary or note block in order if building a running notes panel

### Soniox Direct Translation

When `STT_BACKEND=soniox` and the requested `translation_language` is supported, both `subtitle` and `conversation` can use Soniox final translation directly instead of the OpenAI text translation worker.

Current mapped target languages:

- `ja`
- `en`
- `zh`
- `zh-Hans`
- `ko`

Protocol remains unchanged:

```json
{
  "type": "translation",
  "segment_id": "seg_xxx",
  "transcript": "Source transcript text",
  "translation": "Translated text"
}
```

If the target language is unsupported or is the same as the explicit source language, the server falls back to the existing OpenAI text translation path.

## 4. Conversation Suggestion Flow

### Previous Behavior

The server automatically generated followup suggestions after each final transcript.

### Current Behavior

The server now separates memory extraction from suggestion generation.

#### Automatic

- Final transcripts are buffered
- When buffered text exceeds a threshold, the server updates conversation memory

Config:

```env
CONVERSATION_MEMORY_TRIGGER_CHARS=80
```

#### Manual

The client must explicitly request a suggestion:

```json
{
  "type": "request_suggestion"
}
```

After that, the server returns the usual `suggestion` event.

If the most recent buffered transcript has not yet reached the memory threshold, it is appended to the current `last_user_message` context before suggestion generation so recent content is not lost.

### Opening Suggestion

Opening suggestion is now generated only once per `session_id`.

That means:

- first connection for a session: opening suggestion is sent
- reconnect with the same `session_id`: opening suggestion is not sent again

## 5. STT Disconnect Protection

Both `conversation` and `subtitle` now include relaxed STT disconnect handling.

If the backend STT websocket closes:

- the main client websocket stays open
- the server stops forwarding more audio to the STT backend
- the server sends a fixed error event

Error event:

```json
{
  "type": "error",
  "stage": "stt_connection",
  "message": "transcribe websocket closed",
  "reason": "..."
}
```

Recommended client behavior after receiving this event:

1. Stop sending audio immediately
2. Mark STT as unavailable
3. Keep the main websocket open
4. In `conversation`, continue allowing manual suggestion requests
5. Reconnect the voice session when ready

## 6. Important Config Additions

```env
STT_BACKEND=realtime
TRANSLATION_TEXT_MODEL=gpt-4.1-mini
CONVERSATION_MEMORY_TRIGGER_CHARS=80
SUBTITLE_SUMMARY_MODEL=gpt-4.1-mini
SUBTITLE_CORRECTION_MODEL=gpt-4.1-mini
SUBTITLE_SUMMARY_TRIGGER_CHARS=120
SONIOX_API_KEY=
SONIOX_MODEL=stt-rt-v4
SONIOX_AUDIO_FORMAT=pcm_s16le
SONIOX_SAMPLE_RATE=24000
SONIOX_NUM_CHANNELS=1
SONIOX_ENABLE_ENDPOINT_DETECTION=true
SONIOX_MAX_ENDPOINT_DELAY_MS=1200
SONIOX_KEEPALIVE_INTERVAL_MS=8000
SONIOX_TRAILING_SILENCE_MS=300
SONIOX_FORCE_FINALIZE_AFTER_CHARS=120
WHISPER_MODEL_SIZE=medium
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
WHISPER_WINDOW_SECONDS=6.0
WHISPER_STEP_SECONDS=1.5
WHISPER_OVERLAP_SECONDS=1.0
WHISPER_FINAL_SILENCE_MS=700
WHISPER_SILENCE_THRESHOLD=0.01
WHISPER_SAMPLE_RATE=24000
WHISPER_IDLE_UNLOAD_SECONDS=60
```

## 7. Current Remote Commit

- `a3e12d5 Improve subtitle notes and whisper lifecycle`
