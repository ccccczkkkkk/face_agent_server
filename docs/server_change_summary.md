# Server Change Summary

## Overview

This server update includes four main areas:

1. Optional local Whisper STT backend
2. Text-model translation path
3. Protocol updates for transcript/translation alignment
4. Manual conversation suggestion flow

## 1. STT Backend

The server now supports two STT backends:

```env
STT_BACKEND=realtime
STT_BACKEND=local_whisper
```

Default behavior remains:

```env
STT_BACKEND=realtime
```

New module:

- `local_whisper.py`

Both routes support the new backend:

- `/conversation/ws`
- `/subtitle/ws`

## 2. Translation Path

Translation no longer uses the Realtime websocket path.

It now uses a normal text model request through:

- `text_translation.py`

Default translation model:

```env
TRANSLATION_TEXT_MODEL=gpt-4.1-mini
```

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
WHISPER_MODEL_SIZE=medium
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
WHISPER_WINDOW_SECONDS=6.0
WHISPER_STEP_SECONDS=1.5
WHISPER_OVERLAP_SECONDS=1.0
WHISPER_FINAL_SILENCE_MS=700
WHISPER_SILENCE_THRESHOLD=0.01
WHISPER_SAMPLE_RATE=24000
```

## 7. Current Remote Commit

- `9d8c690 Add local whisper STT and manual conversation suggestions`
