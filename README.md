# Face Agent Server

Backend for the Face Agent Flutter client.

## Features

- Conversation websocket route: `/conversation/ws`
- Subtitle websocket route: `/subtitle/ws`
- OpenAI Realtime-based transcription and response flow

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a local `.env` from `.env.example` and fill in your values.

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or use values from `.env` manually:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Routes

- `GET /`
- `WS /conversation/ws`
- `WS /subtitle/ws`

## Notes

- `server.py` is the current conversation-mode backend.
- `server_subtitle.py` is the current subtitle-mode backend.
- Experimental files are intentionally kept out of the repo for now.
