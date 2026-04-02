import json

from fastapi import WebSocket, WebSocketDisconnect


async def safe_send_envelope(ws: WebSocket, payload: dict) -> bool:
    try:
        await ws.send_text(json.dumps(payload, ensure_ascii=False))
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


async def safe_receive_message(ws: WebSocket):
    try:
        message = await ws.receive()
    except (WebSocketDisconnect, RuntimeError):
        return None

    if message.get("type") == "websocket.disconnect":
        return None

    return message
