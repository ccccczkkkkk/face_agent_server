import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query

try:
    from sync_store import display_db_path, init_db, list_sessions, upsert_sessions
except ModuleNotFoundError:
    from face_server.sync_store import display_db_path, init_db, list_sessions, upsert_sessions


router = APIRouter()


@router.on_event("startup")
async def startup_sync_store() -> None:
    await asyncio.to_thread(init_db)


@router.get("/health")
async def health() -> dict[str, str]:
    await asyncio.to_thread(init_db)
    return {
        "status": "ok",
        "database": "sqlite",
        "path": display_db_path(),
    }


@router.get("/sessions")
async def get_sessions(
    since: str = Query(default=""),
    device_id: str = Query(default=""),
    device_name: str = Query(default=""),
) -> dict[str, list[dict[str, Any]]]:
    await asyncio.to_thread(init_db)
    sessions = await asyncio.to_thread(list_sessions, since, device_id, device_name)
    return {
        "sessions": [
            {
                "id": str(session.get("id") or ""),
                "payload": session,
            }
            for session in sessions
        ]
    }


@router.post("/sessions")
async def post_sessions(body: dict[str, Any]) -> dict[str, Any]:
    device_id = str(body.get("device_id") or body.get("deviceId") or "").strip()
    device_name = str(body.get("device_name") or body.get("deviceName") or "").strip()
    sessions = body.get("sessions")

    if isinstance(sessions, dict):
        sessions = [sessions]
    if not isinstance(sessions, list):
        raise HTTPException(status_code=400, detail="sessions must be a list")

    try:
        await asyncio.to_thread(init_db)
        result = await asyncio.to_thread(upsert_sessions, sessions, device_id, device_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result["count"] = len(sessions)
    return result
