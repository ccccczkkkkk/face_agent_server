import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "face_agent.sqlite3"
_DB_LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get_db_path() -> Path:
    configured = (os.getenv("SYNC_DB_PATH") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_DB_PATH


def display_db_path() -> str:
    try:
        return str(get_db_path().relative_to(Path(__file__).resolve().parent))
    except ValueError:
        return str(get_db_path())


def connect() -> sqlite3.Connection:
    path = get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _DB_LOCK:
        with connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                  id TEXT PRIMARY KEY,
                  mode TEXT NOT NULL,
                  title TEXT NOT NULL,
                  outline TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  deleted_at TEXT NOT NULL DEFAULT '',
                  source_device_id TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS sync_devices (
                  device_id TEXT PRIMARY KEY,
                  name TEXT NOT NULL DEFAULT '',
                  first_seen_at TEXT NOT NULL,
                  last_seen_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_updated_at
                  ON sessions(updated_at);
                """
            )


def _as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def normalize_session_payload(session: dict[str, Any]) -> dict[str, str]:
    if not isinstance(session, dict):
        raise ValueError("session must be a JSON object")

    session_id = _as_text(session.get("id")).strip()
    if not session_id:
        raise ValueError("session.id is required")

    created_at = _as_text(session.get("createdAt") or session.get("created_at")).strip()
    updated_at = _as_text(session.get("updatedAt") or session.get("updated_at")).strip()
    deleted_at = _as_text(session.get("deletedAt") or session.get("deleted_at")).strip()
    timestamp = now_iso()

    return {
        "id": session_id,
        "mode": _as_text(session.get("mode"), "unknown").strip() or "unknown",
        "title": _as_text(session.get("title")).strip(),
        "outline": _as_text(session.get("outline")).strip(),
        "created_at": created_at or updated_at or timestamp,
        "updated_at": updated_at or created_at or timestamp,
        "deleted_at": deleted_at,
        "payload_json": json.dumps(session, ensure_ascii=False, separators=(",", ":")),
    }


def _record_device(conn: sqlite3.Connection, device_id: str, name: str = "") -> None:
    if not device_id:
        return
    timestamp = now_iso()
    conn.execute(
        """
        INSERT INTO sync_devices (device_id, name, first_seen_at, last_seen_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(device_id) DO UPDATE SET
          name = CASE WHEN excluded.name != '' THEN excluded.name ELSE sync_devices.name END,
          last_seen_at = excluded.last_seen_at
        """,
        (device_id, name, timestamp, timestamp),
    )


def _sync_clock(updated_at: str, deleted_at: str = "") -> str:
    if deleted_at and deleted_at > updated_at:
        return deleted_at
    return updated_at


def upsert_sessions(
    sessions: list[dict[str, Any]],
    device_id: str = "",
    device_name: str = "",
) -> dict[str, Any]:
    normalized = [normalize_session_payload(session) for session in sessions]

    inserted = 0
    updated = 0
    skipped = 0

    with _DB_LOCK:
        with connect() as conn:
            conn.execute("BEGIN")
            try:
                _record_device(conn, device_id, device_name)
                for row in normalized:
                    existing = conn.execute(
                        "SELECT updated_at, deleted_at FROM sessions WHERE id = ?",
                        (row["id"],),
                    ).fetchone()
                    incoming_clock = _sync_clock(row["updated_at"], row["deleted_at"])
                    if existing:
                        existing_clock = _sync_clock(
                            _as_text(existing["updated_at"]),
                            _as_text(existing["deleted_at"]),
                        )
                    else:
                        existing_clock = ""
                    if existing and existing_clock > incoming_clock:
                        skipped += 1
                        continue

                    conn.execute(
                        """
                        INSERT INTO sessions (
                          id, mode, title, outline, payload_json,
                          created_at, updated_at, deleted_at, source_device_id
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                          mode = excluded.mode,
                          title = excluded.title,
                          outline = excluded.outline,
                          payload_json = excluded.payload_json,
                          created_at = excluded.created_at,
                          updated_at = excluded.updated_at,
                          deleted_at = excluded.deleted_at,
                          source_device_id = excluded.source_device_id
                        """,
                        (
                            row["id"],
                            row["mode"],
                            row["title"],
                            row["outline"],
                            row["payload_json"],
                            row["created_at"],
                            row["updated_at"],
                            row["deleted_at"],
                            device_id,
                        ),
                    )
                    if existing:
                        updated += 1
                    else:
                        inserted += 1
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    return {
        "status": "ok",
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
    }


def list_sessions(since: str = "", device_id: str = "", device_name: str = "") -> list[dict[str, Any]]:
    with _DB_LOCK:
        with connect() as conn:
            _record_device(conn, device_id, device_name)
            if device_id:
                conn.commit()
            if since:
                rows = conn.execute(
                    """
                    SELECT payload_json FROM sessions
                    WHERE updated_at > ? OR deleted_at > ?
                    ORDER BY updated_at ASC
                    """,
                    (since, since),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT payload_json FROM sessions ORDER BY updated_at ASC"
                ).fetchall()

    payloads: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads
