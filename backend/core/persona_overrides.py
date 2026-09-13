import threading
from datetime import datetime, timezone

from core.db import DB_PATH, TURSO_AUTH_TOKEN, TURSO_DATABASE_URL
from core.db import get_connection as _db_get_connection

_lock = threading.Lock()


def _get_connection():
    return _db_get_connection(DB_PATH, TURSO_DATABASE_URL, TURSO_AUTH_TOKEN)


def init_db() -> None:
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS persona_overrides (
                    agent TEXT PRIMARY KEY,
                    system_prompt TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


def get_override(agent: str) -> str | None:
    with _lock:
        conn = _get_connection()
        try:
            row = conn.execute(
                "SELECT system_prompt FROM persona_overrides WHERE agent = ?", (agent,)
            ).fetchone()
        finally:
            conn.close()
    return row[0] if row else None


def set_override(agent: str, system_prompt: str) -> None:
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                """
                INSERT INTO persona_overrides (agent, system_prompt, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(agent) DO UPDATE SET system_prompt = excluded.system_prompt, updated_at = excluded.updated_at
                """,
                (agent, system_prompt, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()


def clear_override(agent: str) -> None:
    with _lock:
        conn = _get_connection()
        try:
            conn.execute("DELETE FROM persona_overrides WHERE agent = ?", (agent,))
            conn.commit()
        finally:
            conn.close()
