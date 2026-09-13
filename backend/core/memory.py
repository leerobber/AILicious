import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import libsql

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "memory.db"
TURSO_DATABASE_URL = os.environ.get("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")
_lock = threading.Lock()


def _get_connection():
    if TURSO_DATABASE_URL:
        return libsql.connect(TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return libsql.connect(str(DB_PATH))


def init_db() -> None:
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    agent TEXT NOT NULL DEFAULT 'nexus',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN agent TEXT NOT NULL DEFAULT 'nexus'")
            except ValueError:
                pass  # column already exists (fresh table, or already migrated)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_agent ON messages(session_id, agent)")
            conn.commit()
        finally:
            conn.close()


def add_message(session_id: str, agent: str, role: str, content: str) -> None:
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                "INSERT INTO messages (session_id, agent, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, agent, role, content, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()


def get_history(session_id: str, agent: str, limit: int = 20) -> list[dict]:
    with _lock:
        conn = _get_connection()
        try:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? AND agent = ? ORDER BY id DESC LIMIT ?",
                (session_id, agent, limit),
            ).fetchall()
        finally:
            conn.close()
    return [{"role": role, "content": content} for role, content in reversed(rows)]


def get_stats() -> dict:
    with _lock:
        conn = _get_connection()
        try:
            total_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            total_sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM messages").fetchone()[0]
        finally:
            conn.close()
    return {"total_messages": total_messages, "total_sessions": total_sessions}
