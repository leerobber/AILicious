import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "memory.db"
_lock = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock, _get_connection() as conn:
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
        except sqlite3.OperationalError:
            pass  # column already exists (fresh table, or already migrated)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_agent ON messages(session_id, agent)")


def add_message(session_id: str, agent: str, role: str, content: str) -> None:
    with _lock, _get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, agent, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, agent, role, content, datetime.now(timezone.utc).isoformat()),
        )


def get_history(session_id: str, agent: str, limit: int = 20) -> list[dict]:
    with _lock, _get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? AND agent = ? ORDER BY id DESC LIMIT ?",
            (session_id, agent, limit),
        ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def get_stats() -> dict:
    with _lock, _get_connection() as conn:
        total_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        total_sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM messages").fetchone()[0]
    return {"total_messages": total_messages, "total_sessions": total_sessions}
