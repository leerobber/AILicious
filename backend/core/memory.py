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
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN feedback INTEGER")
            except ValueError:
                pass  # column already exists (fresh table, or already migrated)
            try:
                # Historical: fed KAIROS's per-message utility ranking, since replaced by
                # core/digest.py's whole-conversation summarization. Left in place rather
                # than dropped -- libsql/SQLite don't support a clean column drop, and
                # forcing a table rebuild against the live Turso DB isn't worth the risk
                # for a column that's simply unused going forward.
                conn.execute("ALTER TABLE messages ADD COLUMN utility_score REAL NOT NULL DEFAULT 0.0")
            except ValueError:
                pass  # column already exists (fresh table, or already migrated)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_agent ON messages(session_id, agent)")
            conn.commit()
        finally:
            conn.close()


def add_message(session_id: str, agent: str, role: str, content: str) -> int:
    with _lock:
        conn = _get_connection()
        try:
            cursor = conn.execute(
                "INSERT INTO messages (session_id, agent, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, agent, role, content, datetime.now(timezone.utc).isoformat()),
            )
            message_id = cursor.lastrowid
            conn.commit()
        finally:
            conn.close()
    return message_id


def set_feedback(message_id: int, rating: int) -> bool:
    """Record a thumbs up/down on a message.

    Optional: neither context selection nor SAGE's persona analysis depend on this
    anymore (both run from the conversation digest in core/digest.py), but a flagged
    message is folded into the next digest cycle as extra-weighted signal. Returns
    False if no message with that id exists.
    """
    with _lock:
        conn = _get_connection()
        try:
            row = conn.execute("SELECT id FROM messages WHERE id = ?", (message_id,)).fetchone()
            if row is None:
                return False
            conn.execute("UPDATE messages SET feedback = ? WHERE id = ?", (rating, message_id))
            conn.commit()
        finally:
            conn.close()
    return True


def get_history(session_id: str, agent: str, limit: int = 20) -> list[dict]:
    """Plain recency window for one conversation -- the short raw tail appended after
    the agent's conversation digest (core/digest.py) when building chat context. The
    digest carries the bulk of long-range memory now, so this only needs to cover the
    last few turns.
    """
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


def get_messages_since(agent: str, since: str | None) -> list[dict]:
    """All raw messages for this agent across every session, oldest first, created
    after the given ISO timestamp -- the new-turns slice a digest cycle folds into
    the agent's running summary. `since=None` means "everything" (an agent's first
    ever digest cycle, before any digest exists).
    """
    with _lock:
        conn = _get_connection()
        try:
            if since:
                rows = conn.execute(
                    "SELECT role, content, feedback FROM messages WHERE agent = ? AND created_at > ? "
                    "ORDER BY id ASC",
                    (agent, since),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT role, content, feedback FROM messages WHERE agent = ? ORDER BY id ASC",
                    (agent,),
                ).fetchall()
        finally:
            conn.close()
    return [{"role": role, "content": content, "feedback": feedback} for role, content, feedback in rows]


def get_stats() -> dict:
    with _lock:
        conn = _get_connection()
        try:
            total_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            total_sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM messages").fetchone()[0]
        finally:
            conn.close()
    return {"total_messages": total_messages, "total_sessions": total_sessions}
