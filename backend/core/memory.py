import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import libsql

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "memory.db"
TURSO_DATABASE_URL = os.environ.get("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")
_lock = threading.Lock()

# KAIROS: utility-weighted retrieval. A message's stored utility_score combines a small
# write-time heuristic (longer, more substantive turns score slightly higher) with any
# feedback adjustment. get_history() uses it to let a genuinely valuable older message
# survive over worthless-but-recent filler, without abandoning "prefer recent" as the
# default — when every candidate has utility_score == 0 (no feedback yet), ranking
# collapses to pure recency, identical to a plain `ORDER BY id DESC LIMIT`.
CANDIDATE_LIMIT = 60
UTILITY_WEIGHT = 10.0
HEURISTIC_LENGTH_CAP = 200
HEURISTIC_MAX = 0.5
FEEDBACK_WEIGHT = 2.0


def _heuristic_score(content: str) -> float:
    return min(len(content) / HEURISTIC_LENGTH_CAP, 1.0) * HEURISTIC_MAX


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
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN feedback INTEGER")
            except ValueError:
                pass  # column already exists (fresh table, or already migrated)
            try:
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
                "INSERT INTO messages (session_id, agent, role, content, created_at, utility_score) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, agent, role, content, datetime.now(timezone.utc).isoformat(), _heuristic_score(content)),
            )
            message_id = cursor.lastrowid
            conn.commit()
        finally:
            conn.close()
    return message_id


def set_feedback(message_id: int, rating: int) -> bool:
    """Record feedback on a message and adjust its utility_score accordingly.

    Idempotent with respect to re-voting or switching a vote: the score adjustment is
    computed from the *change* between the previous and new feedback, not appended
    blindly, so voting the same way twice (or flipping a vote) never double-counts.
    Returns False if no message with that id exists.
    """
    with _lock:
        conn = _get_connection()
        try:
            row = conn.execute("SELECT feedback FROM messages WHERE id = ?", (message_id,)).fetchone()
            if row is None:
                return False

            old_feedback = row[0]
            old_delta = FEEDBACK_WEIGHT * old_feedback if old_feedback in (1, -1) else 0.0
            new_delta = FEEDBACK_WEIGHT * rating
            adjustment = new_delta - old_delta

            conn.execute(
                "UPDATE messages SET feedback = ?, utility_score = utility_score + ? WHERE id = ?",
                (rating, adjustment, message_id),
            )
            conn.commit()
        finally:
            conn.close()
    return True


def get_history(session_id: str, agent: str, limit: int = 20) -> list[dict]:
    with _lock:
        conn = _get_connection()
        try:
            rows = conn.execute(
                "SELECT id, role, content, utility_score FROM messages "
                "WHERE session_id = ? AND agent = ? ORDER BY id DESC LIMIT ?",
                (session_id, agent, CANDIDATE_LIMIT),
            ).fetchall()
        finally:
            conn.close()

    # rows[0] is the most recent candidate. recency_score decreases monotonically with
    # age; when every utility_score is 0 (the common case, no feedback yet), the ranking
    # below is identical to plain recency — same rows a bare LIMIT would return.
    pool_size = len(rows)
    scored = [
        (pool_size - i + utility_score * UTILITY_WEIGHT, msg_id, role, content)
        for i, (msg_id, role, content, utility_score) in enumerate(rows)
    ]

    top = sorted(scored, key=lambda item: item[0], reverse=True)[:limit]
    top.sort(key=lambda item: item[1])  # back to chronological order for the transcript

    return [{"role": role, "content": content} for _, _, role, content in top]


def get_stats() -> dict:
    with _lock:
        conn = _get_connection()
        try:
            total_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            total_sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM messages").fetchone()[0]
        finally:
            conn.close()
    return {"total_messages": total_messages, "total_sessions": total_sessions}
