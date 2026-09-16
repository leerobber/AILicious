import threading
from datetime import datetime, timezone

from core.db import DB_PATH, TURSO_AUTH_TOKEN, TURSO_DATABASE_URL
from core.db import get_connection as _db_get_connection

_lock = threading.Lock()

# A persisted, queryable trail for the background cycles (digest, profile merge, SAGE
# evaluation, embedding) -- the exact things this project spent real time manually
# grepping Render's raw stdout logs to reconstruct. Render's own logs still exist and
# still matter for request-level detail (a Mistral 500, a stack trace), but "did a digest
# fire for nexus in the last hour, and what happened" now has a real answer via GET /events
# instead of a live log search.


def _get_connection():
    return _db_get_connection(DB_PATH, TURSO_DATABASE_URL, TURSO_AUTH_TOKEN)


def init_db() -> None:
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    category TEXT NOT NULL,
                    event TEXT NOT NULL,
                    agent TEXT,
                    detail TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_system_events_category ON system_events(category)")
            conn.commit()
        finally:
            conn.close()


def record_event(category: str, event: str, agent: str | None = None, detail: str = "") -> None:
    """Best-effort by design, and self-contained about it: every caller sites this at the
    end of a real background cycle (a digest applied, a SAGE proposal evaluated, ...), and
    a failure to persist the *record* of that must never look like a failure of the cycle
    itself, or corrupt/interrupt whatever the caller does next. Swallows its own exceptions
    rather than asking every call site to wrap it.
    """
    try:
        with _lock:
            conn = _get_connection()
            try:
                conn.execute(
                    "INSERT INTO system_events (created_at, category, event, agent, detail) VALUES (?, ?, ?, ?, ?)",
                    (datetime.now(timezone.utc).isoformat(), category, event, agent, detail),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception:
        pass


def list_events(category: str | None = None, agent: str | None = None, limit: int = 50) -> list[dict]:
    query = "SELECT id, created_at, category, event, agent, detail FROM system_events WHERE 1=1"
    params: list = []
    if category is not None:
        query += " AND category = ?"
        params.append(category)
    if agent is not None:
        query += " AND agent = ?"
        params.append(agent)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with _lock:
        conn = _get_connection()
        try:
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()
    return [
        {"id": r[0], "created_at": r[1], "category": r[2], "event": r[3], "agent": r[4], "detail": r[5]}
        for r in rows
    ]
