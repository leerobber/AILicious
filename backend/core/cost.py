import threading
from datetime import datetime, timezone

from core.db import DB_PATH, TURSO_AUTH_TOKEN, TURSO_DATABASE_URL
from core.db import get_connection as _db_get_connection

_lock = threading.Lock()

# Raw Mistral token usage per call, tagged by what the call was for -- "chat" (the live
# request path), "digest", "profile_merge", "sage_evaluation", "sage_analyze", "embedding"
# (every background cycle that calls out to Mistral). Deliberately stores only token
# counts, never a dollar estimate: per-token pricing varies by model and changes over
# time, and baking in a guessed rate here would mean either it goes stale silently or
# this module has to know about pricing, which isn't its job. GET /costs computes an
# estimate from these counts using whatever rate the deployment configures, if any.


def _get_connection():
    return _db_get_connection(DB_PATH, TURSO_DATABASE_URL, TURSO_AUTH_TOKEN)


def init_db() -> None:
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    category TEXT NOT NULL,
                    agent TEXT,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_token_usage_category ON token_usage(category)")
            conn.commit()
        finally:
            conn.close()


def record_usage(
    category: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    agent: str | None = None,
) -> None:
    """Best-effort by design, same posture as core.events.record_event: a failure to log
    usage must never break the Mistral call that generated it."""
    try:
        with _lock:
            conn = _get_connection()
            try:
                conn.execute(
                    "INSERT INTO token_usage "
                    "(created_at, category, agent, model, prompt_tokens, completion_tokens, total_tokens) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        category,
                        agent,
                        model,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception:
        pass


def get_usage_summary(category: str | None = None) -> dict:
    query = (
        "SELECT category, model, COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0), "
        "COALESCE(SUM(total_tokens), 0), COUNT(*) FROM token_usage"
    )
    params: list = []
    if category is not None:
        query += " WHERE category = ?"
        params.append(category)
    query += " GROUP BY category, model ORDER BY category, model"

    with _lock:
        conn = _get_connection()
        try:
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()

    by_category = [
        {
            "category": r[0],
            "model": r[1],
            "prompt_tokens": r[2],
            "completion_tokens": r[3],
            "total_tokens": r[4],
            "calls": r[5],
        }
        for r in rows
    ]
    totals = {
        "prompt_tokens": sum(row["prompt_tokens"] for row in by_category),
        "completion_tokens": sum(row["completion_tokens"] for row in by_category),
        "total_tokens": sum(row["total_tokens"] for row in by_category),
        "calls": sum(row["calls"] for row in by_category),
    }
    return {"by_category": by_category, "totals": totals}
