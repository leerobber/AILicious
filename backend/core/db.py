import os
import time
from pathlib import Path

import libsql

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "memory.db"
TURSO_DATABASE_URL = os.environ.get("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")


def get_connection(db_path: Path, turso_url: str | None, turso_token: str | None):
    if turso_url:
        return libsql.connect(turso_url, auth_token=turso_token)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return libsql.connect(str(db_path))


def execute_with_retry(get_connection_fn, operation, max_retries: int = 2, backoff_seconds: float = 0.15):
    """Runs `operation(conn)` against a connection from get_connection_fn(), retrying with a
    fresh connection if Turso's remote (Hrana) driver raises its own explicitly-retryable
    error. Observed in production: under load, Turso can roll back an "interactive
    transaction" it decides has been idle too long, and libsql-python surfaces that as a
    ValueError whose message literally says to retry -- this does exactly that, rather than
    losing whatever the write was trying to do. Local SQLite (the no-Turso fallback) never
    raises this, so it's a no-op in that mode. Always closes whichever connection it opened,
    whether the operation succeeds, fails for an unrelated reason, or gets retried.
    """
    last_exc: ValueError | None = None
    for attempt in range(max_retries + 1):
        if attempt:
            time.sleep(backoff_seconds * attempt)
        conn = get_connection_fn()
        try:
            return operation(conn)
        except ValueError as exc:
            if "retry the transaction" not in str(exc):
                raise
            last_exc = exc
        finally:
            conn.close()
    raise last_exc
