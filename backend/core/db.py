import os
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
