import threading
from datetime import datetime, timezone

from core.db import DB_PATH, TURSO_AUTH_TOKEN, TURSO_DATABASE_URL
from core.db import get_connection as _db_get_connection

_lock = threading.Lock()

_COLUMNS = "id, agent, rationale, proposed_system_prompt, status, created_at, reviewed_at"


def _get_connection():
    return _db_get_connection(DB_PATH, TURSO_DATABASE_URL, TURSO_AUTH_TOKEN)


def _row_to_dict(row) -> dict:
    return {
        "id": row[0],
        "agent": row[1],
        "rationale": row[2],
        "proposed_system_prompt": row[3],
        "status": row[4],
        "created_at": row[5],
        "reviewed_at": row[6],
    }


def init_db() -> None:
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sage_proposals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    proposed_system_prompt TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sage_proposals_agent ON sage_proposals(agent)")
            conn.commit()
        finally:
            conn.close()


def create_proposal(agent: str, rationale: str, proposed_system_prompt: str) -> int:
    with _lock:
        conn = _get_connection()
        try:
            cursor = conn.execute(
                "INSERT INTO sage_proposals (agent, rationale, proposed_system_prompt, status, created_at) "
                "VALUES (?, ?, ?, 'pending', ?)",
                (agent, rationale, proposed_system_prompt, datetime.now(timezone.utc).isoformat()),
            )
            proposal_id = cursor.lastrowid
            conn.commit()
        finally:
            conn.close()
    return proposal_id


def get_proposal(proposal_id: int) -> dict | None:
    with _lock:
        conn = _get_connection()
        try:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM sage_proposals WHERE id = ?", (proposal_id,)
            ).fetchone()
        finally:
            conn.close()
    return _row_to_dict(row) if row else None


def list_proposals(agent: str | None = None, status: str | None = None) -> list[dict]:
    query = f"SELECT {_COLUMNS} FROM sage_proposals WHERE 1=1"
    params: list[str] = []
    if agent is not None:
        query += " AND agent = ?"
        params.append(agent)
    if status is not None:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY id DESC"

    with _lock:
        conn = _get_connection()
        try:
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()
    return [_row_to_dict(row) for row in rows]


def set_proposal_status(proposal_id: int, status: str) -> bool:
    """Transition a proposal to accepted/rejected. Only a pending proposal can transition;
    returns False if the proposal doesn't exist or has already been reviewed, so a caller
    can distinguish "not found" from "already decided" (both need different HTTP statuses).
    """
    with _lock:
        conn = _get_connection()
        try:
            row = conn.execute("SELECT status FROM sage_proposals WHERE id = ?", (proposal_id,)).fetchone()
            if row is None or row[0] != "pending":
                return False
            conn.execute(
                "UPDATE sage_proposals SET status = ?, reviewed_at = ? WHERE id = ?",
                (status, datetime.now(timezone.utc).isoformat(), proposal_id),
            )
            conn.commit()
        finally:
            conn.close()
    return True
