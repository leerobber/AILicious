import threading
from datetime import datetime, timezone

from core.db import DB_PATH, TURSO_AUTH_TOKEN, TURSO_DATABASE_URL
from core.db import get_connection as _db_get_connection

_lock = threading.Lock()

_COLUMNS = (
    "id, agent, rationale, proposed_system_prompt, status, created_at, reviewed_at, "
    "baseline_signal_quality, prior_override, outcome, outcome_reasoning, evaluated_at"
)


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
        "baseline_signal_quality": row[7],
        "prior_override": row[8],
        "outcome": row[9],
        "outcome_reasoning": row[10],
        "evaluated_at": row[11],
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
            # Closing the SAGE loop: an accepted proposal used to just apply and be
            # forgotten, with nothing checking whether it actually helped. baseline_signal_quality
            # and prior_override snapshot state at accept time; outcome/outcome_reasoning/
            # evaluated_at record what a later judge call decided once fresh signal exists.
            for column, ddl in (
                ("baseline_signal_quality", "ALTER TABLE sage_proposals ADD COLUMN baseline_signal_quality TEXT"),
                ("prior_override", "ALTER TABLE sage_proposals ADD COLUMN prior_override TEXT"),
                ("outcome", "ALTER TABLE sage_proposals ADD COLUMN outcome TEXT NOT NULL DEFAULT ''"),
                ("outcome_reasoning", "ALTER TABLE sage_proposals ADD COLUMN outcome_reasoning TEXT"),
                ("evaluated_at", "ALTER TABLE sage_proposals ADD COLUMN evaluated_at TEXT"),
            ):
                try:
                    conn.execute(ddl)
                except ValueError:
                    pass  # column already exists (fresh table, or already migrated)
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


def record_acceptance_baseline(proposal_id: int, baseline_signal_quality: str, prior_override: str | None) -> None:
    """Snapshots state at the moment a proposal is accepted -- what 'signal_quality' read
    justified the change, and what override (if any) was in effect right before it, so a
    later regression can be reverted to exactly that prior state rather than guessing.
    """
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                "UPDATE sage_proposals SET baseline_signal_quality = ?, prior_override = ? WHERE id = ?",
                (baseline_signal_quality, prior_override, proposal_id),
            )
            conn.commit()
        finally:
            conn.close()


def get_pending_evaluation_proposal(agent: str) -> dict | None:
    """The most recent accepted-but-not-yet-evaluated proposal for this agent, if any.
    Returns None once outcome is set (evaluated) or if nothing has ever been accepted --
    each proposal is only ever up for evaluation once, immediately after acceptance.
    """
    with _lock:
        conn = _get_connection()
        try:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM sage_proposals WHERE agent = ? AND status = 'accepted' AND outcome = '' "
                "ORDER BY id DESC LIMIT 1",
                (agent,),
            ).fetchone()
        finally:
            conn.close()
    return _row_to_dict(row) if row else None


def record_outcome(proposal_id: int, outcome: str, reasoning: str) -> None:
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                "UPDATE sage_proposals SET outcome = ?, outcome_reasoning = ?, evaluated_at = ? WHERE id = ?",
                (outcome, reasoning, datetime.now(timezone.utc).isoformat(), proposal_id),
            )
            conn.commit()
        finally:
            conn.close()
