import asyncio
import json
import threading
from datetime import datetime, timezone

from core.db import DB_PATH, TURSO_AUTH_TOKEN, TURSO_DATABASE_URL
from core.db import get_connection as _db_get_connection

_lock = threading.Lock()

# A single row, shared across every agent -- durable facts about the user that
# outlive any one agent's own conversation (name, ongoing projects, stated
# preferences that apply everywhere), as opposed to core.digest's conversation_digests
# table, which stays one row per agent and private to that agent's own habits/tone.
# This is the one thing every agent gets to see, folded in from whichever agent's
# digest most recently updated -- so a fact one agent learns becomes visible to all
# of them without needing to be re-taught.
_PROFILE_ID = 1

PROFILE_MERGE_SYSTEM_PROMPT = (
    "You maintain one durable profile of a user, shared across several different AI "
    "agents/personas they talk to. You'll be shown the profile as it stands now (if "
    "anything) and a fresh summary of one agent's most recent conversations with this "
    "user. Merge in only facts about the USER that would matter to ANY agent talking to "
    "them -- their name, role, ongoing projects, stated preferences, constraints -- not "
    "this one agent's own conversational habits, tone preferences specific to that "
    "agent, or anything that only makes sense in that agent's own domain. Keep what's "
    "still true, revise anything corrected, add what's genuinely new and durable, drop "
    "anything now stale. Keep it dense and factual, not a transcript. If nothing in the "
    "new summary is durable or cross-agent-relevant, return the profile unchanged.\n\n"
    'Respond with ONLY a JSON object: {"profile": "<the updated profile>"}. No markdown, '
    "no extra text -- valid JSON only."
)


def _get_connection():
    return _db_get_connection(DB_PATH, TURSO_DATABASE_URL, TURSO_AUTH_TOKEN)


def init_db() -> None:
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profile (
                    id INTEGER PRIMARY KEY,
                    profile TEXT NOT NULL DEFAULT '',
                    updated_at TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


def get_profile() -> str:
    with _lock:
        conn = _get_connection()
        try:
            row = conn.execute("SELECT profile FROM user_profile WHERE id = ?", (_PROFILE_ID,)).fetchone()
        finally:
            conn.close()
    return row[0] if row else ""


def get_profile_state() -> dict:
    with _lock:
        conn = _get_connection()
        try:
            row = conn.execute(
                "SELECT profile, updated_at FROM user_profile WHERE id = ?", (_PROFILE_ID,)
            ).fetchone()
        finally:
            conn.close()
    if row is None:
        return {"profile": "", "updated_at": None}
    return {"profile": row[0], "updated_at": row[1]}


def apply_profile(new_profile: str) -> None:
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                "INSERT INTO user_profile (id, profile, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET profile = excluded.profile, updated_at = excluded.updated_at",
                (_PROFILE_ID, new_profile, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()


def _build_merge_messages(prior_profile: str, agent: str, agent_digest: str) -> list[dict]:
    lines = [
        f"Profile as it stands now:\n{prior_profile}" if prior_profile else "No profile exists yet.",
        f"\nFresh summary from the '{agent}' agent's own conversations with this user:\n{agent_digest}",
    ]
    return [
        {"role": "system", "content": PROFILE_MERGE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


async def run_profile_merge(agent: str, agent_digest: str, call_mistral) -> bool:
    """Folds one agent's freshly updated digest into the shared cross-agent profile.

    Best-effort, same posture as core.digest.run_digest_cycle: this always runs after
    a digest cycle already succeeded, so a failure here is swallowed rather than raised
    -- it must never turn an otherwise-successful chat turn into an error. Returns True
    if the profile was actually updated.
    """
    if not agent_digest:
        return False

    prior_profile = await asyncio.to_thread(get_profile)
    messages = _build_merge_messages(prior_profile, agent, agent_digest)
    try:
        raw_reply = await call_mistral(messages)
        parsed = json.loads(raw_reply)
        new_profile = parsed["profile"]
        if not isinstance(new_profile, str):
            raise ValueError("profile must be a string")
    except Exception:
        return False

    await asyncio.to_thread(apply_profile, new_profile)
    return True
