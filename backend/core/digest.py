import asyncio
import json
import threading
from datetime import datetime, timezone

from core.db import DB_PATH, TURSO_AUTH_TOKEN, TURSO_DATABASE_URL
from core.db import get_connection as _db_get_connection
from core.memory import get_messages_since

_lock = threading.Lock()

# In-memory coalescing lock: Render runs this as a single instance, so a plain
# per-agent set is enough to guarantee only one digest cycle runs per agent at a
# time. A trigger that fires while one is already running is a no-op -- the in-flight
# run reads fresh state when it actually executes, so it absorbs whatever accumulated
# in the meantime instead of needing a second call.
_active_agents: set[str] = set()
_active_lock = threading.Lock()

# Whole-conversation digestion, replacing KAIROS's per-message utility scoring.
# Instead of ranking individual messages, this periodically folds new raw turns into
# one rolling summary per agent (not per session -- the agent accumulates
# understanding across every conversation with it). No voting required: the trigger
# is automatic, and votes (when given) just get folded in as stronger signal.
MESSAGE_TRIGGER = 10            # digest once this many new turns have accumulated
PAUSE_TRIGGER_SECONDS = 20 * 60  # ...or once this long has passed since the last one
MIN_TURNS_FOR_PAUSE = 3          # ...but only if at least this many turns are pending
MAX_BACKOFF_MULTIPLIER = 4       # cap on how much Mistral trouble can suppress digestion
TOPIC_SHIFT_DIVISOR = 2          # a self-reported topic shift halves the next threshold

DIGEST_SYSTEM_PROMPT = (
    "You maintain a running memory of your conversations with one user, across every "
    "session. You'll be shown what you currently know (if anything) and the raw "
    "exchanges since that summary was last updated. Merge them into an updated summary: "
    "keep what's still true, revise anything the new exchanges corrected, add what's "
    "genuinely new, and drop anything now irrelevant. Keep it dense and factual -- key "
    "facts, decisions, stated preferences, and open threads, not a transcript. Respond "
    "with ONLY a JSON object: "
    '{"digest": "<the updated summary>", "topic_shift": <true if the new exchanges moved '
    'to a substantially different subject than what came before, else false>}. '
    "No markdown, no extra text -- valid JSON only."
)


def _get_connection():
    return _db_get_connection(DB_PATH, TURSO_DATABASE_URL, TURSO_AUTH_TOKEN)


def init_db() -> None:
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_digests (
                    agent TEXT PRIMARY KEY,
                    digest TEXT NOT NULL DEFAULT '',
                    turns_since_digest INTEGER NOT NULL DEFAULT 0,
                    last_message_at TEXT,
                    last_digested_at TEXT,
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    topic_shift_pending INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


def _row_to_state(row) -> dict:
    return {
        "agent": row[0],
        "digest": row[1],
        "turns_since_digest": row[2],
        "last_message_at": row[3],
        "last_digested_at": row[4],
        "consecutive_failures": row[5],
        "topic_shift_pending": bool(row[6]),
    }


def get_state(agent: str) -> dict:
    with _lock:
        conn = _get_connection()
        try:
            row = conn.execute(
                "SELECT agent, digest, turns_since_digest, last_message_at, last_digested_at, "
                "consecutive_failures, topic_shift_pending FROM conversation_digests WHERE agent = ?",
                (agent,),
            ).fetchone()
            if row is None:
                conn.execute("INSERT INTO conversation_digests (agent) VALUES (?)", (agent,))
                conn.commit()
                row = (agent, "", 0, None, None, 0, 0)
        finally:
            conn.close()
    return _row_to_state(row)


def get_digest_text(agent: str) -> str:
    return get_state(agent)["digest"]


def note_turns(agent: str, count: int, timestamp: str | None = None) -> bool:
    """Record that `count` messages were just stored for this agent. Returns True if
    a natural pause boundary was just crossed -- the gap since the previous message
    exceeds PAUSE_TRIGGER_SECONDS with enough turns pending to be worth digesting --
    which the caller should treat as an immediate trigger regardless of the count
    threshold.
    """
    now = timestamp or datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _get_connection()
        try:
            row = conn.execute(
                "SELECT turns_since_digest, last_message_at FROM conversation_digests WHERE agent = ?",
                (agent,),
            ).fetchone()
            if row is None:
                conn.execute("INSERT INTO conversation_digests (agent) VALUES (?)", (agent,))
                turns_since_digest, last_message_at = 0, None
            else:
                turns_since_digest, last_message_at = row

            pause_triggered = False
            if last_message_at and turns_since_digest >= MIN_TURNS_FOR_PAUSE:
                gap = (datetime.fromisoformat(now) - datetime.fromisoformat(last_message_at)).total_seconds()
                pause_triggered = gap >= PAUSE_TRIGGER_SECONDS

            conn.execute(
                "UPDATE conversation_digests SET turns_since_digest = turns_since_digest + ?, "
                "last_message_at = ? WHERE agent = ?",
                (count, now, agent),
            )
            conn.commit()
        finally:
            conn.close()
    return pause_triggered


def should_digest(agent: str) -> bool:
    state = get_state(agent)
    backoff = min(1 + state["consecutive_failures"], MAX_BACKOFF_MULTIPLIER)
    threshold = MESSAGE_TRIGGER * backoff
    if state["topic_shift_pending"]:
        threshold = max(MIN_TURNS_FOR_PAUSE, threshold / TOPIC_SHIFT_DIVISOR)
    return state["turns_since_digest"] >= threshold


def try_acquire(agent: str) -> bool:
    with _active_lock:
        if agent in _active_agents:
            return False
        _active_agents.add(agent)
        return True


def release(agent: str) -> None:
    with _active_lock:
        _active_agents.discard(agent)


def apply_digest(agent: str, new_digest: str, topic_shift: bool) -> None:
    with _lock:
        conn = _get_connection()
        try:
            conn.execute("INSERT OR IGNORE INTO conversation_digests (agent) VALUES (?)", (agent,))
            conn.execute(
                "UPDATE conversation_digests SET digest = ?, turns_since_digest = 0, last_digested_at = ?, "
                "consecutive_failures = 0, topic_shift_pending = ? WHERE agent = ?",
                (new_digest, datetime.now(timezone.utc).isoformat(), int(topic_shift), agent),
            )
            conn.commit()
        finally:
            conn.close()


def record_failure(agent: str) -> None:
    with _lock:
        conn = _get_connection()
        try:
            conn.execute("INSERT OR IGNORE INTO conversation_digests (agent) VALUES (?)", (agent,))
            conn.execute(
                "UPDATE conversation_digests SET consecutive_failures = consecutive_failures + 1 WHERE agent = ?",
                (agent,),
            )
            conn.commit()
        finally:
            conn.close()


def _build_digest_messages(prior_digest: str, new_turns: list[dict]) -> list[dict]:
    lines = [f"What you currently know:\n{prior_digest}" if prior_digest else "You don't know anything about this user yet."]
    lines.append("\nNew exchanges since then:")
    for turn in new_turns:
        tag = ""
        if turn.get("feedback") == 1:
            tag = " [user liked this]"
        elif turn.get("feedback") == -1:
            tag = " [user disliked this]"
        lines.append(f"{turn['role']}: {turn['content']}{tag}")
    return [
        {"role": "system", "content": DIGEST_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


async def run_digest_cycle(agent: str, call_mistral) -> bool:
    """Runs one digestion cycle for `agent`, if one isn't already in flight.

    Best-effort background enrichment: failures are recorded (feeding the backoff in
    should_digest) but never raised, since this must never affect the chat path it
    runs alongside. Returns True if a digest was actually produced.
    """
    if not try_acquire(agent):
        return False
    try:
        state = await asyncio.to_thread(get_state, agent)
        new_turns = await asyncio.to_thread(get_messages_since, agent, state["last_digested_at"])
        if not new_turns:
            return False

        messages = _build_digest_messages(state["digest"], new_turns)
        try:
            raw_reply = await call_mistral(messages)
            parsed = json.loads(raw_reply)
            new_digest = parsed["digest"]
            topic_shift = bool(parsed.get("topic_shift", False))
            if not isinstance(new_digest, str):
                raise ValueError("digest must be a string")
        except Exception:
            await asyncio.to_thread(record_failure, agent)
            return False

        await asyncio.to_thread(apply_digest, agent, new_digest, topic_shift)
        return True
    finally:
        release(agent)
