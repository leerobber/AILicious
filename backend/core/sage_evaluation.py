import asyncio
import json

from core.events import record_event
from core.persona_overrides import clear_override, set_override
from core.sage_proposals import get_pending_evaluation_proposal, record_outcome

SAGE_EVALUATION_SYSTEM_PROMPT = (
    "You judge whether a persona change made to an AI agent actually helped. You'll be shown "
    "the rationale for the change, a 'signal quality' read on what the user responded well or "
    "poorly to from BEFORE the change, and a fresh signal-quality read from AFTER it took "
    "effect -- both inferred from the user's own behavior (rephrasing, corrections, reuse, "
    "affirmations), not explicit ratings. Judge only whether the after-read shows things "
    "getting better, getting worse, or staying too ambiguous to tell -- not whether the "
    "rationale itself was reasonable.\n\n"
    'Respond with ONLY a JSON object: {"verdict": "improved"|"regressed"|"unclear", '
    '"reasoning": "<one sentence>"}. No markdown, no extra text -- valid JSON only.'
)


def _build_evaluation_messages(
    agent: str, rationale: str, baseline_signal_quality: str, new_signal_quality: str
) -> list[dict]:
    lines = [
        f"Persona change made to '{agent}', with this rationale:",
        rationale,
        "",
        "Signal-quality read from BEFORE the change:",
        baseline_signal_quality or "(none yet)",
        "",
        "Signal-quality read from AFTER the change:",
        new_signal_quality or "(none yet)",
    ]
    return [
        {"role": "system", "content": SAGE_EVALUATION_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


async def run_pending_evaluation(agent: str, new_signal_quality: str, call_mistral) -> bool:
    """Checks whether `agent` has an accepted-but-unevaluated SAGE proposal and, if so,
    judges it against fresh signal quality. A 'regressed' verdict reverts the override back
    to whatever was in effect before acceptance. Best-effort, same posture as
    core.digest.run_digest_cycle and core.user_profile.run_profile_merge: a malformed judge
    response is left unevaluated (retried on the next digest cycle) rather than applied or
    raised into the chat path that triggered it. Returns True if an evaluation was recorded.
    """
    proposal = await asyncio.to_thread(get_pending_evaluation_proposal, agent)
    if proposal is None:
        return False

    messages = _build_evaluation_messages(
        agent, proposal["rationale"], proposal["baseline_signal_quality"] or "", new_signal_quality
    )
    try:
        raw_reply = await call_mistral(messages)
        parsed = json.loads(raw_reply)
        verdict = parsed["verdict"]
        reasoning = parsed.get("reasoning", "")
        if verdict not in ("improved", "regressed", "unclear"):
            raise ValueError(f"unrecognized verdict: {verdict!r}")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)
    except Exception as exc:
        await asyncio.to_thread(record_event, "sage", "evaluation_failed", agent, str(exc)[:200])
        return False

    await asyncio.to_thread(record_outcome, proposal["id"], verdict, reasoning)
    await asyncio.to_thread(
        record_event, "sage", "evaluated", agent, f"proposal_id={proposal['id']} verdict={verdict}"
    )

    if verdict == "regressed":
        prior_override = proposal["prior_override"]
        if prior_override is None:
            await asyncio.to_thread(clear_override, agent)
        else:
            await asyncio.to_thread(set_override, agent, prior_override)
        await asyncio.to_thread(
            record_event, "sage", "reverted", agent, f"proposal_id={proposal['id']}"
        )

    return True
