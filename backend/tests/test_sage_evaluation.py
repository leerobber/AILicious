import asyncio
import json

import pytest

from core import persona_overrides as persona_overrides_module
from core import sage_evaluation as sage_evaluation_module
from core import sage_proposals as sage_proposals_module


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_sage_evaluation.db"
    monkeypatch.setattr(sage_proposals_module, "DB_PATH", db_path)
    monkeypatch.setattr(persona_overrides_module, "DB_PATH", db_path)
    sage_proposals_module.init_db()
    persona_overrides_module.init_db()
    yield


def _run(coro):
    return asyncio.run(coro)


def _accepted_proposal(agent="nexus", baseline_signal_quality="user likes concise answers", prior_override=None):
    proposal_id = sage_proposals_module.create_proposal(agent, "rationale", "Be warmer.")
    sage_proposals_module.set_proposal_status(proposal_id, "accepted")
    sage_proposals_module.record_acceptance_baseline(proposal_id, baseline_signal_quality, prior_override)
    return proposal_id


def test_no_op_when_nothing_pending_evaluation():
    async def fake_call_mistral(messages):
        raise AssertionError("should not call Mistral when there's nothing to evaluate")

    result = _run(sage_evaluation_module.run_pending_evaluation("nexus", "some new signal", fake_call_mistral))

    assert result is False


def test_improved_verdict_records_outcome_and_does_not_touch_override():
    proposal_id = _accepted_proposal(prior_override="You are a pirate.")
    persona_overrides_module.set_override("nexus", "Be warmer.")

    async def fake_call_mistral(messages):
        return json.dumps({"verdict": "improved", "reasoning": "user affirms more often now"})

    result = _run(sage_evaluation_module.run_pending_evaluation("nexus", "user affirms more often", fake_call_mistral))

    assert result is True
    proposal = sage_proposals_module.get_proposal(proposal_id)
    assert proposal["outcome"] == "improved"
    assert proposal["outcome_reasoning"] == "user affirms more often now"
    assert persona_overrides_module.get_override("nexus") == "Be warmer."


def test_unclear_verdict_records_outcome_and_does_not_touch_override():
    _accepted_proposal(prior_override=None)
    persona_overrides_module.set_override("nexus", "Be warmer.")

    async def fake_call_mistral(messages):
        return json.dumps({"verdict": "unclear", "reasoning": "not enough signal yet"})

    result = _run(sage_evaluation_module.run_pending_evaluation("nexus", "", fake_call_mistral))

    assert result is True
    assert persona_overrides_module.get_override("nexus") == "Be warmer."


def test_regressed_verdict_reverts_to_prior_override():
    proposal_id = _accepted_proposal(prior_override="You are a pirate.")
    persona_overrides_module.set_override("nexus", "Be warmer.")

    async def fake_call_mistral(messages):
        return json.dumps({"verdict": "regressed", "reasoning": "user corrects it constantly now"})

    result = _run(sage_evaluation_module.run_pending_evaluation("nexus", "constant corrections", fake_call_mistral))

    assert result is True
    assert persona_overrides_module.get_override("nexus") == "You are a pirate."
    assert sage_proposals_module.get_proposal(proposal_id)["outcome"] == "regressed"


def test_regressed_verdict_with_no_prior_override_clears_to_yaml_default():
    _accepted_proposal(prior_override=None)
    persona_overrides_module.set_override("nexus", "Be warmer.")

    async def fake_call_mistral(messages):
        return json.dumps({"verdict": "regressed", "reasoning": "worse"})

    _run(sage_evaluation_module.run_pending_evaluation("nexus", "worse signal", fake_call_mistral))

    assert persona_overrides_module.get_override("nexus") is None


def test_malformed_response_leaves_proposal_unevaluated_and_override_untouched():
    _accepted_proposal(prior_override="You are a pirate.")
    persona_overrides_module.set_override("nexus", "Be warmer.")

    async def fake_call_mistral(messages):
        return "not json at all"

    result = _run(sage_evaluation_module.run_pending_evaluation("nexus", "some signal", fake_call_mistral))

    assert result is False
    assert persona_overrides_module.get_override("nexus") == "Be warmer."
    # Left unevaluated (outcome still ''), not corrupted with a bogus value -- retried next cycle.
    pending = sage_proposals_module.get_pending_evaluation_proposal("nexus")
    assert pending is not None


def test_unrecognized_verdict_value_is_treated_as_malformed():
    _accepted_proposal(prior_override=None)

    async def fake_call_mistral(messages):
        return json.dumps({"verdict": "definitely great", "reasoning": "great"})

    result = _run(sage_evaluation_module.run_pending_evaluation("nexus", "signal", fake_call_mistral))

    assert result is False
    assert sage_proposals_module.get_pending_evaluation_proposal("nexus") is not None


def test_evaluation_is_scoped_to_agent():
    _accepted_proposal(agent="forge", prior_override=None)

    async def fake_call_mistral(messages):
        raise AssertionError("should not evaluate nexus when only forge has a pending proposal")

    result = _run(sage_evaluation_module.run_pending_evaluation("nexus", "signal", fake_call_mistral))

    assert result is False


def test_sends_rationale_and_both_signal_quality_reads_to_judge():
    _accepted_proposal(baseline_signal_quality="old read", prior_override=None)
    seen_messages = []

    async def fake_call_mistral(messages):
        seen_messages.append(messages)
        return json.dumps({"verdict": "unclear", "reasoning": ""})

    _run(sage_evaluation_module.run_pending_evaluation("nexus", "new read", fake_call_mistral))

    sent_content = seen_messages[0][1]["content"]
    assert "old read" in sent_content
    assert "new read" in sent_content
    assert "rationale" in sent_content
