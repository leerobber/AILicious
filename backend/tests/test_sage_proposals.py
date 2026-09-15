import pytest

from core import sage_proposals as sp_module


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(sp_module, "DB_PATH", tmp_path / "test_sage_proposals.db")
    sp_module.init_db()
    yield


def test_create_and_get_proposal_roundtrip():
    proposal_id = sp_module.create_proposal("nexus", "users liked concise answers", "Be concise.")

    proposal = sp_module.get_proposal(proposal_id)

    assert proposal["agent"] == "nexus"
    assert proposal["rationale"] == "users liked concise answers"
    assert proposal["proposed_system_prompt"] == "Be concise."
    assert proposal["status"] == "pending"
    assert proposal["reviewed_at"] is None


def test_get_proposal_unknown_id_returns_none():
    assert sp_module.get_proposal(999999) is None


def test_list_proposals_returns_all_most_recent_first():
    id1 = sp_module.create_proposal("nexus", "r1", "p1")
    id2 = sp_module.create_proposal("nexus", "r2", "p2")

    proposals = sp_module.list_proposals()

    assert [p["id"] for p in proposals] == [id2, id1]


def test_list_proposals_filters_by_agent():
    sp_module.create_proposal("nexus", "r", "p")
    sp_module.create_proposal("forge", "r", "p")

    assert len(sp_module.list_proposals(agent="forge")) == 1
    assert len(sp_module.list_proposals(agent="nexus")) == 1


def test_list_proposals_filters_by_status():
    accepted_id = sp_module.create_proposal("nexus", "r", "p")
    sp_module.create_proposal("nexus", "r2", "p2")
    sp_module.set_proposal_status(accepted_id, "accepted")

    assert [p["id"] for p in sp_module.list_proposals(status="accepted")] == [accepted_id]
    assert len(sp_module.list_proposals(status="pending")) == 1


def test_set_proposal_status_accept_transition():
    proposal_id = sp_module.create_proposal("nexus", "r", "p")

    assert sp_module.set_proposal_status(proposal_id, "accepted") is True

    proposal = sp_module.get_proposal(proposal_id)
    assert proposal["status"] == "accepted"
    assert proposal["reviewed_at"] is not None


def test_set_proposal_status_on_unknown_id_returns_false():
    assert sp_module.set_proposal_status(999999, "accepted") is False


def test_set_proposal_status_on_already_reviewed_proposal_returns_false():
    proposal_id = sp_module.create_proposal("nexus", "r", "p")
    sp_module.set_proposal_status(proposal_id, "accepted")

    assert sp_module.set_proposal_status(proposal_id, "rejected") is False
    assert sp_module.get_proposal(proposal_id)["status"] == "accepted"


def test_new_proposal_has_empty_outcome_fields():
    proposal_id = sp_module.create_proposal("nexus", "r", "p")

    proposal = sp_module.get_proposal(proposal_id)
    assert proposal["baseline_signal_quality"] is None
    assert proposal["prior_override"] is None
    assert proposal["outcome"] == ""
    assert proposal["outcome_reasoning"] is None
    assert proposal["evaluated_at"] is None


def test_record_acceptance_baseline_stores_snapshot():
    proposal_id = sp_module.create_proposal("nexus", "r", "p")
    sp_module.set_proposal_status(proposal_id, "accepted")

    sp_module.record_acceptance_baseline(proposal_id, "user likes concise answers", "You are a pirate.")

    proposal = sp_module.get_proposal(proposal_id)
    assert proposal["baseline_signal_quality"] == "user likes concise answers"
    assert proposal["prior_override"] == "You are a pirate."


def test_record_acceptance_baseline_allows_none_prior_override():
    proposal_id = sp_module.create_proposal("nexus", "r", "p")
    sp_module.set_proposal_status(proposal_id, "accepted")

    sp_module.record_acceptance_baseline(proposal_id, "signal", None)

    assert sp_module.get_proposal(proposal_id)["prior_override"] is None


def test_get_pending_evaluation_proposal_none_when_nothing_accepted():
    sp_module.create_proposal("nexus", "r", "p")

    assert sp_module.get_pending_evaluation_proposal("nexus") is None


def test_get_pending_evaluation_proposal_returns_accepted_unevaluated_proposal():
    proposal_id = sp_module.create_proposal("nexus", "r", "p")
    sp_module.set_proposal_status(proposal_id, "accepted")
    sp_module.record_acceptance_baseline(proposal_id, "signal", None)

    pending = sp_module.get_pending_evaluation_proposal("nexus")
    assert pending["id"] == proposal_id


def test_get_pending_evaluation_proposal_scoped_to_agent():
    proposal_id = sp_module.create_proposal("forge", "r", "p")
    sp_module.set_proposal_status(proposal_id, "accepted")
    sp_module.record_acceptance_baseline(proposal_id, "signal", None)

    assert sp_module.get_pending_evaluation_proposal("nexus") is None
    assert sp_module.get_pending_evaluation_proposal("forge")["id"] == proposal_id


def test_get_pending_evaluation_proposal_none_after_outcome_recorded():
    proposal_id = sp_module.create_proposal("nexus", "r", "p")
    sp_module.set_proposal_status(proposal_id, "accepted")
    sp_module.record_acceptance_baseline(proposal_id, "signal", None)

    sp_module.record_outcome(proposal_id, "improved", "things got better")

    assert sp_module.get_pending_evaluation_proposal("nexus") is None


def test_record_outcome_sets_fields():
    proposal_id = sp_module.create_proposal("nexus", "r", "p")
    sp_module.set_proposal_status(proposal_id, "accepted")
    sp_module.record_acceptance_baseline(proposal_id, "signal", None)

    sp_module.record_outcome(proposal_id, "regressed", "signal got worse")

    proposal = sp_module.get_proposal(proposal_id)
    assert proposal["outcome"] == "regressed"
    assert proposal["outcome_reasoning"] == "signal got worse"
    assert proposal["evaluated_at"] is not None


def test_get_pending_evaluation_proposal_ignores_rejected_proposals():
    proposal_id = sp_module.create_proposal("nexus", "r", "p")
    sp_module.set_proposal_status(proposal_id, "rejected")

    assert sp_module.get_pending_evaluation_proposal("nexus") is None
