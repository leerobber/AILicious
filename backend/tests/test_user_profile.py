import asyncio
import json

import pytest

from core import events as events_module
from core import user_profile as user_profile_module


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_user_profile.db"
    monkeypatch.setattr(user_profile_module, "DB_PATH", db_path)
    monkeypatch.setattr(events_module, "DB_PATH", db_path)
    user_profile_module.init_db()
    events_module.init_db()
    yield


def _run(coro):
    return asyncio.run(coro)


def test_get_profile_empty_when_none_applied():
    assert user_profile_module.get_profile() == ""


def test_get_profile_state_defaults_when_none_applied():
    assert user_profile_module.get_profile_state() == {"profile": "", "updated_at": None}


def test_apply_profile_round_trips():
    user_profile_module.apply_profile("Name: Lee. Prefers concise answers.")

    assert user_profile_module.get_profile() == "Name: Lee. Prefers concise answers."


def test_apply_profile_overwrites_prior_value():
    user_profile_module.apply_profile("first")
    user_profile_module.apply_profile("second")

    assert user_profile_module.get_profile() == "second"


def test_apply_profile_sets_updated_at():
    user_profile_module.apply_profile("Name: Lee.")

    assert user_profile_module.get_profile_state()["updated_at"] is not None


def test_run_profile_merge_no_op_when_digest_empty():
    async def fake_call_mistral(messages):
        raise AssertionError("should not call Mistral when there's no digest to merge")

    result = _run(user_profile_module.run_profile_merge("nexus", "", fake_call_mistral))

    assert result is False
    assert user_profile_module.get_profile() == ""


def test_run_profile_merge_applies_valid_merge():
    async def fake_call_mistral(messages):
        return json.dumps({"profile": "Name: Lee. Works on AILicious."})

    result = _run(user_profile_module.run_profile_merge("nexus", "User mentioned their name is Lee.", fake_call_mistral))

    assert result is True
    assert user_profile_module.get_profile() == "Name: Lee. Works on AILicious."


def test_run_profile_merge_sends_prior_profile_and_agent_digest():
    user_profile_module.apply_profile("Name: Lee.")
    seen_messages = []

    async def fake_call_mistral(messages):
        seen_messages.append(messages)
        return json.dumps({"profile": "Name: Lee. Prefers dark mode."})

    _run(user_profile_module.run_profile_merge("forge", "User said they prefer dark mode.", fake_call_mistral))

    sent_user_content = seen_messages[0][1]["content"]
    assert "Name: Lee." in sent_user_content
    assert "User said they prefer dark mode." in sent_user_content
    assert "forge" in sent_user_content


def test_run_profile_merge_malformed_response_does_not_raise_or_apply():
    async def fake_call_mistral(messages):
        return "this is not JSON at all"

    result = _run(user_profile_module.run_profile_merge("nexus", "some digest", fake_call_mistral))

    assert result is False
    assert user_profile_module.get_profile() == ""


def test_run_profile_merge_non_string_profile_does_not_apply():
    async def fake_call_mistral(messages):
        return json.dumps({"profile": 12345})

    result = _run(user_profile_module.run_profile_merge("nexus", "some digest", fake_call_mistral))

    assert result is False
    assert user_profile_module.get_profile() == ""


def test_run_profile_merge_preserves_unrelated_prior_facts():
    """Deliberate-break-style check: a merge that only adds a new fact must not silently
    drop what was already known -- this only passes because the fake Mistral reply itself
    includes the prior fact, mirroring what a real incremental-merge prompt is for."""
    user_profile_module.apply_profile("Name: Lee.")

    async def fake_call_mistral(messages):
        return json.dumps({"profile": "Name: Lee. Also prefers Rust for performance-critical code."})

    _run(user_profile_module.run_profile_merge("forge", "User said they like Rust for perf-critical work.", fake_call_mistral))

    profile = user_profile_module.get_profile()
    assert "Name: Lee." in profile
    assert "Rust" in profile


def test_run_profile_merge_records_merged_event_on_success():
    async def fake_call_mistral(messages):
        return json.dumps({"profile": "Name: Lee."})

    _run(user_profile_module.run_profile_merge("nexus", "User's name is Lee.", fake_call_mistral))

    events = events_module.list_events(category="profile", agent="nexus")
    assert events[0]["event"] == "merged"


def test_run_profile_merge_records_merge_failed_event_on_malformed_response():
    async def fake_call_mistral(messages):
        return "not json"

    _run(user_profile_module.run_profile_merge("nexus", "some digest", fake_call_mistral))

    events = events_module.list_events(category="profile", agent="nexus")
    assert events[0]["event"] == "merge_failed"
