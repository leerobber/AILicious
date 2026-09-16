import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from core import digest as digest_module
from core import events as events_module
from core import memory as memory_module


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_digest.db"
    monkeypatch.setattr(digest_module, "DB_PATH", db_path)
    monkeypatch.setattr(memory_module, "DB_PATH", db_path)
    monkeypatch.setattr(events_module, "DB_PATH", db_path)
    digest_module.init_db()
    memory_module.init_db()
    events_module.init_db()
    digest_module._active_agents.clear()
    yield
    digest_module._active_agents.clear()


def test_get_state_defaults_for_unknown_agent():
    state = digest_module.get_state("nexus")

    assert state == {
        "agent": "nexus",
        "digest": "",
        "turns_since_digest": 0,
        "last_message_at": None,
        "last_digested_at": None,
        "consecutive_failures": 0,
        "topic_shift_pending": False,
        "signal_quality": "",
    }


def test_get_digest_text_empty_when_none_applied():
    assert digest_module.get_digest_text("nexus") == ""


def test_note_turns_increments_count():
    digest_module.note_turns("nexus", 2, "2026-01-01T00:00:00+00:00")

    assert digest_module.get_state("nexus")["turns_since_digest"] == 2


def test_note_turns_no_pause_on_first_ever_turn():
    pause = digest_module.note_turns("nexus", 2, "2026-01-01T00:00:00+00:00")

    assert pause is False


def test_note_turns_detects_pause_after_enough_pending_turns():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    digest_module.note_turns("nexus", digest_module.MIN_TURNS_FOR_PAUSE, t0.isoformat())

    long_gap = (t0 + timedelta(seconds=digest_module.PAUSE_TRIGGER_SECONDS + 60)).isoformat()
    pause = digest_module.note_turns("nexus", 2, long_gap)

    assert pause is True


def test_note_turns_no_pause_below_min_turns_threshold():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    digest_module.note_turns("nexus", digest_module.MIN_TURNS_FOR_PAUSE - 1, t0.isoformat())

    long_gap = (t0 + timedelta(seconds=digest_module.PAUSE_TRIGGER_SECONDS + 60)).isoformat()
    pause = digest_module.note_turns("nexus", 2, long_gap)

    assert pause is False


def test_note_turns_no_pause_for_short_gap():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    digest_module.note_turns("nexus", digest_module.MIN_TURNS_FOR_PAUSE, t0.isoformat())

    short_gap = (t0 + timedelta(seconds=5)).isoformat()
    pause = digest_module.note_turns("nexus", 2, short_gap)

    assert pause is False


def test_should_digest_false_below_threshold():
    digest_module.note_turns("nexus", digest_module.MESSAGE_TRIGGER - 1)

    assert digest_module.should_digest("nexus") is False


def test_should_digest_true_at_threshold():
    digest_module.note_turns("nexus", digest_module.MESSAGE_TRIGGER)

    assert digest_module.should_digest("nexus") is True


def test_should_digest_backoff_raises_effective_threshold():
    digest_module.note_turns("nexus", digest_module.MESSAGE_TRIGGER)
    digest_module.record_failure("nexus")  # backoff multiplier now 2x

    assert digest_module.should_digest("nexus") is False


def test_should_digest_topic_shift_lowers_effective_threshold():
    digest_module.apply_digest("nexus", "prior digest", topic_shift=True)
    digest_module.note_turns("nexus", digest_module.MESSAGE_TRIGGER // 2)

    assert digest_module.should_digest("nexus") is True


def test_try_acquire_coalesces_concurrent_triggers():
    assert digest_module.try_acquire("nexus") is True
    assert digest_module.try_acquire("nexus") is False  # already running, coalesced

    digest_module.release("nexus")

    assert digest_module.try_acquire("nexus") is True  # free again after release


def test_try_acquire_scoped_per_agent():
    assert digest_module.try_acquire("nexus") is True
    assert digest_module.try_acquire("forge") is True  # independent lock


def test_apply_digest_resets_turns_and_failures():
    digest_module.note_turns("nexus", 5)
    digest_module.record_failure("nexus")

    digest_module.apply_digest("nexus", "the new digest", topic_shift=True, signal_quality="prefers brevity")

    state = digest_module.get_state("nexus")
    assert state["digest"] == "the new digest"
    assert state["turns_since_digest"] == 0
    assert state["consecutive_failures"] == 0
    assert state["topic_shift_pending"] is True
    assert state["last_digested_at"] is not None
    assert state["signal_quality"] == "prefers brevity"


def test_apply_digest_signal_quality_defaults_to_empty():
    digest_module.apply_digest("nexus", "the new digest", topic_shift=False)

    assert digest_module.get_state("nexus")["signal_quality"] == ""


def test_record_failure_increments_consecutive_failures():
    digest_module.record_failure("nexus")
    digest_module.record_failure("nexus")

    assert digest_module.get_state("nexus")["consecutive_failures"] == 2


def _run(coro):
    return asyncio.run(coro)


def test_run_digest_cycle_applies_a_valid_merged_digest():
    memory_module.add_message("s1", "nexus", "user", "my favorite color is teal")
    memory_module.add_message("s1", "nexus", "assistant", "noted")

    async def fake_call_mistral(messages):
        return json.dumps({"digest": "User's favorite color is teal.", "topic_shift": False})

    result = _run(digest_module.run_digest_cycle("nexus", fake_call_mistral))

    assert result is True
    state = digest_module.get_state("nexus")
    assert state["digest"] == "User's favorite color is teal."
    assert state["turns_since_digest"] == 0


def test_run_digest_cycle_stores_inferred_signal_quality():
    memory_module.add_message("s1", "nexus", "user", "explain X")
    memory_module.add_message("s1", "nexus", "assistant", "a long rambling answer")
    memory_module.add_message("s1", "nexus", "user", "no, simpler please")

    async def fake_call_mistral(messages):
        return json.dumps(
            {
                "digest": "Asked about X.",
                "topic_shift": False,
                "signal_quality": "Prefers short, simple answers over long ones.",
            }
        )

    result = _run(digest_module.run_digest_cycle("nexus", fake_call_mistral))

    assert result is True
    assert digest_module.get_state("nexus")["signal_quality"] == "Prefers short, simple answers over long ones."


def test_run_digest_cycle_missing_signal_quality_keeps_prior_value():
    digest_module.apply_digest("nexus", "prior digest", topic_shift=False, signal_quality="likes code examples")
    memory_module.add_message("s1", "nexus", "user", "another message")

    async def fake_call_mistral(messages):
        return json.dumps({"digest": "updated digest", "topic_shift": False})

    result = _run(digest_module.run_digest_cycle("nexus", fake_call_mistral))

    assert result is True
    assert digest_module.get_state("nexus")["signal_quality"] == "likes code examples"


def test_run_digest_cycle_non_string_signal_quality_keeps_prior_value():
    digest_module.apply_digest("nexus", "prior digest", topic_shift=False, signal_quality="likes code examples")
    memory_module.add_message("s1", "nexus", "user", "another message")

    async def fake_call_mistral(messages):
        return json.dumps({"digest": "updated digest", "topic_shift": False, "signal_quality": 12345})

    result = _run(digest_module.run_digest_cycle("nexus", fake_call_mistral))

    assert result is True
    assert digest_module.get_state("nexus")["signal_quality"] == "likes code examples"


def test_build_digest_messages_includes_prior_signal_quality():
    messages = digest_module._build_digest_messages(
        "prior digest", [{"role": "user", "content": "hi", "feedback": None}], "prefers brevity"
    )

    assert "prefers brevity" in messages[1]["content"]


def test_run_digest_cycle_no_new_turns_is_a_noop():
    calls = []

    async def fake_call_mistral(messages):
        calls.append(messages)
        return json.dumps({"digest": "should not be reached", "topic_shift": False})

    result = _run(digest_module.run_digest_cycle("nexus", fake_call_mistral))

    assert result is False
    assert calls == []  # no wasted Mistral call when there's nothing to digest


def test_run_digest_cycle_malformed_response_records_failure_without_raising():
    memory_module.add_message("s1", "nexus", "user", "hello")

    async def fake_call_mistral(messages):
        return "this is not JSON"

    result = _run(digest_module.run_digest_cycle("nexus", fake_call_mistral))

    assert result is False
    state = digest_module.get_state("nexus")
    assert state["consecutive_failures"] == 1
    assert state["digest"] == ""  # unchanged on failure


def test_run_digest_cycle_skips_when_already_in_flight():
    memory_module.add_message("s1", "nexus", "user", "hello")
    digest_module.try_acquire("nexus")  # simulate a cycle already running

    calls = []

    async def fake_call_mistral(messages):
        calls.append(messages)
        return json.dumps({"digest": "x", "topic_shift": False})

    result = _run(digest_module.run_digest_cycle("nexus", fake_call_mistral))

    assert result is False
    assert calls == []


def test_run_digest_cycle_records_applied_event_on_success():
    memory_module.add_message("s1", "nexus", "user", "hello")

    async def fake_call_mistral(messages):
        return json.dumps({"digest": "x", "topic_shift": True})

    _run(digest_module.run_digest_cycle("nexus", fake_call_mistral))

    events = events_module.list_events(category="digest", agent="nexus")
    assert events[0]["event"] == "applied"
    assert "topic_shift=True" in events[0]["detail"]


def test_run_digest_cycle_records_failed_event_on_malformed_response():
    memory_module.add_message("s1", "nexus", "user", "hello")

    async def fake_call_mistral(messages):
        return "this is not JSON"

    _run(digest_module.run_digest_cycle("nexus", fake_call_mistral))

    events = events_module.list_events(category="digest", agent="nexus")
    assert events[0]["event"] == "failed"


def test_run_digest_cycle_records_skipped_coalescing_event():
    memory_module.add_message("s1", "nexus", "user", "hello")
    digest_module.try_acquire("nexus")

    async def fake_call_mistral(messages):
        return json.dumps({"digest": "x", "topic_shift": False})

    _run(digest_module.run_digest_cycle("nexus", fake_call_mistral))

    events = events_module.list_events(category="digest", agent="nexus")
    assert events[0]["event"] == "skipped_coalescing"


def test_run_digest_cycle_records_skipped_no_new_turns_event():
    async def fake_call_mistral(messages):
        return json.dumps({"digest": "x", "topic_shift": False})

    _run(digest_module.run_digest_cycle("nexus", fake_call_mistral))

    events = events_module.list_events(category="digest", agent="nexus")
    assert events[0]["event"] == "skipped_no_new_turns"


def test_run_digest_cycle_incremental_only_sends_new_turns():
    memory_module.add_message("s1", "nexus", "user", "first message")
    memory_module.add_message("s1", "nexus", "assistant", "first reply")

    async def first_call(messages):
        return json.dumps({"digest": "Knows about the first exchange.", "topic_shift": False})

    _run(digest_module.run_digest_cycle("nexus", first_call))

    memory_module.add_message("s1", "nexus", "user", "second message")
    memory_module.add_message("s1", "nexus", "assistant", "second reply")

    captured = {}

    async def second_call(messages):
        captured["prompt"] = messages[1]["content"]
        return json.dumps({"digest": "Merged digest.", "topic_shift": False})

    _run(digest_module.run_digest_cycle("nexus", second_call))

    # The second cycle's prompt must carry the prior digest forward and include only
    # the new turns -- not "first message"/"first reply" again.
    assert "Knows about the first exchange." in captured["prompt"]
    assert "second message" in captured["prompt"]
    assert "first message" not in captured["prompt"]
