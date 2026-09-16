import pytest

from core import events as events_module


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(events_module, "DB_PATH", tmp_path / "test_events.db")
    events_module.init_db()
    yield


def test_record_and_list_event_roundtrip():
    events_module.record_event("digest", "applied", "nexus", "turns=10 topic_shift=False")

    events = events_module.list_events()

    assert len(events) == 1
    assert events[0]["category"] == "digest"
    assert events[0]["event"] == "applied"
    assert events[0]["agent"] == "nexus"
    assert events[0]["detail"] == "turns=10 topic_shift=False"
    assert events[0]["created_at"] is not None


def test_record_event_without_agent_or_detail():
    events_module.record_event("embedding", "stored")

    events = events_module.list_events()

    assert events[0]["agent"] is None
    assert events[0]["detail"] == ""


def test_list_events_most_recent_first():
    events_module.record_event("digest", "first")
    events_module.record_event("digest", "second")

    events = events_module.list_events()

    assert [e["event"] for e in events] == ["second", "first"]


def test_list_events_filters_by_category():
    events_module.record_event("digest", "applied")
    events_module.record_event("sage", "evaluated")

    assert [e["category"] for e in events_module.list_events(category="sage")] == ["sage"]


def test_list_events_filters_by_agent():
    events_module.record_event("digest", "applied", agent="nexus")
    events_module.record_event("digest", "applied", agent="forge")

    assert [e["agent"] for e in events_module.list_events(agent="forge")] == ["forge"]


def test_list_events_respects_limit():
    for i in range(5):
        events_module.record_event("digest", f"event{i}")

    assert len(events_module.list_events(limit=2)) == 2


def test_record_event_swallows_failure_instead_of_raising(monkeypatch):
    """The table doesn't exist yet in this fresh, un-migrated path -- record_event must
    not raise even so, since a failure to persist the record of a cycle must never look
    like a failure of the cycle itself."""
    monkeypatch.setattr(events_module, "DB_PATH", events_module.DB_PATH.parent / "never_initialized.db")

    events_module.record_event("digest", "applied")  # must not raise
