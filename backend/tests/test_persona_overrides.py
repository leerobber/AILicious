import pytest

from core import persona_overrides as po_module


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(po_module, "DB_PATH", tmp_path / "test_overrides.db")
    po_module.init_db()
    yield


def test_get_override_returns_none_when_absent():
    assert po_module.get_override("nexus") is None


def test_set_and_get_override_roundtrip():
    po_module.set_override("nexus", "You are a pirate.")

    assert po_module.get_override("nexus") == "You are a pirate."


def test_set_override_upserts_existing_agent():
    po_module.set_override("nexus", "first version")
    po_module.set_override("nexus", "second version")

    assert po_module.get_override("nexus") == "second version"


def test_clear_override_reverts_to_none():
    po_module.set_override("nexus", "You are a pirate.")
    po_module.clear_override("nexus")

    assert po_module.get_override("nexus") is None


def test_clear_override_on_agent_without_one_is_a_no_op():
    po_module.clear_override("nexus")  # must not raise

    assert po_module.get_override("nexus") is None


def test_overrides_are_scoped_per_agent():
    po_module.set_override("nexus", "nexus override")

    assert po_module.get_override("forge") is None
    assert po_module.get_override("nexus") == "nexus override"
