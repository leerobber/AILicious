import pytest

from core.persona import get_system_prompt, has_persona, list_personas, load_personas

ALL_AGENTS = {"avery", "codex", "forge", "nexus", "oracle", "sentinel"}


def test_load_and_list_personas():
    load_personas()
    assert set(list_personas()) == ALL_AGENTS


def test_get_system_prompt_identifies_each_agent():
    load_personas()
    for agent in ALL_AGENTS:
        assert agent.upper() in get_system_prompt(agent)


def test_has_persona():
    load_personas()
    assert has_persona("nexus") is True
    assert has_persona("nonexistent") is False


def test_get_system_prompt_unknown_raises_keyerror():
    load_personas()
    with pytest.raises(KeyError):
        get_system_prompt("nonexistent")
