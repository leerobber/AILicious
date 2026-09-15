import json
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

from core.ratelimit import reset as reset_rate_limit
from main import app

VALID_KEY = "test-app-key"
sent_requests: list[dict] = []
mock_reply_content = "mocked reply"


async def _fake_post(self, url, json=None, headers=None, **kwargs):
    sent_requests.append(json)
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": mock_reply_content}}]},
        request=httpx.Request("POST", url),
    )


@pytest.fixture(autouse=True)
def _reset_rate_limit_state():
    reset_rate_limit()
    yield
    reset_rate_limit()


scheduled_digest_agents: list[str] = []


@pytest.fixture
def client(tmp_path, monkeypatch):
    from core import digest as digest_module
    from core import memory as memory_module
    from core import persona_overrides as persona_overrides_module
    from core import sage_proposals as sage_proposals_module
    from core import user_profile as user_profile_module

    db_path = tmp_path / "api_test.db"
    monkeypatch.setattr(memory_module, "DB_PATH", db_path)
    monkeypatch.setattr(persona_overrides_module, "DB_PATH", db_path)
    monkeypatch.setattr(sage_proposals_module, "DB_PATH", db_path)
    monkeypatch.setattr(digest_module, "DB_PATH", db_path)
    monkeypatch.setattr(user_profile_module, "DB_PATH", db_path)
    digest_module._active_agents.clear()
    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    sent_requests.clear()
    monkeypatch.setattr(f"{__name__}.mock_reply_content", "mocked reply")

    import main as main_module

    # Digestion is fire-and-forget (asyncio.create_task); racing that against a
    # synchronous TestClient request is exactly the kind of flakiness that pattern
    # is supposed to avoid. Tests that care about scheduling assert against this
    # recorder instead; tests that need real digest content call core.digest
    # directly (see test_digest.py) rather than relying on background timing here.
    scheduled_digest_agents.clear()
    monkeypatch.setattr(main_module, "_schedule_digest", scheduled_digest_agents.append)

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_agents_requires_api_key(client):
    resp = client.get("/agents")
    assert resp.status_code == 401


def test_agents_lists_all_personas(client):
    resp = client.get("/agents", headers={"X-API-Key": VALID_KEY})
    assert resp.status_code == 200
    assert set(resp.json()["agents"]) == {"avery", "codex", "forge", "nexus", "oracle", "sentinel"}


def test_chat_returns_mocked_reply_and_session_id(client):
    resp = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "mocked reply"
    assert body["agent"] == "nexus"
    assert body["session_id"]


def test_chat_without_key_is_rejected(client):
    resp = client.post("/chat", json={"message": "hello"})
    assert resp.status_code == 401


def test_agents_route_talks_to_named_agent(client):
    resp = client.post("/agents/forge", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"})
    assert resp.status_code == 200
    assert resp.json()["agent"] == "forge"


def test_agents_route_unknown_agent_is_404(client):
    resp = client.post("/agents/doesnotexist", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"})
    assert resp.status_code == 404


def test_chat_response_includes_message_id(client):
    resp = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"})
    assert isinstance(resp.json()["message_id"], int)


def test_feedback_accepts_valid_rating(client):
    message_id = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"}).json()["message_id"]

    resp = client.post("/feedback", headers={"X-API-Key": VALID_KEY}, json={"message_id": message_id, "rating": 1})
    assert resp.status_code == 200


def test_feedback_invalid_rating_is_400(client):
    message_id = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"}).json()["message_id"]

    resp = client.post("/feedback", headers={"X-API-Key": VALID_KEY}, json={"message_id": message_id, "rating": 5})
    assert resp.status_code == 400


def test_feedback_unknown_message_id_is_404(client):
    resp = client.post("/feedback", headers={"X-API-Key": VALID_KEY}, json={"message_id": 999999, "rating": 1})
    assert resp.status_code == 404


def test_feedback_requires_api_key(client):
    resp = client.post("/feedback", json={"message_id": 1, "rating": 1})
    assert resp.status_code == 401


def test_message_over_length_cap_is_400(client, monkeypatch):
    import main as main_module

    monkeypatch.setattr(main_module, "MAX_MESSAGE_LENGTH", 5)
    resp = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "this is too long"})
    assert resp.status_code == 400


def test_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    import main as main_module

    monkeypatch.setattr(main_module, "RATE_LIMIT_PER_MINUTE", 1)

    first = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "one"})
    assert first.status_code == 200

    second = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "two"})
    assert second.status_code == 429
    assert "Retry-After" in second.headers


def test_memory_stats_reflects_chat_activity(client):
    client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"})

    resp = client.get("/memory/stats", headers={"X-API-Key": VALID_KEY})
    assert resp.status_code == 200
    assert resp.json()["total_messages"] == 2  # the user turn + the mocked assistant reply


def test_sage_current_requires_api_key(client):
    resp = client.get("/sage/current/nexus")
    assert resp.status_code == 401


def test_sage_current_unknown_agent_is_404(client):
    resp = client.get("/sage/current/doesnotexist", headers={"X-API-Key": VALID_KEY})
    assert resp.status_code == 404


def test_sage_current_returns_yaml_default_when_no_override(client):
    resp = client.get("/sage/current/nexus", headers={"X-API-Key": VALID_KEY})
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent"] == "nexus"
    assert body["is_override"] is False
    assert "NEXUS" in body["system_prompt"]


def test_sage_current_returns_override_when_set(client):
    client.post(
        "/sage/override/nexus",
        headers={"X-API-Key": VALID_KEY},
        json={"system_prompt": "You are a pirate."},
    )

    resp = client.get("/sage/current/nexus", headers={"X-API-Key": VALID_KEY})
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_override"] is True
    assert body["system_prompt"] == "You are a pirate."


def test_sage_override_requires_api_key(client):
    resp = client.post("/sage/override/nexus", json={"system_prompt": "You are a pirate."})
    assert resp.status_code == 401


def test_sage_override_unknown_agent_is_404(client):
    resp = client.post(
        "/sage/override/doesnotexist",
        headers={"X-API-Key": VALID_KEY},
        json={"system_prompt": "You are a pirate."},
    )
    assert resp.status_code == 404


def test_sage_override_changes_outgoing_system_prompt(client):
    override_resp = client.post(
        "/sage/override/nexus",
        headers={"X-API-Key": VALID_KEY},
        json={"system_prompt": "You are a pirate. Speak only in pirate slang."},
    )
    assert override_resp.status_code == 200

    client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"})

    outgoing_system_message = sent_requests[-1]["messages"][0]
    assert outgoing_system_message["content"] == "You are a pirate. Speak only in pirate slang."


def test_sage_reset_reverts_to_default_persona(client):
    client.post(
        "/sage/override/nexus",
        headers={"X-API-Key": VALID_KEY},
        json={"system_prompt": "You are a pirate."},
    )

    reset_resp = client.post("/sage/reset/nexus", headers={"X-API-Key": VALID_KEY})
    assert reset_resp.status_code == 200

    client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"})

    outgoing_system_message = sent_requests[-1]["messages"][0]
    assert outgoing_system_message["content"] != "You are a pirate."


def test_sage_analyze_requires_api_key(client):
    resp = client.post("/sage/analyze/nexus")
    assert resp.status_code == 401


def test_sage_analyze_unknown_agent_is_404(client):
    resp = client.post("/sage/analyze/doesnotexist", headers={"X-API-Key": VALID_KEY})
    assert resp.status_code == 404


def test_sage_analyze_with_no_digest_is_400(client):
    resp = client.post("/sage/analyze/nexus", headers={"X-API-Key": VALID_KEY})
    assert resp.status_code == 400


def test_sage_analyze_with_digest_but_no_signal_quality_is_400(client):
    from core import digest as digest_module

    digest_module.apply_digest("nexus", "some digest", topic_shift=False)  # no signal_quality

    resp = client.post("/sage/analyze/nexus", headers={"X-API-Key": VALID_KEY})
    assert resp.status_code == 400


def _seed_digest(
    agent: str,
    text: str = "User prefers concise, direct answers.",
    signal_quality: str = "Rephrases when answers are too long; affirms short, direct ones.",
) -> None:
    from core import digest as digest_module

    digest_module.apply_digest(agent, text, topic_shift=False, signal_quality=signal_quality)


def test_sage_analyze_creates_pending_proposal(client, monkeypatch):
    test_api_module = sys.modules[__name__]

    _seed_digest("nexus")

    monkeypatch.setattr(
        test_api_module,
        "mock_reply_content",
        json.dumps({"rationale": "users liked warmth", "proposed_system_prompt": "Be warm and friendly."}),
    )

    resp = client.post("/sage/analyze/nexus", headers={"X-API-Key": VALID_KEY})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending"
    assert body["rationale"] == "users liked warmth"
    assert body["proposed_system_prompt"] == "Be warm and friendly."

    proposals = client.get("/sage/proposals", headers={"X-API-Key": VALID_KEY}).json()["proposals"]
    assert len(proposals) == 1
    assert proposals[0]["id"] == body["id"]


def test_sage_analyze_malformed_llm_response_is_502(client, monkeypatch):
    test_api_module = sys.modules[__name__]

    _seed_digest("nexus")

    monkeypatch.setattr(test_api_module, "mock_reply_content", "this is not JSON at all")

    resp = client.post("/sage/analyze/nexus", headers={"X-API-Key": VALID_KEY})
    assert resp.status_code == 502

    proposals = client.get("/sage/proposals", headers={"X-API-Key": VALID_KEY}).json()["proposals"]
    assert proposals == []


def test_sage_proposals_requires_api_key(client):
    resp = client.get("/sage/proposals")
    assert resp.status_code == 401


def test_sage_accept_proposal_unknown_id_is_404(client):
    resp = client.post("/sage/proposals/999999/accept", headers={"X-API-Key": VALID_KEY})
    assert resp.status_code == 404


def test_sage_reject_proposal_unknown_id_is_404(client):
    resp = client.post("/sage/proposals/999999/reject", headers={"X-API-Key": VALID_KEY})
    assert resp.status_code == 404


def test_sage_accept_proposal_applies_override_and_changes_outgoing_system_prompt(client, monkeypatch):
    test_api_module = sys.modules[__name__]

    _seed_digest("nexus")

    monkeypatch.setattr(
        test_api_module,
        "mock_reply_content",
        json.dumps({"rationale": "r", "proposed_system_prompt": "You are now extremely concise."}),
    )
    proposal_id = client.post("/sage/analyze/nexus", headers={"X-API-Key": VALID_KEY}).json()["id"]

    monkeypatch.setattr(test_api_module, "mock_reply_content", "mocked reply")
    accept_resp = client.post(f"/sage/proposals/{proposal_id}/accept", headers={"X-API-Key": VALID_KEY})
    assert accept_resp.status_code == 200
    assert accept_resp.json()["status"] == "accepted"

    proposal = client.get("/sage/proposals", headers={"X-API-Key": VALID_KEY}).json()["proposals"][0]
    assert proposal["status"] == "accepted"

    client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hi again"})
    outgoing_system_message = sent_requests[-1]["messages"][0]
    assert outgoing_system_message["content"] == "You are now extremely concise."


def test_sage_reject_proposal_does_not_apply_override(client, monkeypatch):
    test_api_module = sys.modules[__name__]

    _seed_digest("nexus")

    monkeypatch.setattr(
        test_api_module,
        "mock_reply_content",
        json.dumps({"rationale": "r", "proposed_system_prompt": "You are now extremely concise."}),
    )
    proposal_id = client.post("/sage/analyze/nexus", headers={"X-API-Key": VALID_KEY}).json()["id"]

    monkeypatch.setattr(test_api_module, "mock_reply_content", "mocked reply")
    reject_resp = client.post(f"/sage/proposals/{proposal_id}/reject", headers={"X-API-Key": VALID_KEY})
    assert reject_resp.status_code == 200
    assert reject_resp.json()["status"] == "rejected"

    client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hi again"})
    outgoing_system_message = sent_requests[-1]["messages"][0]
    assert outgoing_system_message["content"] != "You are now extremely concise."


def test_sage_accept_already_reviewed_proposal_is_409(client, monkeypatch):
    test_api_module = sys.modules[__name__]

    _seed_digest("nexus")

    monkeypatch.setattr(
        test_api_module, "mock_reply_content", json.dumps({"rationale": "r", "proposed_system_prompt": "p"})
    )
    proposal_id = client.post("/sage/analyze/nexus", headers={"X-API-Key": VALID_KEY}).json()["id"]

    client.post(f"/sage/proposals/{proposal_id}/accept", headers={"X-API-Key": VALID_KEY})
    resp = client.post(f"/sage/proposals/{proposal_id}/reject", headers={"X-API-Key": VALID_KEY})
    assert resp.status_code == 409


def test_sage_accept_records_baseline_signal_quality_and_no_prior_override(client, monkeypatch):
    test_api_module = sys.modules[__name__]

    _seed_digest("nexus", signal_quality="Affirms short answers.")

    monkeypatch.setattr(
        test_api_module, "mock_reply_content", json.dumps({"rationale": "r", "proposed_system_prompt": "p"})
    )
    proposal_id = client.post("/sage/analyze/nexus", headers={"X-API-Key": VALID_KEY}).json()["id"]
    client.post(f"/sage/proposals/{proposal_id}/accept", headers={"X-API-Key": VALID_KEY})

    proposal = client.get("/sage/proposals", headers={"X-API-Key": VALID_KEY}).json()["proposals"][0]
    assert proposal["baseline_signal_quality"] == "Affirms short answers."
    assert proposal["prior_override"] is None
    assert proposal["outcome"] == ""


def test_sage_accept_records_prior_override_when_one_already_existed(client, monkeypatch):
    test_api_module = sys.modules[__name__]

    client.post(
        "/sage/override/nexus", headers={"X-API-Key": VALID_KEY}, json={"system_prompt": "You are a pirate."}
    )
    _seed_digest("nexus")

    monkeypatch.setattr(
        test_api_module, "mock_reply_content", json.dumps({"rationale": "r", "proposed_system_prompt": "Be warm."})
    )
    proposal_id = client.post("/sage/analyze/nexus", headers={"X-API-Key": VALID_KEY}).json()["id"]
    client.post(f"/sage/proposals/{proposal_id}/accept", headers={"X-API-Key": VALID_KEY})

    proposal = client.get("/sage/proposals", headers={"X-API-Key": VALID_KEY}).json()["proposals"][0]
    assert proposal["prior_override"] == "You are a pirate."


def _setup_sage_loop_modules(monkeypatch, tmp_path, db_name: str):
    from core import digest as digest_module
    from core import memory as memory_module
    from core import persona_overrides as persona_overrides_module
    from core import sage_proposals as sage_proposals_module
    from core import user_profile as user_profile_module

    db_path = tmp_path / db_name
    for module in (digest_module, memory_module, persona_overrides_module, sage_proposals_module, user_profile_module):
        monkeypatch.setattr(module, "DB_PATH", db_path)
        module.init_db()
    digest_module._active_agents.clear()
    return digest_module, memory_module, persona_overrides_module, sage_proposals_module, user_profile_module


def test_sage_loop_reverts_override_when_evaluation_finds_regression(monkeypatch, tmp_path):
    """End-to-end: closing the SAGE loop should actually revert a persona change once fresh
    signal shows it made things worse -- not just record a verdict nobody acts on."""
    import asyncio

    import main as main_module

    digest_module, memory_module, persona_overrides_module, sage_proposals_module, _ = _setup_sage_loop_modules(
        monkeypatch, tmp_path, "sage_loop_regressed.db"
    )

    persona_overrides_module.set_override("nexus", "Be extremely formal.")
    digest_module.apply_digest(
        "nexus", "User likes formal answers.", topic_shift=False, signal_quality="Affirms formal replies."
    )

    proposal_id = sage_proposals_module.create_proposal("nexus", "try warmth instead", "Be warm and casual.")
    sage_proposals_module.set_proposal_status(proposal_id, "accepted")
    sage_proposals_module.record_acceptance_baseline(proposal_id, "Affirms formal replies.", "Be extremely formal.")
    persona_overrides_module.set_override("nexus", "Be warm and casual.")

    memory_module.add_message("s1", "nexus", "user", "That's too casual, please be formal.")

    replies = iter(
        [
            json.dumps(
                {
                    "digest": "User dislikes the new casual tone.",
                    "topic_shift": False,
                    "signal_quality": "Explicitly corrected the new casual tone back to formal.",
                }
            ),
            json.dumps({"profile": "Name unknown."}),
            json.dumps({"verdict": "regressed", "reasoning": "user explicitly corrected the new tone"}),
        ]
    )

    async def fake_call_mistral(messages):
        return next(replies)

    monkeypatch.setattr(main_module, "_call_mistral", fake_call_mistral)

    asyncio.run(main_module._run_digest_safely("nexus"))

    assert persona_overrides_module.get_override("nexus") == "Be extremely formal."
    assert sage_proposals_module.get_proposal(proposal_id)["outcome"] == "regressed"


def test_sage_loop_leaves_override_in_place_when_evaluation_finds_improvement(monkeypatch, tmp_path):
    import asyncio

    import main as main_module

    digest_module, memory_module, persona_overrides_module, sage_proposals_module, _ = _setup_sage_loop_modules(
        monkeypatch, tmp_path, "sage_loop_improved.db"
    )

    digest_module.apply_digest(
        "nexus", "User likes formal answers.", topic_shift=False, signal_quality="Affirms formal replies."
    )
    proposal_id = sage_proposals_module.create_proposal("nexus", "try warmth instead", "Be warm and casual.")
    sage_proposals_module.set_proposal_status(proposal_id, "accepted")
    sage_proposals_module.record_acceptance_baseline(proposal_id, "Affirms formal replies.", None)
    persona_overrides_module.set_override("nexus", "Be warm and casual.")

    memory_module.add_message("s1", "nexus", "user", "That's much better, thanks!")

    replies = iter(
        [
            json.dumps(
                {
                    "digest": "User likes the new casual tone.",
                    "topic_shift": False,
                    "signal_quality": "Affirms the new casual tone.",
                }
            ),
            json.dumps({"profile": "Name unknown."}),
            json.dumps({"verdict": "improved", "reasoning": "user affirmed the new tone"}),
        ]
    )

    async def fake_call_mistral(messages):
        return next(replies)

    monkeypatch.setattr(main_module, "_call_mistral", fake_call_mistral)

    asyncio.run(main_module._run_digest_safely("nexus"))

    assert persona_overrides_module.get_override("nexus") == "Be warm and casual."
    assert sage_proposals_module.get_proposal(proposal_id)["outcome"] == "improved"


def test_chat_without_digest_sends_only_persona_system_prompt(client):
    client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"})

    system_messages = [m for m in sent_requests[-1]["messages"] if m["role"] == "system"]
    assert len(system_messages) == 1  # just the persona prompt, no digest block yet


def test_chat_with_digest_injects_it_as_extra_context(client):
    _seed_digest("nexus", "User's name is Alex and they prefer short answers.")

    client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"})

    system_messages = [m for m in sent_requests[-1]["messages"] if m["role"] == "system"]
    assert len(system_messages) == 2
    assert "User's name is Alex and they prefer short answers." in system_messages[1]["content"]


def test_chat_does_not_schedule_digest_before_threshold(client):
    resp = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"})
    assert resp.status_code == 200
    assert scheduled_digest_agents == []


def test_chat_schedules_digest_once_threshold_crossed(client, monkeypatch):
    from core import digest as digest_module

    monkeypatch.setattr(digest_module, "MESSAGE_TRIGGER", 4)  # 2 exchanges = 4 turns

    client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "one"})
    assert scheduled_digest_agents == []

    client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "two"})
    assert scheduled_digest_agents == ["nexus"]


def test_profile_requires_api_key(client):
    resp = client.get("/profile")
    assert resp.status_code == 401


def test_profile_defaults_empty_when_none_applied(client):
    resp = client.get("/profile", headers={"X-API-Key": VALID_KEY})
    assert resp.status_code == 200
    assert resp.json() == {"profile": "", "updated_at": None}


def test_profile_returns_applied_value(client):
    from core import user_profile as user_profile_module

    user_profile_module.apply_profile("Name: Lee. Works on AILicious.")

    resp = client.get("/profile", headers={"X-API-Key": VALID_KEY})
    body = resp.json()
    assert body["profile"] == "Name: Lee. Works on AILicious."
    assert body["updated_at"] is not None


def test_chat_includes_profile_in_outgoing_system_prompt_when_present(client):
    from core import user_profile as user_profile_module

    user_profile_module.apply_profile("Name: Lee. Prefers concise answers.")

    client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"})

    system_contents = [m["content"] for m in sent_requests[-1]["messages"] if m["role"] == "system"]
    assert any("Name: Lee. Prefers concise answers." in c for c in system_contents)


def test_chat_omits_profile_system_message_when_none_applied(client):
    client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"})

    system_contents = [m["content"] for m in sent_requests[-1]["messages"] if m["role"] == "system"]
    assert not any("across all AILicious agents" in c for c in system_contents)


def test_profile_is_shared_across_different_agents(client):
    from core import user_profile as user_profile_module

    user_profile_module.apply_profile("Name: Lee.")

    client.post("/agents/forge", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"})

    system_contents = [m["content"] for m in sent_requests[-1]["messages"] if m["role"] == "system"]
    assert any("Name: Lee." in c for c in system_contents)


def test_digest_cycle_merges_into_shared_profile(monkeypatch, tmp_path):
    """The digest -> profile chaining lives in main._run_digest_safely, which is
    stubbed out by the client fixture (see scheduled_digest_agents above) to keep
    HTTP tests deterministic -- so this calls the real function directly instead,
    the same way test_digest.py exercises run_digest_cycle directly."""
    import asyncio

    from core import digest as digest_module
    from core import memory as memory_module
    from core import sage_proposals as sage_proposals_module
    from core import user_profile as user_profile_module
    import main as main_module

    db_path = tmp_path / "digest_profile_test.db"
    monkeypatch.setattr(digest_module, "DB_PATH", db_path)
    monkeypatch.setattr(memory_module, "DB_PATH", db_path)
    monkeypatch.setattr(user_profile_module, "DB_PATH", db_path)
    monkeypatch.setattr(sage_proposals_module, "DB_PATH", db_path)
    digest_module.init_db()
    memory_module.init_db()
    user_profile_module.init_db()
    sage_proposals_module.init_db()
    digest_module._active_agents.clear()

    memory_module.add_message("s1", "nexus", "user", "My name is Lee.")

    replies = iter(
        [
            json.dumps({"digest": "User's name is Lee.", "topic_shift": False, "signal_quality": ""}),
            json.dumps({"profile": "Name: Lee."}),
        ]
    )

    async def fake_call_mistral(messages):
        return next(replies)

    monkeypatch.setattr(main_module, "_call_mistral", fake_call_mistral)

    asyncio.run(main_module._run_digest_safely("nexus"))

    assert user_profile_module.get_profile() == "Name: Lee."


def test_nexus_chat_without_tool_call_behaves_normally(client):
    resp = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "mocked reply"
    assert body["delegated_to"] == []
    # NEXUS's own call carries the delegation tool even when it doesn't use it this turn.
    assert sent_requests[-1]["tools"][0]["function"]["name"] == "delegate_to_agent"


def test_non_nexus_chat_does_not_include_delegation_tool(client):
    client.post("/agents/forge", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"})
    assert "tools" not in sent_requests[-1]


def test_nexus_delegates_to_forge_and_records_forge_history(client, monkeypatch):
    tool_call_args = json.dumps({"agent": "forge", "task": "write a hello world function"})
    call_log: list[dict] = []

    async def sequenced_post(self, url, json=None, headers=None, **kwargs):
        call_log.append(json)
        if len(call_log) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "delegate_to_agent", "arguments": tool_call_args},
                                    }
                                ],
                            }
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )
        if len(call_log) == 2:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "def hello(): print('hello')"}}]},
                request=httpx.Request("POST", url),
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "FORGE wrote: def hello(): print('hello')"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", sequenced_post)

    resp = client.post(
        "/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "ask forge for a hello world function"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["delegated_to"] == ["forge"]
    assert body["response"] == "FORGE wrote: def hello(): print('hello')"
    assert len(call_log) == 3  # NEXUS decides -> FORGE actually runs -> NEXUS synthesizes

    from core.memory import get_history

    forge_history = get_history(body["session_id"], "forge")
    assert any(m["content"] == "write a hello world function" for m in forge_history)
    assert any(m["content"] == "def hello(): print('hello')" for m in forge_history)


def test_nexus_delegation_malformed_arguments_does_not_crash(client, monkeypatch):
    call_log: list[dict] = []

    async def sequenced_post(self, url, json=None, headers=None, **kwargs):
        call_log.append(json)
        if len(call_log) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "delegate_to_agent", "arguments": "not json"},
                                    }
                                ],
                            }
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "Handled gracefully."}}]}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", sequenced_post)

    resp = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"})
    assert resp.status_code == 200
    assert resp.json()["response"] == "Handled gracefully."
    assert resp.json()["delegated_to"] == []


def test_nexus_delegation_unknown_agent_does_not_delegate(client, monkeypatch):
    tool_call_args = json.dumps({"agent": "doesnotexist", "task": "x"})
    call_log: list[dict] = []

    async def sequenced_post(self, url, json=None, headers=None, **kwargs):
        call_log.append(json)
        if len(call_log) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "delegate_to_agent", "arguments": tool_call_args},
                                    }
                                ],
                            }
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Couldn't find that agent."}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", sequenced_post)

    resp = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"})
    assert resp.status_code == 200
    assert resp.json()["delegated_to"] == []

    # Isolates the explicit agent-name check specifically (not just "the turn didn't crash",
    # which a broader exception handler around run_agent_chat would also guarantee): a bogus
    # agent name must never even reach run_agent_chat, so it should leave no trace in the
    # digest table -- proven with a raw read that doesn't itself upsert a row.
    from core import digest as digest_module

    conn = digest_module._get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM conversation_digests WHERE agent = ?", ("doesnotexist",)
        ).fetchone()
    finally:
        conn.close()
    assert row is None


def test_nexus_delegation_loop_is_capped(client, monkeypatch):
    tool_call_args = json.dumps({"agent": "forge", "task": "x"})
    call_log: list[dict] = []

    async def always_delegate_post(self, url, json=None, headers=None, **kwargs):
        call_log.append(json)
        if json.get("tools"):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": f"call_{len(call_log)}",
                                        "type": "function",
                                        "function": {"name": "delegate_to_agent", "arguments": tool_call_args},
                                    }
                                ],
                            }
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "forge reply"}}]}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", always_delegate_post)

    resp = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "loop please"})
    assert resp.status_code == 200
    assert resp.json()["response"] == "forge reply"
    # 2 rounds x (NEXUS-decides + FORGE-runs) + 1 forced final answer = 5 calls total.
    assert len(call_log) == 5


def test_search_tool_omitted_when_tavily_not_configured(client):
    resp = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"})
    assert resp.status_code == 200
    tool_names = {t["function"]["name"] for t in sent_requests[-1]["tools"]}
    assert "search_web" not in tool_names


def test_search_tool_included_when_tavily_configured(client, monkeypatch):
    import main as main_module

    monkeypatch.setattr(main_module, "TAVILY_API_KEY", "fake-tavily-key")

    resp = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"})
    assert resp.status_code == 200
    tool_names = {t["function"]["name"] for t in sent_requests[-1]["tools"]}
    assert "search_web" in tool_names


def test_nexus_executes_real_web_search_and_records_query(client, monkeypatch):
    import main as main_module

    monkeypatch.setattr(main_module, "TAVILY_API_KEY", "fake-tavily-key")

    tool_call_args = json.dumps({"query": "current weather in Paris"})
    call_log: list[dict] = []

    async def sequenced_post(self, url, json=None, headers=None, **kwargs):
        call_log.append({"url": str(url), "json": json})
        if "tavily.com" in str(url):
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"title": "Paris Weather", "content": "Sunny, 22C", "url": "https://example.com/weather"}
                    ]
                },
                request=httpx.Request("POST", url),
            )
        if len(call_log) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "search_web", "arguments": tool_call_args},
                                    }
                                ],
                            }
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "It's sunny in Paris, 22C."}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", sequenced_post)

    resp = client.post(
        "/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "what's the weather in Paris?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["searched_web"] == ["current weather in Paris"]
    assert body["response"] == "It's sunny in Paris, 22C."
    # NEXUS-decides (mistral) + tavily search + NEXUS-synthesizes (mistral) = 3 calls total.
    assert len(call_log) == 3


def test_web_search_formats_multiple_results_as_distinct_attributable_sources(client, monkeypatch):
    import main as main_module

    monkeypatch.setattr(main_module, "TAVILY_API_KEY", "fake-tavily-key")

    tool_call_args = json.dumps({"query": "today's military news"})
    call_log: list[dict] = []

    async def sequenced_post(self, url, json=None, headers=None, **kwargs):
        call_log.append({"url": str(url), "json": json})
        if "tavily.com" in str(url):
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Task & Purpose",
                            "content": "Iran blew the hell out of a base in Bahrain.",
                            "url": "https://taskandpurpose.com",
                        },
                        {
                            "title": "Military Daily News",
                            "content": "Trump pardons a Navy veteran.",
                            "url": "https://www.military.com/daily-news",
                        },
                    ]
                },
                request=httpx.Request("POST", url),
            )
        if len(call_log) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "search_web", "arguments": tool_call_args},
                                    }
                                ],
                            }
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "summary"}}]}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", sequenced_post)

    resp = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "give me today's military news"})
    assert resp.status_code == 200

    # The tool-result message fed back into the third (synthesis) call must keep each source's
    # url paired with its own content, not merged into one blob a model could mis-attribute --
    # this is what actually went wrong live (a real headline got cited to the wrong domain).
    tool_result = call_log[2]["json"]["messages"][-1]
    assert tool_result["role"] == "tool"
    content = tool_result["content"]
    assert "[Source 1] https://taskandpurpose.com" in content
    assert "Iran blew the hell out of a base in Bahrain." in content
    assert "[Source 2] https://www.military.com/daily-news" in content
    assert "Trump pardons a Navy veteran." in content
    # The Bahrain fact must appear before Source 2's block starts, i.e. within Source 1's own
    # block -- not just present somewhere in the string, which a bad merge could still satisfy.
    assert content.index("Iran blew the hell out of a base in Bahrain.") < content.index("[Source 2]")


def test_web_search_malformed_arguments_does_not_delegate_or_search(client, monkeypatch):
    import main as main_module

    monkeypatch.setattr(main_module, "TAVILY_API_KEY", "fake-tavily-key")

    call_log: list[dict] = []

    async def sequenced_post(self, url, json=None, headers=None, **kwargs):
        call_log.append(json)
        if len(call_log) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "search_web", "arguments": "not json"},
                                    }
                                ],
                            }
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", sequenced_post)

    resp = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"})
    assert resp.status_code == 200
    assert resp.json()["searched_web"] == []
    assert resp.json()["delegated_to"] == []


def test_unknown_tool_call_name_reports_failure_without_being_treated_as_delegation(client, monkeypatch):
    import main as main_module

    monkeypatch.setattr(main_module, "TAVILY_API_KEY", "fake-tavily-key")

    call_log: list[dict] = []

    async def sequenced_post(self, url, json=None, headers=None, **kwargs):
        call_log.append(json)
        if len(call_log) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "totally_made_up_tool", "arguments": "{}"},
                                    }
                                ],
                            }
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", sequenced_post)

    resp = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hi"})
    assert resp.status_code == 200
    assert resp.json()["delegated_to"] == []
    assert resp.json()["searched_web"] == []

    # Isolates the explicit unknown-tool-name branch, not just "the turn didn't crash": a name
    # matching neither known tool must produce this exact message, not silently fall through to
    # being interpreted as a (malformed) delegation call, which would also leave delegated_to
    # empty and could look like this test passed for the wrong reason.
    tool_result_message = call_log[1]["messages"][-1]
    assert tool_result_message["role"] == "tool"
    assert tool_result_message["content"] == "Tool call failed: unknown tool 'totally_made_up_tool'."


def test_looks_time_sensitive_matches_common_phrasings():
    from main import _looks_time_sensitive

    assert _looks_time_sensitive("give me today's news") is True
    assert _looks_time_sensitive("what's the weather like right now") is True
    assert _looks_time_sensitive("who won the election") is True
    assert _looks_time_sensitive("write me a python function to reverse a list") is False
    assert _looks_time_sensitive("hello") is False


def test_nexus_forces_search_tool_choice_for_time_sensitive_message(client, monkeypatch):
    import main as main_module

    monkeypatch.setattr(main_module, "TAVILY_API_KEY", "fake-tavily-key")

    resp = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "give me today's news"})
    assert resp.status_code == 200
    assert sent_requests[-1]["tool_choice"] == {"type": "function", "function": {"name": "search_web"}}


def test_nexus_does_not_force_tool_choice_for_non_time_sensitive_message(client, monkeypatch):
    import main as main_module

    monkeypatch.setattr(main_module, "TAVILY_API_KEY", "fake-tavily-key")

    resp = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "hello"})
    assert resp.status_code == 200
    assert sent_requests[-1]["tool_choice"] == "auto"


def test_nexus_does_not_force_tool_choice_without_tavily_key(client):
    resp = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "give me today's news"})
    assert resp.status_code == 200
    assert sent_requests[-1]["tool_choice"] == "auto"


def test_nexus_second_round_tool_choice_reverts_to_auto_after_forced_search(client, monkeypatch):
    import main as main_module

    monkeypatch.setattr(main_module, "TAVILY_API_KEY", "fake-tavily-key")

    tool_call_args = json.dumps({"query": "today's news"})
    call_log: list[dict] = []

    async def sequenced_post(self, url, json=None, headers=None, **kwargs):
        call_log.append(json)
        if "tavily.com" in str(url):
            return httpx.Response(200, json={"results": []}, request=httpx.Request("POST", url))
        if len([c for c in call_log if "model" in c]) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "search_web", "arguments": tool_call_args},
                                    }
                                ],
                            }
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "no news found"}}]}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", sequenced_post)

    resp = client.post("/chat", headers={"X-API-Key": VALID_KEY}, json={"message": "give me today's news"})
    assert resp.status_code == 200

    mistral_calls = [c for c in call_log if "model" in c]
    assert len(mistral_calls) == 2
    assert mistral_calls[0]["tool_choice"] == {"type": "function", "function": {"name": "search_web"}}
    assert mistral_calls[1]["tool_choice"] == "auto"


def test_cross_agent_memory_does_not_leak(client):
    forge_resp = client.post("/agents/forge", headers={"X-API-Key": VALID_KEY}, json={"message": "forge secret"})
    session_id = forge_resp.json()["session_id"]

    oracle_resp = client.post(
        "/agents/oracle",
        headers={"X-API-Key": VALID_KEY},
        json={"message": "anything", "session_id": session_id},
    )
    assert oracle_resp.status_code == 200

    stats = client.get("/memory/stats", headers={"X-API-Key": VALID_KEY}).json()
    # forge's 2 messages (user+assistant) + oracle's 2 messages (user+assistant), same session_id but
    # isolated by agent — asserting count catches a regression that merges them into one thread.
    assert stats["total_messages"] == 4

    # The count above can't catch context leaking into the *outgoing* Mistral request (a constant
    # mocked reply hides that), so inspect what was actually sent for oracle's call.
    oracle_request = sent_requests[-1]
    oracle_messages_text = " ".join(m["content"] for m in oracle_request["messages"])
    assert "forge secret" not in oracle_messages_text
