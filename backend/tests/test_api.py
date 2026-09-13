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

    db_path = tmp_path / "api_test.db"
    monkeypatch.setattr(memory_module, "DB_PATH", db_path)
    monkeypatch.setattr(persona_overrides_module, "DB_PATH", db_path)
    monkeypatch.setattr(sage_proposals_module, "DB_PATH", db_path)
    monkeypatch.setattr(digest_module, "DB_PATH", db_path)
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


def _seed_digest(agent: str, text: str = "User prefers concise, direct answers.") -> None:
    from core import digest as digest_module

    digest_module.apply_digest(agent, text, topic_shift=False)


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
