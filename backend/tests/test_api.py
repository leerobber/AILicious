import httpx
import pytest
from fastapi.testclient import TestClient

from core.ratelimit import reset as reset_rate_limit
from main import app

VALID_KEY = "test-app-key"
sent_requests: list[dict] = []


async def _fake_post(self, url, json=None, headers=None, **kwargs):
    sent_requests.append(json)
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": "mocked reply"}}]},
        request=httpx.Request("POST", url),
    )


@pytest.fixture(autouse=True)
def _reset_rate_limit_state():
    reset_rate_limit()
    yield
    reset_rate_limit()


@pytest.fixture
def client(tmp_path, monkeypatch):
    from core import memory as memory_module

    monkeypatch.setattr(memory_module, "DB_PATH", tmp_path / "api_test.db")
    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    sent_requests.clear()

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
