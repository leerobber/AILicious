import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.memory import add_message, get_feedback_examples, get_history, get_stats, init_db, set_feedback
from core.persona import get_system_prompt, has_persona, list_personas, load_personas
from core.persona_overrides import clear_override, get_override, set_override
from core.persona_overrides import init_db as init_persona_overrides_db
from core.ratelimit import check_rate_limit
from core.sage_proposals import create_proposal, get_proposal, list_proposals, set_proposal_status
from core.sage_proposals import init_db as init_sage_proposals_db

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
APP_API_KEY = os.environ.get("APP_API_KEY")
HISTORY_LIMIT = 20
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "20"))
MAX_MESSAGE_LENGTH = int(os.environ.get("MAX_MESSAGE_LENGTH", "4000"))
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(init_db)
    await asyncio.to_thread(init_persona_overrides_db)
    await asyncio.to_thread(init_sage_proposals_db)
    await asyncio.to_thread(load_personas)
    yield


app = FastAPI(title="AILicious Backend", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    agent: str
    message_id: int


class FeedbackRequest(BaseModel):
    message_id: int
    rating: int


class PersonaOverrideRequest(BaseModel):
    system_prompt: str


def require_api_key(x_api_key: str | None) -> None:
    if APP_API_KEY and x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def enforce_rate_limit(identifier: str) -> None:
    allowed, retry_after = check_rate_limit(identifier, RATE_LIMIT_PER_MINUTE)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {RATE_LIMIT_PER_MINUTE} requests per minute. Try again shortly.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )


async def _call_mistral(messages: list[dict]) -> str:
    if not MISTRAL_API_KEY:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY is not configured on the server")

    payload = {"model": MISTRAL_MODEL, "messages": messages}
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(MISTRAL_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=502, detail=f"Mistral API error: {exc.response.status_code} {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to reach Mistral API: {exc}") from exc

    data = resp.json()
    return data["choices"][0]["message"]["content"]


async def run_agent_chat(agent: str, message: str, session_id: str | None) -> ChatResponse:
    if len(message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Message too long ({len(message)} chars, max {MAX_MESSAGE_LENGTH}).",
        )

    session_id = session_id or str(uuid.uuid4())

    history = await asyncio.to_thread(get_history, session_id, agent, HISTORY_LIMIT)
    override = await asyncio.to_thread(get_override, agent)
    system_prompt = override if override is not None else get_system_prompt(agent)
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]

    reply = await _call_mistral(messages)

    await asyncio.to_thread(add_message, session_id, agent, "user", message)
    message_id = await asyncio.to_thread(add_message, session_id, agent, "assistant", reply)

    return ChatResponse(response=reply, session_id=session_id, agent=agent, message_id=message_id)


SAGE_ANALYSIS_SYSTEM_PROMPT = (
    "You are SAGE, a persona-tuning assistant for an AI agent platform. You will be shown an "
    "agent's current system prompt and a set of past exchanges the agent had, each marked with "
    "user feedback (liked or disliked). Propose a revised system prompt that reinforces what "
    "earned positive feedback and steers away from what earned negative feedback, while "
    "preserving the agent's core identity and role. Respond with ONLY a JSON object of the form "
    '{"rationale": "<one paragraph explaining the change>", '
    '"proposed_system_prompt": "<the full revised system prompt>"}. '
    "No markdown, no code fences, no extra text — valid JSON only."
)


def _build_sage_analysis_messages(agent: str, current_system_prompt: str, examples: list[dict]) -> list[dict]:
    lines = [f"Current system prompt for '{agent}':", current_system_prompt, "", "Past exchanges with feedback:"]
    for example in examples:
        verdict = "liked" if example["feedback"] == 1 else "disliked"
        lines.append(
            f"- [{verdict}] User: {example['user_message']!r} -> Assistant: {example['assistant_reply']!r}"
        )
    return [
        {"role": "system", "content": SAGE_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "mistral_configured": bool(MISTRAL_API_KEY)}


@app.get("/agents")
async def agents(x_api_key: str | None = Header(default=None)) -> dict:
    require_api_key(x_api_key)
    return {"agents": list_personas()}


@app.get("/memory/stats")
async def memory_stats(x_api_key: str | None = Header(default=None)) -> dict:
    require_api_key(x_api_key)
    return await asyncio.to_thread(get_stats)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request, x_api_key: str | None = Header(default=None)) -> ChatResponse:
    require_api_key(x_api_key)
    enforce_rate_limit(x_api_key or (request.client.host if request.client else "unknown"))
    return await run_agent_chat("nexus", req.message, req.session_id)


@app.post("/agents/{name}", response_model=ChatResponse)
async def chat_with_agent(
    name: str, req: ChatRequest, request: Request, x_api_key: str | None = Header(default=None)
) -> ChatResponse:
    require_api_key(x_api_key)
    enforce_rate_limit(x_api_key or (request.client.host if request.client else "unknown"))
    if not has_persona(name):
        raise HTTPException(status_code=404, detail=f"Unknown agent '{name}'. Available: {list_personas()}")
    return await run_agent_chat(name, req.message, req.session_id)


@app.post("/feedback")
async def feedback(req: FeedbackRequest, request: Request, x_api_key: str | None = Header(default=None)) -> dict:
    require_api_key(x_api_key)
    enforce_rate_limit(x_api_key or (request.client.host if request.client else "unknown"))
    if req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="rating must be 1 (up) or -1 (down)")
    updated = await asyncio.to_thread(set_feedback, req.message_id, req.rating)
    if not updated:
        raise HTTPException(status_code=404, detail=f"No message with id {req.message_id}")
    return {"status": "ok"}


@app.post("/sage/override/{agent}")
async def sage_override(
    agent: str, req: PersonaOverrideRequest, x_api_key: str | None = Header(default=None)
) -> dict:
    require_api_key(x_api_key)
    if not has_persona(agent):
        raise HTTPException(status_code=404, detail=f"Unknown agent '{agent}'. Available: {list_personas()}")
    await asyncio.to_thread(set_override, agent, req.system_prompt)
    return {"status": "ok"}


@app.post("/sage/reset/{agent}")
async def sage_reset(agent: str, x_api_key: str | None = Header(default=None)) -> dict:
    require_api_key(x_api_key)
    if not has_persona(agent):
        raise HTTPException(status_code=404, detail=f"Unknown agent '{agent}'. Available: {list_personas()}")
    await asyncio.to_thread(clear_override, agent)
    return {"status": "ok"}


@app.post("/sage/analyze/{agent}")
async def sage_analyze(agent: str, x_api_key: str | None = Header(default=None)) -> dict:
    require_api_key(x_api_key)
    if not has_persona(agent):
        raise HTTPException(status_code=404, detail=f"Unknown agent '{agent}'. Available: {list_personas()}")

    examples = await asyncio.to_thread(get_feedback_examples, agent, 50)
    if not examples:
        raise HTTPException(status_code=400, detail=f"No feedback recorded yet for agent '{agent}' to analyze.")

    override = await asyncio.to_thread(get_override, agent)
    current_system_prompt = override if override is not None else get_system_prompt(agent)

    messages = _build_sage_analysis_messages(agent, current_system_prompt, examples)
    raw_reply = await _call_mistral(messages)

    try:
        parsed = json.loads(raw_reply)
        rationale = parsed["rationale"]
        proposed_system_prompt = parsed["proposed_system_prompt"]
        if not isinstance(rationale, str) or not isinstance(proposed_system_prompt, str):
            raise ValueError("rationale and proposed_system_prompt must both be strings")
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=502, detail=f"SAGE analysis returned an unparseable response: {exc}"
        ) from exc

    proposal_id = await asyncio.to_thread(create_proposal, agent, rationale, proposed_system_prompt)
    return {
        "id": proposal_id,
        "agent": agent,
        "rationale": rationale,
        "proposed_system_prompt": proposed_system_prompt,
        "status": "pending",
    }


@app.get("/sage/proposals")
async def sage_list_proposals(
    agent: str | None = None, status: str | None = None, x_api_key: str | None = Header(default=None)
) -> dict:
    require_api_key(x_api_key)
    return {"proposals": await asyncio.to_thread(list_proposals, agent, status)}


@app.post("/sage/proposals/{proposal_id}/accept")
async def sage_accept_proposal(proposal_id: int, x_api_key: str | None = Header(default=None)) -> dict:
    require_api_key(x_api_key)
    proposal = await asyncio.to_thread(get_proposal, proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail=f"No SAGE proposal with id {proposal_id}")
    updated = await asyncio.to_thread(set_proposal_status, proposal_id, "accepted")
    if not updated:
        raise HTTPException(
            status_code=409, detail=f"Proposal {proposal_id} is not pending (status: {proposal['status']})"
        )
    await asyncio.to_thread(set_override, proposal["agent"], proposal["proposed_system_prompt"])
    return {"status": "accepted"}


@app.post("/sage/proposals/{proposal_id}/reject")
async def sage_reject_proposal(proposal_id: int, x_api_key: str | None = Header(default=None)) -> dict:
    require_api_key(x_api_key)
    proposal = await asyncio.to_thread(get_proposal, proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail=f"No SAGE proposal with id {proposal_id}")
    updated = await asyncio.to_thread(set_proposal_status, proposal_id, "rejected")
    if not updated:
        raise HTTPException(
            status_code=409, detail=f"Proposal {proposal_id} is not pending (status: {proposal['status']})"
        )
    return {"status": "rejected"}


# Mounted last so it only catches paths not already matched by an API route above.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
