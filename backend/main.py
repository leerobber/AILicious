import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.digest import get_digest_text, get_state as get_digest_state, note_turns, run_digest_cycle, should_digest
from core.digest import init_db as init_digest_db
from core.memory import add_message, get_history, get_stats, init_db, set_feedback
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
TAVILY_API_URL = "https://api.tavily.com/search"
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")  # optional: web search tool is omitted if unset
RECENT_TAIL_LIMIT = 6  # short raw window; the conversation digest carries the rest
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "20"))
MAX_MESSAGE_LENGTH = int(os.environ.get("MAX_MESSAGE_LENGTH", "4000"))
STATIC_DIR = Path(__file__).resolve().parent / "static"

_background_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(init_db)
    await asyncio.to_thread(init_persona_overrides_db)
    await asyncio.to_thread(init_sage_proposals_db)
    await asyncio.to_thread(init_digest_db)
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
    delegated_to: list[str] = []
    searched_web: list[str] = []


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


async def _call_mistral_raw(messages: list[dict], tools: list[dict] | None = None) -> dict:
    if not MISTRAL_API_KEY:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY is not configured on the server")

    payload = {"model": MISTRAL_MODEL, "messages": messages}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
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

    return resp.json()


async def _call_mistral(messages: list[dict]) -> str:
    data = await _call_mistral_raw(messages)
    return data["choices"][0]["message"]["content"]


# Real agent-to-agent delegation: NEXUS gets a tool letting it hand a task to one of the
# specialized agents and get their actual response back, rather than just telling the user
# which agent to switch to. Only NEXUS carries this tool -- the specialists themselves don't
# delegate further, which structurally rules out delegation loops between agents. NEXUS may
# also get a web search tool (below) when TAVILY_API_KEY is configured -- same NEXUS-only
# pattern, same round cap.
MAX_TOOL_ROUNDS_PER_TURN = 2  # forces a plain final answer if the model won't stop calling tools


def _delegatable_agents() -> list[str]:
    return [name for name in list_personas() if name != "nexus"]


def _delegate_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "delegate_to_agent",
            "description": (
                "Hand a task to one of AILicious's specialized agents and get their real "
                "response back. Use this when a request clearly fits a specialist better than "
                "general conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": _delegatable_agents(),
                        "description": "Which specialized agent to delegate to.",
                    },
                    "task": {
                        "type": "string",
                        "description": "The task or question to hand to that agent, in its own words.",
                    },
                },
                "required": ["agent", "task"],
            },
        },
    }


async def _execute_delegation(call: dict, session_id: str) -> tuple[str | None, str]:
    """Runs one delegate_to_agent tool call for real -- through the same run_agent_chat path
    a direct user message would take, so the delegate agent's own memory and digest see it
    exactly as if the user had asked it themselves. Never raises: a delegation failure comes
    back as a tool result NEXUS can react to, not an error that kills the whole turn.
    """
    try:
        args = json.loads(call["function"]["arguments"])
        agent = args["agent"]
        task = args["task"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return None, f"Delegation failed: malformed tool call ({exc})"

    if agent not in _delegatable_agents():
        return None, f"Delegation failed: '{agent}' is not a valid agent to delegate to."

    try:
        delegate_response = await run_agent_chat(agent, task, session_id)
    except HTTPException as exc:
        return None, f"Delegation to {agent} failed: {exc.detail}"
    except Exception as exc:  # noqa: BLE001 - a delegation hiccup must never break NEXUS's turn
        return None, f"Delegation to {agent} failed: {exc}"

    return agent, f"{agent.upper()} responded: {delegate_response.response}"


def _web_search_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the live web for current information. Use this when a question needs "
                "up-to-date, real-time, or post-training-cutoff information you wouldn't "
                "otherwise have."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                },
                "required": ["query"],
            },
        },
    }


async def _execute_web_search(call: dict) -> tuple[str | None, str]:
    """Runs one search_web tool call for real via Tavily. Never raises: a search failure comes
    back as a tool result NEXUS can react to (e.g. say search isn't working right now), not an
    error that kills the whole turn.
    """
    try:
        args = json.loads(call["function"]["arguments"])
        query = args["query"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return None, f"Web search failed: malformed tool call ({exc})"

    if not TAVILY_API_KEY:
        return None, "Web search failed: no search provider is configured on the server."

    payload = {"api_key": TAVILY_API_KEY, "query": query, "search_depth": "basic", "max_results": 5}
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.post(TAVILY_API_URL, json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return query, f"Web search failed: {exc.response.status_code} {exc.response.text}"
        except httpx.RequestError as exc:
            return query, f"Web search failed: {exc}"

    results = resp.json().get("results", [])
    if not results:
        return query, "No results found."

    lines = [f"- {r.get('title', '')}: {r.get('content', '')} ({r.get('url', '')})" for r in results[:5]]
    return query, "\n".join(lines)


def _nexus_tools() -> list[dict]:
    tools = [_delegate_tool_schema()]
    if TAVILY_API_KEY:
        tools.append(_web_search_tool_schema())
    return tools


async def _run_nexus_turn(messages: list[dict], session_id: str) -> tuple[str, list[str], list[str]]:
    conversation = list(messages)
    delegated_to: list[str] = []
    searched_web: list[str] = []

    for _ in range(MAX_TOOL_ROUNDS_PER_TURN):
        data = await _call_mistral_raw(conversation, tools=_nexus_tools())
        choice_message = data["choices"][0]["message"]
        tool_calls = choice_message.get("tool_calls")

        if not tool_calls:
            return choice_message.get("content") or "", delegated_to, searched_web

        conversation.append(choice_message)
        for call in tool_calls:
            name = call["function"]["name"]
            if name == "delegate_to_agent":
                agent_name, result_text = await _execute_delegation(call, session_id)
                if agent_name:
                    delegated_to.append(agent_name)
            elif name == "search_web":
                query, result_text = await _execute_web_search(call)
                if query:
                    searched_web.append(query)
            else:
                result_text = f"Tool call failed: unknown tool '{name}'."
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": result_text,
                }
            )

    # Ran out of tool-call rounds -- force a plain final answer without further tool access.
    final = await _call_mistral(conversation)
    return final, delegated_to, searched_web


def _schedule_digest(agent: str) -> None:
    """Fires a digest cycle in the background -- never awaited by the request that
    triggers it, so it adds zero latency to the chat response. A module-level set
    keeps a reference to the task so it isn't garbage-collected mid-run.
    """
    task = asyncio.create_task(_run_digest_safely(agent))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _run_digest_safely(agent: str) -> None:
    await run_digest_cycle(agent, _call_mistral)


async def run_agent_chat(agent: str, message: str, session_id: str | None) -> ChatResponse:
    if len(message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Message too long ({len(message)} chars, max {MAX_MESSAGE_LENGTH}).",
        )

    session_id = session_id or str(uuid.uuid4())
    request_started_at = datetime.now(timezone.utc).isoformat()

    recent_tail = await asyncio.to_thread(get_history, session_id, agent, RECENT_TAIL_LIMIT)
    digest_text = await asyncio.to_thread(get_digest_text, agent)
    override = await asyncio.to_thread(get_override, agent)
    system_prompt = override if override is not None else get_system_prompt(agent)

    messages = [{"role": "system", "content": system_prompt}]
    if digest_text:
        messages.append({"role": "system", "content": f"What you know about this user so far:\n{digest_text}"})
    messages += recent_tail + [{"role": "user", "content": message}]

    if agent == "nexus":
        reply, delegated_to, searched_web = await _run_nexus_turn(messages, session_id)
    else:
        reply, delegated_to, searched_web = await _call_mistral(messages), [], []

    await asyncio.to_thread(add_message, session_id, agent, "user", message)
    message_id = await asyncio.to_thread(add_message, session_id, agent, "assistant", reply)

    # Bookkeeping for the next digest cycle -- note_turns compares request_started_at
    # (captured before the Mistral call) against the *previous* exchange's timestamp,
    # so a slow reply from Mistral this turn never gets mistaken for a conversational
    # pause on the user's part.
    pause_triggered = await asyncio.to_thread(note_turns, agent, 2, request_started_at)
    if pause_triggered or await asyncio.to_thread(should_digest, agent):
        _schedule_digest(agent)

    return ChatResponse(
        response=reply,
        session_id=session_id,
        agent=agent,
        message_id=message_id,
        delegated_to=delegated_to,
        searched_web=searched_web,
    )


SAGE_ANALYSIS_SYSTEM_PROMPT = (
    "You are SAGE, a persona-tuning assistant for an AI agent platform. You will be shown an "
    "agent's current system prompt, a synthesized summary of what the agent has learned about "
    "its user across every conversation, and a separate read on what this user implicitly "
    "responds well or poorly to -- inferred from their behavior (rephrasing, corrections, "
    "reuse, affirmations), not from explicit ratings. Weight that signal-quality read as your "
    "primary evidence for what to change; the summary is context, not a mandate to cover every "
    "topic discussed. Propose a revised system prompt that better serves this user based on "
    "that signal, while preserving the agent's core identity and role. If the signal read is "
    "empty or uninformative, say so in the rationale and propose only a minimal, low-risk "
    "change rather than inventing a rewrite from topic content alone. "
    "Respond with ONLY a JSON object of the form "
    '{"rationale": "<one paragraph explaining the change, or why little should change>", '
    '"proposed_system_prompt": "<the full revised system prompt>"}. '
    "No markdown, no code fences, no extra text — valid JSON only."
)


def _build_sage_analysis_messages(agent: str, current_system_prompt: str, digest: str, signal_quality: str) -> list[dict]:
    lines = [
        f"Current system prompt for '{agent}':",
        current_system_prompt,
        "",
        "What this agent has learned about its user, synthesized from every conversation:",
        digest,
        "",
        "What this user implicitly responds well or poorly to (inferred from behavior, not ratings):",
        signal_quality or "(no informative signal yet)",
    ]
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

    state = await asyncio.to_thread(get_digest_state, agent)
    if not state["digest"]:
        raise HTTPException(
            status_code=400, detail=f"No conversation digest yet for agent '{agent}' -- chat with it a bit first."
        )
    if not state["signal_quality"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No read yet on what works or doesn't for '{agent}' -- nothing in the conversation so far "
                "gave an implicit signal (a rephrased question, a correction, reusing an answer, an "
                "affirmation). Keep chatting naturally and this fills in on its own."
            ),
        )

    override = await asyncio.to_thread(get_override, agent)
    current_system_prompt = override if override is not None else get_system_prompt(agent)

    messages = _build_sage_analysis_messages(agent, current_system_prompt, state["digest"], state["signal_quality"])
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
