# AILicious

Personality-driven, multi-agent AI system with cloud-based inference and memory, accessible from Android via a web client. All compute runs server-side — no local hardware or model downloads required.

## Android / web client

A single-page PWA is served directly by the backend at `/` (`backend/static/index.html` — no build step, no separate host). Open the Render URL on an Android phone in Chrome, enter the API key in Settings (gear icon), pick an agent from the dropdown, and chat. "Add to Home Screen" installs it as a standalone app icon via the manifest + service worker in `backend/static/`.

The service worker (`sw.js`) fetches network-first, caching only as an offline fallback — every successful load re-fetches from the live server, so a new deploy reaches an already-installed client on its very next load. (An earlier cache-first version pinned whatever `index.html` a client first cached indefinitely, since `CACHE_NAME` never changed between deploys — a real fix could ship and a phone would keep silently serving the old app shell forever. If a very old client still looks stale after this fix ships, one manual site-data clear resets it; every load after that self-heals.)

Client-side notes:
- The API key is stored in the browser's `localStorage`, sent only as an `X-API-Key` header to this same-origin backend — never committed, never sent anywhere else.
- Each agent gets its own `session_id` and message history in `localStorage`, matching the backend's per-agent memory scoping — switching agents shows that agent's own conversation, not a mixed one.
- Verified end to end with a real browser (Playwright): entering a key, picking an agent, sending a message, and getting back a real Mistral-backed reply, plus confirming history correctly persists per agent when switching back and forth.
- A **SAGE panel** (Settings → "Persona (SAGE)") lets you trigger feedback analysis, review/accept/reject proposals, and reset an agent's persona — all from the phone, no API calls by hand. See SAGE below.

## Backend (Phase 1: minimal walking skeleton)

A FastAPI service that proxies chat messages to Mistral, with SQLite-compatible conversation memory (local file, or Turso for durability — see below) and six personas (NEXUS, FORGE, ORACLE, SENTINEL, CODEX, AVERY) loaded from YAML at startup.

- `POST /chat` — always talks to NEXUS, the default entry point.
- `POST /agents/{name}` — talk to a specific agent directly (`forge`, `oracle`, `sentinel`, `codex`, or `avery`); 404 if the name isn't a loaded persona.
- `GET /agents` — lists the loaded persona names.

Both chat endpoints accept an optional `session_id`; when included, the last 20 messages for that `(session_id, agent)` pair are sent back to Mistral as context. **Memory is scoped per agent, not just per session** — reusing the same `session_id` across different agents does not leak one agent's conversation into another's; each keeps its own thread of history even under a shared session. Persona text lives in `backend/config/personas/*.yaml` — edit those files to change how an agent talks, no code changes needed.

NEXUS's prompt is explicit that none of the six agents have real-time data, live web access, or tool use of any kind — nothing in this backend does function-calling or web search — and that FORGE/ORACLE/SENTINEL/CODEX/AVERY are real, reachable agents (switch in the app) rather than something NEXUS "routes" a request to on your behalf. An earlier version of the prompt predated the other five agents and described them as not-yet-online, which led NEXUS to invent plausible-sounding fictional capabilities ("Research Agent is fetching...") instead of just naming the real agent to switch to, or a real capability gap.

Both chat endpoints are also rate-limited (default 20 requests/minute, per API key — or per client IP locally if `APP_API_KEY` is unset) and cap message length (default 4000 characters), to keep a leaked key or a client bug from running up unbounded Mistral usage. Tune via `RATE_LIMIT_PER_MINUTE` and `MAX_MESSAGE_LENGTH`; exceeding either returns a normal error the PWA already displays inline (429 with a `Retry-After` header, or 400).

Every chat response includes a `message_id`; `POST /feedback` (`{message_id, rating}`, rating `1` or `-1`) records a thumbs up/down on that specific reply. The PWA shows this as small thumb icons under each assistant message. This is entirely optional now — see KAIROS below — but a flagged message still gets folded into the next digest cycle as extra-weighted signal if you use it.

### KAIROS (automatic conversation digestion)

KAIROS originally scored individual messages by a length heuristic plus feedback and ranked which ones to keep in context. It's since been redesigned around a simpler idea: **understand the conversation as a whole, automatically, with no voting required**, rather than ranking messages one at a time.

Instead of a per-message score, each agent keeps one rolling summary — a `conversation_digests` table (`core/digest.py`), one row per agent (not per session: the agent's understanding accumulates across every conversation you have with it). Periodically, new raw turns get folded into that summary by an incremental Mistral call: *"here's what you currently know, here are the new exchanges since then — merge them: keep what's still true, revise what got corrected, add what's new, drop what's stale."* Because only the new slice is ever sent (never the whole transcript), the cost of each digest cycle stays flat no matter how long the relationship with an agent gets.

**Triggers** (checked after every message, no extra API calls to decide):
- **Count** — once 10 new turns have accumulated since the last digest.
- **Pause** — a request whose gap since the previous message exceeds 20 minutes (with at least 3 turns pending) triggers a digest of everything before it, treating the gap itself as the natural conversation boundary. No poller needed — this piggybacks on the very next message.
- **Adaptive** — the digest call reports back whether the new turns represented a topic shift; if so, the next threshold is halved, so a real change in subject gets picked up faster without a dedicated (and costly) drift detector.

**Coalescing** — an in-memory per-agent lock (`core/digest.py`) means at most one digest cycle runs per agent at a time; a trigger that fires while one is already running is a no-op, since the in-flight run reads current state when it actually executes and absorbs whatever accumulated.

**Backoff** — a failed digest call (Mistral down or slow) is never surfaced to the user; it's recorded and doubles the effective trigger threshold (capped at 4×), so a struggling API gets digested against less often rather than retried aggressively. A success resets it.

**Fire-and-forget** — digestion is scheduled via `asyncio.create_task` *after* the chat response is already built, so it adds zero latency to the reply the user sees.

At chat time, `run_agent_chat` injects the agent's current digest (if any) as extra context ahead of a short raw tail (the last 6 messages — the digest now carries the rest), instead of KAIROS's old ranked-message window. With no digest yet (a brand-new agent), this collapses to exactly the original plain-recency behavior — nothing changes until the agent has actually accumulated something to say.

This isn't a scheduled background worker — Render's free tier has no free cron, and the service sleeps when idle anyway — so every trigger, coalesce, and backoff decision above is computed inline, synchronously, from local state, with the one actual Mistral call always fired off in the background.

### SAGE (persona evolution)

An agent's `system_prompt` can be overridden at runtime without touching its YAML file or redeploying. Overrides live in a `persona_overrides` table (`agent` primary key, one row per agent) and take precedence over the YAML default whenever present; `run_agent_chat` checks the override table before falling back to `get_system_prompt`.

- `POST /sage/override/{agent}` (`{system_prompt}`) — sets or replaces an agent's override directly. A manual/testing escape hatch — the proposal flow below calls this same storage after a human accepts a proposal.
- `POST /sage/reset/{agent}` — clears the override, reverting the agent to its YAML-defined persona.

An override never touches the YAML files themselves — which wouldn't survive Render's ephemeral filesystem anyway — so resetting is always instant and always available.

SAGE proposes revisions itself, on demand, from the agent's KAIROS digest — not from isolated feedback examples:

- `POST /sage/analyze/{agent}` — sends the agent's current system prompt and its whole conversation digest to Mistral, and asks it to propose a revision based on the goals, preferences, and corrections that digest has accumulated. 400 if the agent has no digest yet — chat with it a bit first. The LLM's response is parsed strictly as `{"rationale": "...", "proposed_system_prompt": "..."}`; anything that doesn't parse as that exact shape is a 502, never silently stored as a proposal. A successful analysis is stored as a `pending` row in `sage_proposals` and returned.
- `GET /sage/proposals` (optional `?agent=&status=` filters) — lists proposals, most recent first.
- `POST /sage/proposals/{id}/accept` — writes the proposed prompt into `persona_overrides` (so it takes effect immediately) and marks the proposal `accepted`.
- `POST /sage/proposals/{id}/reject` — marks it `rejected`; the override table is untouched.
- Both transitions only apply to a `pending` proposal — acting on one that's already been accepted or rejected returns 409, so a proposal can't be double-applied or re-decided.

Since a digest builds up automatically from ordinary use (see KAIROS above), SAGE now has real material to analyze from turn one — no votes required, though a flagged message still shows up in the digest as extra-weighted signal if you do vote.

Nothing here is autonomous: analysis only runs when `/sage/analyze/{agent}` is called, and a proposal only changes live behavior once a human explicitly accepts it. The PWA panel (Settings → "Persona (SAGE)") drives all of this: an "Analyze feedback" button, a list of pending proposals with Accept/Reject, and "Reset persona to default" — the currently selected agent throughout, matching the chat view. Verified end to end with a real browser (Playwright): a seeded pending proposal renders correctly and shows its rationale/proposed prompt, accepting one actually writes the override (confirmed against the database, not just the UI), and rejecting one leaves the override untouched; the no-digest-yet error path from `/sage/analyze` also surfaces correctly in the panel instead of failing silently.

### Run locally

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # fill in MISTRAL_API_KEY from https://console.mistral.ai
export $(cat .env | xargs)
uvicorn main:app --reload
```

Test it:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/agents -H "X-API-Key: $APP_API_KEY"

# first turn — no session_id, server creates one and returns it
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $APP_API_KEY" \
  -d '{"message": "My favorite color is teal."}'

# second turn — pass the returned session_id back to keep context
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $APP_API_KEY" \
  -d '{"message": "What is my favorite color?", "session_id": "<paste session_id here>"}'

# talk to a specific agent directly instead of NEXUS
curl -X POST http://127.0.0.1:8000/agents/forge \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $APP_API_KEY" \
  -d '{"message": "Sketch a plan for a rate limiter."}'

curl http://127.0.0.1:8000/memory/stats -H "X-API-Key: $APP_API_KEY"
```

`APP_API_KEY` is optional locally (auth is skipped if unset) but should always be set in production.

### Tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest -v
```

Covers persona loading, memory (round-trips, per-agent/per-session isolation, the schema-migration guard, fetching an agent's messages since a given point for digestion), KAIROS's digest engine (`core/digest.py`: count/pause/backoff/topic-shift trigger math, per-agent coalescing — including a deliberate-break check confirming the coalescing lock actually prevents a concurrent second run rather than just looking like it does — incremental merge-not-append prompt construction, and malformed-response handling that records a failure without ever raising into the chat path), persona overrides (round-trip, upsert, reset, per-agent scoping), SAGE proposals (create/list/filter, accept/reject state transitions, rejecting a double-decision), rate limiting, and the API layer end to end (auth, 404s, the length cap, the 429 path, a digest correctly appearing as extra context in the outgoing Mistral request once one exists and correctly absent before then, digest scheduling actually firing once the turn threshold is crossed, SAGE override/reset and accept/reject actually changing the outgoing Mistral request, a malformed LLM analysis response correctly 502ing instead of being stored as a proposal, and — the one that matters most — that one agent's conversation never leaks into another's outgoing request to Mistral, checked by inspecting the mocked request payload itself rather than just row counts). The Mistral call is mocked so tests run offline with no API key or network access needed; everything else exercises real code paths, including a real local `libsql` database per test.

Runs automatically on every PR and push to `main` via `.github/workflows/ci.yml`.

### Persistent memory (Turso)

Conversation history is read/written through the [`libsql`](https://pypi.org/project/libsql/) Python package, which is drop-in DB-API-2.0-compatible with SQLite:

- **`TURSO_DATABASE_URL` / `TURSO_AUTH_TOKEN` unset (default)** — falls back to a local file at `backend/data/memory.db` (gitignored). On Render's free tier this filesystem is ephemeral: the file is wiped on every deploy and periodically on restart, so conversations don't survive. Fine for local dev or quick experiments.
- **Both set** — connects to a remote [Turso](https://turso.tech) database instead, which does survive redeploys and restarts.

To set it up:
1. Create a free account at [turso.tech](https://turso.tech) and a database (via their dashboard or `turso db create ailicious`).
2. Get the URL (`turso db show ailicious --url`, looks like `libsql://ailicious-<org>.turso.io`) and an auth token (`turso db tokens create ailicious`).
3. Set `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN` in Render's dashboard (or `.env` locally) — no code changes needed, no schema migration to run by hand. The same `CREATE TABLE IF NOT EXISTS` / idempotent `ALTER TABLE` logic that runs on every startup handles a fresh database or an existing one transparently.

### Deploy to Render

1. Push this repo to GitHub and create a new **Blueprint** on [Render](https://render.com) pointing at it — it will pick up `render.yaml` automatically.
2. Set `MISTRAL_API_KEY` and `APP_API_KEY` in the Render dashboard (marked `sync: false` in the blueprint so they aren't committed). Optionally set `TURSO_DATABASE_URL` / `TURSO_AUTH_TOKEN` too, per above, for memory that survives redeploys.
3. Once deployed, verify with `curl https://<your-service>.onrender.com/health`.

### Next steps

The original plan's phases are now all in place: cloud backend, remote inference, durable memory, personas, full agent swarm, the Android/web client, basic hardening (rate limiting, message-length caps), and a CI-backed test suite.

KAIROS/SAGE — the self-improvement loops from the original blueprint — are in place and have already evolved once: KAIROS started as per-message utility scoring driven by explicit thumbs up/down, then was redesigned into automatic whole-conversation digestion (count/pause/adaptive triggers, per-agent coalescing, backoff, all fire-and-forget) that needs no voting at all, and SAGE's proposal generation was rewired to analyze that digest instead of isolated feedback examples. Neither is a scheduled background worker — Render's free tier doesn't support that without cost, and the service sleeps when idle anyway — so every trigger decision is computed inline from local state, with the one actual Mistral call per cycle always fired off in the background. Eventually, real per-user auth if this is ever used by more than one person, rather than a single shared API key; a semantic (embedding-based) upgrade to KAIROS's retrieval is also on the table, using libSQL's native vector support so context selection isn't just digest-plus-recency but genuinely relevance-aware.