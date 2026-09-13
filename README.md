# AILicious

Personality-driven, multi-agent AI system with cloud-based inference and memory, accessible from Android via a web client. All compute runs server-side — no local hardware or model downloads required.

## Android / web client

A single-page PWA is served directly by the backend at `/` (`backend/static/index.html` — no build step, no separate host). Open the Render URL on an Android phone in Chrome, enter the API key in Settings (gear icon), pick an agent from the dropdown, and chat. "Add to Home Screen" installs it as a standalone app icon via the manifest + service worker in `backend/static/`.

Client-side notes:
- The API key is stored in the browser's `localStorage`, sent only as an `X-API-Key` header to this same-origin backend — never committed, never sent anywhere else.
- Each agent gets its own `session_id` and message history in `localStorage`, matching the backend's per-agent memory scoping — switching agents shows that agent's own conversation, not a mixed one.
- Verified end to end with a real browser (Playwright): entering a key, picking an agent, sending a message, and getting back a real Mistral-backed reply, plus confirming history correctly persists per agent when switching back and forth.

## Backend (Phase 1: minimal walking skeleton)

A FastAPI service that proxies chat messages to Mistral, with SQLite-compatible conversation memory (local file, or Turso for durability — see below) and six personas (NEXUS, FORGE, ORACLE, SENTINEL, CODEX, AVERY) loaded from YAML at startup.

- `POST /chat` — always talks to NEXUS, the default entry point.
- `POST /agents/{name}` — talk to a specific agent directly (`forge`, `oracle`, `sentinel`, `codex`, or `avery`); 404 if the name isn't a loaded persona.
- `GET /agents` — lists the loaded persona names.

Both chat endpoints accept an optional `session_id`; when included, the last 20 messages for that `(session_id, agent)` pair are sent back to Mistral as context. **Memory is scoped per agent, not just per session** — reusing the same `session_id` across different agents does not leak one agent's conversation into another's; each keeps its own thread of history even under a shared session. Persona text lives in `backend/config/personas/*.yaml` — edit those files to change how an agent talks, no code changes needed.

Both chat endpoints are also rate-limited (default 20 requests/minute, per API key — or per client IP locally if `APP_API_KEY` is unset) and cap message length (default 4000 characters), to keep a leaked key or a client bug from running up unbounded Mistral usage. Tune via `RATE_LIMIT_PER_MINUTE` and `MAX_MESSAGE_LENGTH`; exceeding either returns a normal error the PWA already displays inline (429 with a `Retry-After` header, or 400).

Every chat response includes a `message_id`; `POST /feedback` (`{message_id, rating}`, rating `1` or `-1`) records a thumbs up/down on that specific reply. The PWA shows this as small thumb icons under each assistant message, feeding into KAIROS below.

### KAIROS (utility-weighted memory retrieval)

Every stored message carries a `utility_score`, seeded at write time by a small heuristic (longer, more substantive turns score slightly higher, capped so a wall of text doesn't dominate) and nudged by feedback (`+2.0` on a thumbs up, `-2.0` on a thumbs down, applied as a delta so re-voting or switching a vote never double-counts). `get_history` uses this score to let a genuinely valuable older message survive in context over worthless-but-recent filler, instead of always dropping to a flat recency window — it pulls a larger recency-ordered candidate pool, ranks by recency + utility, takes the top N, then re-sorts back into chronological order so the transcript the model sees still reads naturally.

This isn't a scheduled background job — Render's free tier has no free cron, and the service sleeps when idle anyway — so KAIROS is inline scoring computed synchronously at write/read time. When nothing has any feedback yet (the common case for a new or quiet agent), ranking collapses to exactly what a plain `ORDER BY id DESC LIMIT` would return — zero behavior change until real signal exists.

### SAGE (persona evolution)

An agent's `system_prompt` can be overridden at runtime without touching its YAML file or redeploying. Overrides live in a `persona_overrides` table (`agent` primary key, one row per agent) and take precedence over the YAML default whenever present; `run_agent_chat` checks the override table before falling back to `get_system_prompt`.

- `POST /sage/override/{agent}` (`{system_prompt}`) — sets or replaces an agent's override directly. A manual/testing escape hatch — the proposal flow below calls this same storage after a human accepts a proposal.
- `POST /sage/reset/{agent}` — clears the override, reverting the agent to its YAML-defined persona.

An override never touches the YAML files themselves — which wouldn't survive Render's ephemeral filesystem anyway — so resetting is always instant and always available.

SAGE proposes revisions itself, on demand, from real feedback:

- `POST /sage/analyze/{agent}` — pulls up to 50 of the agent's most recent feedback'd exchanges (thumbs up/down, paired with the user message that prompted each), sends them to Mistral along with the agent's current system prompt, and asks it to propose a revision that reinforces what worked and steers away from what didn't. 400 if the agent has no feedback yet — nothing to analyze. The LLM's response is parsed strictly as `{"rationale": "...", "proposed_system_prompt": "..."}`; anything that doesn't parse as that exact shape is a 502, never silently stored as a proposal. A successful analysis is stored as a `pending` row in `sage_proposals` and returned.
- `GET /sage/proposals` (optional `?agent=&status=` filters) — lists proposals, most recent first.
- `POST /sage/proposals/{id}/accept` — writes the proposed prompt into `persona_overrides` (so it takes effect immediately) and marks the proposal `accepted`.
- `POST /sage/proposals/{id}/reject` — marks it `rejected`; the override table is untouched.
- Both transitions only apply to a `pending` proposal — acting on one that's already been accepted or rejected returns 409, so a proposal can't be double-applied or re-decided.

Nothing here is autonomous: analysis only runs when `/sage/analyze/{agent}` is called, and a proposal only changes live behavior once a human explicitly accepts it. The next step is a PWA panel to trigger analysis and review proposals without curling the API by hand.

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

Covers persona loading, memory (round-trips, per-agent/per-session isolation, the schema-migration guard, KAIROS's utility-weighted ranking — including a case a plain-recency implementation would get wrong, and pairing feedback'd exchanges for SAGE), persona overrides (round-trip, upsert, reset, per-agent scoping), SAGE proposals (create/list/filter, accept/reject state transitions, rejecting a double-decision), rate limiting, and the API layer end to end (auth, 404s, the length cap, the 429 path, SAGE override/reset and accept/reject actually changing the outgoing Mistral request, a malformed LLM analysis response correctly 502ing instead of being stored as a proposal, and — the one that matters most — that one agent's conversation never leaks into another's outgoing request to Mistral, checked by inspecting the mocked request payload itself rather than just row counts). The Mistral call is mocked so tests run offline with no API key or network access needed; everything else exercises real code paths, including a real local `libsql` database per test.

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

KAIROS/SAGE — the self-improvement loops from the original blueprint — are underway, shipped as incremental steps: feedback capture, utility-weighted memory retrieval (KAIROS), override storage, and now SAGE's actual proposal generation (this PR) are done. Still to come: a PWA panel to trigger analysis and review/accept/reject proposals without curling the API by hand. Neither KAIROS nor SAGE is a scheduled background worker — Render's free tier doesn't support that without cost, and the service sleeps when idle anyway — so KAIROS is inline scoring computed at write/read time, and SAGE analysis only runs when explicitly triggered, never autonomously. Eventually, real per-user auth if this is ever used by more than one person, rather than a single shared API key.