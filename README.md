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

The original plan's phases are now all in place: cloud backend, remote inference, durable memory, personas, full agent swarm, the Android/web client, and basic hardening (rate limiting, message-length caps). What's left is the KAIROS/SAGE self-improvement loops from the original blueprint — a design task before it's a coding task — and eventually real per-user auth if this is ever used by more than one person, rather than a single shared API key.