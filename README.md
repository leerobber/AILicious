# AILicious

Personality-driven, multi-agent AI system with cloud-based inference and memory, accessible from Android via a web client. All compute runs server-side — no local hardware or model downloads required.

## Backend (Phase 1: minimal walking skeleton)

A FastAPI service that proxies chat messages to Mistral, with SQLite-backed conversation memory and six personas (NEXUS, FORGE, ORACLE, SENTINEL, CODEX, AVERY) loaded from YAML at startup.

- `POST /chat` — always talks to NEXUS, the default entry point.
- `POST /agents/{name}` — talk to a specific agent directly (`forge`, `oracle`, `sentinel`, `codex`, or `avery`); 404 if the name isn't a loaded persona.
- `GET /agents` — lists the loaded persona names.

Both chat endpoints accept an optional `session_id`; when included, the last 20 messages for that `(session_id, agent)` pair are sent back to Mistral as context. **Memory is scoped per agent, not just per session** — reusing the same `session_id` across different agents does not leak one agent's conversation into another's; each keeps its own thread of history even under a shared session. Persona text lives in `backend/config/personas/*.yaml` — edit those files to change how an agent talks, no code changes needed.

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

`APP_API_KEY` is optional locally (auth is skipped if unset) but should always be set in production. Conversation history is stored in `backend/data/memory.db` (SQLite, gitignored).

**Render free-tier caveat:** the free plan's filesystem is ephemeral, so `memory.db` is wiped on every deploy and periodically on restart. Conversations won't survive across deploys until this moves to a persistent store (a paid Render Disk, or an external DB like Turso/Supabase, per the original plan). Fine for now while iterating; worth revisiting before this is a real daily-driver.

### Deploy to Render

1. Push this repo to GitHub and create a new **Blueprint** on [Render](https://render.com) pointing at it — it will pick up `render.yaml` automatically.
2. Set the `MISTRAL_API_KEY` and `APP_API_KEY` environment variables in the Render dashboard (marked `sync: false` in the blueprint so they aren't committed).
3. Once deployed, verify with `curl https://<your-service>.onrender.com/health`.

### Next steps

Build the Android PWA client.