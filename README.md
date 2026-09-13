# AILicious

Personality-driven, multi-agent AI system with cloud-based inference and memory, accessible from Android via a web client. All compute runs server-side — no local hardware or model downloads required.

## Backend (Phase 1: minimal walking skeleton)

A FastAPI service that proxies chat messages to Mistral, with SQLite-backed conversation memory. `POST /chat` accepts an optional `session_id`; when included, the last 20 messages for that session are sent back to Mistral as context, so the model actually remembers earlier turns in the same conversation.

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

curl http://127.0.0.1:8000/memory/stats -H "X-API-Key: $APP_API_KEY"
```

`APP_API_KEY` is optional locally (auth is skipped if unset) but should always be set in production. Conversation history is stored in `backend/data/memory.db` (SQLite, gitignored).

**Render free-tier caveat:** the free plan's filesystem is ephemeral, so `memory.db` is wiped on every deploy and periodically on restart. Conversations won't survive across deploys until this moves to a persistent store (a paid Render Disk, or an external DB like Turso/Supabase, per the original plan). Fine for now while iterating; worth revisiting before this is a real daily-driver.

### Deploy to Render

1. Push this repo to GitHub and create a new **Blueprint** on [Render](https://render.com) pointing at it — it will pick up `render.yaml` automatically.
2. Set the `MISTRAL_API_KEY` and `APP_API_KEY` environment variables in the Render dashboard (marked `sync: false` in the blueprint so they aren't committed).
3. Once deployed, verify with `curl https://<your-service>.onrender.com/health`.

### Next steps

Load the NEXUS persona from YAML, wire up the remaining agents (FORGE, ORACLE, SENTINEL, CODEX, AVERY), then build the Android PWA client.