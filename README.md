# AILicious

Personality-driven, multi-agent AI system with cloud-based inference and memory, accessible from Android via a web client. All compute runs server-side — no local hardware or model downloads required.

## Backend (Phase 1: minimal walking skeleton)

A single-route FastAPI service that proxies chat messages to Groq. This is step one of the build-out: prove the deploy + provider integration before adding agents, persona, and memory.

### Run locally

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY; get one free at https://console.groq.com
export $(cat .env | xargs)
uvicorn main:app --reload
```

Test it:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $APP_API_KEY" \
  -d '{"message": "hello"}'
```

`APP_API_KEY` is optional locally (auth is skipped if unset) but should always be set in production.

### Deploy to Render

1. Push this repo to GitHub and create a new **Blueprint** on [Render](https://render.com) pointing at it — it will pick up `render.yaml` automatically.
2. Set the `GROQ_API_KEY` and `APP_API_KEY` environment variables in the Render dashboard (marked `sync: false` in the blueprint so they aren't committed).
3. Once deployed, verify with `curl https://<your-service>.onrender.com/health`.

### Next steps

Once `/chat` is confirmed working in production: add SQLite-backed memory, load the NEXUS persona from YAML, wire up the remaining agents (FORGE, ORACLE, SENTINEL, CODEX, AVERY), then build the Android PWA client.