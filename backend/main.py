import os

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
APP_API_KEY = os.environ.get("APP_API_KEY")

app = FastAPI(title="AILicious Backend")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


def require_api_key(x_api_key: str | None) -> None:
    if APP_API_KEY and x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "groq_configured": bool(GROQ_API_KEY)}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, x_api_key: str | None = Header(default=None)) -> ChatResponse:
    require_api_key(x_api_key)

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on the server")

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": req.message}],
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(GROQ_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=502, detail=f"Groq API error: {exc.response.status_code} {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to reach Groq API: {exc}") from exc

    data = resp.json()
    reply = data["choices"][0]["message"]["content"]
    return ChatResponse(response=reply)
