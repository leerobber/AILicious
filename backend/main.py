import os

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
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
    return {"status": "ok", "mistral_configured": bool(MISTRAL_API_KEY)}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, x_api_key: str | None = Header(default=None)) -> ChatResponse:
    require_api_key(x_api_key)

    if not MISTRAL_API_KEY:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY is not configured on the server")

    payload = {
        "model": MISTRAL_MODEL,
        "messages": [{"role": "user", "content": req.message}],
    }
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
    reply = data["choices"][0]["message"]["content"]
    return ChatResponse(response=reply)
