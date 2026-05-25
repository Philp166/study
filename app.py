import asyncio
import base64
import io
import os
from pathlib import Path

from dotenv import load_dotenv
import edge_tts
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import anthropic

from agent import Agent

BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR / ".env", override=True)

if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit("Открой файл .env и вставь свой ключ в строку ANTHROPIC_API_KEY=")

client = anthropic.Anthropic()
cos_agent = Agent(client)

app = FastAPI()

_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
if _origins:
    allowed_origins = [o.strip() for o in _origins.split(",") if o.strip()]
else:
    allowed_origins = ["*"]
    print(
        "ВНИМАНИЕ: ALLOWED_ORIGINS не задан — /chat открыт всем. Любой, кто "
        "узнает адрес, сможет тратить твои токены Anthropic. На проде укажи "
        "ALLOWED_ORIGINS = адрес твоего фронтенда."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_MODELS = {
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
}
DEFAULT_MODEL = "claude-opus-4-6"

TTS_VOICE = "ru-RU-DmitryNeural"


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None


class VoiceChatRequest(BaseModel):
    text: str
    messages: list[Message]
    model: str | None = None


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "index.html")


@app.post("/chat")
def chat(req: ChatRequest):
    model = req.model if req.model in ALLOWED_MODELS else DEFAULT_MODEL
    cos_agent.model = model
    try:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        text = cos_agent.respond(messages)
        return {"reply": text}
    except anthropic.APIError as e:
        return {"reply": f"Ошибка обращения к Claude: {e}"}


@app.get("/voice")
def voice_page():
    return FileResponse(BASE_DIR / "voice.html")


@app.post("/voice/chat")
async def voice_chat(req: VoiceChatRequest):
    model = req.model if req.model in ALLOWED_MODELS else DEFAULT_MODEL
    cos_agent.model = model

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    messages.append({"role": "user", "content": req.text})

    try:
        reply_text = cos_agent.respond(messages)
    except Exception as e:
        return {"error": f"Ошибка агента: {e}"}

    reply_audio_b64 = ""
    try:
        communicate = edge_tts.Communicate(reply_text, TTS_VOICE)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        reply_audio_b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        pass

    return {
        "reply_text": reply_text,
        "reply_audio": reply_audio_b64,
    }
