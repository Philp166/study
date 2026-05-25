import asyncio
import base64
import io
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
import edge_tts
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import anthropic

from agent import Agent
import db

BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR / ".env", override=True)

if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit("Открой файл .env и вставь свой ключ в строку ANTHROPIC_API_KEY=")

client = anthropic.Anthropic()
cos_agent = Agent(client)

db.init_db()

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

TTS_VOICE = "ru-RU-SvetlanaNeural"
VOICE_MODEL = "claude-haiku-4-5"
VOICE_MAX_TOKENS = 500

VOICE_SYSTEM_ADDENDUM = """

## Режим голосового чата
Сейчас ты общаешься голосом. Правила:
- Отвечай КРАТКО: 1-3 предложения. Максимум 4-5 для сложных тем.
- НЕ используй markdown, списки, заголовки, блоки кода — ответ будет озвучен.
- Говори естественно, как в живом разговоре.
- Если тема требует развёрнутого ответа, предложи перейти в текстовый чат.
"""


def strip_markdown(text: str) -> str:
    """Убирает markdown-разметку, чтобы TTS не читал спецсимволы вслух."""
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[(){}[\]]", "", text)
    text = re.sub(r"[/\\|~^<>]", " ", text)
    text = re.sub(r"—|--|–", ", ", text)
    text = re.sub(r"[;:…•·«»\"'\"\"'']", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def tts_text(text: str) -> str:
    """Озвучивает текст через edge-tts, возвращает base64 mp3."""
    try:
        clean = strip_markdown(text)
        if not clean.strip():
            return ""
        communicate = edge_tts.Communicate(clean, TTS_VOICE)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        if buf.tell() == 0:
            return ""
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None


class StreamRequest(BaseModel):
    chat_id: str
    model: str | None = None


class VoiceChatRequest(BaseModel):
    text: str
    chat_id: str | None = None
    messages: list[Message] = []


class ChatCreateRequest(BaseModel):
    title: str = ""
    model: str = DEFAULT_MODEL


class ChatUpdateRequest(BaseModel):
    title: str | None = None
    pinned: bool | None = None
    model: str | None = None


class MigrateChat(BaseModel):
    id: str
    title: str = ""
    pinned: bool = False
    model: str = DEFAULT_MODEL
    createdAt: float
    updatedAt: float
    messages: list[Message] = []


class MigrateRequest(BaseModel):
    chats: list[MigrateChat]


# ── Pages ──

@app.get("/")
def index():
    return FileResponse(BASE_DIR / "index.html")


# ── Chat CRUD API ──

@app.get("/api/chats")
def api_list_chats():
    return db.list_chats()


@app.post("/api/chats")
def api_create_chat(req: ChatCreateRequest):
    model = req.model if req.model in ALLOWED_MODELS else DEFAULT_MODEL
    cid = db.create_chat(title=req.title, model=model)
    return {"id": cid}


@app.get("/api/chats/{chat_id}")
def api_get_chat(chat_id: str):
    chat = db.get_chat(chat_id)
    if not chat:
        return {"error": "not_found"}
    return chat


@app.patch("/api/chats/{chat_id}")
def api_update_chat(chat_id: str, req: ChatUpdateRequest):
    fields = {}
    if req.title is not None:
        fields["title"] = req.title
    if req.pinned is not None:
        fields["pinned"] = req.pinned
    if req.model is not None:
        fields["model"] = req.model if req.model in ALLOWED_MODELS else DEFAULT_MODEL
    if fields:
        db.update_chat(chat_id, **fields)
    return {"ok": True}


@app.delete("/api/chats/{chat_id}")
def api_delete_chat(chat_id: str):
    db.delete_chat(chat_id)
    return {"ok": True}


@app.post("/api/chats/{chat_id}/messages")
def api_add_message(chat_id: str, msg: Message):
    db.add_message(chat_id, msg.role, msg.content)
    return {"ok": True}


@app.post("/api/migrate")
def api_migrate(req: MigrateRequest):
    for c in req.chats:
        msgs = [{"role": m.role, "content": m.content} for m in c.messages]
        db.import_chat(
            chat_id=c.id, title=c.title, pinned=c.pinned,
            model=c.model if c.model in ALLOWED_MODELS else DEFAULT_MODEL,
            created_at=c.createdAt / 1000, updated_at=c.updatedAt / 1000,
            messages=msgs,
        )
    return {"ok": True, "imported": len(req.chats)}


# ── LLM Streaming (now with chat_id) ──

@app.post("/chat/stream")
async def chat_stream(req: StreamRequest):
    chat = db.get_chat(req.chat_id)
    if not chat:
        return {"error": "chat not found"}
    model = req.model if req.model in ALLOWED_MODELS else chat.get("model", DEFAULT_MODEL)
    messages = chat["messages"]

    def generate():
        full_text = ""
        try:
            for chunk in cos_agent.respond_stream(messages, model=model):
                full_text += chunk
                yield f"data: {json.dumps({'type': 'text', 'chunk': chunk})}\n\n"
            db.add_message(req.chat_id, "assistant", full_text)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/voice")
def voice_page():
    return FileResponse(BASE_DIR / "voice.html")


@app.post("/voice/chat/stream")
async def voice_chat_stream(req: VoiceChatRequest):
    from agent import SYSTEM_PROMPT
    voice_system = SYSTEM_PROMPT + VOICE_SYSTEM_ADDENDUM

    if req.chat_id:
        chat = db.get_chat(req.chat_id)
        messages = chat["messages"] if chat else []
    else:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
    messages.append({"role": "user", "content": req.text})

    if req.chat_id:
        db.add_message(req.chat_id, "user", req.text)

    async def generate():
        text_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def run_claude():
            try:
                for chunk in cos_agent.respond_stream(
                    messages,
                    model=VOICE_MODEL,
                    max_tokens=VOICE_MAX_TOKENS,
                    system_prompt=voice_system,
                ):
                    loop.call_soon_threadsafe(text_queue.put_nowait, ("chunk", chunk))
            except Exception as e:
                loop.call_soon_threadsafe(text_queue.put_nowait, ("error", str(e)))
            loop.call_soon_threadsafe(text_queue.put_nowait, ("end", ""))

        loop.run_in_executor(None, run_claude)

        sentence_buffer = ""
        full_text = ""

        while True:
            try:
                msg_type, data = await asyncio.wait_for(text_queue.get(), timeout=60)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Timeout'})}\n\n"
                break

            if msg_type == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': data})}\n\n"
                break

            if msg_type == "end":
                if sentence_buffer.strip():
                    audio = await tts_text(sentence_buffer.strip())
                    if audio:
                        yield f"data: {json.dumps({'type': 'audio', 'data': audio})}\n\n"
                if req.chat_id and full_text:
                    db.add_message(req.chat_id, "assistant", full_text)
                yield f"data: {json.dumps({'type': 'done', 'full_text': full_text})}\n\n"
                break

            chunk = data
            full_text += chunk
            sentence_buffer += chunk

            yield f"data: {json.dumps({'type': 'text', 'chunk': chunk})}\n\n"

            sentences = re.split(r'(?<=[.!?])\s+', sentence_buffer)
            if len(sentences) > 1:
                for s in sentences[:-1]:
                    if s.strip():
                        audio = await tts_text(s)
                        if audio:
                            yield f"data: {json.dumps({'type': 'audio', 'data': audio})}\n\n"
                sentence_buffer = sentences[-1]

    return StreamingResponse(generate(), media_type="text/event-stream")
