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


class VoiceChatRequest(BaseModel):
    text: str
    messages: list[Message]


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "index.html")


@app.post("/chat")
def chat(req: ChatRequest):
    model = req.model if req.model in ALLOWED_MODELS else DEFAULT_MODEL
    try:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        text = cos_agent.respond(messages, model=model)
        return {"reply": text}
    except anthropic.APIError as e:
        return {"reply": f"Ошибка обращения к Claude: {e}"}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    model = req.model if req.model in ALLOWED_MODELS else DEFAULT_MODEL
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    def generate():
        try:
            for chunk in cos_agent.respond_stream(messages, model=model):
                yield f"data: {json.dumps({'type': 'text', 'chunk': chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/voice")
def voice_page():
    return FileResponse(BASE_DIR / "voice.html")


@app.post("/voice/chat")
async def voice_chat(req: VoiceChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    messages.append({"role": "user", "content": req.text})

    from agent import SYSTEM_PROMPT
    voice_system = SYSTEM_PROMPT + VOICE_SYSTEM_ADDENDUM

    try:
        reply_text = cos_agent.respond(
            messages, model=VOICE_MODEL, max_tokens=VOICE_MAX_TOKENS,
            system_prompt=voice_system,
        )
    except Exception as e:
        return {"error": f"Ошибка агента: {e}"}

    reply_audio_b64 = await tts_text(reply_text)

    return {
        "reply_text": reply_text,
        "reply_audio": reply_audio_b64,
    }


@app.post("/voice/chat/stream")
async def voice_chat_stream(req: VoiceChatRequest):
    from agent import SYSTEM_PROMPT
    voice_system = SYSTEM_PROMPT + VOICE_SYSTEM_ADDENDUM

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    messages.append({"role": "user", "content": req.text})

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
