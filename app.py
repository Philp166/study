import base64
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import anthropic

from agent import Agent
from salute_speech import SaluteSpeech

# Папка, где лежит этот файл. Привязываемся к ней, а не к папке запуска,
# чтобы .env и index.html находились независимо от того, откуда стартуем сервер.
BASE_DIR = Path(__file__).parent

# Берём ключ из .env. override=True — файл .env главнее, чем случайная
# пустая переменная ANTHROPIC_API_KEY в окружении (иначе она его «затеняет»).
load_dotenv(BASE_DIR / ".env", override=True)

if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit("Открой файл .env и вставь свой ключ в строку ANTHROPIC_API_KEY=")

client = anthropic.Anthropic()
cos_agent = Agent(client)
speech = SaluteSpeech()

app = FastAPI()

# CORS: фронт и бэк на разных адресах, поэтому браузеру нужно явно
# разрешить запросы с домена фронта. Список доменов — в ALLOWED_ORIGINS
# (через запятую). На Render задай его в переменных окружения.
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


# Какие модели разрешены к выбору. Это и фильтр от мусора с публичного
# эндпоинта (чтобы нельзя было прислать произвольный model id), и
# единый источник правды о доступных моделях.
ALLOWED_MODELS = {
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
}
DEFAULT_MODEL = "claude-opus-4-6"


# Одна реплика диалога: кто сказал (user/assistant) и что.
class Message(BaseModel):
    role: str
    content: str


# Фронт присылает всю историю — так Claude «помнит» предыдущие сообщения.
class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None


# GET / — браузер заходит на эту страницу и получает HTML-файл.
@app.get("/")
def index():
    return FileResponse(BASE_DIR / "index.html")


# POST /chat — фронт шлёт всю историю, мы спрашиваем Claude и возвращаем ответ.
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
async def voice_chat(
    audio: UploadFile,
    messages: str = Form("[]"),
    model: str = Form(None),
):
    if not speech.is_configured:
        return {"error": "SaluteSpeech не настроен. Добавь SALUTE_SPEECH_AUTH в .env"}

    model = model if model in ALLOWED_MODELS else DEFAULT_MODEL
    cos_agent.model = model

    audio_bytes = await audio.read()

    try:
        user_text = speech.recognize(audio_bytes)
    except Exception as e:
        return {"error": f"Ошибка распознавания речи: {e}"}

    if not user_text:
        return {"error": "Не удалось распознать речь. Попробуй ещё раз."}

    history = json.loads(messages)
    history.append({"role": "user", "content": user_text})

    try:
        reply_text = cos_agent.respond(history)
    except Exception as e:
        return {"error": f"Ошибка агента: {e}"}

    reply_audio_b64 = ""
    try:
        reply_audio = speech.synthesize(reply_text)
        reply_audio_b64 = base64.b64encode(reply_audio).decode()
    except Exception:
        pass

    return {
        "user_text": user_text,
        "reply_text": reply_text,
        "reply_audio": reply_audio_b64,
    }
