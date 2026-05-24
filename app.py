import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Папка, где лежит этот файл. Привязываемся к ней, а не к папке запуска,
# чтобы .env и index.html находились независимо от того, откуда стартуем сервер.
BASE_DIR = Path(__file__).parent

# Берём ключ из .env. override=True — файл .env главнее, чем случайная
# пустая переменная ANTHROPIC_API_KEY в окружении (иначе она его «затеняет»).
load_dotenv(BASE_DIR / ".env", override=True)

if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit("Открой файл .env и вставь свой ключ в строку ANTHROPIC_API_KEY=")

client = anthropic.Anthropic()

# app — это наше веб-приложение. uvicorn будет его запускать.
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


# Одна реплика диалога: кто сказал (user/assistant) и что.
class Message(BaseModel):
    role: str
    content: str


# Фронт присылает всю историю — так Claude «помнит» предыдущие сообщения.
class ChatRequest(BaseModel):
    messages: list[Message]


# GET / — браузер заходит на эту страницу и получает HTML-файл.
@app.get("/")
def index():
    return FileResponse(BASE_DIR / "index.html")


# POST /chat — фронт шлёт всю историю, мы спрашиваем Claude и возвращаем ответ.
@app.post("/chat")
def chat(req: ChatRequest):
    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            # 1024 было для урока 1; для развёрнутых ответов и таблиц мало.
            max_tokens=8192,
            messages=[{"role": m.role, "content": m.content} for m in req.messages],
        )
        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return {"reply": text}
    except anthropic.APIError as e:
        # Чтобы в браузере был понятный текст, а не падение сервера.
        return {"reply": f"Ошибка обращения к Claude: {e}"}
