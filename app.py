import asyncio
import base64
import hashlib
import hmac
import io
import json
import logging
import os
import re
import secrets
import time
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env", override=True)

import edge_tts
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger("cos.branch")

import anthropic

from agent import Agent
import db
import summarizer

if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit("Открой файл .env и вставь свой ключ в строку ANTHROPIC_API_KEY=")

client = anthropic.Anthropic()
cos_agent = Agent(client)

db.init_db()

app = FastAPI()

# ── Пароль на весь сайт через форму входа (cookie-сессия) ──
# Пароль берётся ТОЛЬКО из переменной окружения (никогда не из кода/БД/.env-в-репо).
# Нет переменной → вход выключен (удобно для локальной разработки).
# Есть → все страницы и эндпоинты доступны только после ввода пароля на /login.
SITE_PASSWORD = os.getenv("SITE_PASSWORD")

AUTH_COOKIE = "cos_auth"
AUTH_MAX_AGE = 60 * 60 * 24 * 30  # срок жизни сессии — 30 дней

# Пути, доступные без входа: health-check (его пингует Render — под паролем
# сервис считался бы упавшим), favicon и сама страница входа/выхода.
AUTH_EXEMPT_PATHS = {"/healthz", "/favicon.ico", "/login", "/logout"}

if not SITE_PASSWORD:
    logging.warning(
        "SITE_PASSWORD не задан — вход по паролю ОТКЛЮЧЁН, сайт открыт без пароля. "
        "Это нормально для локальной разработки. На проде задай SITE_PASSWORD "
        "в переменных окружения (например, в дашборде Render)."
    )


def _make_auth_token() -> str:
    """Подписанный токен сессии вида `<expiry>.<hmac>`. Ключ подписи — сам пароль,
    поэтому смена SITE_PASSWORD автоматически инвалидирует все старые сессии."""
    expiry = str(int(time.time()) + AUTH_MAX_AGE)
    sig = hmac.new(SITE_PASSWORD.encode(), expiry.encode(), hashlib.sha256).hexdigest()
    return f"{expiry}.{sig}"


def _valid_auth_token(token: str) -> bool:
    """Проверяет подпись и срок токена из cookie (timing-safe)."""
    if not token or "." not in token:
        return False
    expiry, _, sig = token.partition(".")
    expected = hmac.new(SITE_PASSWORD.encode(), expiry.encode(), hashlib.sha256).hexdigest()
    if not secrets.compare_digest(sig, expected):
        return False
    try:
        return int(expiry) > int(time.time())
    except ValueError:
        return False


class SitePasswordMiddleware(BaseHTTPMiddleware):
    """Закрывает все роуты (страницы, API, статика, SSE) до ввода пароля на /login.

    Авторизация хранится в подписанной HttpOnly-cookie, которая браузер сам шлёт
    с каждым запросом, включая fetch-стриминг (same-origin) — SSE не ломается.
    Middleware НЕ читает и НЕ буферизует тело ответа, поэтому StreamingResponse
    проходит чанками без изменений.
    """

    async def dispatch(self, request: Request, call_next):
        # Вход выключен (нет пароля) — пропускаем всё.
        if not SITE_PASSWORD:
            return await call_next(request)

        # CORS preflight и пути, не требующие входа.
        if request.method == "OPTIONS" or request.url.path in AUTH_EXEMPT_PATHS:
            return await call_next(request)

        if _valid_auth_token(request.cookies.get(AUTH_COOKIE, "")):
            return await call_next(request)

        # Не авторизован: навигацию по страницам уводим на форму входа,
        # запросы fetch/API получают 401 (фронтенд сам решит, что показать).
        accept = request.headers.get("accept", "")
        if request.method == "GET" and "text/html" in accept:
            return RedirectResponse("/login", status_code=302)
        return JSONResponse({"error": "unauthorized"}, status_code=401)


# Регистрируем ДО CORS, чтобы CORS остался внешним middleware.
app.add_middleware(SitePasswordMiddleware)

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

TTS_VOICE_DEFAULT = "ru-RU-SvetlanaNeural"
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
        voice = db.get_setting("tts_voice", TTS_VOICE_DEFAULT)
        communicate = edge_tts.Communicate(clean, voice)
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


class AddMessageRequest(BaseModel):
    role: str
    content: str
    expected_parent_message_id: int | None = None


class BranchCreateRequest(BaseModel):
    from_message_id: int
    user_message: str


class SwitchBranchRequest(BaseModel):
    leaf_message_id: int


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


# ── Health-check (без пароля — пингует Render) ──

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# ── Вход по паролю (форма + cookie-сессия) ──

LOGIN_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Вход</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh; display: flex;
      align-items: center; justify-content: center;
      background: #0f1115; color: #e8e8ea;
      font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif;
    }
    .card {
      width: 100%; max-width: 360px; padding: 32px;
      background: #1a1d24; border: 1px solid #2a2e38; border-radius: 14px;
      box-shadow: 0 12px 40px rgba(0,0,0,.4);
    }
    h1 { margin: 0 0 4px; font-size: 20px; }
    p { margin: 0 0 20px; color: #9aa0ab; font-size: 14px; }
    input {
      width: 100%; padding: 12px 14px; font-size: 15px;
      background: #0f1115; color: #e8e8ea;
      border: 1px solid #2a2e38; border-radius: 9px; outline: none;
    }
    input:focus { border-color: #5b8cff; }
    button {
      width: 100%; margin-top: 12px; padding: 12px 14px; font-size: 15px;
      font-weight: 600; color: #fff; cursor: pointer;
      background: #5b8cff; border: none; border-radius: 9px;
    }
    button:hover { background: #4a7bf0; }
    button:disabled { opacity: .6; cursor: default; }
    .err { margin-top: 12px; color: #ff6b6b; font-size: 14px; min-height: 18px; }
  </style>
</head>
<body>
  <form class="card" id="f">
    <h1>Вход</h1>
    <p>Введите пароль, чтобы продолжить.</p>
    <input type="password" id="pwd" placeholder="Пароль" autofocus autocomplete="current-password">
    <button type="submit" id="btn">Войти</button>
    <div class="err" id="err"></div>
  </form>
  <script>
    const f = document.getElementById('f');
    const pwd = document.getElementById('pwd');
    const err = document.getElementById('err');
    const btn = document.getElementById('btn');
    f.addEventListener('submit', async (e) => {
      e.preventDefault();
      err.textContent = '';
      btn.disabled = true;
      try {
        const res = await fetch('/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ password: pwd.value }),
        });
        if (res.ok) {
          window.location.replace('/');
        } else {
          err.textContent = 'Неверный пароль';
          pwd.select();
        }
      } catch (_) {
        err.textContent = 'Ошибка сети, попробуйте ещё раз';
      } finally {
        btn.disabled = false;
      }
    });
  </script>
</body>
</html>"""


class LoginRequest(BaseModel):
    password: str


@app.get("/login")
def login_page(request: Request):
    # Вход выключен или уже авторизован — нечего показывать, ведём на сайт.
    if not SITE_PASSWORD or _valid_auth_token(request.cookies.get(AUTH_COOKIE, "")):
        return RedirectResponse("/", status_code=302)
    return HTMLResponse(LOGIN_HTML)


@app.post("/login")
def login_submit(req: LoginRequest):
    if not SITE_PASSWORD:
        return JSONResponse({"ok": True})
    if secrets.compare_digest(req.password, SITE_PASSWORD):
        resp = JSONResponse({"ok": True})
        resp.set_cookie(
            AUTH_COOKIE, _make_auth_token(),
            max_age=AUTH_MAX_AGE, httponly=True, samesite="lax",
        )
        return resp
    return JSONResponse({"ok": False, "error": "wrong_password"}, status_code=401)


@app.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie(AUTH_COOKIE)
    return resp


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
def api_add_message(chat_id: str, msg: AddMessageRequest):
    # Защита от случайных веток: если клиент знает ожидаемого родителя
    # (current_leaf на момент открытия вкладки), а он разошёлся с актуальным —
    # значит ветку уже сдвинули в другой вкладке / при refresh во время стрима.
    if msg.expected_parent_message_id is not None:
        current = db.get_current_leaf(chat_id)
        if msg.expected_parent_message_id != current:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "stale_leaf",
                    "current_leaf_message_id": current,
                    "chat": db.get_chat(chat_id),
                },
            )
    mid = db.add_message(chat_id, msg.role, msg.content)
    return {"ok": True, "id": mid, "current_leaf_message_id": mid}


@app.get("/api/token-stats")
def api_token_stats():
    return {
        **cos_agent.token_stats.to_dict(),
        "context": cos_agent.get_context_pressure(),
        "all_time": db.get_global_usage(),
    }


@app.get("/api/chats/{chat_id}/usage")
def api_chat_usage(chat_id: str):
    return db.get_chat_usage(chat_id)


@app.get("/api/settings")
def api_get_settings():
    return db.get_all_settings()


@app.patch("/api/settings")
def api_update_settings(updates: dict):
    allowed = {"tts_voice", "voice_enabled"}
    for key, value in updates.items():
        if key in allowed:
            db.set_setting(key, str(value))
    return db.get_all_settings()


@app.get("/api/voices")
async def api_voices():
    voices = await edge_tts.list_voices()
    ru_voices = [
        {"id": v["ShortName"], "name": v["FriendlyName"], "gender": v["Gender"]}
        for v in voices
        if v["Locale"].startswith("ru")
    ]
    return ru_voices


@app.get("/api/token-stats/by-model")
def api_token_stats_by_model():
    return db.get_usage_by_model()


# ── Strategy Settings per-chat ──

_STRATEGY_RANGES = {
    "rolling_summary_threshold_pct": (5, 90),
    "sliding_window_size": (5, 100),
}
_STRATEGY_BOOLS = {"rolling_summary_enabled", "sliding_window_enabled", "sticky_facts_enabled"}
_STRATEGY_EXCLUSIVE = {"rolling_summary_enabled", "sliding_window_enabled", "sticky_facts_enabled"}


@app.get("/api/chats/{chat_id}/strategy-settings")
def api_get_strategy_settings(chat_id: str):
    return db.get_strategy_settings(chat_id)


@app.patch("/api/chats/{chat_id}/strategy-settings")
def api_update_strategy_settings(chat_id: str, updates: dict):
    clean = {}
    for k, v in updates.items():
        if k in _STRATEGY_BOOLS:
            clean[k] = 1 if v else 0
        elif k in _STRATEGY_RANGES:
            lo, hi = _STRATEGY_RANGES[k]
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            if iv < lo or iv > hi:
                continue
            clean[k] = iv
    if not clean:
        return db.get_strategy_settings(chat_id)

    # Mutual exclusivity: enabling one strategy disables the others
    for key in _STRATEGY_EXCLUSIVE:
        if clean.get(key) == 1:
            for other in _STRATEGY_EXCLUSIVE:
                if other != key:
                    clean[other] = 0
            break

    return db.update_strategy_settings(chat_id, **clean)


@app.get("/api/chats/{chat_id}/facts")
def api_get_facts(chat_id: str):
    leaf = db.get_current_leaf(chat_id)
    return db.get_active_facts(chat_id, leaf)


@app.delete("/api/chats/{chat_id}/facts/{fact_id}")
def api_delete_fact(chat_id: str, fact_id: int):
    db.delete_chat_fact_by_id(fact_id)
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

def _stream_chat_response(chat_id: str, model: str,
                          user_message_id: int | None,
                          user_message_text: str | None) -> StreamingResponse:
    """Общий конвейер ответа агента для обычной отправки и для ветвления.

    Контекст собирается по активной ветке (leaf = user_message_id — это
    сообщение, на которое отвечаем). Ответ ассистента сохраняется с
    parent_message_id = user_message_id НАПРЯМУЮ, без повторного чтения
    current_leaf после стрима — это защищает от смены ветки во время стрима.
    """
    system_prompt, messages = summarizer.prepare_context(
        client, model, chat_id,
        cos_agent.system_prompt,
        user_message_id,
        user_message=user_message_text,
    )

    def generate():
        full_text = ""
        try:
            for chunk in cos_agent.respond_stream(messages, model=model, system_prompt=system_prompt):
                full_text += chunk
                yield f"data: {json.dumps({'type': 'text', 'chunk': chunk})}\n\n"
            # parent = id user-сообщения, зафиксированный до стрима;
            # add_message также двигает current_leaf на id ответа.
            assistant_id = db.add_message(
                chat_id, "assistant", full_text,
                parent_message_id=user_message_id,
            )
            u = cos_agent.token_stats.last_usage
            db.save_usage(
                chat_id, model,
                u.input_tokens, u.output_tokens,
                u.cache_creation_input_tokens, u.cache_read_input_tokens,
            )
            usage_data = {
                "type": "usage",
                **cos_agent.token_stats.to_dict(),
                "context": cos_agent.get_context_pressure(model),
                "chat_totals": db.get_chat_usage(chat_id),
            }
            yield f"data: {json.dumps(usage_data)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'leaf_message_id': assistant_id})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat/stream")
async def chat_stream(req: StreamRequest):
    chat = db.get_chat(req.chat_id)
    if not chat:
        return {"error": "chat not found"}
    model = req.model if req.model in ALLOWED_MODELS else chat.get("model", DEFAULT_MODEL)

    msgs = chat.get("messages") or []
    leaf = chat.get("current_leaf_message_id")
    user_message = msgs[-1]["content"] if msgs and msgs[-1]["role"] == "user" else None

    return _stream_chat_response(req.chat_id, model, leaf, user_message)


@app.post("/api/chats/{chat_id}/branch")
async def api_create_branch(chat_id: str, req: BranchCreateRequest):
    """Намеренное ветвление: новое user-сообщение-ребёнок от from_message_id,
    затем обычный конвейер ответа. Проверки expected_parent здесь нет.
    """
    chat = db.get_chat(chat_id)
    if not chat:
        return JSONResponse(status_code=404, content={"error": "chat_not_found"})
    if not db.message_in_chat(chat_id, req.from_message_id):
        return JSONResponse(status_code=404, content={"error": "from_message_not_in_chat"})

    model = chat.get("model", DEFAULT_MODEL)
    if model not in ALLOWED_MODELS:
        model = DEFAULT_MODEL

    new_user_id = db.create_branch_message(chat_id, req.from_message_id, req.user_message)
    log.info("branch chat=%s from=%s new_user_msg=%s", chat_id, req.from_message_id, new_user_id)
    return _stream_chat_response(chat_id, model, new_user_id, req.user_message)


@app.post("/api/chats/{chat_id}/switch-branch")
def api_switch_branch(chat_id: str, req: SwitchBranchRequest):
    """Переключение активной ветки. От переданного узла спускаемся вниз по
    последнему добавленному ребёнку каждого уровня до листа (навигатор).
    """
    if not db.message_in_chat(chat_id, req.leaf_message_id):
        return JSONResponse(
            status_code=409,
            content={"error": "not_in_chat", "chat": db.get_chat(chat_id)},
        )
    final_leaf = db.descend_to_leaf(chat_id, req.leaf_message_id)
    db.set_current_leaf(chat_id, final_leaf)
    log.info(
        "switch-branch chat=%s requested=%s final_leaf=%s",
        chat_id, req.leaf_message_id, final_leaf,
    )
    return db.get_chat(chat_id)


@app.delete("/api/messages/{message_id}/branch")
def api_delete_branch(message_id: int):
    """Каскадно удаляет сообщение и потомков. Если в поддереве был current_leaf —
    он переезжает на родителя удалённого.
    """
    res = db.delete_message_subtree(message_id)
    if not res.get("chat_id"):
        return JSONResponse(status_code=404, content={"error": "not_found"})
    log.info(
        "delete-branch msg=%s chat=%s new_leaf=%s",
        message_id, res["chat_id"], res["new_leaf"],
    )
    return {"ok": True, **res, "chat": db.get_chat(res["chat_id"])}


@app.get("/styles.css")
def serve_css():
    return FileResponse(BASE_DIR / "styles.css")


@app.get("/agent.js")
def serve_agent_js():
    return FileResponse(BASE_DIR / "agent.js", media_type="application/javascript")


@app.get("/voice")
def voice_page():
    if db.get_setting("voice_enabled", "true") == "false":
        return RedirectResponse("/")
    return FileResponse(BASE_DIR / "voice.html")


@app.get("/settings")
def settings_page():
    return FileResponse(BASE_DIR / "settings.html")


@app.post("/voice/chat/stream")
async def voice_chat_stream(req: VoiceChatRequest):
    from agent import SYSTEM_PROMPT
    voice_base = SYSTEM_PROMPT + VOICE_SYSTEM_ADDENDUM

    voice_user_id = None
    if req.chat_id:
        voice_user_id = db.add_message(req.chat_id, "user", req.text)

        voice_system, messages = summarizer.prepare_context(
            client, VOICE_MODEL, req.chat_id,
            voice_base,
            voice_user_id,
            user_message=req.text,
        )
    else:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        messages.append({"role": "user", "content": req.text})
        voice_system = voice_base

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
                    db.add_message(req.chat_id, "assistant", full_text,
                                   parent_message_id=voice_user_id)
                    u = cos_agent.token_stats.last_usage
                    db.save_usage(
                        req.chat_id, VOICE_MODEL,
                        u.input_tokens, u.output_tokens,
                        u.cache_creation_input_tokens, u.cache_read_input_tokens,
                    )
                usage_data = {
                    "type": "usage",
                    **cos_agent.token_stats.to_dict(),
                    "context": cos_agent.get_context_pressure(VOICE_MODEL),
                    "chat_totals": db.get_chat_usage(req.chat_id) if req.chat_id else None,
                }
                yield f"data: {json.dumps(usage_data)}\n\n"
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
