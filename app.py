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
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger("cos.branch")

import anthropic

from agent import Agent, MEMORY_TOOLS, build_suggestion
import db
import summarizer

if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit("Открой файл .env и вставь свой ключ в строку ANTHROPIC_API_KEY=")

client = anthropic.Anthropic()
cos_agent = Agent(client)

db.init_db()

app = FastAPI()

# ── Пароль на весь сайт (Bearer-токен, спрашивается при каждом заходе) ──
# Пароль берётся ТОЛЬКО из переменной окружения (никогда не из кода/БД/.env-в-репо).
# Нет переменной → вход выключен (удобно для локальной разработки).
#
# Модель: токен хранится в sessionStorage вкладки (auth-gate.js) и шлётся в
# заголовке `Authorization: Bearer <token>`. Вход РАЗОВЫЙ на сессию — токен
# переживает reload (F5) и переходы между разделами, но очищается при закрытии
# браузера. Cookie не используем сознательно: токен виден только нашему JS и не
# уходит на сервер автоматически с каждым запросом (меньше surface для CSRF).
SITE_PASSWORD = os.getenv("SITE_PASSWORD")

AUTH_TTL = 60 * 60 * 6  # срок жизни токена — 6 часов (страховка от протухания)

# Публичные пути: только «оболочка» сайта (она не содержит данных — данные грузятся
# через /api после ввода пароля) + статика для отрисовки оверлёя входа + health-check.
# Всё остальное (/api/*, стримы) требует валидный Bearer-токен.
PUBLIC_PATHS = {
    "/", "/voice", "/settings", "/memory",
    "/login", "/healthz", "/favicon.ico",
    "/styles.css", "/agent.js", "/auth-gate.js",
}

if not SITE_PASSWORD:
    logging.warning(
        "SITE_PASSWORD не задан — вход по паролю ОТКЛЮЧЁН, сайт открыт без пароля. "
        "Это нормально для локальной разработки. На проде задай SITE_PASSWORD "
        "в переменных окружения (например, в дашборде Render)."
    )


def _make_auth_token() -> str:
    """Подписанный токен вида `<expiry>.<hmac>`. Ключ подписи — сам пароль,
    поэтому смена SITE_PASSWORD автоматически инвалидирует все старые токены."""
    expiry = str(int(time.time()) + AUTH_TTL)
    sig = hmac.new(SITE_PASSWORD.encode(), expiry.encode(), hashlib.sha256).hexdigest()
    return f"{expiry}.{sig}"


def _valid_auth_token(token: str) -> bool:
    """Проверяет подпись и срок токена (timing-safe)."""
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
    """Требует Bearer-токен на всех данных/эндпоинтах (кроме PUBLIC_PATHS).

    Оболочка страниц отдаётся свободно, чтобы клиентский auth-gate.js успел
    показать оверлей входа; секретные данные (/api/*, стримы) — только с токеном.
    Middleware НЕ читает и НЕ буферизует тело ответа, поэтому StreamingResponse /
    SSE проходит чанками без изменений.
    """

    async def dispatch(self, request: Request, call_next):
        # Вход выключен (нет пароля) — пропускаем всё.
        if not SITE_PASSWORD:
            return await call_next(request)

        # CORS preflight и публичная оболочка/статика.
        if request.method == "OPTIONS" or request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and _valid_auth_token(auth[7:]):
            return await call_next(request)

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

# Память в голосе: типизированные инструменты (save_memory/save_task/close_task)
# доступны как в тексте, но карточки подтверждения нет — подтверждение только
# голосом. Какой блок добавить к системнику, решаем по memory_autosave на запрос.
VOICE_MEMORY_AUTOSAVE_ON = """
## Сохранение в память (голос, автосохранение ВКЛ)
Когда пользователь явно просит запомнить/сохранить/записать что-то или отметить/закрыть задачу — сразу вызови нужный инструмент (save_memory / save_task / close_task), заполнив поля самостоятельно, и коротко подтверди вслух: «Сохранил». Не переспрашивай.
"""

VOICE_MEMORY_AUTOSAVE_OFF = """
## Сохранение в память (голос, подтверждение голосом)
Автосохранение выключено, в голосе нет кнопок — подтверждение голосом, и его обеспечивает система.
Когда пользователь явно просит запомнить/сохранить/записать что-то или отметить/закрыть задачу:
1. Сразу вызови нужный инструмент (save_memory / save_task / close_task). Он НЕ сохранит сразу, а подготовит запись и попросит переспросить.
2. Коротко переспроси вслух: «Сохранить это в память?».
3. Если в следующей реплике пользователь подтвердил («да», «давай», «сохрани», «ага», «угу», «верно») — запись применится САМА, просто скажи «Сохранил». НЕ вызывай инструмент повторно.
4. Если пользователь отказался, передумал или сменил тему — ничего не делай, подготовленная запись отменится сама.
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

class LoginRequest(BaseModel):
    password: str


@app.get("/login")
def login_page():
    # Отдельной страницы входа нет — пароль спрашивает оверлей (auth-gate.js)
    # прямо поверх приложения. Прямой заход на /login просто ведём на сайт.
    return RedirectResponse("/", status_code=302)


@app.post("/login")
def login_submit(req: LoginRequest):
    # Возвращаем токен в ТЕЛЕ ответа (не в cookie): фронтенд держит его в памяти
    # вкладки и шлёт в заголовке Authorization. Reload памяти не переживает →
    # пароль спрашивается заново.
    if not SITE_PASSWORD:
        return JSONResponse({"ok": True, "token": ""})
    if secrets.compare_digest(req.password, SITE_PASSWORD):
        return JSONResponse({"ok": True, "token": _make_auth_token()})
    return JSONResponse({"ok": False, "error": "wrong_password"}, status_code=401)


@app.get("/auth-gate.js")
def serve_auth_gate_js():
    return FileResponse(BASE_DIR / "auth-gate.js", media_type="application/javascript")


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
    allowed = {"tts_voice", "voice_enabled", "memory_autosave"}
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


# ── Долговременная память API ──
# Тела запросов — простыми dict (как api_update_settings). Валидация обязательных
# полей — здесь, в app-слое. Папки (folder) — свободная строка, без enum.

MEMORY_STATUSES = {"active", "inactive"}


def _norm_folder(v):
    """Пустая строка/пробелы → None (заметка без папки). Иначе обрезанная строка."""
    if v is None:
        return None
    v = str(v).strip()
    return v or None


@app.get("/api/memory")
def api_list_memory(folder: str | None = None, q: str | None = None):
    return db.list_memory(folder=folder, q=q)


@app.post("/api/memory")
def api_create_memory(body: dict):
    title = (body.get("title") or "").strip()
    if not title:
        return JSONResponse(status_code=400, content={"error": "title_required"})
    content = body.get("content") or ""
    folder = _norm_folder(body.get("folder"))
    row = db.create_memory(title=title, content=content, folder=folder)
    if row is None:
        return JSONResponse(status_code=409, content={"error": "title_exists"})
    return row


# ВАЖНО: /search, /links, /folders регистрируются ДО /{memory_id}, иначе FastAPI
# поймает их как memory_id и упадёт на валидации int (422).
@app.get("/api/memory/search")
def api_search_memory(q: str = ""):
    return db.list_memory(q=q)


@app.get("/api/memory/links")
def api_memory_links():
    return db.get_memory_links()


@app.get("/api/memory/folders")
def api_memory_folders():
    return db.get_memory_folders()


def _resolve_task_id(title) -> int | None:
    """id активной задачи по названию (агент close/update задаёт title, не id)."""
    t = (title or "").strip()
    if not t:
        return None
    found = db.find_active_task_by_title(t)
    return found["id"] if found else None


def _tool_result_msg(saved: dict) -> str:
    """Короткий честный итог для агента по результату _apply_suggestion."""
    parts = []
    mem_txt = {
        "created": "сохранил заметку",
        "merged": "дополнил существующую заметку",
        "duplicate": "заметка уже была записана",
        "updated": "обновил заметку",
    }.get(saved.get("memory_status"))
    if saved.get("memory") and mem_txt:
        parts.append(mem_txt)
    task_txt = {
        "created": "создал задачу",
        "updated": "обновил задачу",
        "closed": "закрыл задачу",
    }.get(saved.get("task_status"))
    if saved.get("task") and task_txt:
        parts.append(task_txt)
    if parts:
        return "Готово: " + ", ".join(parts) + "."
    if saved.get("task_status") == "not_found":
        return "Не нашёл активную задачу с таким названием — ничего не закрыл. Уточни название."
    if saved.get("memory_status") == "title_exists":
        return "Заметку не записал — заголовок уже занят другой заметкой."
    return "Сохранять было нечего."


def _format_search_results(results: list) -> str:
    """Читаемый агентом текст из результатов db.search_memory (для tool-результата)."""
    if not results:
        return "По запросу в памяти ничего не нашёл."
    lines = []
    for r in results:
        meta = []
        if r.get("folder"):
            meta.append("папка: " + r["folder"])
        head = r.get("title") or ""
        if meta:
            head += " (" + ", ".join(meta) + ")"
        block = "• " + head
        body = (r.get("content") or "").strip()
        if body:
            block += "\n  " + body
        links = r.get("links") or []
        if links:
            block += "\n  связи: " + ", ".join("[[" + t + "]]" for t in links)
        lines.append(block)
    return "Нашёл в памяти:\n" + "\n".join(lines)


def _apply_suggestion(task, memory) -> dict:
    """Применяет структурированное предложение (от типизированных инструментов агента
    или кнопки подтверждения) к БД. При коллизии title у memory.create НЕ теряем
    данные: дописываем в существующую заметку. Задачи close/update резолвятся по
    title→id, если id не передан. saved["memory"]/["task"] == True ⟺ данные реально
    сохранены; *_status поясняет как именно. Резолв id идёт ДО записи; каждая
    db-операция атомарна сама по себе (своя транзакция)."""
    saved = {"task": False, "task_status": None, "memory": False, "memory_status": None}
    if isinstance(task, dict):
        a = task.get("action")
        if a == "create" and (task.get("title") or "").strip():
            db.create_task(title=task["title"].strip(), context=task.get("context") or "")
            saved["task"] = True
            saved["task_status"] = "created"
        elif a == "update":
            tid = task.get("id") or _resolve_task_id(task.get("title"))
            fields = {}
            if task.get("context") is not None:
                fields["context"] = task["context"]
            if (task.get("title") or "").strip():
                fields["title"] = task["title"].strip()
            if tid and fields:
                db.update_task(tid, **fields)
                saved["task"] = True
                saved["task_status"] = "updated"
            elif not tid:
                saved["task_status"] = "not_found"
        elif a == "close":
            tid = task.get("id") or _resolve_task_id(task.get("title"))
            if tid:
                db.update_task(tid, status="completed", outcome=task.get("outcome") or "")
                saved["task"] = True
                saved["task_status"] = "closed"
            else:
                saved["task_status"] = "not_found"
    if isinstance(memory, dict):
        a = memory.get("action")
        if a == "create" and (memory.get("title") or "").strip():
            title = memory["title"].strip()
            content = memory.get("content") or ""
            row = db.create_memory(
                title=title,
                content=content,
                folder=_norm_folder(memory.get("folder")),
            )
            if row is not None:
                saved["memory"] = True
                saved["memory_status"] = "created"
            else:
                # title занят: дописываем в существующую, а не роняем с ложным «✓».
                merged, how = db.merge_memory_content(title, content)
                if merged is not None:
                    saved["memory"] = True
                    saved["memory_status"] = how        # "merged" | "duplicate"
                else:
                    saved["memory_status"] = "title_exists"
        elif a == "update" and memory.get("id"):
            fields = {}
            if memory.get("content") is not None:
                fields["content"] = memory["content"]
            if (memory.get("title") or "").strip():
                fields["title"] = memory["title"].strip()
            if "folder" in memory:
                fields["folder"] = _norm_folder(memory.get("folder"))
            if fields:
                db.update_memory(memory["id"], **fields)
            saved["memory"] = True
            saved["memory_status"] = "updated"
    return saved


@app.post("/api/suggestion/apply")
def api_apply_suggestion(body: dict):
    """Применить предложение(я) памяти (кнопка «Подтвердить» на фронте). Принимает
    список `items` (карточка-список) ИЛИ одиночные task/memory (back-compat).
    Возвращает saved-список по каждому элементу."""
    items = body.get("items")
    if not isinstance(items, list):
        items = [{"task": body.get("task"), "memory": body.get("memory")}]
    saved = [
        _apply_suggestion(it.get("task"), it.get("memory"))
        for it in items if isinstance(it, dict)
    ]
    return {"ok": True, "saved": saved}


@app.get("/api/memory/{memory_id}")
def api_get_memory(memory_id: int):
    row = db.get_memory(memory_id)
    if not row:
        return JSONResponse(status_code=404, content={"error": "not_found"})
    return row


@app.patch("/api/memory/{memory_id}")
def api_update_memory(memory_id: int, body: dict):
    clean = {}
    if "title" in body:
        title = (body["title"] or "").strip()
        if not title:
            return JSONResponse(status_code=400, content={"error": "title_empty"})
        clean["title"] = title
    if "content" in body:
        clean["content"] = body["content"] or ""
    if "folder" in body:
        clean["folder"] = _norm_folder(body["folder"])
    if "status" in body:
        if body["status"] not in MEMORY_STATUSES:
            return JSONResponse(status_code=400, content={"error": "bad_status"})
        clean["status"] = body["status"]
    row = db.update_memory(memory_id, **clean)
    if not row:
        return JSONResponse(status_code=404, content={"error": "not_found"})
    return row


@app.delete("/api/memory/{memory_id}")
def api_delete_memory(memory_id: int):
    db.delete_memory(memory_id)
    return {"ok": True}


# ── Задачи API ──

TASK_STATUSES = {"active", "completed", "archived"}


@app.get("/api/tasks")
def api_list_tasks(status: str | None = None):
    return db.list_tasks(status=status)


@app.post("/api/tasks")
def api_create_task(body: dict):
    title = (body.get("title") or "").strip()
    if not title:
        return JSONResponse(status_code=400, content={"error": "title_required"})
    context = body.get("context") or ""
    return db.create_task(title=title, context=context)


@app.get("/api/tasks/{task_id}")
def api_get_task(task_id: int):
    row = db.get_task(task_id)
    if not row:
        return JSONResponse(status_code=404, content={"error": "not_found"})
    return row


@app.patch("/api/tasks/{task_id}")
def api_update_task(task_id: int, body: dict):
    clean = {}
    if "title" in body:
        title = (body["title"] or "").strip()
        if not title:
            return JSONResponse(status_code=400, content={"error": "title_empty"})
        clean["title"] = title
    if "context" in body:
        clean["context"] = body["context"] or ""
    if "outcome" in body:
        clean["outcome"] = body["outcome"] or ""
    if "status" in body:
        if body["status"] not in TASK_STATUSES:
            return JSONResponse(status_code=400, content={"error": "bad_status"})
        clean["status"] = body["status"]
    row = db.update_task(task_id, **clean)
    if not row:
        return JSONResponse(status_code=404, content={"error": "not_found"})
    return row


@app.delete("/api/tasks/{task_id}")
def api_delete_task(task_id: int):
    db.delete_task(task_id)
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

    Контекст собирается по активной ветке (leaf = user_message_id). Ответ
    ассистента сохраняется с parent_message_id = user_message_id напрямую.

    Память: агент сам структурирует и зовёт типизированные инструменты
    save_memory/save_task/close_task (только по явной просьбе пользователя),
    заполняя поля сам. memory_autosave включён —
    пишем сразу (event suggestion_applied), иначе показываем карточку
    подтверждения (event memory_suggestion). Память также инжектится в
    системпромпт для чтения.
    """
    system_prompt, messages = summarizer.prepare_context(
        client, model, chat_id,
        cos_agent.system_prompt,
        user_message_id,
        user_message=user_message_text,
        inject_memory=True,
    )

    def generate():
        applied = []         # автосохранённые (тост)
        pending_items = []   # СПИСОК отложенных предложений за ход (карточка-список)

        def tool_handler(name, tool_input):
            if name == "search_memory":
                return _format_search_results(db.search_memory((tool_input or {}).get("query") or ""))
            sug = build_suggestion(name, tool_input)
            if sug is None:
                return ("Не передал обязательные поля (как минимум title) или неизвестный "
                        "инструмент — ничего не сохранил.")
            if db.get_setting("memory_autosave", "false") == "true":
                saved = _apply_suggestion(sug.get("task"), sug.get("memory"))
                applied.append(saved)
                return _tool_result_msg(saved)
            # autosave off: копим КАЖДЫЙ вызов отдельным пунктом одной карточки
            # подтверждения (раньше одиночные слоты схлопывали несколько записей).
            pending_items.append(sug)
            return "Подготовил запись и показал пользователю карточку подтверждения."

        full_text = ""
        try:
            for chunk in cos_agent.respond_stream(
                messages, model=model, system_prompt=system_prompt,
                tools=MEMORY_TOOLS, tool_handler=tool_handler,
            ):
                full_text += chunk
                yield f"data: {json.dumps({'type': 'text', 'chunk': chunk})}\n\n"

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

            for saved in applied:
                yield f"data: {json.dumps({'type': 'suggestion_applied', 'saved': saved})}\n\n"
            if pending_items:
                yield f"data: {json.dumps({'type': 'memory_suggestion', 'items': pending_items})}\n\n"

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


@app.get("/memory")
def memory_page():
    return FileResponse(BASE_DIR / "memory.html")


# Отложенные голосовые сохранения при autosave=OFF: в голосе нет кнопки, поэтому
# подтверждение даётся устным «да» на СЛЕДУЮЩЕЙ реплике. Храним предложенную запись
# на бэке (ключ — chat_id) до подтверждения. Эфемерно (живёт в процессе) — для
# короткой голосовой сессии этого достаточно.
_VOICE_PENDING: dict[str, dict] = {}

_AFFIRMATIVE_RE = re.compile(
    r"\b(да|ага|угу|угум|давай|давайте|сохрани|сохраните|запиши|запишите|верно|"
    r"точно|именно|конечно|ок|окей|yes|сохраняй|подтверждаю|ладно|можно|"
    r"согласен|согласна)\b",
    re.IGNORECASE,
)
# Отрицание перекрывает согласие: при «не/нет/отмена/передумал» — НЕ подтверждение.
# Цена ложного «да» — молчаливая запись против воли; цена ложного «нет» — переспрос.
_NEGATION_RE = re.compile(
    r"\b(не|нет|нету|неа|отмена|отмени|против|передумал|передумала)\b",
    re.IGNORECASE,
)


def _is_affirmative(text: str) -> bool:
    """Грубый детектор устного согласия для голосового подтверждения. Отрицание имеет
    приоритет: «да не надо», «конечно нет», «не сохраняй» → НЕ подтверждение."""
    t = text or ""
    if _NEGATION_RE.search(t):
        return False
    return bool(_AFFIRMATIVE_RE.search(t))


def _resolve_voice_pending(chat_id, text) -> list:
    """Если для chat_id есть отложенные записи (список): применить ВСЕ при устном
    «да», иначе снять (не подтвердил/сменил тему). Возвращает список saved-словарей
    (пустой, если нечего применять или не подтвердили)."""
    if not chat_id or chat_id not in _VOICE_PENDING:
        return []
    pend = _VOICE_PENDING.pop(chat_id)
    if not _is_affirmative(text):
        return []
    return [_apply_suggestion(s.get("task"), s.get("memory")) for s in pend]


@app.post("/voice/chat/stream")
async def voice_chat_stream(req: VoiceChatRequest):
    from agent import SYSTEM_PROMPT
    # Голос пишет в память теми же типизированными инструментами, что и текст.
    # Поведение зависит от memory_autosave: ВКЛ — пишем сразу и говорим «сохранил»;
    # ВЫКЛ — сначала переспрашиваем голосом и сохраняем только после устного «да».
    voice_autosave = db.get_setting("memory_autosave", "false") == "true"
    voice_base = SYSTEM_PROMPT + VOICE_SYSTEM_ADDENDUM + (
        VOICE_MEMORY_AUTOSAVE_ON if voice_autosave else VOICE_MEMORY_AUTOSAVE_OFF
    )
    # Устное «да» применяет запись, отложенную на прошлой реплике (только при OFF).
    # Если тумблер успели включить — снимаем устаревшее отложенное (иначе оно зависло
    # бы и применилось позже на несвязанном «да»).
    voice_pending_saved = []
    if voice_autosave:
        _VOICE_PENDING.pop(req.chat_id, None)
    else:
        voice_pending_saved = _resolve_voice_pending(req.chat_id, req.text)

    voice_user_id = None
    if req.chat_id:
        voice_user_id = db.add_message(req.chat_id, "user", req.text)

        voice_system, messages = summarizer.prepare_context(
            client, VOICE_MODEL, req.chat_id,
            voice_base,
            voice_user_id,
            user_message=req.text,
            inject_memory=True,
        )
    else:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        messages.append({"role": "user", "content": req.text})
        voice_system = voice_base

    async def generate():
        # Сообщаем фронту о записях, применённых устным «да» в начале этой реплики.
        for _saved in voice_pending_saved:
            yield f"data: {json.dumps({'type': 'memory_saved', 'saved': _saved})}\n\n"
        text_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def voice_tool_handler(name, tool_input):
            if name == "search_memory":
                return _format_search_results(db.search_memory((tool_input or {}).get("query") or ""))
            sug = build_suggestion(name, tool_input)
            if sug is None:
                return ("Не передал обязательные поля (как минимум title) или неизвестный "
                        "инструмент — ничего не сохранил.")
            if not voice_autosave:
                # OFF: не пишем сразу — копим в pending (список), применим ВСЕ по
                # устному «да» на следующей реплике (тумблер реально работает).
                if req.chat_id:
                    _VOICE_PENDING.setdefault(req.chat_id, []).append(sug)
                    return ("Подготовил запись, но пока НЕ сохранил. Переспроси у пользователя "
                            "вслух «Сохранить это в память?» — применю после устного «да».")
                return ("Без активного чата не могу отложить подтверждение голосом — "
                        "предложи сохранить в текстовом чате.")
            saved = _apply_suggestion(sug.get("task"), sug.get("memory"))
            loop.call_soon_threadsafe(text_queue.put_nowait, ("memory_saved", saved))
            return _tool_result_msg(saved)

        def run_claude():
            try:
                for chunk in cos_agent.respond_stream(
                    messages,
                    model=VOICE_MODEL,
                    max_tokens=VOICE_MAX_TOKENS,
                    system_prompt=voice_system,
                    tools=MEMORY_TOOLS,
                    tool_handler=voice_tool_handler,
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

            if msg_type == "memory_saved":
                yield f"data: {json.dumps({'type': 'memory_saved', 'saved': data})}\n\n"
                continue

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
