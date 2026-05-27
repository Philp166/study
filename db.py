"""
Слой хранения данных.

DATABASE_URL задана → PostgreSQL (Render, прод).
Не задана             → SQLite   (локальная разработка).
"""

import os
import time
import uuid
from pathlib import Path
from contextlib import contextmanager

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    import psycopg2
    import psycopg2.extras
    _PG = True
else:
    import sqlite3
    _PG = False

DB_PATH = Path(__file__).parent / "cos.db"

# Плейсхолдер параметра: %s для PostgreSQL, ? для SQLite
_P = "%s" if _PG else "?"


@contextmanager
def _connect():
    if _PG:
        conn = psycopg2.connect(DATABASE_URL)
        try:
            yield conn
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()


def _fetchone(cur) -> dict | None:
    if _PG:
        row = cur.fetchone()
        return dict(row) if row else None
    else:
        row = cur.fetchone()
        return dict(row) if row else None


def _fetchall(cur) -> list[dict]:
    return [dict(r) for r in cur.fetchall()]


def _execute(conn, sql, params=()):
    if _PG:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params)
        return cur
    else:
        return conn.execute(sql, params)


# ── Schema ──

_SCHEMA_PG = """
CREATE TABLE IF NOT EXISTS chats (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT '',
    pinned     INTEGER NOT NULL DEFAULT 0,
    model      TEXT NOT NULL DEFAULT 'claude-sonnet-4-6',
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id      SERIAL PRIMARY KEY,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role    TEXT NOT NULL,
    content TEXT NOT NULL,
    ts      DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);

CREATE TABLE IF NOT EXISTS token_usage (
    id                    SERIAL PRIMARY KEY,
    chat_id               TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    model                 TEXT NOT NULL,
    input_tokens          INTEGER NOT NULL DEFAULT 0,
    output_tokens         INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens     INTEGER NOT NULL DEFAULT 0,
    ts                    DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_token_usage_chat ON token_usage(chat_id);

CREATE TABLE IF NOT EXISTS summaries (
    chat_id                  TEXT PRIMARY KEY REFERENCES chats(id) ON DELETE CASCADE,
    summary_text             TEXT NOT NULL,
    created_at               DOUBLE PRECISION NOT NULL,
    covers_messages_up_to_id INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS usage_archive (
    model                 TEXT NOT NULL,
    request_count         INTEGER NOT NULL DEFAULT 0,
    input_tokens          INTEGER NOT NULL DEFAULT 0,
    output_tokens         INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens     INTEGER NOT NULL DEFAULT 0
);
"""

_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS chats (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT '',
    pinned     INTEGER NOT NULL DEFAULT 0,
    model      TEXT NOT NULL DEFAULT 'claude-sonnet-4-6',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role    TEXT NOT NULL,
    content TEXT NOT NULL,
    ts      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);

CREATE TABLE IF NOT EXISTS token_usage (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id               TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    model                 TEXT NOT NULL,
    input_tokens          INTEGER NOT NULL DEFAULT 0,
    output_tokens         INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens     INTEGER NOT NULL DEFAULT 0,
    ts                    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_token_usage_chat ON token_usage(chat_id);

CREATE TABLE IF NOT EXISTS summaries (
    chat_id                  TEXT PRIMARY KEY REFERENCES chats(id) ON DELETE CASCADE,
    summary_text             TEXT NOT NULL,
    created_at               REAL NOT NULL,
    covers_messages_up_to_id INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS usage_archive (
    model                 TEXT NOT NULL,
    request_count         INTEGER NOT NULL DEFAULT 0,
    input_tokens          INTEGER NOT NULL DEFAULT 0,
    output_tokens         INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens     INTEGER NOT NULL DEFAULT 0
);
"""


def init_db():
    with _connect() as conn:
        if _PG:
            cur = conn.cursor()
            cur.execute(_SCHEMA_PG)
            conn.commit()
        else:
            conn.executescript(_SCHEMA_SQLITE)
    engine = "PostgreSQL" if _PG else "SQLite"
    print(f"DB: {engine} ready")


def new_id():
    return uuid.uuid4().hex[:12]


# ── Chats CRUD ──

def list_chats():
    with _connect() as conn:
        cur = _execute(
            conn,
            "SELECT id, title, pinned, model, created_at, updated_at FROM chats "
            "ORDER BY pinned DESC, updated_at DESC",
        )
        return _fetchall(cur)


def get_chat(chat_id: str):
    with _connect() as conn:
        cur = _execute(conn, f"SELECT * FROM chats WHERE id={_P}", (chat_id,))
        chat = _fetchone(cur)
        if not chat:
            return None
        cur = _execute(
            conn,
            f"SELECT role, content FROM messages WHERE chat_id={_P} ORDER BY id",
            (chat_id,),
        )
        chat["messages"] = _fetchall(cur)
        return chat


def create_chat(*, chat_id=None, title="", pinned=False, model="claude-sonnet-4-6",
                created_at=None, updated_at=None):
    cid = chat_id or new_id()
    now = time.time()
    ca = created_at or now
    ua = updated_at or now
    with _connect() as conn:
        _execute(
            conn,
            f"INSERT INTO chats (id, title, pinned, model, created_at, updated_at) "
            f"VALUES ({_P},{_P},{_P},{_P},{_P},{_P})",
            (cid, title, int(pinned), model, ca, ua),
        )
        conn.commit()
    return cid


def update_chat(chat_id: str, **fields):
    allowed = {"title", "pinned", "model", "updated_at"}
    sets = []
    vals = []
    for k, v in fields.items():
        if k not in allowed:
            continue
        if k == "pinned":
            v = int(v)
        sets.append(f"{k}={_P}")
        vals.append(v)
    if not sets:
        return
    vals.append(chat_id)
    with _connect() as conn:
        _execute(conn, f"UPDATE chats SET {', '.join(sets)} WHERE id={_P}", vals)
        conn.commit()


def delete_chat(chat_id: str):
    with _connect() as conn:
        _archive_usage(conn, chat_id)
        _execute(conn, f"DELETE FROM chats WHERE id={_P}", (chat_id,))
        conn.commit()


def _archive_usage(conn, chat_id: str):
    """Переносит расход удаляемого чата в usage_archive (по моделям)."""
    cur = _execute(
        conn,
        "SELECT model, "
        "  COUNT(*) AS cnt, "
        "  COALESCE(SUM(input_tokens), 0)  AS inp, "
        "  COALESCE(SUM(output_tokens), 0) AS out, "
        "  COALESCE(SUM(cache_creation_tokens), 0) AS cw, "
        "  COALESCE(SUM(cache_read_tokens), 0)     AS cr "
        f"FROM token_usage WHERE chat_id={_P} GROUP BY model",
        (chat_id,),
    )
    rows = _fetchall(cur)
    for r in rows:
        if _PG:
            sql = (
                f"INSERT INTO usage_archive (model, request_count, input_tokens, output_tokens, "
                f"cache_creation_tokens, cache_read_tokens) VALUES ({_P},{_P},{_P},{_P},{_P},{_P}) "
                f"ON CONFLICT DO NOTHING"
            )
            _execute(conn, sql, (r["model"], r["cnt"], r["inp"], r["out"], r["cw"], r["cr"]))
            _execute(
                conn,
                f"UPDATE usage_archive SET "
                f"request_count=request_count+{_P}, input_tokens=input_tokens+{_P}, "
                f"output_tokens=output_tokens+{_P}, cache_creation_tokens=cache_creation_tokens+{_P}, "
                f"cache_read_tokens=cache_read_tokens+{_P} WHERE model={_P}",
                (r["cnt"], r["inp"], r["out"], r["cw"], r["cr"], r["model"]),
            )
        else:
            existing = _execute(
                conn, f"SELECT rowid FROM usage_archive WHERE model={_P}", (r["model"],)
            )
            if _fetchone(existing):
                _execute(
                    conn,
                    f"UPDATE usage_archive SET "
                    f"request_count=request_count+{_P}, input_tokens=input_tokens+{_P}, "
                    f"output_tokens=output_tokens+{_P}, cache_creation_tokens=cache_creation_tokens+{_P}, "
                    f"cache_read_tokens=cache_read_tokens+{_P} WHERE model={_P}",
                    (r["cnt"], r["inp"], r["out"], r["cw"], r["cr"], r["model"]),
                )
            else:
                _execute(
                    conn,
                    f"INSERT INTO usage_archive (model, request_count, input_tokens, output_tokens, "
                    f"cache_creation_tokens, cache_read_tokens) VALUES ({_P},{_P},{_P},{_P},{_P},{_P})",
                    (r["model"], r["cnt"], r["inp"], r["out"], r["cw"], r["cr"]),
                )


# ── Messages ──

def add_message(chat_id: str, role: str, content: str):
    now = time.time()
    with _connect() as conn:
        _execute(
            conn,
            f"INSERT INTO messages (chat_id, role, content, ts) VALUES ({_P},{_P},{_P},{_P})",
            (chat_id, role, content, now),
        )
        _execute(conn, f"UPDATE chats SET updated_at={_P} WHERE id={_P}", (now, chat_id))
        conn.commit()


def get_messages(chat_id: str):
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT role, content FROM messages WHERE chat_id={_P} ORDER BY id",
            (chat_id,),
        )
        return _fetchall(cur)


def get_messages_with_ids(chat_id: str) -> list[dict]:
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT id, role, content FROM messages WHERE chat_id={_P} ORDER BY id",
            (chat_id,),
        )
        return _fetchall(cur)


# ── Summaries ──

def get_summary(chat_id: str) -> dict | None:
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT summary_text, covers_messages_up_to_id FROM summaries WHERE chat_id={_P}",
            (chat_id,),
        )
        return _fetchone(cur)


def save_summary(chat_id: str, summary_text: str, covers_up_to_id: int):
    now = time.time()
    if _PG:
        sql = (
            f"INSERT INTO summaries (chat_id, summary_text, created_at, covers_messages_up_to_id) "
            f"VALUES ({_P},{_P},{_P},{_P}) "
            f"ON CONFLICT (chat_id) DO UPDATE SET "
            f"summary_text=EXCLUDED.summary_text, created_at=EXCLUDED.created_at, "
            f"covers_messages_up_to_id=EXCLUDED.covers_messages_up_to_id"
        )
    else:
        sql = (
            f"INSERT OR REPLACE INTO summaries "
            f"(chat_id, summary_text, created_at, covers_messages_up_to_id) "
            f"VALUES ({_P},{_P},{_P},{_P})"
        )
    with _connect() as conn:
        _execute(conn, sql, (chat_id, summary_text, now, covers_up_to_id))
        conn.commit()


# ── Token usage ──

def save_usage(chat_id: str, model: str, input_tokens: int, output_tokens: int,
               cache_creation: int = 0, cache_read: int = 0):
    now = time.time()
    with _connect() as conn:
        _execute(
            conn,
            f"INSERT INTO token_usage "
            f"(chat_id, model, input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens, ts) "
            f"VALUES ({_P},{_P},{_P},{_P},{_P},{_P},{_P})",
            (chat_id, model, input_tokens, output_tokens, cache_creation, cache_read, now),
        )
        conn.commit()


def get_chat_usage(chat_id: str) -> dict:
    with _connect() as conn:
        cur = _execute(
            conn,
            "SELECT "
            "  COUNT(*) AS request_count, "
            "  COALESCE(SUM(input_tokens), 0)  AS total_input, "
            "  COALESCE(SUM(output_tokens), 0) AS total_output, "
            "  COALESCE(SUM(cache_creation_tokens), 0) AS total_cache_creation, "
            "  COALESCE(SUM(cache_read_tokens), 0)     AS total_cache_read "
            f"FROM token_usage WHERE chat_id={_P}",
            (chat_id,),
        )
        return _fetchone(cur) or {
            "request_count": 0, "total_input": 0, "total_output": 0,
            "total_cache_creation": 0, "total_cache_read": 0,
        }


def get_global_usage() -> dict:
    with _connect() as conn:
        cur = _execute(
            conn,
            "SELECT "
            "  COALESCE(SUM(request_count), 0) AS request_count, "
            "  COALESCE(SUM(input_tokens), 0)  AS total_input, "
            "  COALESCE(SUM(output_tokens), 0) AS total_output, "
            "  COALESCE(SUM(cache_creation_tokens), 0) AS total_cache_creation, "
            "  COALESCE(SUM(cache_read_tokens), 0)     AS total_cache_read "
            "FROM ("
            "  SELECT COUNT(*) AS request_count, "
            "    SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens, "
            "    SUM(cache_creation_tokens) AS cache_creation_tokens, "
            "    SUM(cache_read_tokens) AS cache_read_tokens "
            "  FROM token_usage "
            "  UNION ALL "
            "  SELECT SUM(request_count), SUM(input_tokens), SUM(output_tokens), "
            "    SUM(cache_creation_tokens), SUM(cache_read_tokens) "
            "  FROM usage_archive"
            ") t",
        )
        return _fetchone(cur) or {
            "request_count": 0, "total_input": 0, "total_output": 0,
            "total_cache_creation": 0, "total_cache_read": 0,
        }


# ── Migration (localStorage → server) ──

def import_chat(chat_id: str, title: str, pinned: bool, model: str,
                created_at: float, updated_at: float, messages: list[dict]):
    insert_sql = (
        f"INSERT INTO chats (id, title, pinned, model, created_at, updated_at) "
        f"VALUES ({_P},{_P},{_P},{_P},{_P},{_P}) "
    )
    if _PG:
        insert_sql += "ON CONFLICT (id) DO NOTHING"
    else:
        insert_sql = insert_sql.replace("INSERT INTO", "INSERT OR IGNORE INTO")

    with _connect() as conn:
        _execute(conn, insert_sql,
                 (chat_id, title, int(pinned), model, created_at, updated_at))
        for m in messages:
            _execute(
                conn,
                f"INSERT INTO messages (chat_id, role, content, ts) VALUES ({_P},{_P},{_P},{_P})",
                (chat_id, m["role"], m["content"], updated_at),
            )
        conn.commit()


# ── Settings (key-value) ──

_SETTING_DEFAULTS = {
    "tts_voice": "ru-RU-SvetlanaNeural",
    "compression_method": "rolling_summary",
    "voice_enabled": "true",
}


def get_setting(key: str, default: str | None = None) -> str | None:
    fallback = default if default is not None else _SETTING_DEFAULTS.get(key)
    with _connect() as conn:
        cur = _execute(conn, f"SELECT value FROM settings WHERE key={_P}", (key,))
        row = _fetchone(cur)
        return row["value"] if row else fallback


def get_all_settings() -> dict:
    with _connect() as conn:
        cur = _execute(conn, "SELECT key, value FROM settings")
        rows = _fetchall(cur)
    result = dict(_SETTING_DEFAULTS)
    for r in rows:
        result[r["key"]] = r["value"]
    return result


def set_setting(key: str, value: str):
    if _PG:
        sql = (
            f"INSERT INTO settings (key, value) VALUES ({_P},{_P}) "
            f"ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value"
        )
    else:
        sql = f"INSERT OR REPLACE INTO settings (key, value) VALUES ({_P},{_P})"
    with _connect() as conn:
        _execute(conn, sql, (key, value))
        conn.commit()


# ── Token usage by model ──

def get_usage_by_model() -> list[dict]:
    with _connect() as conn:
        cur = _execute(
            conn,
            "SELECT model, "
            "  SUM(request_count) AS request_count, "
            "  SUM(input_tokens)  AS total_input, "
            "  SUM(output_tokens) AS total_output, "
            "  SUM(cache_creation_tokens) AS total_cache_creation, "
            "  SUM(cache_read_tokens)     AS total_cache_read "
            "FROM ("
            "  SELECT model, COUNT(*) AS request_count, "
            "    SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens, "
            "    SUM(cache_creation_tokens) AS cache_creation_tokens, "
            "    SUM(cache_read_tokens) AS cache_read_tokens "
            "  FROM token_usage GROUP BY model "
            "  UNION ALL "
            "  SELECT model, request_count, input_tokens, output_tokens, "
            "    cache_creation_tokens, cache_read_tokens "
            "  FROM usage_archive"
            ") t GROUP BY model ORDER BY total_input DESC",
        )
        return _fetchall(cur)
