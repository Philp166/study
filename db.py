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
        _execute(conn, f"DELETE FROM chats WHERE id={_P}", (chat_id,))
        conn.commit()


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
            "  COUNT(*) AS request_count, "
            "  COALESCE(SUM(input_tokens), 0)  AS total_input, "
            "  COALESCE(SUM(output_tokens), 0) AS total_output, "
            "  COALESCE(SUM(cache_creation_tokens), 0) AS total_cache_creation, "
            "  COALESCE(SUM(cache_read_tokens), 0)     AS total_cache_read "
            "FROM token_usage",
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
