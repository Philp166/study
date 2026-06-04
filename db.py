"""
Слой хранения данных.

DATABASE_URL задана → PostgreSQL (Render, прод).
Не задана             → SQLite   (локальная разработка).

История чата — дерево: каждое сообщение знает родителя (parent_message_id).
Активная ветка чата — путь от chats.current_leaf_message_id вверх до корня.
Линейный чат — вырожденное дерево, где у каждого сообщения один ребёнок.
"""

import json
import os
import time
import uuid
from pathlib import Path
from contextlib import contextmanager

import memory_schema

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    import psycopg2
    import psycopg2.extras
    _PG = True
else:
    import sqlite3
    _PG = False

DB_PATH = Path(os.getenv("COS_DB_PATH") or (Path(__file__).parent / "cos.db"))

# Плейсхолдер параметра: %s для PostgreSQL, ? для SQLite
_P = "%s" if _PG else "?"

# Глубина рекурсивного подъёма по дереву (защита от циклов/переполнения).
MAX_BRANCH_DEPTH = 1000

# Значение-надгробие: факт «удалён» в этой ветке, не затрагивая предков/сиблингов.
FACT_TOMBSTONE = "\x00__deleted__"


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


def _insert_message(conn, chat_id: str, role: str, content: str, ts: float,
                    parent_message_id: int | None) -> int:
    """Вставляет сообщение, возвращает его id (кросс-СУБД)."""
    sql = (
        f"INSERT INTO messages (chat_id, role, content, ts, parent_message_id) "
        f"VALUES ({_P},{_P},{_P},{_P},{_P})"
    )
    if _PG:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql + " RETURNING id", (chat_id, role, content, ts, parent_message_id))
        return cur.fetchone()["id"]
    else:
        cur = conn.execute(sql, (chat_id, role, content, ts, parent_message_id))
        return cur.lastrowid


# ── Schema (для НОВЫХ БД создаются старые таблицы; _migrate() приводит к финалу) ──

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

CREATE TABLE IF NOT EXISTS chat_strategy_settings (
    chat_id                       TEXT PRIMARY KEY REFERENCES chats(id) ON DELETE CASCADE,
    rolling_summary_enabled       INTEGER NOT NULL DEFAULT 1,
    rolling_summary_threshold_pct INTEGER NOT NULL DEFAULT 35,
    sliding_window_enabled        INTEGER NOT NULL DEFAULT 0,
    sliding_window_size           INTEGER NOT NULL DEFAULT 20,
    sticky_facts_enabled          INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chat_facts (
    id         SERIAL PRIMARY KEY,
    chat_id    TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL,
    UNIQUE(chat_id, key)
);

-- ── Долговременная память (граф): глобальная, без chat_id как ключа изоляции.
--    source_* — только провенанс (из какого чата/сообщения пришёл узел/ребро). ──
CREATE TABLE IF NOT EXISTS mem_nodes (
    id                SERIAL PRIMARY KEY,
    type              TEXT NOT NULL,
    name              TEXT NOT NULL,
    attributes        TEXT NOT NULL DEFAULT '{}',
    origin            TEXT NOT NULL DEFAULT 'manual',
    basis             TEXT NOT NULL DEFAULT '',
    confidence        DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    status            TEXT NOT NULL DEFAULT 'active',
    source_chat_id    TEXT,
    source_message_id INTEGER,
    embedding         BYTEA,
    created_at        DOUBLE PRECISION NOT NULL,
    updated_at        DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_nodes_type ON mem_nodes(type, status);
CREATE INDEX IF NOT EXISTS idx_mem_nodes_name ON mem_nodes(name);

CREATE TABLE IF NOT EXISTS mem_edges (
    id                SERIAL PRIMARY KEY,
    src_id            INTEGER NOT NULL REFERENCES mem_nodes(id) ON DELETE CASCADE,
    dst_id            INTEGER NOT NULL REFERENCES mem_nodes(id) ON DELETE CASCADE,
    type              TEXT NOT NULL,
    origin            TEXT NOT NULL DEFAULT 'manual',
    basis             TEXT NOT NULL DEFAULT '',
    confidence        DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    status            TEXT NOT NULL DEFAULT 'active',
    source_chat_id    TEXT,
    source_message_id INTEGER,
    created_at        DOUBLE PRECISION NOT NULL,
    updated_at        DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_edges_src ON mem_edges(src_id, status);
CREATE INDEX IF NOT EXISTS idx_mem_edges_dst ON mem_edges(dst_id, status);

-- ── Слой знания (RAG): свободный текст, не лёгший в граф. embedding — BYTEA float32. ──
CREATE TABLE IF NOT EXISTS mem_chunks (
    id                SERIAL PRIMARY KEY,
    text              TEXT NOT NULL,
    embedding         BYTEA,
    origin            TEXT NOT NULL DEFAULT 'manual',
    status            TEXT NOT NULL DEFAULT 'active',
    node_id           INTEGER REFERENCES mem_nodes(id) ON DELETE SET NULL,
    source_chat_id    TEXT,
    source_message_id INTEGER,
    created_at        DOUBLE PRECISION NOT NULL,
    updated_at        DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_chunks_status ON mem_chunks(status);
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

CREATE TABLE IF NOT EXISTS chat_strategy_settings (
    chat_id                       TEXT PRIMARY KEY REFERENCES chats(id) ON DELETE CASCADE,
    rolling_summary_enabled       INTEGER NOT NULL DEFAULT 1,
    rolling_summary_threshold_pct INTEGER NOT NULL DEFAULT 35,
    sliding_window_enabled        INTEGER NOT NULL DEFAULT 0,
    sliding_window_size           INTEGER NOT NULL DEFAULT 20,
    sticky_facts_enabled          INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chat_facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(chat_id, key)
);

-- ── Долговременная память (граф): глобальная, без chat_id как ключа изоляции.
--    source_* — только провенанс (из какого чата/сообщения пришёл узел/ребро). ──
CREATE TABLE IF NOT EXISTS mem_nodes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    type              TEXT NOT NULL,
    name              TEXT NOT NULL,
    attributes        TEXT NOT NULL DEFAULT '{}',
    origin            TEXT NOT NULL DEFAULT 'manual',
    basis             TEXT NOT NULL DEFAULT '',
    confidence        REAL NOT NULL DEFAULT 1.0,
    status            TEXT NOT NULL DEFAULT 'active',
    source_chat_id    TEXT,
    source_message_id INTEGER,
    embedding         BLOB,
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_nodes_type ON mem_nodes(type, status);
CREATE INDEX IF NOT EXISTS idx_mem_nodes_name ON mem_nodes(name);

CREATE TABLE IF NOT EXISTS mem_edges (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    src_id            INTEGER NOT NULL REFERENCES mem_nodes(id) ON DELETE CASCADE,
    dst_id            INTEGER NOT NULL REFERENCES mem_nodes(id) ON DELETE CASCADE,
    type              TEXT NOT NULL,
    origin            TEXT NOT NULL DEFAULT 'manual',
    basis             TEXT NOT NULL DEFAULT '',
    confidence        REAL NOT NULL DEFAULT 1.0,
    status            TEXT NOT NULL DEFAULT 'active',
    source_chat_id    TEXT,
    source_message_id INTEGER,
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_edges_src ON mem_edges(src_id, status);
CREATE INDEX IF NOT EXISTS idx_mem_edges_dst ON mem_edges(dst_id, status);

-- ── Слой знания (RAG): свободный текст, не лёгший в граф. embedding — BLOB float32. ──
CREATE TABLE IF NOT EXISTS mem_chunks (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    text              TEXT NOT NULL,
    embedding         BLOB,
    origin            TEXT NOT NULL DEFAULT 'manual',
    status            TEXT NOT NULL DEFAULT 'active',
    node_id           INTEGER REFERENCES mem_nodes(id) ON DELETE SET NULL,
    source_chat_id    TEXT,
    source_message_id INTEGER,
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_chunks_status ON mem_chunks(status);
"""


# ── Migration helpers ──

def _table_exists(conn, table: str) -> bool:
    if _PG:
        cur = _execute(
            conn,
            "SELECT 1 FROM information_schema.tables WHERE table_name=" + _P,
            (table,),
        )
        return _fetchone(cur) is not None
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=" + _P, (table,)
    )
    return cur.fetchone() is not None


def _column_exists(conn, table: str, col: str) -> bool:
    if _PG:
        cur = _execute(
            conn,
            "SELECT 1 FROM information_schema.columns "
            f"WHERE table_name={_P} AND column_name={_P}",
            (table, col),
        )
        return _fetchone(cur) is not None
    else:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return any(r[1] == col for r in cur.fetchall())


def _migrate(conn):
    """Идемпотентно приводит схему к дереву и переносит легаси-данные."""
    int_type = "INTEGER" if not _PG else "INTEGER"

    # 1. messages.parent_message_id
    if not _column_exists(conn, "messages", "parent_message_id"):
        _execute(
            conn,
            f"ALTER TABLE messages ADD COLUMN parent_message_id {int_type} "
            f"REFERENCES messages(id) ON DELETE CASCADE",
        )
    # 2. chats.current_leaf_message_id
    if not _column_exists(conn, "chats", "current_leaf_message_id"):
        _execute(
            conn,
            f"ALTER TABLE chats ADD COLUMN current_leaf_message_id {int_type} "
            f"REFERENCES messages(id) ON DELETE SET NULL",
        )
    conn.commit()

    # 3. indexes
    _execute(conn, "CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id)")
    _execute(conn, "CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(parent_message_id)")
    _execute(conn, "CREATE INDEX IF NOT EXISTS idx_chats_current_leaf ON chats(current_leaf_message_id)")
    conn.commit()

    # 4. backfill parent links: каждый чат, где parent ещё не проставлен
    cur = _execute(conn, "SELECT id FROM chats")
    chat_ids = [r["id"] for r in _fetchall(cur)]
    for cid in chat_ids:
        cur = _execute(
            conn,
            f"SELECT COUNT(*) AS n FROM messages WHERE chat_id={_P}",
            (cid,),
        )
        total = _fetchone(cur)["n"]
        if total <= 1:
            continue
        cur = _execute(
            conn,
            f"SELECT COUNT(*) AS n FROM messages "
            f"WHERE chat_id={_P} AND parent_message_id IS NOT NULL",
            (cid,),
        )
        if _fetchone(cur)["n"] > 0:
            continue  # уже мигрирован
        cur = _execute(
            conn,
            f"SELECT id FROM messages WHERE chat_id={_P} ORDER BY id",
            (cid,),
        )
        ids = [r["id"] for r in _fetchall(cur)]
        prev = None
        for mid in ids:
            if prev is not None:
                _execute(
                    conn,
                    f"UPDATE messages SET parent_message_id={_P} WHERE id={_P}",
                    (prev, mid),
                )
            prev = mid
    conn.commit()

    # 5. backfill current_leaf (только где NULL) = последнее сообщение чата
    _execute(
        conn,
        "UPDATE chats SET current_leaf_message_id = "
        "(SELECT MAX(id) FROM messages WHERE messages.chat_id = chats.id) "
        "WHERE current_leaf_message_id IS NULL",
    )
    conn.commit()

    # 6. summaries rebuild → composite PK (chat_id, branch_anchor_message_id)
    if not _column_exists(conn, "summaries", "branch_anchor_message_id"):
        _rebuild_summaries(conn)
        conn.commit()

    # 7. chat_facts rebuild → +branch_anchor_message_id, UNIQUE(chat_id, anchor, key)
    if not _column_exists(conn, "chat_facts", "branch_anchor_message_id"):
        _rebuild_chat_facts(conn)
        conn.commit()

    # 8. memory: колонка embedding на mem_nodes (если таблица создана до фазы 2)
    blob_type = "BYTEA" if _PG else "BLOB"
    if _table_exists(conn, "mem_nodes") and not _column_exists(conn, "mem_nodes", "embedding"):
        _execute(conn, f"ALTER TABLE mem_nodes ADD COLUMN embedding {blob_type}")
        conn.commit()


def _rebuild_summaries(conn):
    real = "DOUBLE PRECISION" if _PG else "REAL"
    _execute(
        conn,
        f"CREATE TABLE summaries_new ("
        f"  chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,"
        f"  branch_anchor_message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,"
        f"  summary_text TEXT NOT NULL,"
        f"  created_at {real} NOT NULL,"
        f"  PRIMARY KEY (chat_id, branch_anchor_message_id))",
    )
    # anchor = covers_messages_up_to_id (если >0), иначе current_leaf чата;
    # осиротевшие (нет якоря) — отбрасываем.
    _execute(
        conn,
        "INSERT INTO summaries_new (chat_id, branch_anchor_message_id, summary_text, created_at) "
        "SELECT s.chat_id, "
        "       COALESCE(NULLIF(s.covers_messages_up_to_id, 0), c.current_leaf_message_id), "
        "       s.summary_text, s.created_at "
        "FROM summaries s JOIN chats c ON c.id = s.chat_id "
        "WHERE COALESCE(NULLIF(s.covers_messages_up_to_id, 0), c.current_leaf_message_id) IS NOT NULL",
    )
    _execute(conn, "DROP TABLE summaries")
    _execute(conn, "ALTER TABLE summaries_new RENAME TO summaries")


def _rebuild_chat_facts(conn):
    real = "DOUBLE PRECISION" if _PG else "REAL"
    pk = "SERIAL PRIMARY KEY" if _PG else "INTEGER PRIMARY KEY AUTOINCREMENT"
    _execute(
        conn,
        f"CREATE TABLE chat_facts_new ("
        f"  id {pk},"
        f"  chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,"
        f"  key TEXT NOT NULL,"
        f"  value TEXT NOT NULL,"
        f"  created_at {real} NOT NULL,"
        f"  updated_at {real} NOT NULL,"
        f"  branch_anchor_message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,"
        f"  UNIQUE(chat_id, branch_anchor_message_id, key))",
    )
    # anchor = корневое сообщение чата; осиротевшие факты (нет сообщений) — отбрасываем.
    _execute(
        conn,
        "INSERT INTO chat_facts_new (chat_id, key, value, created_at, updated_at, branch_anchor_message_id) "
        "SELECT f.chat_id, f.key, f.value, f.created_at, f.updated_at, "
        "  (SELECT MIN(m.id) FROM messages m "
        "   WHERE m.chat_id = f.chat_id AND m.parent_message_id IS NULL) "
        "FROM chat_facts f "
        "WHERE (SELECT MIN(m.id) FROM messages m "
        "       WHERE m.chat_id = f.chat_id AND m.parent_message_id IS NULL) IS NOT NULL",
    )
    _execute(conn, "DROP TABLE chat_facts")
    _execute(conn, "ALTER TABLE chat_facts_new RENAME TO chat_facts")


def init_db():
    with _connect() as conn:
        if _PG:
            cur = conn.cursor()
            cur.execute(_SCHEMA_PG)
            conn.commit()
        else:
            conn.executescript(_SCHEMA_SQLITE)

        _migrate(conn)

        # Migrate: ensure every existing chat has a strategy_settings row
        if _PG:
            _execute(
                conn,
                "INSERT INTO chat_strategy_settings (chat_id) "
                "SELECT id FROM chats WHERE id NOT IN "
                "(SELECT chat_id FROM chat_strategy_settings) "
                "ON CONFLICT DO NOTHING",
            )
        else:
            _execute(
                conn,
                "INSERT OR IGNORE INTO chat_strategy_settings (chat_id) "
                "SELECT id FROM chats WHERE id NOT IN "
                "(SELECT chat_id FROM chat_strategy_settings)",
            )
        conn.commit()

    engine = "PostgreSQL" if _PG else "SQLite"
    print(f"DB: {engine} ready")


def new_id():
    return uuid.uuid4().hex[:12]


# ── Branch tree core ──

def _get_current_leaf(conn, chat_id: str) -> int | None:
    cur = _execute(
        conn,
        f"SELECT current_leaf_message_id FROM chats WHERE id={_P}",
        (chat_id,),
    )
    row = _fetchone(cur)
    return row["current_leaf_message_id"] if row else None


def get_current_leaf(chat_id: str) -> int | None:
    with _connect() as conn:
        return _get_current_leaf(conn, chat_id)


def set_current_leaf(chat_id: str, leaf_message_id: int | None):
    now = time.time()
    with _connect() as conn:
        _execute(
            conn,
            f"UPDATE chats SET current_leaf_message_id={_P}, updated_at={_P} WHERE id={_P}",
            (leaf_message_id, now, chat_id),
        )
        conn.commit()


def _branch_path_rows(conn, chat_id: str, leaf_message_id: int) -> list[dict]:
    """Путь root→leaf: список {id, role, content} через рекурсивный CTE."""
    sql = (
        f"WITH RECURSIVE branch_path(id, role, content, parent_message_id, depth) AS ("
        f"  SELECT id, role, content, parent_message_id, 0 "
        f"  FROM messages WHERE id={_P} AND chat_id={_P} "
        f"  UNION ALL "
        f"  SELECT m.id, m.role, m.content, m.parent_message_id, bp.depth + 1 "
        f"  FROM messages m JOIN branch_path bp ON m.id = bp.parent_message_id "
        f"  WHERE bp.depth < {MAX_BRANCH_DEPTH} "
        f") SELECT id, role, content, parent_message_id FROM branch_path ORDER BY depth DESC"
    )
    cur = _execute(conn, sql, (leaf_message_id, chat_id))
    return _fetchall(cur)


def get_branch_history(chat_id: str, leaf_message_id: int | None) -> list[dict]:
    """История активной ветки: [{id, role, content}] от корня до leaf."""
    if leaf_message_id is None:
        return []
    with _connect() as conn:
        return _branch_path_rows(conn, chat_id, leaf_message_id)


def _branch_path_ids(conn, chat_id: str, leaf_message_id: int | None) -> set[int]:
    if leaf_message_id is None:
        return set()
    return {r["id"] for r in _branch_path_rows(conn, chat_id, leaf_message_id)}


def get_branch_tree(chat_id: str) -> list[dict]:
    """Все сообщения чата (плоско) с parent — фронт строит дерево/навигаторы."""
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT id, role, content, parent_message_id, ts "
            f"FROM messages WHERE chat_id={_P} ORDER BY id",
            (chat_id,),
        )
        return _fetchall(cur)


def _descend_to_leaf(conn, from_message_id: int) -> int:
    """Внутри открытого соединения: от узла вниз по последнему ребёнку до листа."""
    current = from_message_id
    for _ in range(MAX_BRANCH_DEPTH):
        cur = _execute(
            conn,
            f"SELECT id FROM messages WHERE parent_message_id={_P} "
            f"ORDER BY id DESC LIMIT 1",
            (current,),
        )
        row = _fetchone(cur)
        if not row:
            break
        current = row["id"]
    return current


def descend_to_leaf(chat_id: str, from_message_id: int) -> int:
    """От узла вниз по последнему добавленному ребёнку каждого уровня до листа."""
    with _connect() as conn:
        return _descend_to_leaf(conn, from_message_id)


def message_in_chat(chat_id: str, message_id: int) -> bool:
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT 1 FROM messages WHERE id={_P} AND chat_id={_P}",
            (message_id, chat_id),
        )
        return _fetchone(cur) is not None


def get_message(message_id: int) -> dict | None:
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT id, chat_id, role, content, parent_message_id FROM messages WHERE id={_P}",
            (message_id,),
        )
        return _fetchone(cur)


def create_branch_message(chat_id: str, from_message_id: int, user_message: str) -> int:
    """Создаёт user-сообщение-ребёнка от from_message_id и делает его current_leaf.

    Намеренное ветвление — без проверки expected_parent.
    """
    now = time.time()
    with _connect() as conn:
        mid = _insert_message(conn, chat_id, "user", user_message, now, from_message_id)
        _execute(
            conn,
            f"UPDATE chats SET current_leaf_message_id={_P}, updated_at={_P} WHERE id={_P}",
            (mid, now, chat_id),
        )
        conn.commit()
        return mid


def delete_message_subtree(message_id: int) -> dict:
    """Каскадно удаляет сообщение и потомков. Если в поддереве был current_leaf —
    переключает leaf на родителя удалённого. Возвращает {chat_id, new_leaf}.
    """
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT chat_id, parent_message_id FROM messages WHERE id={_P}",
            (message_id,),
        )
        msg = _fetchone(cur)
        if not msg:
            return {"chat_id": None, "new_leaf": None}
        chat_id = msg["chat_id"]
        parent = msg["parent_message_id"]

        leaf = _get_current_leaf(conn, chat_id)
        leaf_affected = leaf is not None and message_id in _branch_path_ids(conn, chat_id, leaf)

        _execute(conn, f"DELETE FROM messages WHERE id={_P}", (message_id,))

        new_leaf = leaf
        if leaf_affected:
            # Лист уехал из удалённого поддерева. Спускаемся по уцелевшему
            # ребёнку родителя на соседнюю ветку (после удаления своей развилки
            # пользователь должен оказаться на оставшейся ветке целиком, а не
            # «зависнуть» на родителе с обрезанной историей).
            new_leaf = None if parent is None else _descend_to_leaf(conn, parent)
            _execute(
                conn,
                f"UPDATE chats SET current_leaf_message_id={_P}, updated_at={_P} WHERE id={_P}",
                (new_leaf, time.time(), chat_id),
            )
        conn.commit()
        return {"chat_id": chat_id, "new_leaf": new_leaf}


# ── Chats CRUD ──

def list_chats():
    with _connect() as conn:
        cur = _execute(
            conn,
            "SELECT id, title, pinned, model, created_at, updated_at, current_leaf_message_id "
            "FROM chats ORDER BY pinned DESC, updated_at DESC",
        )
        return _fetchall(cur)


def get_chat(chat_id: str):
    """Чат + история АКТИВНОЙ ВЕТКИ (messages) + плоское дерево (tree)."""
    with _connect() as conn:
        cur = _execute(conn, f"SELECT * FROM chats WHERE id={_P}", (chat_id,))
        chat = _fetchone(cur)
        if not chat:
            return None
        leaf = chat.get("current_leaf_message_id")
        chat["messages"] = _branch_path_rows(conn, chat_id, leaf) if leaf else []
        cur = _execute(
            conn,
            f"SELECT id, role, content, parent_message_id, ts "
            f"FROM messages WHERE chat_id={_P} ORDER BY id",
            (chat_id,),
        )
        chat["tree"] = _fetchall(cur)
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
        if _PG:
            _execute(
                conn,
                f"INSERT INTO chat_strategy_settings (chat_id) VALUES ({_P}) "
                f"ON CONFLICT DO NOTHING",
                (cid,),
            )
        else:
            _execute(
                conn,
                f"INSERT OR IGNORE INTO chat_strategy_settings (chat_id) VALUES ({_P})",
                (cid,),
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

def add_message(chat_id: str, role: str, content: str,
                parent_message_id: int | None = None) -> int:
    """Добавляет сообщение в активную ветку. Если parent не задан — чейнит к
    текущему current_leaf. Двигает current_leaf на новое сообщение. Возвращает id.
    """
    now = time.time()
    with _connect() as conn:
        if parent_message_id is None:
            parent_message_id = _get_current_leaf(conn, chat_id)
        mid = _insert_message(conn, chat_id, role, content, now, parent_message_id)
        _execute(
            conn,
            f"UPDATE chats SET updated_at={_P}, current_leaf_message_id={_P} WHERE id={_P}",
            (now, mid, chat_id),
        )
        conn.commit()
        return mid


def get_messages(chat_id: str):
    """История активной ветки в формате [{role, content}] (для совместимости)."""
    with _connect() as conn:
        leaf = _get_current_leaf(conn, chat_id)
        rows = _branch_path_rows(conn, chat_id, leaf) if leaf else []
        return [{"role": r["role"], "content": r["content"]} for r in rows]


def get_messages_with_ids(chat_id: str) -> list[dict]:
    """История активной ветки [{id, role, content}] (для совместимости)."""
    with _connect() as conn:
        leaf = _get_current_leaf(conn, chat_id)
        return _branch_path_rows(conn, chat_id, leaf) if leaf else []


# ── Summaries (branch-aware) ──

def get_active_summary(chat_id: str, leaf_message_id: int | None) -> dict | None:
    """Summary, чей anchor лежит НА ПУТИ от leaf до корня; берём ближайший к листу.

    Anchor не на пути → summary охватывает события, которых в этой ветке не было,
    значит неприменим — игнорируем.
    """
    if leaf_message_id is None:
        return None
    with _connect() as conn:
        path = _branch_path_ids(conn, chat_id, leaf_message_id)
        if not path:
            return None
        cur = _execute(
            conn,
            f"SELECT branch_anchor_message_id, summary_text FROM summaries "
            f"WHERE chat_id={_P} ORDER BY branch_anchor_message_id DESC",
            (chat_id,),
        )
        for r in _fetchall(cur):
            if r["branch_anchor_message_id"] in path:
                return {
                    "summary_text": r["summary_text"],
                    "covers_messages_up_to_id": r["branch_anchor_message_id"],
                    "branch_anchor_message_id": r["branch_anchor_message_id"],
                }
        return None


def save_summary(chat_id: str, summary_text: str, branch_anchor_message_id: int):
    now = time.time()
    if _PG:
        sql = (
            f"INSERT INTO summaries (chat_id, branch_anchor_message_id, summary_text, created_at) "
            f"VALUES ({_P},{_P},{_P},{_P}) "
            f"ON CONFLICT (chat_id, branch_anchor_message_id) DO UPDATE SET "
            f"summary_text=EXCLUDED.summary_text, created_at=EXCLUDED.created_at"
        )
    else:
        sql = (
            f"INSERT INTO summaries (chat_id, branch_anchor_message_id, summary_text, created_at) "
            f"VALUES ({_P},{_P},{_P},{_P}) "
            f"ON CONFLICT (chat_id, branch_anchor_message_id) DO UPDATE SET "
            f"summary_text=excluded.summary_text, created_at=excluded.created_at"
        )
    with _connect() as conn:
        _execute(conn, sql, (chat_id, branch_anchor_message_id, summary_text, now))
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
        parent = None
        last_id = None
        for m in messages:
            last_id = _insert_message(conn, chat_id, m["role"], m["content"], updated_at, parent)
            parent = last_id
        if last_id is not None:
            _execute(
                conn,
                f"UPDATE chats SET current_leaf_message_id={_P} WHERE id={_P}",
                (last_id, chat_id),
            )
        conn.commit()


# ── Settings (key-value) ──

_SETTING_DEFAULTS = {
    "tts_voice": "ru-RU-SvetlanaNeural",
    "voice_enabled": "true",
    "memory_read_enabled": "true",
    "memory_token_budget": "600",
    # По умолчанию ВЫКЛ: загрузка локальной модели эмбеддингов (~150 МБ) тяжела
    # для Render free-tier и в read-пути подвесила бы первый запрос. Включается
    # тумблером «Вектор» на экране /memory, когда подтверждён запас памяти.
    "memory_vector_enabled": "false",
    "memory_write_enabled": "true",
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


# ── Chat Strategy Settings ──

_STRATEGY_DEFAULTS = {
    "rolling_summary_enabled": 1,
    "rolling_summary_threshold_pct": 35,
    "sliding_window_enabled": 0,
    "sliding_window_size": 20,
    "sticky_facts_enabled": 0,
}

_STRATEGY_FIELDS = set(_STRATEGY_DEFAULTS.keys())


def get_strategy_settings(chat_id: str) -> dict:
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT * FROM chat_strategy_settings WHERE chat_id={_P}",
            (chat_id,),
        )
        row = _fetchone(cur)
    if row:
        return row
    return {"chat_id": chat_id, **_STRATEGY_DEFAULTS}


def update_strategy_settings(chat_id: str, **fields) -> dict:
    sets = []
    vals = []
    for k, v in fields.items():
        if k not in _STRATEGY_FIELDS:
            continue
        sets.append(f"{k}={_P}")
        vals.append(int(v))
    if not sets:
        return get_strategy_settings(chat_id)
    vals.append(chat_id)
    with _connect() as conn:
        _execute(
            conn,
            f"UPDATE chat_strategy_settings SET {', '.join(sets)} WHERE chat_id={_P}",
            vals,
        )
        conn.commit()
    return get_strategy_settings(chat_id)


# ── Chat Facts (branch-aware) ──

def get_chat_facts(chat_id: str) -> list[dict]:
    """ВСЕ факты чата (по всем веткам). Для UI активной ветки см. get_active_facts."""
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT id, key, value, branch_anchor_message_id, created_at, updated_at "
            f"FROM chat_facts WHERE chat_id={_P} ORDER BY id",
            (chat_id,),
        )
        return _fetchall(cur)


def get_active_facts(chat_id: str, leaf_message_id: int | None) -> list[dict]:
    """Факты активной ветки: подъём от leaf к корню, по каждому ключу берётся
    ближайшая к листу запись. Надгробия (FACT_TOMBSTONE) скрывают ключ.
    """
    if leaf_message_id is None:
        return []
    with _connect() as conn:
        path = _branch_path_ids(conn, chat_id, leaf_message_id)
        if not path:
            return []
        cur = _execute(
            conn,
            f"SELECT id, key, value, branch_anchor_message_id, created_at, updated_at "
            f"FROM chat_facts WHERE chat_id={_P}",
            (chat_id,),
        )
        best: dict[str, dict] = {}
        for r in _fetchall(cur):
            anchor = r["branch_anchor_message_id"]
            if anchor not in path:
                continue
            cur_best = best.get(r["key"])
            if cur_best is None or anchor > cur_best["branch_anchor_message_id"]:
                best[r["key"]] = r
        result = [r for r in best.values() if r["value"] != FACT_TOMBSTONE]
        result.sort(key=lambda r: r["id"])
        return result


def add_chat_fact(chat_id: str, key: str, value: str, branch_anchor_message_id: int):
    """Upsert факта в точке ветки (anchor). add и update в экстракторе — оба сюда."""
    now = time.time()
    if _PG:
        sql = (
            f"INSERT INTO chat_facts (chat_id, key, value, created_at, updated_at, branch_anchor_message_id) "
            f"VALUES ({_P},{_P},{_P},{_P},{_P},{_P}) "
            f"ON CONFLICT (chat_id, branch_anchor_message_id, key) "
            f"DO UPDATE SET value=EXCLUDED.value, updated_at=EXCLUDED.updated_at"
        )
    else:
        sql = (
            f"INSERT INTO chat_facts (chat_id, key, value, created_at, updated_at, branch_anchor_message_id) "
            f"VALUES ({_P},{_P},{_P},{_P},{_P},{_P}) "
            f"ON CONFLICT (chat_id, branch_anchor_message_id, key) "
            f"DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at"
        )
    with _connect() as conn:
        _execute(conn, sql, (chat_id, key, value, now, now, branch_anchor_message_id))
        conn.commit()


def update_chat_fact(chat_id: str, key: str, value: str, branch_anchor_message_id: int):
    """Семантически = upsert в точке ветки."""
    add_chat_fact(chat_id, key, value, branch_anchor_message_id)


def tombstone_chat_fact(chat_id: str, key: str, branch_anchor_message_id: int):
    """Удаляет факт в текущей ветке через надгробие (не трогая предков/сиблингов)."""
    add_chat_fact(chat_id, key, FACT_TOMBSTONE, branch_anchor_message_id)


def delete_chat_fact_by_id(fact_id: int):
    with _connect() as conn:
        _execute(conn, f"DELETE FROM chat_facts WHERE id={_P}", (fact_id,))
        conn.commit()


# ── Долговременная память: граф (mem_nodes / mem_edges) ──
#
# Глобальная (на пользователя), вне окна контекста. Запись — только через эти
# функции; create_edge ОБЯЗАТЕЛЬНО прогоняет пару через memory_schema.validate_edge,
# поэтому невалидное ребро физически нельзя создать ни ручным API, ни экстрактором.
# Удаление — мягкое (status='active' → 'inactive'), узлы/рёбра физически не стираем.

def _bin(blob):
    """Оборачивает bytes для bytea в PostgreSQL; для SQLite возвращает как есть."""
    if blob is None:
        return None
    return psycopg2.Binary(blob) if _PG else blob


def _node_from_row(row: dict | None) -> dict | None:
    """Парсит attributes из JSON-строки в dict; убирает сырой embedding-BLOB,
    чтобы он не попал в JSON-ответы API (не сериализуется и не нужен клиенту)."""
    if not row:
        return None
    row.pop("embedding", None)
    raw = row.get("attributes")
    if isinstance(raw, str):
        try:
            row["attributes"] = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            row["attributes"] = {}
    return row


def _insert_returning_id(conn, sql: str, params) -> int:
    """INSERT с возвратом id (кросс-СУБД), как _insert_message, но для любой таблицы."""
    if _PG:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql + " RETURNING id", params)
        return cur.fetchone()["id"]
    cur = conn.execute(sql, params)
    return cur.lastrowid


def _get_node_row(conn, node_id: int) -> dict | None:
    cur = _execute(conn, f"SELECT * FROM mem_nodes WHERE id={_P}", (node_id,))
    return _fetchone(cur)


# ── Nodes ──

def create_node(node_type: str, name: str, *, attributes: dict | None = None,
                origin: str = "manual", basis: str = "", confidence: float = 1.0,
                status: str = "active", source_chat_id: str | None = None,
                source_message_id: int | None = None,
                embedding: bytes | None = None) -> int:
    ok, reason = memory_schema.validate_node_type(node_type)
    if not ok:
        raise ValueError(reason)
    now = time.time()
    attrs = json.dumps(attributes or {}, ensure_ascii=False)
    sql = (
        f"INSERT INTO mem_nodes (type, name, attributes, origin, basis, confidence, "
        f"status, source_chat_id, source_message_id, embedding, created_at, updated_at) "
        f"VALUES ({_P},{_P},{_P},{_P},{_P},{_P},{_P},{_P},{_P},{_P},{_P},{_P})"
    )
    params = (node_type, name, attrs, origin, basis, confidence, status,
              source_chat_id, source_message_id, _bin(embedding), now, now)
    with _connect() as conn:
        nid = _insert_returning_id(conn, sql, params)
        conn.commit()
        return nid


def set_node_embedding(node_id: int, embedding: bytes | None) -> None:
    with _connect() as conn:
        _execute(
            conn,
            f"UPDATE mem_nodes SET embedding={_P}, updated_at={_P} WHERE id={_P}",
            (_bin(embedding), time.time(), node_id),
        )
        conn.commit()


def get_node_vectors(*, active_only: bool = True) -> list[tuple[int, bytes]]:
    """[(node_id, embedding_blob)] для векторного поиска. Без узлов без вектора."""
    where = " WHERE status='active'" if active_only else ""
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT id, embedding FROM mem_nodes{where}",
        )
        rows = _fetchall(cur)
    out = []
    for r in rows:
        blob = r.get("embedding")
        if blob is not None:
            out.append((r["id"], bytes(blob)))
    return out


def get_node(node_id: int) -> dict | None:
    with _connect() as conn:
        return _node_from_row(_get_node_row(conn, node_id))


def get_active_nodes() -> list[dict]:
    with _connect() as conn:
        cur = _execute(conn, "SELECT * FROM mem_nodes WHERE status='active' ORDER BY id")
        return [_node_from_row(r) for r in _fetchall(cur)]


def get_nodes_by_type(node_type: str, *, active_only: bool = True) -> list[dict]:
    clauses = [f"type={_P}"]
    params: list = [node_type]
    if active_only:
        clauses.append("status='active'")
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT * FROM mem_nodes WHERE {' AND '.join(clauses)} ORDER BY id",
            params,
        )
        return [_node_from_row(r) for r in _fetchall(cur)]


def find_nodes_by_name(query: str, *, node_type: str | None = None,
                       limit: int = 20, active_only: bool = True) -> list[dict]:
    """Поиск по подстроке имени (именной индекс для дедупа/ручного поиска).

    SQLite LIKE регистронезависим только для ASCII; для кириллицы подъём узлов
    в пути чтения делает memory._find_seeds через Python .lower(). Здесь — для
    ручного API/поиска, этого достаточно.
    """
    op = "ILIKE" if _PG else "LIKE"
    clauses = [f"name {op} {_P}"]
    params: list = [f"%{query}%"]
    if active_only:
        clauses.append("status='active'")
    if node_type:
        clauses.append(f"type={_P}")
        params.append(node_type)
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT * FROM mem_nodes WHERE {' AND '.join(clauses)} "
            f"ORDER BY id LIMIT {int(limit)}",
            params,
        )
        return [_node_from_row(r) for r in _fetchall(cur)]


def update_node(node_id: int, **fields) -> None:
    allowed = {"name", "attributes", "basis", "confidence", "status"}
    sets, vals = [], []
    for k, v in fields.items():
        if k not in allowed:
            continue
        if k == "attributes":
            v = json.dumps(v or {}, ensure_ascii=False)
        sets.append(f"{k}={_P}")
        vals.append(v)
    if not sets:
        return
    sets.append(f"updated_at={_P}")
    vals.append(time.time())
    vals.append(node_id)
    with _connect() as conn:
        _execute(conn, f"UPDATE mem_nodes SET {', '.join(sets)} WHERE id={_P}", vals)
        conn.commit()


def soft_delete_node(node_id: int) -> None:
    """Мягкое удаление: status='inactive'. Узел физически остаётся."""
    with _connect() as conn:
        _execute(
            conn,
            f"UPDATE mem_nodes SET status='inactive', updated_at={_P} WHERE id={_P}",
            (time.time(), node_id),
        )
        conn.commit()


# ── Edges ──

def create_edge(src_id: int, dst_id: int, edge_type: str, *, origin: str = "manual",
                basis: str = "", confidence: float = 1.0, status: str = "active",
                source_chat_id: str | None = None,
                source_message_id: int | None = None) -> int:
    """Создаёт ребро ТОЛЬКО после проверки пары (src_type, dst_type) валидатором.

    Бросает ValueError, если узлов нет или пара недопустима. Это и есть
    «предохранитель» — единая точка, через которую проходит любая запись ребра.
    """
    now = time.time()
    with _connect() as conn:
        src = _get_node_row(conn, src_id)
        dst = _get_node_row(conn, dst_id)
        if not src:
            raise ValueError(f"Узел-источник #{src_id} не найден")
        if not dst:
            raise ValueError(f"Узел-цель #{dst_id} не найден")
        ok, reason = memory_schema.validate_edge(edge_type, src["type"], dst["type"])
        if not ok:
            raise ValueError(reason)
        sql = (
            f"INSERT INTO mem_edges (src_id, dst_id, type, origin, basis, confidence, "
            f"status, source_chat_id, source_message_id, created_at, updated_at) "
            f"VALUES ({_P},{_P},{_P},{_P},{_P},{_P},{_P},{_P},{_P},{_P},{_P})"
        )
        params = (src_id, dst_id, edge_type, origin, basis, confidence, status,
                  source_chat_id, source_message_id, now, now)
        eid = _insert_returning_id(conn, sql, params)
        conn.commit()
        return eid


def soft_delete_edge(edge_id: int) -> None:
    with _connect() as conn:
        _execute(
            conn,
            f"UPDATE mem_edges SET status='inactive', updated_at={_P} WHERE id={_P}",
            (time.time(), edge_id),
        )
        conn.commit()


# ── Конфликт версий: supersedes ──

def supersede_node(old_id: int, *, name: str | None = None,
                   attributes: dict | None = None, basis: str = "",
                   confidence: float = 1.0, origin: str = "manual",
                   source_chat_id: str | None = None,
                   source_message_id: int | None = None) -> int:
    """Конфликт факта: новое значение НЕ затирает старое.

    Создаём новый узел того же типа → ребро supersedes (new → old) → старый
    помечаем inactive. Источник правды = активная (новая) версия.
    """
    old = get_node(old_id)
    if not old:
        raise ValueError(f"Узел #{old_id} не найден")
    ntype = old["type"]
    if ntype not in memory_schema.SUPERSEDABLE_TYPES:
        raise ValueError(
            f"supersedes разрешён только для {', '.join(sorted(memory_schema.SUPERSEDABLE_TYPES))}, "
            f"а узел #{old_id} имеет тип '{ntype}'."
        )
    new_id = create_node(
        ntype,
        name if name is not None else old["name"],
        attributes=attributes if attributes is not None else old.get("attributes"),
        origin=origin, basis=basis, confidence=confidence,
        source_chat_id=source_chat_id, source_message_id=source_message_id,
    )
    create_edge(new_id, old_id, "supersedes", origin=origin, basis=basis,
                confidence=confidence, source_chat_id=source_chat_id,
                source_message_id=source_message_id)
    soft_delete_node(old_id)
    return new_id


# ── Запросы подграфов (путь чтения и фронт) ──

def get_subgraph(node_id: int, hops: int = 1, *, active_only: bool = True) -> dict:
    """BFS от узла на `hops` шагов по рёбрам. Возвращает {nodes, edges}.

    Рёбра — только те, у которых ОБА конца попали в выборку узлов (без висячих).
    """
    status_clause = " AND status='active'" if active_only else ""
    with _connect() as conn:
        visited: set[int] = set()
        frontier: set[int] = {node_id}
        edges: dict[int, dict] = {}
        for _ in range(max(0, hops)):
            if not frontier:
                break
            nxt: set[int] = set()
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                cur = _execute(
                    conn,
                    f"SELECT * FROM mem_edges WHERE (src_id={_P} OR dst_id={_P}){status_clause}",
                    (nid, nid),
                )
                for e in _fetchall(cur):
                    edges[e["id"]] = e
                    other = e["dst_id"] if e["src_id"] == nid else e["src_id"]
                    if other not in visited:
                        nxt.add(other)
            frontier = nxt
        visited |= frontier

        nodes = []
        for nid in visited:
            r = _get_node_row(conn, nid)
            if r and (not active_only or r["status"] == "active"):
                nodes.append(_node_from_row(r))
        keep = {n["id"] for n in nodes}
        clean_edges = [e for e in edges.values()
                       if e["src_id"] in keep and e["dst_id"] in keep]
        return {"nodes": nodes, "edges": clean_edges}


def get_topic_subgraph(topic_id: int, *, active_only: bool = True) -> dict:
    """Подграф темы: узлы, связанные ребром `about` с этой темой, + сама тема,
    + активные рёбра между ними. Для экрана памяти (фокус по папке-теме)."""
    status_clause = " AND status='active'" if active_only else ""
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT * FROM mem_edges WHERE type='about' AND dst_id={_P}{status_clause}",
            (topic_id,),
        )
        about_edges = _fetchall(cur)
        ids = {topic_id} | {e["src_id"] for e in about_edges}

        nodes = []
        for nid in ids:
            r = _get_node_row(conn, nid)
            if r and (not active_only or r["status"] == "active"):
                nodes.append(_node_from_row(r))
        keep = {n["id"] for n in nodes}

        edges: dict[int, dict] = {}
        for nid in keep:
            cur = _execute(
                conn,
                f"SELECT * FROM mem_edges WHERE (src_id={_P} OR dst_id={_P}){status_clause}",
                (nid, nid),
            )
            for e in _fetchall(cur):
                if e["src_id"] in keep and e["dst_id"] in keep:
                    edges[e["id"]] = e
        return {"nodes": nodes, "edges": list(edges.values())}


def get_full_graph(*, active_only: bool = True) -> dict:
    """Весь граф (для визуализации на экране памяти)."""
    nfilter = " WHERE status='active'" if active_only else ""
    with _connect() as conn:
        cur = _execute(conn, f"SELECT * FROM mem_nodes{nfilter} ORDER BY id")
        nodes = [_node_from_row(r) for r in _fetchall(cur)]
        cur = _execute(conn, f"SELECT * FROM mem_edges{nfilter} ORDER BY id")
        edges = _fetchall(cur)
        return {"nodes": nodes, "edges": edges}


# ── Слой знания: чанки (mem_chunks) ──

_CHUNK_COLS = ("id, text, origin, status, node_id, source_chat_id, "
               "source_message_id, created_at, updated_at")


def create_chunk(text: str, *, embedding: bytes | None = None, origin: str = "manual",
                 node_id: int | None = None, source_chat_id: str | None = None,
                 source_message_id: int | None = None) -> int:
    now = time.time()
    sql = (
        f"INSERT INTO mem_chunks (text, embedding, origin, status, node_id, "
        f"source_chat_id, source_message_id, created_at, updated_at) "
        f"VALUES ({_P},{_P},{_P},'active',{_P},{_P},{_P},{_P},{_P})"
    )
    params = (text, _bin(embedding), origin, node_id,
              source_chat_id, source_message_id, now, now)
    with _connect() as conn:
        cid = _insert_returning_id(conn, sql, params)
        conn.commit()
        return cid


def get_chunk(chunk_id: int) -> dict | None:
    with _connect() as conn:
        cur = _execute(
            conn, f"SELECT {_CHUNK_COLS} FROM mem_chunks WHERE id={_P}", (chunk_id,)
        )
        return _fetchone(cur)


def get_active_chunks() -> list[dict]:
    """Чанки без embedding-BLOB — безопасно для JSON-ответов."""
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT {_CHUNK_COLS} FROM mem_chunks WHERE status='active' ORDER BY id",
        )
        return _fetchall(cur)


def soft_delete_chunk(chunk_id: int) -> None:
    with _connect() as conn:
        _execute(
            conn,
            f"UPDATE mem_chunks SET status='inactive', updated_at={_P} WHERE id={_P}",
            (time.time(), chunk_id),
        )
        conn.commit()


def get_chunk_vectors(*, active_only: bool = True) -> list[tuple[int, bytes]]:
    """[(chunk_id, embedding_blob)] для векторного поиска."""
    where = " WHERE status='active'" if active_only else ""
    with _connect() as conn:
        cur = _execute(conn, f"SELECT id, embedding FROM mem_chunks{where}")
        rows = _fetchall(cur)
    out = []
    for r in rows:
        blob = r.get("embedding")
        if blob is not None:
            out.append((r["id"], bytes(blob)))
    return out
