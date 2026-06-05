"""
Слой хранения данных.

DATABASE_URL задана → PostgreSQL (Render, прод).
Не задана             → SQLite   (локальная разработка).

История чата — дерево: каждое сообщение знает родителя (parent_message_id).
Активная ветка чата — путь от chats.current_leaf_message_id вверх до корня.
Линейный чат — вырожденное дерево, где у каждого сообщения один ребёнок.
"""

import os
import re
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
        # Встроенный SQLite LOWER() понижает регистр только ASCII — кириллицу не
        # трогает, из-за чего поиск по русским словам с заглавной буквы не находит.
        # Подменяем на юникод-корректный (PostgreSQL LOWER() и так юникодный).
        conn.create_function(
            "lower", 1, lambda s: s.lower() if isinstance(s, str) else s,
            deterministic=True,
        )
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


def _insert_returning_id(conn, sql, params):
    """Вставка с возвратом нового id (кросс-СУБД)."""
    if _PG:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql + " RETURNING id", params)
        return cur.fetchone()["id"]
    else:
        return conn.execute(sql, params).lastrowid


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

CREATE TABLE IF NOT EXISTS long_term_memory (
    id         SERIAL PRIMARY KEY,
    title      TEXT NOT NULL,
    content    TEXT NOT NULL DEFAULT '',
    folder     TEXT,
    status     TEXT NOT NULL DEFAULT 'active',
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ltm_status   ON long_term_memory(status);

CREATE TABLE IF NOT EXISTS tasks (
    id         SERIAL PRIMARY KEY,
    title      TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'active',
    context    TEXT NOT NULL DEFAULT '',
    outcome    TEXT NOT NULL DEFAULT '',
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL,
    closed_at  DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
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

CREATE TABLE IF NOT EXISTS long_term_memory (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL,
    content    TEXT NOT NULL DEFAULT '',
    folder     TEXT,
    status     TEXT NOT NULL DEFAULT 'active',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ltm_status   ON long_term_memory(status);

CREATE TABLE IF NOT EXISTS tasks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'active',
    context    TEXT NOT NULL DEFAULT '',
    outcome    TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    closed_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
"""


# ── Migration helpers ──

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

    # 8. long_term_memory: category → folder (свободные пользовательские папки, nullable)
    if (_column_exists(conn, "long_term_memory", "category")
            and not _column_exists(conn, "long_term_memory", "folder")):
        _migrate_ltm_folder(conn)
        conn.commit()
    if _column_exists(conn, "long_term_memory", "folder"):
        _execute(conn, "DROP INDEX IF EXISTS idx_ltm_category")
        _execute(conn, "CREATE INDEX IF NOT EXISTS idx_ltm_folder ON long_term_memory(folder)")
        # rebuild на SQLite пересоздаёт таблицу без индексов — восстанавливаем status-индекс
        _execute(conn, "CREATE INDEX IF NOT EXISTS idx_ltm_status ON long_term_memory(status)")
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


def _migrate_ltm_folder(conn):
    """long_term_memory.category → folder (TEXT, nullable). Старые значения категорий
    сохраняются как имена папок. На SQLite — rebuild (нельзя ALTER COLUMN, чтобы снять
    NOT NULL/DEFAULT и сделать колонку nullable); на PostgreSQL — RENAME + DROP NOT NULL/DEFAULT.
    """
    if _PG:
        _execute(conn, "ALTER TABLE long_term_memory RENAME COLUMN category TO folder")
        _execute(conn, "ALTER TABLE long_term_memory ALTER COLUMN folder DROP NOT NULL")
        _execute(conn, "ALTER TABLE long_term_memory ALTER COLUMN folder DROP DEFAULT")
    else:
        _execute(
            conn,
            "CREATE TABLE long_term_memory_new ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  title TEXT NOT NULL,"
            "  content TEXT NOT NULL DEFAULT '',"
            "  folder TEXT,"
            "  status TEXT NOT NULL DEFAULT 'active',"
            "  created_at REAL NOT NULL,"
            "  updated_at REAL NOT NULL)",
        )
        _execute(
            conn,
            "INSERT INTO long_term_memory_new "
            "(id, title, content, folder, status, created_at, updated_at) "
            "SELECT id, title, content, category, status, created_at, updated_at "
            "FROM long_term_memory",
        )
        _execute(conn, "DROP TABLE long_term_memory")
        _execute(conn, "ALTER TABLE long_term_memory_new RENAME TO long_term_memory")


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


# ── Долговременная память (long_term_memory) ──

_MEMORY_FIELDS = {"title", "content", "folder", "status"}

# Wiki-ссылка [[Заголовок]] или [[Заголовок|алиас]] в markdown-теле заметки.
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

_MEMORY_COLS = "id, title, content, folder, status, created_at, updated_at"


def list_memory(folder: str | None = None, q: str | None = None) -> list[dict]:
    """Активные заметки. Опц. фильтр по folder и полнотекстовый поиск q
    (LOWER(title) LIKE OR LOWER(content) LIKE). Сортировка: свежие сверху."""
    where = ["status='active'"]
    params: list = []
    if folder:
        where.append(f"folder={_P}")
        params.append(folder)
    if q:
        like = f"%{q.lower()}%"
        where.append(f"(LOWER(title) LIKE {_P} OR LOWER(content) LIKE {_P})")
        params.extend([like, like])
    sql = (
        f"SELECT {_MEMORY_COLS} FROM long_term_memory "
        f"WHERE {' AND '.join(where)} ORDER BY updated_at DESC"
    )
    with _connect() as conn:
        cur = _execute(conn, sql, params)
        return _fetchall(cur)


def get_memory(memory_id: int) -> dict | None:
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT {_MEMORY_COLS} FROM long_term_memory WHERE id={_P}",
            (memory_id,),
        )
        return _fetchone(cur)


def create_memory(title: str, content: str = "", folder: str | None = None) -> dict | None:
    """Создаёт заметку. Если active-заметка с таким title уже есть → None (=409).
    Проверка и вставка в одной транзакции (защита от гонки). folder=None → без папки."""
    now = time.time()
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT id FROM long_term_memory WHERE status='active' AND title={_P}",
            (title,),
        )
        if _fetchone(cur):
            return None
        new_id = _insert_returning_id(
            conn,
            f"INSERT INTO long_term_memory "
            f"(title, content, folder, status, created_at, updated_at) "
            f"VALUES ({_P},{_P},{_P},'active',{_P},{_P})",
            (title, content, folder, now, now),
        )
        conn.commit()
        cur = _execute(
            conn,
            f"SELECT {_MEMORY_COLS} FROM long_term_memory WHERE id={_P}",
            (new_id,),
        )
        return _fetchone(cur)


def update_memory(memory_id: int, **fields) -> dict | None:
    """Whitelist {title, content, folder, status}. Всегда обновляет updated_at.
    Возвращает обновлённую запись или None, если id не найден."""
    sets = []
    vals = []
    for k, v in fields.items():
        if k not in _MEMORY_FIELDS:
            continue
        sets.append(f"{k}={_P}")
        vals.append(v)
    if not sets:
        return get_memory(memory_id)
    sets.append(f"updated_at={_P}")
    vals.append(time.time())
    vals.append(memory_id)
    with _connect() as conn:
        _execute(
            conn,
            f"UPDATE long_term_memory SET {', '.join(sets)} WHERE id={_P}",
            vals,
        )
        conn.commit()
    return get_memory(memory_id)


def delete_memory(memory_id: int):
    """Физическое удаление (для крайних случаев; основной способ — status=inactive)."""
    with _connect() as conn:
        _execute(conn, f"DELETE FROM long_term_memory WHERE id={_P}", (memory_id,))
        conn.commit()


def get_memory_links() -> list[dict]:
    """Пары wiki-связей по всем active-заметкам: парсит [[...]] из content,
    резолвит target по точному title. target_id=None, если заметка не найдена."""
    with _connect() as conn:
        cur = _execute(
            conn,
            "SELECT id, title, content FROM long_term_memory WHERE status='active'",
        )
        notes = _fetchall(cur)
    title_to_id = {n["title"]: n["id"] for n in notes}
    links: list[dict] = []
    for n in notes:
        for raw in _WIKILINK_RE.findall(n["content"] or ""):
            target_title = raw.split("|")[0].strip()  # [[Заголовок|алиас]] → Заголовок
            if not target_title:
                continue
            links.append({
                "source_id": n["id"],
                "source_title": n["title"],
                "target_title": target_title,
                "target_id": title_to_id.get(target_title),
            })
    return links


def get_memory_folders() -> list[str]:
    """Уникальные папки активных заметок (без NULL), по алфавиту. Папки не хранятся
    отдельной таблицей — выводятся из DISTINCT folder."""
    with _connect() as conn:
        cur = _execute(
            conn,
            "SELECT DISTINCT folder FROM long_term_memory "
            "WHERE status='active' AND folder IS NOT NULL ORDER BY folder",
        )
        return [r["folder"] for r in _fetchall(cur)]


def relevant_memory(text: str, limit: int = 5) -> list[dict]:
    """Релевантные активные заметки для классификатора памяти: поиск по ключевым
    словам сообщения (OR LIKE по title/content). Если совпадений нет — последние
    обновлённые. Возвращает [{id, title, content_preview}]."""
    words = [w for w in re.findall(r"\w+", (text or "").lower()) if len(w) >= 4]
    words = list(dict.fromkeys(words))[:8]  # уникальные, не больше 8
    lim = max(1, int(limit))
    with _connect() as conn:
        rows = []
        if words:
            clauses, params = [], []
            for w in words:
                like = f"%{w}%"
                clauses.append(f"(LOWER(title) LIKE {_P} OR LOWER(content) LIKE {_P})")
                params.extend([like, like])
            cur = _execute(
                conn,
                f"SELECT id, title, content FROM long_term_memory "
                f"WHERE status='active' AND ({' OR '.join(clauses)}) "
                f"ORDER BY updated_at DESC LIMIT {lim}",
                params,
            )
            rows = _fetchall(cur)
        if not rows:
            cur = _execute(
                conn,
                f"SELECT id, title, content FROM long_term_memory "
                f"WHERE status='active' ORDER BY updated_at DESC LIMIT {lim}",
            )
            rows = _fetchall(cur)
    return [
        {"id": r["id"], "title": r["title"], "content_preview": (r["content"] or "")[:300]}
        for r in rows
    ]


# ── Буфер задач (tasks) ──

_TASK_FIELDS = {"title", "context", "outcome", "status"}


def list_tasks(status: str | None = None) -> list[dict]:
    """Все задачи или фильтр по status. Сортировка: свежие сверху."""
    with _connect() as conn:
        if status:
            cur = _execute(
                conn,
                f"SELECT * FROM tasks WHERE status={_P} ORDER BY updated_at DESC",
                (status,),
            )
        else:
            cur = _execute(conn, "SELECT * FROM tasks ORDER BY updated_at DESC")
        return _fetchall(cur)


def get_task(task_id: int) -> dict | None:
    with _connect() as conn:
        cur = _execute(conn, f"SELECT * FROM tasks WHERE id={_P}", (task_id,))
        return _fetchone(cur)


def create_task(title: str, context: str = "") -> dict:
    now = time.time()
    with _connect() as conn:
        new_id = _insert_returning_id(
            conn,
            f"INSERT INTO tasks (title, status, context, outcome, created_at, updated_at) "
            f"VALUES ({_P},'active',{_P},'',{_P},{_P})",
            (title, context, now, now),
        )
        conn.commit()
        cur = _execute(conn, f"SELECT * FROM tasks WHERE id={_P}", (new_id,))
        return _fetchone(cur)


def update_task(task_id: int, **fields) -> dict | None:
    """Whitelist {title, context, outcome, status}. Всегда обновляет updated_at.
    closed_at: ставится при переходе в completed/archived, обнуляется при active.
    Возвращает обновлённую задачу или None, если id не найден."""
    sets = []
    vals = []
    for k, v in fields.items():
        if k not in _TASK_FIELDS:
            continue
        sets.append(f"{k}={_P}")
        vals.append(v)
    if "status" in fields:
        if fields["status"] in ("completed", "archived"):
            sets.append(f"closed_at={_P}")
            vals.append(time.time())
        elif fields["status"] == "active":
            sets.append("closed_at=NULL")
    if not sets:
        return get_task(task_id)
    sets.append(f"updated_at={_P}")
    vals.append(time.time())
    vals.append(task_id)
    with _connect() as conn:
        _execute(conn, f"UPDATE tasks SET {', '.join(sets)} WHERE id={_P}", vals)
        conn.commit()
    return get_task(task_id)
