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

CREATE TABLE IF NOT EXISTS pending_suggestions (
    chat_id    TEXT PRIMARY KEY,
    items      TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL
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

CREATE TABLE IF NOT EXISTS pending_suggestions (
    chat_id    TEXT PRIMARY KEY,
    items      TEXT NOT NULL,
    created_at REAL NOT NULL
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


def _table_exists(conn, table: str) -> bool:
    if _PG:
        cur = _execute(
            conn,
            f"SELECT 1 FROM information_schema.tables "
            f"WHERE table_schema = current_schema() AND table_name={_P}",
            (table,),
        )
        return _fetchone(cur) is not None
    else:
        cur = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        return cur.fetchone() is not None


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

    # 9. Память v2 Этап 2: проекты-контейнеры + материализованный граф связей.
    _migrate_projects_and_links(conn)
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


def _migrate_projects_and_links(conn):
    """Идемпотентно: таблица memory_links (материализованный граф [[..]],
    UNIQUE(source_id,target_title) для upsert) + одноразовый бэкфилл из заметок при
    первом создании. ТАКЖЕ создаёт таблицу projects и колонки project_id — они теперь
    ИНЕРТНЫ (сущность «проекты» убрана 2026-06-08, группировка = папки): код их не
    читает/не пишет, но схему оставляем (дроп на живой dual-СУБД рискованнее пустого
    столбца). Кросс-СУБД; идиома ADD COLUMN ... REFERENCES (как parent_message_id)."""
    real = "DOUBLE PRECISION" if _PG else "REAL"
    pk = "SERIAL PRIMARY KEY" if _PG else "INTEGER PRIMARY KEY AUTOINCREMENT"

    # Коммитим по блокам (как шаги 1-8): на Postgres ошибка в середине иначе
    # отравила бы всю транзакцию миграции; на SQLite — безвредно.
    _execute(
        conn,
        f"CREATE TABLE IF NOT EXISTS projects ("
        f"  id {pk},"
        f"  title TEXT NOT NULL UNIQUE,"
        f"  description TEXT NOT NULL DEFAULT '',"
        f"  status TEXT NOT NULL DEFAULT 'active',"
        f"  created_at {real} NOT NULL,"
        f"  updated_at {real} NOT NULL)",
    )
    _execute(conn, "CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status)")
    conn.commit()

    if not _column_exists(conn, "long_term_memory", "project_id"):
        _execute(
            conn,
            "ALTER TABLE long_term_memory ADD COLUMN project_id INTEGER "
            "REFERENCES projects(id) ON DELETE SET NULL",
        )
    if not _column_exists(conn, "tasks", "project_id"):
        _execute(
            conn,
            "ALTER TABLE tasks ADD COLUMN project_id INTEGER "
            "REFERENCES projects(id) ON DELETE SET NULL",
        )
    _execute(conn, "CREATE INDEX IF NOT EXISTS idx_ltm_project ON long_term_memory(project_id)")
    _execute(conn, "CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id)")
    conn.commit()

    links_existed = _table_exists(conn, "memory_links")
    _execute(
        conn,
        f"CREATE TABLE IF NOT EXISTS memory_links ("
        f"  id {pk},"
        f"  source_id INTEGER NOT NULL REFERENCES long_term_memory(id) ON DELETE CASCADE,"
        f"  target_id INTEGER REFERENCES long_term_memory(id) ON DELETE SET NULL,"
        f"  target_title TEXT NOT NULL,"
        f"  UNIQUE(source_id, target_title))",
    )
    _execute(conn, "CREATE INDEX IF NOT EXISTS idx_memlinks_source ON memory_links(source_id)")
    _execute(conn, "CREATE INDEX IF NOT EXISTS idx_memlinks_target ON memory_links(target_id)")
    conn.commit()

    if not links_existed:
        # одноразовый бэкфилл графа из существующих active-заметок
        cur = _execute(conn, "SELECT id, content FROM long_term_memory WHERE status='active'")
        for r in _fetchall(cur):
            _upsert_links_for_note(conn, r["id"], r["content"] or "")
        _resolve_all_link_targets(conn)
        conn.commit()


def init_db():
    with _connect() as conn:
        if _PG:
            cur = conn.cursor()
            cur.execute(_SCHEMA_PG)
            conn.commit()
        else:
            conn.executescript(_SCHEMA_SQLITE)

        _migrate(conn)

        # Мёртвые таблицы откатанного графового модуля памяти (Фаза 5).
        # Пустые и не используются; дропаем идемпотентно. Порядок: сначала
        # рёбра/чанки, затем узлы (на случай FK от edges к nodes).
        for _legacy in ("mem_edges", "mem_chunks", "mem_nodes"):
            _execute(conn, f"DROP TABLE IF EXISTS {_legacy}")
        conn.commit()

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

    # Защищённая заметка профиля: создаём, если в папке 'Профиль' ещё ничего нет.
    ensure_profile_exists()

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


def create_chat(*, chat_id=None, title="", pinned=False, model="claude-opus-4-6",
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



# ── Settings (key-value) ──

_SETTING_DEFAULTS = {
    "tts_voice": "ru-RU-SvetlanaNeural",
    "voice_enabled": "true",
    "memory_autosave": "false",
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


# ── Chat Facts (branch-aware) ──

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


# ── Pending-предложения памяти ──
# Неподтверждённая карточка (autosave=OFF) хранится в БД, а не в JS/процессе:
# переживает reload, переход между разделами, редеплой и multi-worker. Одна
# карточка на чат; новое предложение заменяет старое.

def set_pending_suggestions(chat_id: str, items: list) -> None:
    if not chat_id or not items:
        return
    payload = json.dumps(items, ensure_ascii=False)
    now = time.time()
    with _connect() as conn:
        if _PG:
            _execute(
                conn,
                f"INSERT INTO pending_suggestions (chat_id, items, created_at) "
                f"VALUES ({_P},{_P},{_P}) ON CONFLICT (chat_id) DO UPDATE "
                f"SET items=EXCLUDED.items, created_at=EXCLUDED.created_at",
                (chat_id, payload, now),
            )
        else:
            _execute(
                conn,
                f"INSERT OR REPLACE INTO pending_suggestions (chat_id, items, created_at) "
                f"VALUES ({_P},{_P},{_P})",
                (chat_id, payload, now),
            )
        conn.commit()


def get_pending_suggestions(chat_id: str) -> list:
    if not chat_id:
        return []
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT items FROM pending_suggestions WHERE chat_id={_P}",
            (chat_id,),
        )
        row = _fetchone(cur)
    if not row:
        return []
    try:
        items = json.loads(row["items"])
        return items if isinstance(items, list) else []
    except (ValueError, TypeError):
        return []


def clear_pending_suggestions(chat_id: str) -> None:
    if not chat_id:
        return
    with _connect() as conn:
        _execute(conn, f"DELETE FROM pending_suggestions WHERE chat_id={_P}", (chat_id,))
        conn.commit()


# ── Долговременная память (long_term_memory) ──

_MEMORY_FIELDS = {"title", "content", "folder", "status"}

# Защищённый профиль: заметки этой папки всегда инжектятся в контекст и не
# удаляются (только редактируются). «Профиль» — обычная папка (folder), отдельной
# колонки pinned/system нет; принадлежность к профилю определяется по folder.
PROFILE_FOLDER = "Профиль"
_PROFILE_TITLE = "Профиль пользователя"
_PROFILE_PLACEHOLDER = (
    "Здесь можно описать себя: имя, роль, предпочтения по стилю ответов, "
    "любимые инструменты, ограничения."
)

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


def find_active_memory_by_title(title: str) -> dict | None:
    """Active-заметка по нормализованному (trim+lower) title; свежайшая при
    нескольких совпадениях. Нужно для update_memory/delete_memory по названию:
    агент оперирует заголовками, а не id (зеркалит find_active_task_by_title)."""
    norm = (title or "").strip().lower()
    if not norm:
        return None
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT {_MEMORY_COLS} FROM long_term_memory "
            f"WHERE status='active' AND LOWER(TRIM(title))={_P} "
            f"ORDER BY updated_at DESC, id DESC",
            (norm,),
        )
        return _fetchone(cur)


# ── Профиль (защищённые заметки папки 'Профиль') ──

def get_pinned_memory(folder: str = PROFILE_FOLDER) -> list[dict]:
    """ВСЕ active-заметки из указанной папки (по умолчанию 'Профиль'). Для
    безусловного инжекта профиля в контекст агента. Свежие сверху."""
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT {_MEMORY_COLS} FROM long_term_memory "
            f"WHERE status='active' AND folder={_P} ORDER BY updated_at DESC",
            (folder,),
        )
        return _fetchall(cur)


def ensure_profile_exists():
    """Гарантирует наличие хотя бы одной active-заметки профиля (папка 'Профиль').
    Если её нет — создаёт одну с placeholder-текстом. Идемпотентно; вызывается из
    init_db. Проверка ведётся по папке (а не по title): если юзер сознательно увёл
    заметку профиля в другую папку, папка 'Профиль' опустеет и здесь создастся
    новая — это ок (см. урок 12)."""
    now = time.time()
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT 1 FROM long_term_memory "
            f"WHERE status='active' AND folder={_P} LIMIT 1",
            (PROFILE_FOLDER,),
        )
        if _fetchone(cur):
            return
        _execute(
            conn,
            f"INSERT INTO long_term_memory "
            f"(title, content, folder, status, created_at, updated_at) "
            f"VALUES ({_P},{_P},{_P},'active',{_P},{_P})",
            (_PROFILE_TITLE, _PROFILE_PLACEHOLDER, PROFILE_FOLDER, now, now),
        )
        conn.commit()


def is_profile_note(memory_id: int) -> bool:
    """True, если заметка существует и лежит в защищённой папке 'Профиль'.
    Единый источник истины для защиты от удаления (API-эндпоинты и путь агента)."""
    row = get_memory(memory_id)
    return bool(row) and row.get("folder") == PROFILE_FOLDER


# ── Материализованный граф связей [[..]] (memory_links) ──

def _parse_wikilink_titles(content: str) -> list[str]:
    """Уникальные (по нормализованному виду) target-заголовки из [[..]] в content,
    в порядке появления; [[Заголовок|алиас]] → Заголовок."""
    out, seen = [], set()
    for raw in _WIKILINK_RE.findall(content or ""):
        t = raw.split("|")[0].strip()
        k = t.lower()
        if t and k not in seen:
            seen.add(k)
            out.append(t)
    return out


def _upsert_links_for_note(conn, source_id: int, content: str) -> None:
    """UPSERT исходящих связей заметки + удаление исчезнувших (в рамках conn, без
    commit). target_id здесь не резолвится — это делает _resolve_all_link_targets."""
    titles = _parse_wikilink_titles(content)
    for t in titles:
        _execute(
            conn,
            f"INSERT INTO memory_links (source_id, target_id, target_title) "
            f"VALUES ({_P}, NULL, {_P}) "
            f"ON CONFLICT (source_id, target_title) DO NOTHING",
            (source_id, t),
        )
    if titles:
        ph = ",".join([_P] * len(titles))
        _execute(
            conn,
            f"DELETE FROM memory_links WHERE source_id={_P} AND target_title NOT IN ({ph})",
            (source_id, *titles),
        )
    else:
        _execute(conn, f"DELETE FROM memory_links WHERE source_id={_P}", (source_id,))


def _resolve_all_link_targets(conn) -> None:
    """Пересчитывает target_id всех связей по нормализованному (trim+lower) title
    активных заметок. Покрывает бэклинки (появилась целевая заметка) и
    переименование (target съезжает/становится ghost)."""
    cur = _execute(conn, "SELECT id, title FROM long_term_memory WHERE status='active'")
    norm_to_id = {(r["title"] or "").strip().lower(): r["id"] for r in _fetchall(cur)}
    cur = _execute(conn, "SELECT id, target_title FROM memory_links")
    for r in _fetchall(cur):
        tid = norm_to_id.get((r["target_title"] or "").strip().lower())
        _execute(conn, f"UPDATE memory_links SET target_id={_P} WHERE id={_P}", (tid, r["id"]))


def sync_memory_links(source_id: int, content: str) -> None:
    """Материализует граф после записи заметки source_id: upsert её исходящих
    [[..]], удаление исчезнувших и глобальный ре-резолв target_id (бэклинки +
    переименование). Свой connection + commit."""
    with _connect() as conn:
        _upsert_links_for_note(conn, source_id, content or "")
        _resolve_all_link_targets(conn)
        conn.commit()


def create_memory(title: str, content: str = "", folder: str | None = None) -> dict | None:
    """Создаёт заметку. Если active-заметка с таким title уже есть → None (=409).
    Проверка и вставка в одной транзакции (защита от гонки). folder=None → без папки.
    После создания материализует граф связей (исходящие [[..]] + бэклинки на неё)."""
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
        row = _fetchone(cur)
    sync_memory_links(new_id, content or "")
    return row


def update_memory(memory_id: int, **fields) -> dict | None:
    """Whitelist {title, content, folder, status}. Всегда обновляет updated_at.
    Возвращает обновлённую запись или None, если id не найден. Любое изменение
    (в т.ч. title/status) ре-материализует граф связей."""
    # Защита профиля (db-слой, закрывает и путь агента): заметку из папки 'Профиль'
    # нельзя деактивировать (= мягко удалить) — но редактировать можно. Снимаем
    # только status=inactive, остальные поля применяются как обычно.
    if fields.get("status") == "inactive" and is_profile_note(memory_id):
        fields = {k: v for k, v in fields.items() if k != "status"}
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
    row = get_memory(memory_id)
    if row is not None:
        sync_memory_links(memory_id, row.get("content") or "")
    return row


def merge_memory_content(title: str, content: str) -> tuple[dict | None, str]:
    """Дописывает content в уже существующую active-заметку с таким title.
    Нужно при коллизии create (title занят): вместо тихой потери данных дополняем
    существующую заметку. Резолв по точному title, как в create_memory.
    Возвращает (заметка|None, status):
      'merged'    — content дописан в существующую заметку (заметка обновлена);
      'duplicate' — дописывать было нечего (content пуст или уже содержится), не меняли;
      'absent'    — active-заметки с таким title нет (заметка=None)."""
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT {_MEMORY_COLS} FROM long_term_memory "
            f"WHERE status='active' AND title={_P}",
            (title,),
        )
        existing = _fetchone(cur)
    if existing is None:
        return None, "absent"
    new = (content or "").strip()
    old = existing.get("content") or ""
    if not new or new in old:
        return existing, "duplicate"
    merged = (old.rstrip() + "\n\n" + new).strip()
    return update_memory(existing["id"], content=merged), "merged"


def delete_memory(memory_id: int) -> bool:
    """Физическое удаление (для крайних случаев; основной способ — status=inactive).
    Заметку профиля (папка 'Профиль') не удаляет → возвращает False; иначе удаляет и
    возвращает True (защита db-слоя, закрывает и путь агента)."""
    if is_profile_note(memory_id):
        return False
    with _connect() as conn:
        _execute(conn, f"DELETE FROM long_term_memory WHERE id={_P}", (memory_id,))
        conn.commit()
    return True


def get_memory_links() -> list[dict]:
    """Пары wiki-связей из материализованной таблицы memory_links (а не парсингом
    текста). Только связи активных source-заметок; target_id показываем лишь когда
    целевая заметка тоже активна (иначе ghost: target_id=None). Форма прежняя:
    {source_id, source_title, target_title, target_id}."""
    with _connect() as conn:
        cur = _execute(
            conn,
            "SELECT ml.source_id AS source_id, s.title AS source_title, "
            "       ml.target_title AS target_title, "
            "       CASE WHEN t.status='active' THEN ml.target_id ELSE NULL END AS target_id "
            "FROM memory_links ml "
            "JOIN long_term_memory s ON s.id = ml.source_id "
            "LEFT JOIN long_term_memory t ON t.id = ml.target_id "
            "WHERE s.status='active' "
            "ORDER BY ml.source_id, ml.id",
        )
        return [
            {"source_id": r["source_id"], "source_title": r["source_title"],
             "target_title": r["target_title"], "target_id": r["target_id"]}
            for r in _fetchall(cur)
        ]


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


# Сущность «проекты» убрана по решению пользователя (2026-06-08): группировка =
# папки (folder). Таблица projects и колонки project_id оставлены инертными в схеме
# (см. _migrate_projects_and_links) — дроп на живой dual-СУБД рискованнее пустого
# столбца; код их больше не читает и не пишет.


# Частотные служебные слова: при пороге ≥3 они иначе матчат %LIKE% почти всё и
# тянут слабо-связанные заметки в дедуп/инжект (шум, против которого Этап 0).
# Порог ≥3 нужен ради имён/аббревиатур (API, СЕО), а не ради «что/как/для».
_STOPWORDS = frozenset({
    # ru
    "что", "как", "так", "это", "эта", "эти", "тот", "том", "вот", "там", "тут",
    "где", "кто", "чем", "для", "под", "над", "при", "без", "про", "его", "её",
    "ее", "они", "она", "оно", "мне", "нам", "вам", "был", "уже", "ещё", "еще",
    "или", "была", "было", "этот", "того", "тоже", "есть", "надо", "чтобы",
    "когда", "потом", "очень", "если",
    # en
    "the", "and", "for", "you", "are", "was", "not", "but", "has", "had", "can",
    "all", "any", "our", "its", "his", "her", "who", "why", "how", "what", "this",
    "that", "with", "from",
})


# Простой стемминг русского: усечение частых словоизменительных окончаний, чтобы
# «диагностику» матчила «диагностика». Не лингвистически точно — задача лишь свести
# словоформы к общему началу для LIKE-поиска. Длиннейший суффикс первым; стем ≥4.
_RU_SUFFIXES = sorted([
    "ами", "ями", "ого", "его", "ему", "ому", "ыми", "ими", "ах", "ях", "ов",
    "ев", "ёв", "ам", "ям", "ом", "ем", "ой", "ый", "ий", "ая", "яя", "ую", "юю",
    "ые", "ие", "их", "ых", "ее", "ть", "а", "я", "о", "е", "у", "ю", "ы", "и",
    "й", "ь",
], key=len, reverse=True)


def _stem(w: str) -> str:
    for suf in _RU_SUFFIXES:
        if len(w) - len(suf) >= 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _query_stems(text: str) -> list[str]:
    """Стемы значимых слов запроса: без стоп-слов, длина исходного слова ≥3,
    уникальные, не больше 8. Стем ≥3 символов."""
    out, seen = [], set()
    for w in re.findall(r"\w+", (text or "").lower()):
        if len(w) < 3 or w in _STOPWORDS:
            continue
        s = _stem(w)
        if len(s) >= 3 and s not in seen:
            seen.add(s)
            out.append(s)
    return out[:8]


def _match_active_notes(conn, stems: list[str], cols: str) -> list[dict]:
    """active-заметки, где встречается хотя бы один стем (LIKE по title/content).
    LIKE — лишь грубый префильтр (символ `_` в стеме работал бы как wildcard);
    истинный отбор/ранжирование делает _score_by_stems по литеральному вхождению.
    LIMIT — предохранитель от загрузки гигантской выборки в память."""
    clauses, params = [], []
    for s in stems:
        like = f"%{s}%"
        clauses.append(f"(LOWER(title) LIKE {_P} OR LOWER(content) LIKE {_P})")
        params.extend([like, like])
    cur = _execute(
        conn,
        f"SELECT {cols} FROM long_term_memory "
        f"WHERE status='active' AND ({' OR '.join(clauses)}) "
        f"ORDER BY updated_at DESC LIMIT 500",
        params,
    )
    return _fetchall(cur)


def _score_by_stems(rows: list[dict], stems: list[str]) -> list[dict]:
    """Сортировка по числу совпавших уникальных стемов (затем свежесть). Строки с
    нулём совпадений ОТБРАСЫВАЕМ: это ложные срабатывания грубого LIKE-префильтра
    (напр. `_` как wildcard) — иначе нерелевантные заметки утекли бы в инжект
    (реанимация канала бага B). Python-вхождение здесь — источник истины."""
    scored = []
    for r in rows:
        hay = ((r.get("title") or "") + " " + (r.get("content") or "")).lower()
        score = sum(1 for s in stems if s in hay)
        if score:
            scored.append((score, r.get("updated_at") or 0, r))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [r for _, _, r in scored]


def relevant_memory(text: str, limit: int = 5) -> list[dict]:
    """Релевантные активные заметки для пред-инжекта в системпромпт. Поиск по стемам
    значимых слов сообщения (LIKE по title/content), ранг по числу совпавших слов
    (не по свежести). Если ни одно слово не совпало — [] (без фолбэка «последние
    свежие» — канал бага B). Возвращает [{id, title, content_preview, folder}].
    Дедуп при записи теперь делает БД по title — отдельного memory_for_dedup не нужно."""
    stems = _query_stems(text)
    lim = max(1, int(limit))
    if not stems:
        return []
    with _connect() as conn:
        rows = _match_active_notes(conn, stems, "id, title, content, folder, updated_at")
    ranked = _score_by_stems(rows, stems)[:lim]
    return [
        {"id": r["id"], "title": r["title"], "content_preview": (r["content"] or "")[:300],
         "folder": r["folder"]}
        for r in ranked
    ]


def search_memory(query: str, limit: int = 8) -> list[dict]:
    """Агентный retrieval: ищет заметки по запросу (стемминг + ранг по совпавшим
    словам) и обогащает результат исходящими связями [[..]] — чтобы агент мог достать
    факт по требованию и идти по связям. Возвращает [{id, title, content, folder,
    links:[target_title], score}]. [] если запрос пуст/нет совпадений."""
    stems = _query_stems(query)
    lim = max(1, int(limit))
    if not stems:
        return []
    with _connect() as conn:
        rows = _match_active_notes(
            conn, stems, "id, title, content, folder, updated_at")
        ranked = _score_by_stems(rows, stems)[:lim]
        if not ranked:
            return []
        ids = [r["id"] for r in ranked]
        ph = ",".join([_P] * len(ids))
        # исходящие связи каждой найденной заметки
        cur = _execute(
            conn,
            f"SELECT source_id, target_title FROM memory_links WHERE source_id IN ({ph}) "
            f"ORDER BY id",
            ids,
        )
        links_by_src = {}
        for r in _fetchall(cur):
            links_by_src.setdefault(r["source_id"], []).append(r["target_title"])
    out = []
    for r in ranked:
        hay = ((r.get("title") or "") + " " + (r.get("content") or "")).lower()
        out.append({
            "id": r["id"],
            "title": r["title"],
            "content": (r["content"] or "")[:600],
            "folder": r["folder"],
            "links": links_by_src.get(r["id"], []),
            "score": sum(1 for s in stems if s in hay),
        })
    return out


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


def find_task_by_title(title: str, statuses: tuple = ("active",)) -> dict | None:
    """Задача по нормализованному (trim+lower) title среди указанных статусов;
    свежайшая, если совпало несколько (по updated_at, затем id — детерминированно).
    Нужно для close/update/archive/delete по названию: агент видит задачи по title,
    но не знает их id. LOWER кросс-СУБД (SQLite — юникод-функция). NB: при двух
    задачах с одинаковым нормализованным title затронется только свежайшая."""
    norm = (title or "").strip().lower()
    if not norm or not statuses:
        return None
    placeholders = ",".join([_P] * len(statuses))
    with _connect() as conn:
        cur = _execute(
            conn,
            f"SELECT * FROM tasks WHERE status IN ({placeholders}) "
            f"AND LOWER(TRIM(title))={_P} "
            f"ORDER BY updated_at DESC, id DESC",
            (*statuses, norm),
        )
        return _fetchone(cur)


def find_active_task_by_title(title: str) -> dict | None:
    """Обёртка совместимости: активная задача по title (см. find_task_by_title)."""
    return find_task_by_title(title, ("active",))


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


def delete_task(task_id: int):
    """Физическое удаление задачи."""
    with _connect() as conn:
        _execute(conn, f"DELETE FROM tasks WHERE id={_P}", (task_id,))
        conn.commit()
