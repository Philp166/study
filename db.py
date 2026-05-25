import sqlite3
import time
import uuid
from pathlib import Path

DB_PATH = Path(__file__).parent / "cos.db"

def _connect():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    conn = _connect()
    conn.executescript("""
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
    """)
    conn.close()

def new_id():
    return uuid.uuid4().hex[:12]

def list_chats():
    conn = _connect()
    rows = conn.execute(
        "SELECT id, title, pinned, model, created_at, updated_at FROM chats "
        "ORDER BY pinned DESC, updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_chat(chat_id: str):
    conn = _connect()
    chat = conn.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
    if not chat:
        conn.close()
        return None
    msgs = conn.execute(
        "SELECT role, content FROM messages WHERE chat_id=? ORDER BY id", (chat_id,)
    ).fetchall()
    conn.close()
    result = dict(chat)
    result["messages"] = [dict(m) for m in msgs]
    return result

def create_chat(*, chat_id=None, title="", pinned=False, model="claude-sonnet-4-6",
                created_at=None, updated_at=None):
    cid = chat_id or new_id()
    now = time.time()
    ca = created_at or now
    ua = updated_at or now
    conn = _connect()
    conn.execute(
        "INSERT INTO chats (id, title, pinned, model, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        (cid, title, int(pinned), model, ca, ua),
    )
    conn.commit()
    conn.close()
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
        sets.append(f"{k}=?")
        vals.append(v)
    if not sets:
        return
    vals.append(chat_id)
    conn = _connect()
    conn.execute(f"UPDATE chats SET {', '.join(sets)} WHERE id=?", vals)
    conn.commit()
    conn.close()

def delete_chat(chat_id: str):
    conn = _connect()
    conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    conn.commit()
    conn.close()

def add_message(chat_id: str, role: str, content: str):
    now = time.time()
    conn = _connect()
    conn.execute(
        "INSERT INTO messages (chat_id, role, content, ts) VALUES (?,?,?,?)",
        (chat_id, role, content, now),
    )
    conn.execute("UPDATE chats SET updated_at=? WHERE id=?", (now, chat_id))
    conn.commit()
    conn.close()

def get_messages(chat_id: str):
    conn = _connect()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE chat_id=? ORDER BY id", (chat_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def import_chat(chat_id: str, title: str, pinned: bool, model: str,
                created_at: float, updated_at: float, messages: list[dict]):
    conn = _connect()
    conn.execute(
        "INSERT OR IGNORE INTO chats (id, title, pinned, model, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        (chat_id, title, int(pinned), model, created_at, updated_at),
    )
    for m in messages:
        conn.execute(
            "INSERT INTO messages (chat_id, role, content, ts) VALUES (?,?,?,?)",
            (chat_id, m["role"], m["content"], updated_at),
        )
    conn.commit()
    conn.close()
