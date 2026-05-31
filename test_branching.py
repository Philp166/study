"""
Тесты ветвления диалога (дерево сообщений) и трёх branch-aware стратегий.

Каждый тест работает на изолированной временной SQLite-БД: db.DB_PATH
подменяется через monkeypatch, db.init_db() прогоняет миграцию на чистом файле.
"""

import sqlite3

import pytest

import db


# ── fixtures ──

@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Свежая мигрированная БД на каждый тест."""
    dbfile = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", dbfile)
    db.init_db()
    return dbfile


def _new_chat():
    return db.create_chat(title="t", model="claude-sonnet-4-6")


def _linear(chat, *contents):
    """Добавляет цепочку сообщений в активную ветку, возвращает список id."""
    ids = []
    role = "user"
    for c in contents:
        ids.append(db.add_message(chat, role, c))
        role = "assistant" if role == "user" else "user"
    return ids


# ── 1. Линейное поведение (дерево == плоская история) ──

def test_linear_chain_preserves_order(fresh_db):
    chat = _new_chat()
    m1, m2, m3, m4 = _linear(chat, "q1", "a1", "q2", "a2")

    assert db.get_current_leaf(chat) == m4

    hist = db.get_branch_history(chat, m4)
    assert [h["id"] for h in hist] == [m1, m2, m3, m4]
    assert [h["content"] for h in hist] == ["q1", "a1", "q2", "a2"]
    assert [h["role"] for h in hist] == ["user", "assistant", "user", "assistant"]

    # каждый узел знает родителя; корень — NULL
    assert hist[0]["parent_message_id"] is None
    assert hist[1]["parent_message_id"] == m1
    assert hist[3]["parent_message_id"] == m3


# ── 2. Ветвление + переключение ──

def test_branch_creates_sibling_and_switch(fresh_db):
    chat = _new_chat()
    m1, m2, m3, m4 = _linear(chat, "q1", "a1", "q2", "a2")

    # развилка от a1 (m2): альтернативный вопрос
    m5 = db.create_branch_message(chat, m2, "q2-alt")
    m6 = db.add_message(chat, "assistant", "a2-alt")  # чейнится к leaf=m5

    # m2 теперь точка ветвления: дети m3 и m5
    tree = db.get_branch_tree(chat)
    kids_of_m2 = sorted(t["id"] for t in tree if t["parent_message_id"] == m2)
    assert kids_of_m2 == [m3, m5]

    # активная ветка = m1,m2,m5,m6
    assert [h["id"] for h in db.get_branch_history(chat, m6)] == [m1, m2, m5, m6]
    # старая ветка по-прежнему достижима
    assert [h["id"] for h in db.get_branch_history(chat, m4)] == [m1, m2, m3, m4]

    # навигатор: переключение на ребёнка m3 спускается до листа m4
    assert db.descend_to_leaf(chat, m3) == m4
    assert db.descend_to_leaf(chat, m5) == m6

    db.set_current_leaf(chat, db.descend_to_leaf(chat, m3))
    assert db.get_current_leaf(chat) == m4


# ── 3. Sticky Facts: наследование по ветке, ближайший к листу выигрывает ──

def test_facts_inheritance_nearest_wins_and_sibling_isolation(fresh_db):
    chat = _new_chat()
    m1 = db.add_message(chat, "user", "привет")
    db.add_chat_fact(chat, "name", "Алиса", m1)

    m2 = db.add_message(chat, "assistant", "ok")
    m3 = db.add_message(chat, "user", "дедлайн 15 июня")
    db.add_chat_fact(chat, "deadline", "15 июня", m3)

    facts = {f["key"]: f["value"] for f in db.get_active_facts(chat, m3)}
    assert facts == {"name": "Алиса", "deadline": "15 июня"}

    # переопределение ближе к листу: name=Боб на m3 перебивает name=Алиса на m1
    db.add_chat_fact(chat, "name", "Боб", m3)
    facts = {f["key"]: f["value"] for f in db.get_active_facts(chat, m3)}
    assert facts["name"] == "Боб"

    # соседняя ветка от m1: видит только name=Алиса (deadline и Боб не на пути)
    m4 = db.create_branch_message(chat, m1, "другой вопрос")
    facts_sib = {f["key"]: f["value"] for f in db.get_active_facts(chat, m4)}
    assert facts_sib == {"name": "Алиса"}


def test_fact_tombstone_hides_inherited_key(fresh_db):
    chat = _new_chat()
    m1 = db.add_message(chat, "user", "привет")
    db.add_chat_fact(chat, "name", "Алиса", m1)
    m2 = db.add_message(chat, "assistant", "ok")
    m3 = db.add_message(chat, "user", "забудь имя")

    db.tombstone_chat_fact(chat, "name", m3)
    facts = db.get_active_facts(chat, m3)
    assert all(f["key"] != "name" for f in facts)

    # но в соседней ветке (без надгробия) имя живо
    m4 = db.create_branch_message(chat, m1, "ещё вопрос")
    facts_sib = {f["key"]: f["value"] for f in db.get_active_facts(chat, m4)}
    assert facts_sib.get("name") == "Алиса"


# ── 4. Rolling Summary: anchor должен лежать на пути ветки ──

def test_summary_must_be_on_path(fresh_db):
    chat = _new_chat()
    m1, m2, m3, m4 = _linear(chat, "q1", "a1", "q2", "a2")

    db.save_summary(chat, "S-mid", m3)
    s = db.get_active_summary(chat, m4)
    assert s is not None and s["summary_text"] == "S-mid"
    assert s["branch_anchor_message_id"] == m3

    # ближайший к листу выигрывает
    db.save_summary(chat, "S-root", m1)
    s = db.get_active_summary(chat, m4)
    assert s["summary_text"] == "S-mid"  # m3 > m1, оба на пути

    # ветка от m1: summary с anchor=m3 НЕ на пути → игнор; остаётся S-root
    m5 = db.create_branch_message(chat, m1, "alt")
    s_sib = db.get_active_summary(chat, m5)
    assert s_sib is not None and s_sib["summary_text"] == "S-root"


def test_summary_off_path_ignored_returns_none(fresh_db):
    chat = _new_chat()
    m1 = db.add_message(chat, "user", "q1")
    m2 = db.add_message(chat, "assistant", "a1")
    db.save_summary(chat, "S-deep", m2)

    # ветка от корня m1 — m2 не на её пути
    m3 = db.create_branch_message(chat, m1, "alt")
    assert db.get_active_summary(chat, m3) is None


# ── 5. Каскадное удаление поддерева ──

def test_delete_subtree_on_active_path_moves_leaf_to_parent(fresh_db):
    chat = _new_chat()
    m1, m2, m3, m4 = _linear(chat, "q1", "a1", "q2", "a2")

    res = db.delete_message_subtree(m3)
    assert res["chat_id"] == chat
    assert res["new_leaf"] == m2
    assert db.get_current_leaf(chat) == m2

    remaining = {t["id"] for t in db.get_branch_tree(chat)}
    assert remaining == {m1, m2}  # m3, m4 удалены каскадом


def test_delete_subtree_off_path_keeps_active_leaf(fresh_db):
    chat = _new_chat()
    m1, m2, m3, m4 = _linear(chat, "q1", "a1", "q2", "a2")
    m5 = db.create_branch_message(chat, m2, "q2-alt")
    m6 = db.add_message(chat, "assistant", "a2-alt")
    assert db.get_current_leaf(chat) == m6

    # удаляем m3 (вне активной ветки m1,m2,m5,m6)
    res = db.delete_message_subtree(m3)
    assert res["new_leaf"] == m6  # активный leaf не тронут
    remaining = {t["id"] for t in db.get_branch_tree(chat)}
    assert remaining == {m1, m2, m5, m6}


# ── 6. Краевые случаи ──

def test_empty_chat_edges(fresh_db):
    chat = _new_chat()
    assert db.get_current_leaf(chat) is None
    assert db.get_branch_history(chat, None) == []
    assert db.get_active_facts(chat, None) == []
    assert db.get_active_summary(chat, None) is None
    got = db.get_chat(chat)
    assert got["messages"] == []
    assert got["tree"] == []


# ── 7. Защита глубины ──

def test_depth_guard_constant_and_deep_chain(fresh_db):
    assert db.MAX_BRANCH_DEPTH == 1000
    chat = _new_chat()
    contents = [f"m{i}" for i in range(60)]
    ids = _linear(chat, *contents)
    hist = db.get_branch_history(chat, ids[-1])
    assert len(hist) == 60
    assert [h["id"] for h in hist] == ids


# ── 8. Миграция legacy-БД (плоская история → дерево) ──

def test_migration_backfills_tree_from_legacy(tmp_path, monkeypatch):
    dbfile = tmp_path / "legacy.db"

    # 1) создаём ОРИГИНАЛЬНУЮ (домиграционную) схему и кладём легаси-данные
    raw = sqlite3.connect(str(dbfile))
    raw.executescript(db._SCHEMA_SQLITE)  # IF NOT EXISTS, без новых колонок
    raw.execute(
        "INSERT INTO chats (id,title,pinned,model,created_at,updated_at) "
        "VALUES ('c1','old',0,'claude-sonnet-4-6',1.0,1.0)"
    )
    msg_ids = []
    for i, (role, content) in enumerate(
        [("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2")]
    ):
        cur = raw.execute(
            "INSERT INTO messages (chat_id,role,content,ts) VALUES (?,?,?,?)",
            ("c1", role, content, float(i)),
        )
        msg_ids.append(cur.lastrowid)
    # старый summary (PK=chat_id), покрывает до 3-го сообщения
    raw.execute(
        "INSERT INTO summaries (chat_id,summary_text,created_at,covers_messages_up_to_id) "
        "VALUES ('c1','old-summary',1.0,?)",
        (msg_ids[2],),
    )
    # старый факт (UNIQUE chat_id,key)
    raw.execute(
        "INSERT INTO chat_facts (chat_id,key,value,created_at,updated_at) "
        "VALUES ('c1','name','Алиса',1.0,1.0)"
    )
    raw.commit()
    raw.close()

    # 2) прогоняем миграцию через публичный init_db
    monkeypatch.setattr(db, "DB_PATH", dbfile)
    db.init_db()

    root, second, third, leaf = msg_ids

    # parent_message_id зачейнен по порядку id; корень — NULL
    hist = db.get_branch_history("c1", leaf)
    assert [h["id"] for h in hist] == msg_ids
    assert hist[0]["parent_message_id"] is None
    assert hist[1]["parent_message_id"] == root
    assert hist[3]["parent_message_id"] == third

    # current_leaf = последнее сообщение
    assert db.get_current_leaf("c1") == leaf

    # summary заякорен на covers_up_to (3-е сообщение) и лежит на пути
    s = db.get_active_summary("c1", leaf)
    assert s is not None and s["summary_text"] == "old-summary"
    assert s["branch_anchor_message_id"] == third

    # факт заякорен на корень и наследуется до листа
    facts = {f["key"]: f["value"] for f in db.get_active_facts("c1", leaf)}
    assert facts == {"name": "Алиса"}


def test_migration_is_idempotent(fresh_db):
    """Повторный init_db на уже мигрированной БД не ломает данные."""
    chat = _new_chat()
    ids = _linear(chat, "q1", "a1", "q2")
    before = [h["id"] for h in db.get_branch_history(chat, ids[-1])]

    db.init_db()  # второй прогон миграции
    after = [h["id"] for h in db.get_branch_history(chat, ids[-1])]
    assert before == after
    assert db.get_current_leaf(chat) == ids[-1]
