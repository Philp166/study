"""Тесты детерминированного ядра памяти (План доработок v2 — Этап 0).

Изолированная временная SQLite-БД на каждый тест: db.DB_PATH подменяется через
monkeypatch, db.init_db() прогоняет миграцию на чистом файле (как в test_branching).

Покрываем то, что чинит баг B и тихую потерю данных на db-слое:
  - relevant_memory: нет фолбэка «последние свежие» при 0 совпадений + порог ≥3;
  - merge_memory_content: при коллизии title данные дописываются, а не теряются.
Scope-правило связей и _RECENT_LIMIT — поведение LLM/формата, проверяется вживую.
"""

import os

# app.py при импорте требует ANTHROPIC_API_KEY и прогоняет db.init_db() на COS_DB_PATH.
# Ставим заглушки ДО импорта app: dummy-ключ (клиент не ходит в сеть при создании
# объекта) и временный COS_DB_PATH (реальная per-тест БД всё равно подменяется
# фикстурой fresh_db через monkeypatch db.DB_PATH). Локально БД — SQLite.
import sqlite3
import tempfile
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-for-tests")
os.environ.setdefault("COS_DB_PATH", str(Path(tempfile.gettempdir()) / "cos_apptest_import.db"))

import pytest

import agent
import app
import db
import summarizer


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Свежая мигрированная БД на каждый тест."""
    dbfile = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", dbfile)
    db.init_db()
    return dbfile


# ── relevant_memory: нет фолбэка при 0 совпадений ──

def test_relevant_memory_no_match_returns_empty(fresh_db):
    """Канал бага B: при 0 совпадений раньше подставлялись «последние свежие»
    (несвязанная заметка про Алёну прилипала к запросу про другое). Теперь — []."""
    db.create_memory(title="Алёна Котова", content="коллега с прошлой работы")
    assert db.relevant_memory("купить молоко завтра") == []


def test_relevant_memory_empty_query_returns_empty(fresh_db):
    db.create_memory(title="Что-то", content="контент")
    assert db.relevant_memory("") == []
    assert db.relevant_memory("   ") == []


def test_relevant_memory_matches_on_keyword(fresh_db):
    db.create_memory(title="Проект Офис", content="переезд в новый офис")
    res = db.relevant_memory("что там по проекту офис")
    assert [r["title"] for r in res] == ["Проект Офис"]


# ── relevant_memory: порог длины слова ≥3 (был ≥4) ──

def test_relevant_memory_three_letter_word_matches(fresh_db):
    """Слово из 3 букв (аббревиатура/короткое имя) теперь участвует в поиске."""
    note = db.create_memory(title="API", content="наш REST-сервис")
    res = db.relevant_memory("как там api")
    assert [r["id"] for r in res] == [note["id"]]


def test_relevant_memory_two_letter_word_ignored(fresh_db):
    """Слова короче 3 букв по-прежнему отбрасываются → нет слов → []
    (а не фолбэк на свежие)."""
    db.create_memory(title="ок", content="ок")
    assert db.relevant_memory("ок") == []


def test_relevant_memory_stopwords_only_returns_empty(fresh_db):
    """Запрос только из служебных слов → нет содержательных слов → []
    (порог ≥3 не должен тянуть «что/как/для» в поиск)."""
    db.create_memory(title="API", content="сервис")
    assert db.relevant_memory("что как для это") == []


def test_relevant_memory_stopwords_dropped_keeps_content_word(fresh_db):
    """Служебные слова отфильтрованы, матчит только содержательное слово:
    «для» не должно подтянуть заметку «Для дома», ищем по «api»."""
    api = db.create_memory(title="API", content="наш сервис")
    db.create_memory(title="Для дома", content="бытовое")
    res = db.relevant_memory("что там для api")
    assert [r["id"] for r in res] == [api["id"]]


# ── merge_memory_content: коллизия title не теряет данные ──

def test_create_memory_collision_returns_none(fresh_db):
    """Базовый инвариант: дубль active-title → None (=409)."""
    db.create_memory(title="Сервер", content="prod-сервер")
    assert db.create_memory(title="Сервер", content="ещё раз") is None


def test_merge_appends_to_existing(fresh_db):
    row = db.create_memory(title="Сервер", content="prod-сервер")
    merged, status = db.merge_memory_content("Сервер", "добавил мониторинг")
    assert status == "merged"
    assert merged["id"] == row["id"]
    assert merged["content"] == "prod-сервер\n\nдобавил мониторинг"


def test_merge_duplicate_is_noop(fresh_db):
    db.create_memory(title="Сервер", content="prod-сервер")
    # тот же текст уже содержится → не дублируем
    merged, status = db.merge_memory_content("Сервер", "prod-сервер")
    assert status == "duplicate"
    assert merged["content"] == "prod-сервер"


def test_merge_empty_content_is_noop(fresh_db):
    db.create_memory(title="Сервер", content="prod-сервер")
    merged, status = db.merge_memory_content("Сервер", "   ")
    assert status == "duplicate"
    assert merged["content"] == "prod-сервер"


def test_merge_absent_title_returns_none(fresh_db):
    merged, status = db.merge_memory_content("Нет такой заметки", "что-то")
    assert merged is None
    assert status == "absent"


def test_merge_only_active_notes(fresh_db):
    """Мягко удалённая (inactive) заметка не считается коллизией — merge её не видит."""
    row = db.create_memory(title="Старая", content="архив")
    db.update_memory(row["id"], status="inactive")
    merged, status = db.merge_memory_content("Старая", "новое")
    assert merged is None and status == "absent"
    # и create с тем же title теперь снова возможен
    assert db.create_memory(title="Старая", content="свежая") is not None


# ── Этап 1: build_suggestion (чистый маппер типизированных инструментов) ──

def test_build_suggestion_save_memory_full():
    sug = agent.build_suggestion(
        "save_memory", {"title": " Сервер ", "content": " prod-узел ", "folder": " Инфра "})
    assert sug == {"task": None, "project": None, "memory": {
        "action": "create", "title": "Сервер", "content": "prod-узел", "folder": "Инфра"}}


def test_build_suggestion_save_memory_no_folder():
    sug = agent.build_suggestion("save_memory", {"title": "X", "content": "y"})
    assert sug["memory"] == {"action": "create", "title": "X", "content": "y"}
    assert "folder" not in sug["memory"]


def test_build_suggestion_blank_folder_dropped():
    sug = agent.build_suggestion("save_memory", {"title": "X", "content": "y", "folder": "   "})
    assert "folder" not in sug["memory"]


def test_build_suggestion_save_task():
    sug = agent.build_suggestion("save_task", {"title": "Диагностика", "context": "до пятницы"})
    assert sug == {"task": {"action": "create", "title": "Диагностика", "context": "до пятницы"},
                   "memory": None, "project": None}


def test_build_suggestion_close_task():
    sug = agent.build_suggestion("close_task", {"title": "Диагностика", "outcome": "готово"})
    assert sug == {"task": {"action": "close", "title": "Диагностика", "outcome": "готово"},
                   "memory": None, "project": None}


def test_build_suggestion_requires_title():
    assert agent.build_suggestion("save_memory", {"content": "y"}) is None
    assert agent.build_suggestion("save_task", {"title": "   "}) is None
    assert agent.build_suggestion("close_task", {}) is None
    assert agent.build_suggestion("save_memory", None) is None


def test_build_suggestion_unknown_tool():
    assert agent.build_suggestion("remember", {"title": "X", "content": "y"}) is None


# ── Этап 1: find_active_task_by_title (резолв close/update по названию) ──

def test_find_active_task_by_title_normalized(fresh_db):
    t = db.create_task(title="Диагностика сети", context="x")
    assert db.find_active_task_by_title("  диагностика СЕТИ ")["id"] == t["id"]


def test_find_active_task_by_title_absent(fresh_db):
    db.create_task(title="Другая")
    assert db.find_active_task_by_title("Нет такой") is None
    assert db.find_active_task_by_title("   ") is None


def test_find_active_task_by_title_only_active(fresh_db):
    t = db.create_task(title="Закрыть")
    db.update_task(t["id"], status="completed", outcome="done")
    assert db.find_active_task_by_title("Закрыть") is None


def test_find_active_task_by_title_freshest_wins(fresh_db):
    t1 = db.create_task(title="Дубль")
    t2 = db.create_task(title="дубль")        # тот же нормализованный title
    db.update_task(t1["id"], context="bump")  # делаем t1 свежее детерминированно
    assert db.find_active_task_by_title("Дубль")["id"] == t1["id"]


def test_build_suggestion_optional_fields_default_empty():
    assert agent.build_suggestion("save_task", {"title": "T"})["task"]["context"] == ""
    assert agent.build_suggestion("close_task", {"title": "T"})["task"]["outcome"] == ""
    assert agent.build_suggestion("save_memory", {"title": "T"})["memory"]["content"] == ""


# ── Этап 1: _apply_suggestion как исполнитель (через app, dummy-окружение) ──

def test_apply_close_by_title_resolves_and_closes(fresh_db):
    t = db.create_task(title="Диагностика", context="x")
    saved = app._apply_suggestion(
        {"action": "close", "title": "  диагностика ", "outcome": "готово"}, None)
    assert saved["task"] is True and saved["task_status"] == "closed"
    closed = db.get_task(t["id"])
    assert closed["status"] == "completed" and closed["outcome"] == "готово"


def test_apply_close_not_found_is_honest(fresh_db):
    saved = app._apply_suggestion({"action": "close", "title": "Нет такой"}, None)
    assert saved["task"] is False and saved["task_status"] == "not_found"


def test_apply_save_memory_create_then_merge(fresh_db):
    s1 = app._apply_suggestion(None, {"action": "create", "title": "Сервер", "content": "prod"})
    assert s1["memory"] is True and s1["memory_status"] == "created"
    s2 = app._apply_suggestion(None, {"action": "create", "title": "Сервер", "content": "мониторинг"})
    assert s2["memory"] is True and s2["memory_status"] == "merged"
    note = db.list_memory(q="Сервер")[0]
    assert "prod" in note["content"] and "мониторинг" in note["content"]


def test_apply_update_without_resolvable_target_is_noop(fresh_db):
    # update по несуществующему title → ничего не меняем, честный not_found
    saved = app._apply_suggestion({"action": "update", "title": "Несуществующая"}, None)
    assert saved["task"] is False and saved["task_status"] == "not_found"


def test_tool_result_msg_honest():
    assert "закрыл задачу" in app._tool_result_msg(
        {"task": True, "task_status": "closed", "memory": False, "memory_status": None})
    assert "не закрыл" in app._tool_result_msg(
        {"task": False, "task_status": "not_found", "memory": False, "memory_status": None})
    assert app._tool_result_msg(
        {"task": False, "task_status": None, "memory": True, "memory_status": "merged"}
    ).startswith("Готово")


# ── Этап 2a: миграции (projects, memory_links, project_id) ──

def test_migration_creates_tables_and_columns(fresh_db):
    with sqlite3.connect(str(fresh_db)) as raw:
        tables = {r[0] for r in raw.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert {"projects", "memory_links"} <= tables
        ltm_cols = {r[1] for r in raw.execute("PRAGMA table_info(long_term_memory)").fetchall()}
        task_cols = {r[1] for r in raw.execute("PRAGMA table_info(tasks)").fetchall()}
        assert "project_id" in ltm_cols and "project_id" in task_cols


def test_migration_idempotent(fresh_db):
    p = db.create_project(title="Компьютер")
    db.init_db()  # повторный прогон не должен ломать данные/схему
    assert db.get_project(p["id"])["title"] == "Компьютер"


# ── Этап 2a: projects CRUD ──

def test_project_create_unique_and_lookup(fresh_db):
    p = db.create_project(title="Офис", description="переезд")
    assert p["title"] == "Офис" and p["status"] == "active"
    assert db.create_project(title="Офис") is None        # точный дубль title → None
    assert db.get_project_by_title("  офис ")["id"] == p["id"]   # резолв нормализованный
    assert db.create_project(title="   ") is None          # пустой title → None
    assert [pr["id"] for pr in db.list_projects()] == [p["id"]]


def test_project_update(fresh_db):
    p = db.create_project(title="Офис")
    db.update_project(p["id"], status="archived")
    assert db.list_projects() == []                        # active по умолчанию
    assert db.get_project(p["id"])["status"] == "archived"


# ── Этап 2a: материализованный граф memory_links ──

def test_links_ghost_then_backlink_resolves(fresh_db):
    a = db.create_memory(title="Алёна", content="коллега из [[Проект Икс]]")
    links = db.get_memory_links()
    assert len(links) == 1
    assert links[0]["source_id"] == a["id"]
    assert links[0]["target_title"] == "Проект Икс"
    assert links[0]["target_id"] is None                  # ghost: целевой заметки ещё нет
    # появилась целевая заметка → бэклинк резолвится
    x = db.create_memory(title="Проект Икс", content="описание")
    links = db.get_memory_links()
    link = [l for l in links if l["source_id"] == a["id"]][0]
    assert link["target_id"] == x["id"]


def test_links_pruned_on_content_change(fresh_db):
    a = db.create_memory(title="A", content="[[Б]] и [[В]]")
    assert len(db.get_memory_links()) == 2
    db.update_memory(a["id"], content="только [[Б]]")
    links = db.get_memory_links()
    assert [l["target_title"] for l in links] == ["Б"]


def test_links_reresolve_on_rename(fresh_db):
    a = db.create_memory(title="A", content="см. [[Сервер]]")
    s = db.create_memory(title="Сервер", content="prod")
    assert [l for l in db.get_memory_links() if l["source_id"] == a["id"]][0]["target_id"] == s["id"]
    # переименование цели → связь A снова ghost
    db.update_memory(s["id"], title="Сервер 2")
    link = [l for l in db.get_memory_links() if l["source_id"] == a["id"]][0]
    assert link["target_id"] is None


def test_links_ghost_when_target_soft_deleted(fresh_db):
    a = db.create_memory(title="A", content="[[Б]]")
    b = db.create_memory(title="Б", content="x")
    assert [l for l in db.get_memory_links() if l["source_id"] == a["id"]][0]["target_id"] == b["id"]
    db.update_memory(b["id"], status="inactive")          # мягкое удаление
    link = [l for l in db.get_memory_links() if l["source_id"] == a["id"]][0]
    assert link["target_id"] is None                      # target неактивен → ghost


def test_links_source_inactive_hidden(fresh_db):
    a = db.create_memory(title="A", content="[[Б]]")
    db.update_memory(a["id"], status="inactive")
    assert db.get_memory_links() == []                    # source неактивен → связь не показываем


def test_links_hard_delete_cascades(fresh_db):
    a = db.create_memory(title="A", content="[[Б]]")
    db.create_memory(title="Б", content="x")
    assert len(db.get_memory_links()) == 1
    db.delete_memory(a["id"])                              # FK ON DELETE CASCADE
    assert db.get_memory_links() == []


def test_links_backfill_from_existing_notes(tmp_path, monkeypatch):
    """memory_links отсутствовала → миграция создаёт её и бэкфиллит из заметок."""
    dbfile = tmp_path / "bf.db"
    raw = sqlite3.connect(str(dbfile))
    raw.executescript(db._SCHEMA_SQLITE)   # старые таблицы, без memory_links/projects
    for title, content in [("Алёна", "коллега [[Проект Икс]]"), ("Проект Икс", "описание")]:
        raw.execute(
            "INSERT INTO long_term_memory (title,content,folder,status,created_at,updated_at) "
            "VALUES (?,?,NULL,'active',1.0,1.0)", (title, content))
    raw.commit()
    raw.close()

    monkeypatch.setattr(db, "DB_PATH", dbfile)
    db.init_db()                            # миграция: создаёт memory_links + бэкфилл

    links = db.get_memory_links()
    assert len(links) == 1
    assert links[0]["source_title"] == "Алёна"
    assert links[0]["target_title"] == "Проект Икс"
    assert links[0]["target_id"] is not None    # бэкфилл + резолв связал target


# ── Этап 2b: проекты — build_suggestion ──

def test_build_suggestion_create_project():
    sug = agent.build_suggestion("create_project", {"title": " Компьютер ", "description": " сборка "})
    assert sug == {"task": None, "memory": None,
                   "project": {"action": "create", "title": "Компьютер", "description": "сборка"}}


def test_build_suggestion_project_field():
    m = agent.build_suggestion("save_memory", {"title": "T", "content": "c", "project": " Офис "})
    assert m["memory"]["project"] == "Офис"
    t = agent.build_suggestion("save_task", {"title": "T", "project": "Офис"})
    assert t["task"]["project"] == "Офис"
    m2 = agent.build_suggestion("save_memory", {"title": "T", "content": "c", "project": "  "})
    assert "project" not in m2["memory"]          # пустой project не добавляется


# ── Этап 2b: _apply_suggestion проекты (закрытие бага A) ──

def test_apply_create_project_slot(fresh_db):
    saved = app._apply_suggestion(None, None, {"action": "create", "title": "Компьютер"})
    assert saved["project"] is True and saved["project_status"] == "created"
    assert db.get_project_by_title("компьютер")["title"] == "Компьютер"
    saved2 = app._apply_suggestion(None, None, {"action": "create", "title": "Компьютер"})
    assert saved2["project_status"] == "exists"   # повторно — не дубль


def test_apply_bug_a_create_project_not_note(fresh_db):
    """Acceptance бага A: «создай проект Компьютер» → строка в projects, НЕ заметка."""
    app._apply_suggestion(None, None, {"action": "create", "title": "Компьютер"})
    assert len(db.list_projects()) == 1
    assert db.list_memory() == []                 # обычной заметки НЕ создано


def test_apply_memory_with_project_resolves_or_creates(fresh_db):
    saved = app._apply_suggestion(
        None, {"action": "create", "title": "Сервер", "content": "prod", "project": "Инфра"}, None)
    assert saved["memory"] is True
    proj = db.get_project_by_title("Инфра")
    assert proj is not None                        # проект создан автоматически
    assert db.list_memory(q="Сервер")[0]["project_id"] == proj["id"]


def test_apply_task_with_project(fresh_db):
    p = db.create_project(title="Инфра")
    app._apply_suggestion({"action": "create", "title": "Чинить", "project": "инфра"}, None, None)
    assert db.find_active_task_by_title("Чинить")["project_id"] == p["id"]


# ── Этап 2b: backfill + delete_project ──

def test_backfill_folders_to_projects(fresh_db):
    db.create_memory(title="A", content="x", folder="Офис")
    db.create_memory(title="B", content="y", folder="Офис")
    db.create_memory(title="C", content="z", folder=None)
    res = db.backfill_folders_to_projects()
    assert res == {"projects_created": 1, "notes_linked": 2}
    proj = db.get_project_by_title("Офис")
    linked = {n["title"] for n in db.list_memory() if n["project_id"] == proj["id"]}
    assert linked == {"A", "B"}
    assert db.backfill_folders_to_projects() == {"projects_created": 0, "notes_linked": 0}  # идемпотентно


def test_delete_project_detaches_notes(fresh_db):
    p = db.create_project(title="Офис")
    n = db.create_memory(title="A", content="x")
    db.update_memory(n["id"], project_id=p["id"])
    assert db.get_memory(n["id"])["project_id"] == p["id"]
    db.delete_project(p["id"])
    assert db.get_memory(n["id"])["project_id"] is None   # FK ON DELETE SET NULL


# ── Этап 2b: API проектов (прямые вызовы ручек) ──

def test_api_projects_crud(fresh_db):
    row = app.api_create_project({"title": "Офис", "description": "переезд"})
    assert isinstance(row, dict) and row["title"] == "Офис"
    dup = app.api_create_project({"title": "Офис"})
    assert getattr(dup, "status_code", None) == 409
    assert any(p["title"] == "Офис" for p in app.api_list_projects())
    bad = app.api_create_project({"title": "  "})
    assert getattr(bad, "status_code", None) == 400


def test_create_project_normalized_dedup(fresh_db):
    p = db.create_project(title="Инфра")
    assert db.create_project(title="  инфра ") is None     # нормализованный дубль не создаём
    assert app._resolve_or_create_project_id("ИНФРА") == p["id"]
    assert len(db.list_projects()) == 1


# ── Этап 3: стемминг ──

def test_stem_inflections_collapse():
    assert db._stem("диагностику") == db._stem("диагностика")
    assert db._stem("проекту") == db._stem("проект") == "проект"
    assert db._stem("офис") == "офис"        # коротких/без окончания не трогаем
    assert db._stem("api") == "api"


def test_query_stems_filters_and_stems():
    stems = db._query_stems("Провёл диагностику по проекту, ок")
    assert "ок" not in stems                  # <3
    assert db._stem("диагностику") in stems
    assert db._stem("проекту") in stems


# ── Этап 3: relevant_memory стемминг + скоринг ──

def test_relevant_memory_matches_inflected_form(fresh_db):
    db.create_memory(title="Диагностика сети", content="проверка узлов")
    res = db.relevant_memory("надо сделать диагностику сети")
    assert [r["title"] for r in res] == ["Диагностика сети"]


def test_relevant_memory_ranks_by_matched_words(fresh_db):
    a = db.create_memory(title="Проект Альфа", content="дедлайн пятница")
    b = db.create_memory(title="Проект Бета", content="встреча")
    res = db.relevant_memory("проект альфа дедлайн")
    # обе содержат «проект», но Альфа совпадает по 3 словам → выше
    assert res[0]["id"] == a["id"]
    assert b["id"] in [r["id"] for r in res]


# ── Этап 3: search_memory (агентный retrieval) ──

def test_search_memory_empty_and_no_match(fresh_db):
    db.create_memory(title="Сервер", content="prod")
    assert db.search_memory("") == []
    assert db.search_memory("совершенно несвязанный зонтик") == []


def test_search_memory_returns_content_project_links(fresh_db):
    p = db.create_project(title="Инфра")
    a = db.create_memory(title="Альфа", content="связана с [[Бета]]")
    db.update_memory(a["id"], project_id=p["id"])
    db.create_memory(title="Бета", content="деталь")
    res = db.search_memory("Альфа")
    item = [r for r in res if r["title"] == "Альфа"][0]
    assert item["content"].startswith("связана")
    assert item["project"] == "Инфра"
    assert "Бета" in item["links"]
    assert item["score"] >= 1


def test_format_search_results():
    assert "ничего не нашёл" in app._format_search_results([])
    txt = app._format_search_results([
        {"title": "X", "content": "тело", "folder": "F", "project": "P", "links": ["Y"]}])
    assert "X" in txt and "проект: P" in txt and "[[Y]]" in txt


# ── Этап 4: голос уважает autosave (pending + устное «да») ──

def test_is_affirmative():
    assert app._is_affirmative("да, сохрани")
    assert app._is_affirmative("Ага")
    assert app._is_affirmative("конечно, давай")
    assert app._is_affirmative("ладно")
    assert not app._is_affirmative("нет, не надо")
    assert not app._is_affirmative("расскажи про погоду")
    # отрицание перекрывает согласие (иначе молчаливая запись против воли)
    assert not app._is_affirmative("не сохраняй")
    assert not app._is_affirmative("конечно нет")
    assert not app._is_affirmative("да не надо")
    assert not app._is_affirmative("передумал")


def test_resolve_voice_pending_applies_on_yes(fresh_db):
    cid = "voice-yes"
    app._VOICE_PENDING[cid] = [
        {"task": None, "project": None,
         "memory": {"action": "create", "title": "Сервер", "content": "prod"}}]
    saved = app._resolve_voice_pending(cid, "да, сохрани")
    assert len(saved) == 1 and saved[0]["memory"] is True
    assert cid not in app._VOICE_PENDING            # очищено
    assert db.list_memory(q="Сервер")               # реально записано


def test_resolve_voice_pending_applies_multiple(fresh_db):
    cid = "voice-multi"
    app._VOICE_PENDING[cid] = [
        {"task": None, "project": None,
         "memory": {"action": "create", "title": "Аня", "content": "дизайнер"}},
        {"task": {"action": "create", "title": "Демо", "context": "в пятницу"},
         "memory": None, "project": None},
    ]
    saved = app._resolve_voice_pending(cid, "ага, давай")
    assert len(saved) == 2
    assert db.list_memory(q="Аня") and db.find_active_task_by_title("Демо")


def test_resolve_voice_pending_drops_on_no(fresh_db):
    cid = "voice-no"
    app._VOICE_PENDING[cid] = [
        {"task": None, "project": None,
         "memory": {"action": "create", "title": "X", "content": "y"}}]
    assert app._resolve_voice_pending(cid, "нет, потом") == []
    assert cid not in app._VOICE_PENDING            # снято (не подтвердили)
    assert db.list_memory() == []                   # не записано


def test_resolve_voice_pending_absent(fresh_db):
    assert app._resolve_voice_pending("nope", "да") == []


# ── Карточка-список: api_apply_suggestion с items[] ──

def test_api_apply_suggestion_items_list(fresh_db):
    body = {"items": [
        {"memory": {"action": "create", "title": "Команда", "content": "Аня, Петя"}},
        {"task": {"action": "create", "title": "Оффер", "context": "среда"}},
        {"project": {"action": "create", "title": "Запуск"}},
    ]}
    res = app.api_apply_suggestion(body)
    assert res["ok"] is True and len(res["saved"]) == 3
    assert db.list_memory(q="Команда") and db.find_active_task_by_title("Оффер")
    assert db.get_project_by_title("Запуск") is not None


def test_api_apply_suggestion_backcompat_single(fresh_db):
    res = app.api_apply_suggestion({"memory": {"action": "create", "title": "X", "content": "y"}})
    assert res["ok"] is True and len(res["saved"]) == 1 and res["saved"][0]["memory"] is True


# ── Этап 4: кап активных задач в инжекте ──

def test_tasks_block_truncation_note():
    tasks = [{"title": f"T{i}", "context": ""} for i in range(3)]
    assert "показаны" in summarizer._render_tasks_block(tasks, total=10)
    assert "из 10" in summarizer._render_tasks_block(tasks, total=10)
    assert "показаны" not in summarizer._render_tasks_block(tasks, total=3)


def test_build_memory_blocks_passes_tasks_total():
    _, tb = summarizer.build_memory_blocks([], [{"title": "T1", "context": ""}], tasks_total=5)
    assert tb is not None and "показаны" in tb


def test_max_injected_tasks_constant():
    assert isinstance(summarizer.MAX_INJECTED_TASKS, int) and summarizer.MAX_INJECTED_TASKS > 0
