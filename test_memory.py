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
    assert sug == {"task": None, "memory": {
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
                   "memory": None}


def test_build_suggestion_close_task():
    sug = agent.build_suggestion("close_task", {"title": "Диагностика", "outcome": "готово"})
    assert sug == {"task": {"action": "close", "title": "Диагностика", "outcome": "готово"},
                   "memory": None}


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
    assert "Не нашёл задачу" in app._tool_result_msg(
        {"task": False, "task_status": "not_found", "memory": False, "memory_status": None})
    assert app._tool_result_msg(
        {"task": False, "task_status": None, "memory": True, "memory_status": "merged"}
    ).startswith("Готово")


# ── Блок 1 (ревью 2026-06-10): update/delete заметок, archive/delete задач ──

def test_build_suggestion_update_memory_full():
    sug = agent.build_suggestion("update_memory", {
        "title": "Сервер", "new_content": "новый текст",
        "new_title": "Прод-сервер", "folder": "Инфра"})
    m = sug["memory"]
    assert m["action"] == "update" and m["title"] == "Сервер"
    assert m["content"] == "новый текст" and m["new_title"] == "Прод-сервер"
    assert m["folder"] == "Инфра" and sug["task"] is None


def test_build_suggestion_update_memory_requires_changes():
    # без единого изменяемого поля предлагать нечего
    assert agent.build_suggestion("update_memory", {"title": "Сервер"}) is None
    assert agent.build_suggestion("update_memory", {"title": "Сервер", "new_content": "  "}) is None


def test_build_suggestion_delete_and_archive_tools():
    assert agent.build_suggestion("delete_memory", {"title": "X"})["memory"] == {
        "action": "delete", "title": "X"}
    assert agent.build_suggestion("archive_task", {"title": "T"})["task"] == {
        "action": "archive", "title": "T"}
    assert agent.build_suggestion("delete_task", {"title": "T"})["task"] == {
        "action": "delete", "title": "T"}


def test_apply_update_memory_by_title_replaces_content(fresh_db):
    db.create_memory(title="Сервер", content="старый", folder="Инфра")
    saved = app._apply_suggestion(None, {"action": "update", "title": " сервер ", "content": "новый"})
    assert saved["memory"] is True and saved["memory_status"] == "updated"
    note = db.find_active_memory_by_title("Сервер")
    assert note["content"] == "новый"
    assert note["folder"] == "Инфра"  # папку не трогали — осталась


def test_apply_update_memory_rename_and_move(fresh_db):
    n = db.create_memory(title="Сервер", content="x")
    saved = app._apply_suggestion(None, {
        "action": "update", "title": "Сервер", "new_title": "Прод", "folder": "Инфра"})
    assert saved["memory_status"] == "updated"
    note = db.get_memory(n["id"])
    assert note["title"] == "Прод" and note["folder"] == "Инфра"


def test_apply_update_memory_not_found(fresh_db):
    saved = app._apply_suggestion(None, {"action": "update", "title": "Нет", "content": "x"})
    assert saved["memory"] is False and saved["memory_status"] == "not_found"


def test_apply_delete_memory_soft(fresh_db):
    n = db.create_memory(title="Лишняя", content="x")
    saved = app._apply_suggestion(None, {"action": "delete", "title": "лишняя"})
    assert saved["memory"] is True and saved["memory_status"] == "deleted"
    assert db.get_memory(n["id"])["status"] == "inactive"     # мягкое: строка жива
    assert db.find_active_memory_by_title("Лишняя") is None   # из active ушла


def test_apply_delete_memory_not_found(fresh_db):
    saved = app._apply_suggestion(None, {"action": "delete", "title": "Нет"})
    assert saved["memory"] is False and saved["memory_status"] == "not_found"


def test_apply_archive_task_works_for_completed(fresh_db):
    t = db.create_task(title="Чек-лист", context="")
    db.update_task(t["id"], status="completed", outcome="готово")
    saved = app._apply_suggestion({"action": "archive", "title": "чек-лист"}, None)
    assert saved["task"] is True and saved["task_status"] == "archived"
    assert db.get_task(t["id"])["status"] == "archived"


def test_apply_delete_task_is_two_stage(fresh_db):
    # живая задача: «удалить» = убрать в архив (восстановима), строка остаётся
    t = db.create_task(title="Мусор", context="")
    s1 = app._apply_suggestion({"action": "delete", "title": "Мусор"}, None)
    assert s1["task"] is True and s1["task_status"] == "archived"
    assert db.get_task(t["id"])["status"] == "archived"
    # повторное удаление уже архивной — безвозвратно
    s2 = app._apply_suggestion({"action": "delete", "title": "Мусор"}, None)
    assert s2["task"] is True and s2["task_status"] == "deleted"
    assert db.get_task(t["id"]) is None


def test_apply_archive_task_not_found(fresh_db):
    saved = app._apply_suggestion({"action": "archive", "title": "Нет"}, None)
    assert saved["task"] is False and saved["task_status"] == "not_found"


def test_find_task_by_title_statuses(fresh_db):
    t = db.create_task(title="Отчёт", context="")
    db.update_task(t["id"], status="completed")
    assert db.find_task_by_title("Отчёт") is None  # среди активных нет
    assert db.find_task_by_title("Отчёт", ("active", "completed"))["id"] == t["id"]


def test_find_active_memory_by_title_normalized(fresh_db):
    n = db.create_memory(title="Прод-сервер", content="x")
    assert db.find_active_memory_by_title("  прод-сервер ")["id"] == n["id"]
    assert db.find_active_memory_by_title("нет такой") is None


def test_resolve_memory_strips_folder_suffix(fresh_db):
    # модель копирует «Title (Папка)» из среза памяти — резолвер должен пережить
    n = db.create_memory(title="Пилотное ревью CoS", content="x", folder="Проекты")
    assert app._resolve_memory_id("Пилотное ревью CoS (Проекты)") == n["id"]
    # но заметка, у которой скобки — часть настоящего title, резолвится точно
    m = db.create_memory(title="Отчёт (черновик)", content="y")
    assert app._resolve_memory_id("Отчёт (черновик)") == m["id"]


def test_tool_result_msg_new_statuses():
    assert "удалил заметку" in app._tool_result_msg(
        {"task": False, "task_status": None, "memory": True, "memory_status": "deleted"})
    assert "архив" in app._tool_result_msg(
        {"task": True, "task_status": "archived", "memory": False, "memory_status": None})
    assert "Не нашёл заметку" in app._tool_result_msg(
        {"task": False, "task_status": None, "memory": False, "memory_status": "not_found"})


# ── Блок 6: персистентный pending (карточка переживает reload/редеплой) ──

def _mk_chat():
    return db.create_chat(title="t")


def test_pending_set_get_clear(fresh_db):
    cid = _mk_chat()
    items = [{"task": None, "memory": {"action": "create", "title": "X", "content": "y"}}]
    db.set_pending_suggestions(cid, items)
    assert db.get_pending_suggestions(cid) == items
    db.clear_pending_suggestions(cid)
    assert db.get_pending_suggestions(cid) == []


def test_pending_replaced_not_appended(fresh_db):
    cid = _mk_chat()
    db.set_pending_suggestions(cid, [{"task": {"action": "create", "title": "A"}, "memory": None}])
    db.set_pending_suggestions(cid, [{"task": {"action": "create", "title": "B"}, "memory": None}])
    pend = db.get_pending_suggestions(cid)
    assert len(pend) == 1 and pend[0]["task"]["title"] == "B"


def test_voice_pending_applies_on_yes(fresh_db):
    cid = _mk_chat()
    db.set_pending_suggestions(cid, [
        {"task": {"action": "create", "title": "Аренда офиса"}, "memory": None}])
    saved = app._resolve_voice_pending(cid, "да, сохраняй")
    assert len(saved) == 1 and saved[0]["task_status"] == "created"
    assert db.find_active_task_by_title("Аренда офиса") is not None
    assert db.get_pending_suggestions(cid) == []  # применённое очищено


def test_voice_pending_survives_topic_change(fresh_db):
    # смена темы больше НЕ выбрасывает подготовленное — карточка остаётся
    cid = _mk_chat()
    items = [{"task": {"action": "create", "title": "Закуп канцелярии"}, "memory": None}]
    db.set_pending_suggestions(cid, items)
    saved = app._resolve_voice_pending(cid, "кстати, какая погода в офисе?")
    assert saved == []
    assert db.get_pending_suggestions(cid) == items  # живо
    assert db.find_active_task_by_title("Закуп канцелярии") is None  # но не применено


def test_voice_pending_cleared_on_explicit_no(fresh_db):
    cid = _mk_chat()
    db.set_pending_suggestions(cid, [
        {"task": {"action": "create", "title": "Лишнее"}, "memory": None}])
    saved = app._resolve_voice_pending(cid, "нет, не надо")
    assert saved == []
    assert db.get_pending_suggestions(cid) == []  # явный отказ очищает
    assert db.find_active_task_by_title("Лишнее") is None


def test_apply_endpoint_clears_pending(fresh_db):
    cid = _mk_chat()
    items = [{"task": None, "memory": {"action": "create", "title": "Офис", "content": "10 чел"}}]
    db.set_pending_suggestions(cid, items)
    r = app.api_apply_suggestion({"items": items, "chat_id": cid})
    assert r["saved"][0]["memory_status"] == "created"
    assert db.get_pending_suggestions(cid) == []


# ── Этап 2a: миграции (memory_links; projects/project_id оставлены инертными) ──

def test_migration_creates_tables_and_columns(fresh_db):
    with sqlite3.connect(str(fresh_db)) as raw:
        tables = {r[0] for r in raw.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "memory_links" in tables           # граф связей
        assert "projects" in tables               # инертная (сущность убрана, схема осталась)


def test_migration_idempotent(fresh_db):
    n = db.create_memory(title="Компьютер", content="сборка")
    db.init_db()  # повторный прогон не должен ломать данные/схему
    assert db.get_memory(n["id"])["title"] == "Компьютер"


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


def test_search_memory_returns_content_and_links(fresh_db):
    db.create_memory(title="Альфа", content="связана с [[Бета]]", folder="Инфра")
    db.create_memory(title="Бета", content="деталь")
    res = db.search_memory("Альфа")
    item = [r for r in res if r["title"] == "Альфа"][0]
    assert item["content"].startswith("связана")
    assert item["folder"] == "Инфра"
    assert "Бета" in item["links"]
    assert item["score"] >= 1


def test_format_search_results():
    assert "ничего не нашёл" in app._format_search_results([])
    txt = app._format_search_results([
        {"title": "X", "content": "тело", "folder": "F", "links": ["Y"]}])
    assert "X" in txt and "папка: F" in txt and "[[Y]]" in txt


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
    db.set_pending_suggestions(cid, [
        {"task": None, "memory": {"action": "create", "title": "Сервер", "content": "prod"}}])
    saved = app._resolve_voice_pending(cid, "да, сохрани")
    assert len(saved) == 1 and saved[0]["memory"] is True
    assert db.get_pending_suggestions(cid) == []    # очищено
    assert db.list_memory(q="Сервер")               # реально записано


def test_resolve_voice_pending_applies_multiple(fresh_db):
    cid = "voice-multi"
    db.set_pending_suggestions(cid, [
        {"task": None, "memory": {"action": "create", "title": "Аня", "content": "дизайнер"}},
        {"task": {"action": "create", "title": "Демо", "context": "в пятницу"}, "memory": None},
    ])
    saved = app._resolve_voice_pending(cid, "ага, давай")
    assert len(saved) == 2
    assert db.list_memory(q="Аня") and db.find_active_task_by_title("Демо")


def test_resolve_voice_pending_drops_on_no(fresh_db):
    cid = "voice-no"
    db.set_pending_suggestions(cid, [
        {"task": None, "memory": {"action": "create", "title": "X", "content": "y"}}])
    assert app._resolve_voice_pending(cid, "нет, потом") == []
    assert db.get_pending_suggestions(cid) == []    # явный отказ очищает
    assert db.list_memory() == []                   # не записано


def test_resolve_voice_pending_absent(fresh_db):
    assert app._resolve_voice_pending("nope", "да") == []


# ── Карточка-список: api_apply_suggestion с items[] ──

def test_api_apply_suggestion_items_list(fresh_db):
    body = {"items": [
        {"memory": {"action": "create", "title": "Команда", "content": "Аня, Петя"}},
        {"task": {"action": "create", "title": "Оффер", "context": "среда"}},
    ]}
    res = app.api_apply_suggestion(body)
    assert res["ok"] is True and len(res["saved"]) == 2
    assert db.list_memory(q="Команда") and db.find_active_task_by_title("Оффер")


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
