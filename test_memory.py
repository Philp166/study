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
import tempfile
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-for-tests")
os.environ.setdefault("COS_DB_PATH", str(Path(tempfile.gettempdir()) / "cos_apptest_import.db"))

import pytest

import agent
import app
import db


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
    assert "не закрыл" in app._tool_result_msg(
        {"task": False, "task_status": "not_found", "memory": False, "memory_status": None})
    assert app._tool_result_msg(
        {"task": False, "task_status": None, "memory": True, "memory_status": "merged"}
    ).startswith("Готово")
