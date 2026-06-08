"""Тесты детерминированного ядра памяти (План доработок v2 — Этап 0).

Изолированная временная SQLite-БД на каждый тест: db.DB_PATH подменяется через
monkeypatch, db.init_db() прогоняет миграцию на чистом файле (как в test_branching).

Покрываем то, что чинит баг B и тихую потерю данных на db-слое:
  - relevant_memory: нет фолбэка «последние свежие» при 0 совпадений + порог ≥3;
  - merge_memory_content: при коллизии title данные дописываются, а не теряются.
Scope-правило связей и _RECENT_LIMIT — поведение LLM/формата, проверяется вживую.
"""

import pytest

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
