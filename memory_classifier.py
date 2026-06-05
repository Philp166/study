"""Воркер инструмента `remember` (главный агент → классификатор).

Когда агент по явной просьбе пользователя вызывает `remember`, этот модуль
превращает его текст (+ контекст разговора + существующую память + активные задачи)
в структурированные операции над БД: заметка памяти и/или задача. Один синхронный
вызов Haiku, fail-safe: нет конкретики → {"task": None, "memory": None}.
"""

import json
import logging
import re

from prompts.memory_classifier import MEMORY_CLASSIFIER_SYSTEM_PROMPT

log = logging.getLogger("cos.memory")

COMPOSER_MODEL = "claude-haiku-4-5"

_TASK_ACTIONS = {"create", "update", "close"}
_MEMORY_ACTIONS = {"create", "update"}

_MSG_CAP = 1000      # макс. символов на сообщение контекста
_RECENT_LIMIT = 12   # сколько последних сообщений отдаём


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _valid_task(t):
    if not isinstance(t, dict):
        return None
    action = t.get("action")
    if action not in _TASK_ACTIONS:
        log.error("memory composer: bad task action %r", action)
        return None
    if action in ("update", "close") and not t.get("id"):
        log.error("memory composer: task %s without id", action)
        return None
    if action == "create" and not (t.get("title") or "").strip():
        log.error("memory composer: task create without title")
        return None
    return t


def _valid_memory(m):
    if not isinstance(m, dict):
        return None
    action = m.get("action")
    if action not in _MEMORY_ACTIONS:
        log.error("memory composer: bad memory action %r", action)
        return None
    if action == "update" and not m.get("id"):
        log.error("memory composer: memory update without id")
        return None
    if action == "create" and not (m.get("title") or "").strip():
        log.error("memory composer: memory create without title")
        return None
    return m


def _build_input(save_text, recent_messages, existing_memory, active_tasks) -> str:
    parts = [f"<SAVE_REQUEST>\n{save_text}\n</SAVE_REQUEST>"]
    if recent_messages:
        lines = []
        for mmsg in recent_messages[-_RECENT_LIMIT:]:
            label = "Пользователь" if mmsg.get("role") == "user" else "CoS"
            content = (mmsg.get("content") or "")[:_MSG_CAP]
            lines.append(f"{label}: {content}")
        parts.append("<RECENT_CONTEXT>\n" + "\n".join(lines) + "\n</RECENT_CONTEXT>")
    parts.append("<EXISTING_MEMORY>\n" + json.dumps(existing_memory or [], ensure_ascii=False) + "\n</EXISTING_MEMORY>")
    parts.append("<ACTIVE_TASKS>\n" + json.dumps(active_tasks or [], ensure_ascii=False) + "\n</ACTIVE_TASKS>")
    return "\n\n".join(parts)


def compose_save(client, save_text, recent_messages=None, existing_memory=None,
                 active_tasks=None) -> dict:
    """Структурирует явный запрос на сохранение. Возвращает {"task": ..|None,
    "memory": ..|None}. Fail-safe: при любой ошибке/нехватке данных — оба None."""
    null = {"task": None, "memory": None}
    if not (save_text or "").strip():
        return null

    user_input = _build_input(save_text, recent_messages, existing_memory, active_tasks)
    try:
        response = client.messages.create(
            model=COMPOSER_MODEL,
            max_tokens=1024,
            temperature=0,
            system=MEMORY_CLASSIFIER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_input}],
        )
        raw = "".join(b.text for b in response.content if b.type == "text")
    except Exception:
        log.exception("memory composer: API call failed")
        return null

    try:
        data = json.loads(_strip_fences(raw))
    except (json.JSONDecodeError, ValueError):
        log.error("memory composer: invalid JSON. Raw response: %s", raw)
        return null
    if not isinstance(data, dict):
        log.error("memory composer: response is not an object: %s", data)
        return null

    return {
        "task": _valid_task(data["task"]) if data.get("task") else None,
        "memory": _valid_memory(data["memory"]) if data.get("memory") else None,
    }
