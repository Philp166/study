"""Субагент-классификатор памяти (Фаза 3).

После сообщения пользователя решает, есть ли что предложить к сохранению в
долговременную память / буфер задач, и подтверждает ли пользователь словами
ранее показанное предложение. Один синхронный вызов Haiku (temperature=0),
fail-open: любая ошибка → ничего не предлагаем. Аналог extract_and_apply_facts
в summarizer.py, но НЕ пишет в БД сам и НЕ трогает token_stats главного агента.
"""

import json
import logging
import re

from prompts.memory_classifier import MEMORY_CLASSIFIER_SYSTEM_PROMPT

log = logging.getLogger("cos.memory")

CLASSIFIER_MODEL = "claude-haiku-4-5"

_TASK_ACTIONS = {"create", "update", "close"}
_MEMORY_ACTIONS = {"create", "update"}


def _null() -> dict:
    return {"task": None, "memory": None, "confirm_pending": False}


def _strip_fences(text: str) -> str:
    """Срезает markdown-обёртку ```json ... ```, если модель её добавила."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _build_user_input(user_message, active_tasks, relevant_memory, pending_suggestion) -> str:
    parts = [
        f"<USER_MESSAGE>\n{user_message}\n</USER_MESSAGE>",
        "<ACTIVE_TASKS>\n" + json.dumps(active_tasks, ensure_ascii=False) + "\n</ACTIVE_TASKS>",
        "<EXISTING_MEMORY>\n" + json.dumps(relevant_memory, ensure_ascii=False) + "\n</EXISTING_MEMORY>",
    ]
    if pending_suggestion:
        parts.append(
            "<PENDING_SUGGESTION>\n" + json.dumps(pending_suggestion, ensure_ascii=False) + "\n</PENDING_SUGGESTION>"
        )
    return "\n\n".join(parts)


def _valid_task(t):
    if not isinstance(t, dict):
        return None
    action = t.get("action")
    if action not in _TASK_ACTIONS:
        log.error("classifier: bad task action %r", action)
        return None
    if action in ("update", "close") and not t.get("id"):
        log.error("classifier: task %s without id", action)
        return None
    if action == "create" and not (t.get("title") or "").strip():
        log.error("classifier: task create without title")
        return None
    return t


def _valid_memory(m):
    if not isinstance(m, dict):
        return None
    action = m.get("action")
    if action not in _MEMORY_ACTIONS:
        log.error("classifier: bad memory action %r", action)
        return None
    if action == "update" and not m.get("id"):
        log.error("classifier: memory update without id")
        return None
    if action == "create" and not (m.get("title") or "").strip():
        log.error("classifier: memory create without title")
        return None
    return m


def classify_message(client, user_message, active_tasks, relevant_memory,
                     pending_suggestion=None) -> dict:
    """Возвращает {"task": ..|None, "memory": ..|None, "confirm_pending": bool}.
    Fail-open: при любой ошибке — всё None/False."""
    if not user_message or not user_message.strip():
        return _null()

    user_input = _build_user_input(user_message, active_tasks, relevant_memory, pending_suggestion)
    try:
        response = client.messages.create(
            model=CLASSIFIER_MODEL,
            max_tokens=1024,
            temperature=0,
            system=MEMORY_CLASSIFIER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_input}],
        )
        raw = "".join(b.text for b in response.content if b.type == "text")
    except Exception:
        log.exception("classifier: API call failed")
        return _null()

    try:
        data = json.loads(_strip_fences(raw))
    except (json.JSONDecodeError, ValueError):
        log.error("classifier: invalid JSON. Raw response: %s", raw)
        return _null()
    if not isinstance(data, dict):
        log.error("classifier: response is not an object: %s", data)
        return _null()

    return {
        "task": _valid_task(data["task"]) if data.get("task") else None,
        "memory": _valid_memory(data["memory"]) if data.get("memory") else None,
        "confirm_pending": bool(data.get("confirm_pending")),
    }
