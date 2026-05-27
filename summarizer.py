"""
Context management — три независимые стратегии сжатия/обогащения контекста.

1. Rolling Summary  — сжимает старые сообщения через Haiku
2. Sliding Window   — срезает историю до последних N сообщений
3. Sticky Facts     — key-value факты, инжектятся в системпромпт
"""

import json
import logging
import re

import anthropic

import db
from agent import MODEL_CONTEXT_WINDOWS
from prompts.summarizer import SUMMARIZER_SYSTEM_PROMPT
from prompts.facts_extractor import FACTS_EXTRACTOR_SYSTEM_PROMPT

log = logging.getLogger(__name__)

SUMMARIZER_MODEL = "claude-haiku-4-5"
FRESH_WINDOW_SIZE = 10
CONTEXT_THRESHOLD_DEFAULT = 0.35


# ── Rolling Summary ──

def get_recent_window(
    all_msgs: list[dict], n: int = FRESH_WINDOW_SIZE
) -> tuple[list[dict], int | None]:
    """Последние n сообщений, начиная с user (не разрывая пару).

    Возвращает (recent_msgs, window_start_id).
    window_start_id — id первого сообщения в окне.
    """
    if not all_msgs:
        return [], None
    if len(all_msgs) <= n:
        return all_msgs, all_msgs[0]["id"]

    idx = len(all_msgs) - n
    if all_msgs[idx]["role"] == "assistant" and idx > 0:
        idx -= 1

    recent = all_msgs[idx:]
    return recent, recent[0]["id"]


def build_system_prompt(
    base_prompt: str,
    summary_text: str | None,
    facts: list[dict] | None = None,
) -> str:
    """base_prompt + summary (если есть) + блок фактов (если есть)."""
    parts = [base_prompt]

    if summary_text:
        parts.append(
            "\n\n# Текущее краткое содержание разговора\n\n" + summary_text
        )

    if facts:
        facts_header = (
            "\n\n# Известные факты\n\n"
            "Ниже структурированные факты о пользователе и текущем диалоге "
            "-- твоя долговременная база знаний об этом юзере, сохраняется "
            "между сообщениями независимо от того, попало ли исходное сообщение "
            "в свежее окно.\n"
            "Используй эти факты как достоверный контекст. Если в фактах указано "
            '`deadline: "15 июня"` -- это не предположение, это подтверждённая '
            "договорённость.\n"
            "Если факты противоречат свежим сообщениям -- свежие сообщения важнее.\n\n"
        )
        lines = [f"- {f['key']}: {f['value']}" for f in facts]
        parts.append(facts_header + "\n".join(lines))

    return "".join(parts)


def _count_tokens(
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    messages: list[dict],
) -> int:
    resp = client.beta.messages.count_tokens(
        model=model,
        system=system_prompt,
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
    )
    return resp.input_tokens


def should_summarize(
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    messages: list[dict],
    threshold_pct: int = 35,
) -> bool:
    """True, если payload превышает threshold_pct % окна модели."""
    window = MODEL_CONTEXT_WINDOWS.get(model, 200_000)
    threshold = int(window * threshold_pct / 100)
    token_count = _count_tokens(client, model, system_prompt, messages)
    return token_count > threshold


def run_summarizer(
    client: anthropic.Anthropic,
    previous_summary: str,
    messages_to_compress: list[dict],
) -> str:
    """Вызывает Haiku для создания/обновления summary."""
    parts = [
        "<PREVIOUS_SUMMARY>",
        previous_summary if previous_summary else "(пусто -- первое сжатие)",
        "</PREVIOUS_SUMMARY>",
        "",
        "<NEW_MESSAGES>",
    ]
    for msg in messages_to_compress:
        label = "Пользователь" if msg["role"] == "user" else "CoS"
        parts.append(f"{label}: {msg['content']}")
    parts.append("</NEW_MESSAGES>")

    response = client.messages.create(
        model=SUMMARIZER_MODEL,
        max_tokens=2048,
        temperature=0,
        system=SUMMARIZER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "\n".join(parts)}],
    )
    return "".join(
        block.text for block in response.content if block.type == "text"
    )


def maybe_summarize(
    client: anthropic.Anthropic,
    model: str,
    chat_id: str,
    system_prompt: str,
    all_msgs: list[dict],
    recent: list[dict],
    window_start_id: int | None,
    summary_record: dict | None,
    settings: dict | None = None,
) -> str | None:
    """Проверяет триггер и при необходимости запускает сжатие.

    Возвращает актуальный summary_text (новый или старый) или None.
    """
    if settings is None:
        settings = db.get_strategy_settings(chat_id)

    if not settings.get("rolling_summary_enabled", 1):
        return summary_record["summary_text"] if summary_record else None

    threshold_pct = settings.get("rolling_summary_threshold_pct", 35)
    summary_text = summary_record["summary_text"] if summary_record else None
    full_system = build_system_prompt(system_prompt, summary_text)

    if not should_summarize(client, model, full_system, recent, threshold_pct):
        return summary_text

    covers_up_to = (
        summary_record["covers_messages_up_to_id"] if summary_record else 0
    )
    msgs_to_compress = [
        m
        for m in all_msgs
        if m["id"] > covers_up_to
        and (window_start_id is None or m["id"] < window_start_id)
    ]
    if not msgs_to_compress:
        return summary_text

    new_summary = run_summarizer(client, summary_text or "", msgs_to_compress)
    db.save_summary(chat_id, new_summary, msgs_to_compress[-1]["id"])
    return new_summary


# ── Sliding Window ──

def apply_sliding_window(
    messages: list[dict], window_size: int
) -> list[dict]:
    """Срезает историю до последних window_size сообщений, не разрывая пары."""
    if len(messages) <= window_size:
        return messages

    idx = len(messages) - window_size
    if messages[idx]["role"] == "assistant" and idx > 0:
        idx -= 1

    return messages[idx:]


# ── Sticky Facts (extractor) ──

def _strip_markdown_json(text: str) -> str:
    """Убирает ```json ... ``` обёртку, если Haiku добавил."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _validate_facts_response(data: dict) -> bool:
    """Проверяет структуру ответа фактоэкстрактора."""
    if not isinstance(data, dict):
        return False
    for arr_key in ("add", "update"):
        items = data.get(arr_key, [])
        if not isinstance(items, list):
            return False
        for item in items:
            if not isinstance(item, dict) or "key" not in item or "value" not in item:
                return False
    deletes = data.get("delete", [])
    if not isinstance(deletes, list):
        return False
    for d in deletes:
        if not isinstance(d, str):
            return False
    return True


def extract_and_apply_facts(
    client: anthropic.Anthropic,
    chat_id: str,
    user_message: str,
) -> None:
    """Вызывает фактоэкстрактор (Haiku) и применяет операции к chat_facts."""
    current_facts = db.get_chat_facts(chat_id)
    facts_text = "\n".join(f"{f['key']}: {f['value']}" for f in current_facts)
    if not facts_text:
        facts_text = "(пусто)"

    user_input = (
        f"<CURRENT_FACTS>\n{facts_text}\n</CURRENT_FACTS>\n\n"
        f"<NEW_USER_MESSAGE>\n{user_message}\n</NEW_USER_MESSAGE>"
    )

    try:
        response = client.messages.create(
            model=SUMMARIZER_MODEL,
            max_tokens=1024,
            temperature=0,
            system=FACTS_EXTRACTOR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_input}],
        )
        raw_text = "".join(
            block.text for block in response.content if block.type == "text"
        )
    except Exception:
        log.exception("Facts extractor: API call failed for chat %s", chat_id)
        return

    cleaned = _strip_markdown_json(raw_text)

    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        log.error(
            "Facts extractor: invalid JSON for chat %s. Raw response: %s",
            chat_id, raw_text,
        )
        return

    if not _validate_facts_response(data):
        log.error(
            "Facts extractor: schema mismatch for chat %s. Parsed: %s",
            chat_id, data,
        )
        return

    for item in data.get("delete", []):
        db.delete_chat_fact_by_key(chat_id, item)

    for item in data.get("update", []):
        db.update_chat_fact(chat_id, item["key"], item["value"])

    for item in data.get("add", []):
        db.add_chat_fact(chat_id, item["key"], item["value"])


# ── Orchestrator: собирает messages с учётом всех трёх стратегий ──

def prepare_context(
    client: anthropic.Anthropic,
    model: str,
    chat_id: str,
    base_system_prompt: str,
    user_message: str | None = None,
) -> tuple[str, list[dict]]:
    """Возвращает (system_prompt, messages) для отправки в Claude.

    Порядок:
    A. Достаём историю и настройки
    B. Если sticky_facts_enabled и есть user_message — извлекаем факты
    C. Если rolling_summary_enabled и порог превышен — суммаризатор
    D. Если sliding_window_enabled — срезаем окно
    E. Собираем system prompt = base + summary + facts
    """
    settings = db.get_strategy_settings(chat_id)
    all_msgs = db.get_messages_with_ids(chat_id)

    # B: Sticky Facts — extract before building prompt
    if settings.get("sticky_facts_enabled") and user_message:
        extract_and_apply_facts(client, chat_id, user_message)

    # C: Rolling Summary
    summary_record = db.get_summary(chat_id)
    recent, window_start_id = get_recent_window(all_msgs)

    summary_text = maybe_summarize(
        client, model, chat_id,
        base_system_prompt, all_msgs, recent,
        window_start_id, summary_record,
        settings=settings,
    )

    # D: Sliding Window
    messages = [{"role": m["role"], "content": m["content"]} for m in recent]

    if settings.get("sliding_window_enabled"):
        window_size = settings.get("sliding_window_size", 20)
        messages = apply_sliding_window(messages, window_size)

    # E: Build system prompt with facts
    facts = None
    if settings.get("sticky_facts_enabled"):
        facts_rows = db.get_chat_facts(chat_id)
        if facts_rows:
            facts = facts_rows

    system_prompt = build_system_prompt(base_system_prompt, summary_text, facts)

    return system_prompt, messages
