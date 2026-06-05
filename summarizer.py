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
    memory_block: str | None = None,
    tasks_block: str | None = None,
) -> str:
    """base_prompt + долговременная память + задачи + summary + блок фактов.

    Память и задачи идут ПЕРЕД summary/facts: они стабильнее (живут между чатами)
    и менее «свежие», чем summary/facts текущего диалога.
    """
    parts = [base_prompt]

    if memory_block:
        parts.append(memory_block)
    if tasks_block:
        parts.append(tasks_block)

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

    Работает по истории АКТИВНОЙ ВЕТКИ (all_msgs/recent — путь root→leaf).
    Вдоль ветки id монотонно растут (ребёнок создаётся позже родителя),
    поэтому фильтры по id корректно вырезают средний сегмент ветки.
    Новый summary якорится на id последнего сжатого сообщения ветки.

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
    leaf_message_id: int,
) -> None:
    """Вызывает фактоэкстрактор (Haiku) и применяет операции к chat_facts.

    Факты читаются из АКТИВНОЙ ВЕТКИ (подъём от leaf к корню), а записываются
    с branch_anchor_message_id = leaf_message_id (это id текущего user-сообщения,
    т.к. оно уже сохранено и стало current_leaf). Удаление — через надгробие,
    чтобы не затронуть предков и соседние ветки.
    """
    current_facts = db.get_active_facts(chat_id, leaf_message_id)
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
        db.tombstone_chat_fact(chat_id, item, leaf_message_id)

    for item in data.get("update", []):
        db.update_chat_fact(chat_id, item["key"], item["value"], leaf_message_id)

    for item in data.get("add", []):
        db.add_chat_fact(chat_id, item["key"], item["value"], leaf_message_id)


# ── Долговременная память + активные задачи в системпромпт (Фаза 4) ──

MEMORY_TOKEN_BUDGET = 1500   # суммарно на блок памяти + блок задач
TASK_CONTEXT_CAP = 500       # защитный кап на context одной задачи (живой документ)


def _estimate_tokens(text: str) -> int:
    """Консервативная верхняя оценка токенов (≈2 симв/токен для кириллицы).
    Без сетевого вызова на горячем пути — блоки и так малы. Точный аналог —
    client.beta.messages.count_tokens, но он добавил бы вызов на каждый запрос."""
    return (len(text) + 1) // 2


def _render_memory_block(notes: list[dict], content_cap: int | None = None) -> str:
    lines = ["\n\n# Долговременная память\n"]
    for n in notes:
        folder = n.get("folder")
        head = f"\n## {n['title']} ({folder})" if folder else f"\n## {n['title']}"
        content = n.get("content_preview") or ""
        if content_cap is not None:
            content = content[:content_cap]
        lines.append(head + ("\n" + content if content else ""))
    return "".join(lines)


def _render_tasks_block(tasks: list[dict]) -> str:
    lines = ["\n\n# Активные задачи\n"]
    for t in tasks:
        ctx = (t.get("context") or "")[:TASK_CONTEXT_CAP]
        lines.append(f"\n## {t['title']}" + ("\n" + ctx if ctx else ""))
    return "".join(lines)


def build_memory_blocks(
    notes: list[dict], tasks: list[dict]
) -> tuple[str | None, str | None]:
    """Строит (memory_block, tasks_block) под токен-бюджет MEMORY_TOKEN_BUDGET.

    Урезание при переполнении: (1) контент заметок → 100 симв; (2) выкидываем
    заметки с конца (они отсортированы по релевантности/свежести). Задачи не
    урезаем (но context каждой капится TASK_CONTEXT_CAP в рендере). Если обоих
    нет — (None, None), пустые заголовки не добавляем."""
    if not notes and not tasks:
        return None, None

    tasks_block = _render_tasks_block(tasks) if tasks else None
    tasks_tokens = _estimate_tokens(tasks_block) if tasks_block else 0
    if not notes:
        return None, tasks_block

    mem = _render_memory_block(notes)
    if _estimate_tokens(mem) + tasks_tokens <= MEMORY_TOKEN_BUDGET:
        return mem, tasks_block

    mem = _render_memory_block(notes, content_cap=100)
    if _estimate_tokens(mem) + tasks_tokens <= MEMORY_TOKEN_BUDGET:
        return mem, tasks_block

    kept = list(notes)
    while kept:
        kept.pop()
        if not kept:
            return None, tasks_block
        mem = _render_memory_block(kept, content_cap=100)
        if _estimate_tokens(mem) + tasks_tokens <= MEMORY_TOKEN_BUDGET:
            return mem, tasks_block
    return None, tasks_block


# ── Orchestrator: собирает messages с учётом всех трёх стратегий ──

def prepare_context(
    client: anthropic.Anthropic,
    model: str,
    chat_id: str,
    base_system_prompt: str,
    leaf_message_id: int | None,
    user_message: str | None = None,
    inject_memory: bool = False,
) -> tuple[str, list[dict]]:
    """Возвращает (system_prompt, messages) для отправки в Claude.

    inject_memory=True добавляет в системпромпт глобальные блоки долговременной
    памяти (релевантные заметки) и активных задач — ПЕРЕД summary/facts. По
    умолчанию False, чтобы голосовой путь (своя prepare_context) остался без них.

    Все три стратегии работают по истории АКТИВНОЙ ВЕТКИ — пути от
    leaf_message_id до корня (рекурсивный CTE в db.get_branch_history).
    Для линейного чата ветка == плоская история (поведение не меняется).

    Порядок:
    A. Достаём историю ветки и настройки
    B. Если sticky_facts_enabled и есть user_message — извлекаем факты (anchor=leaf)
    C. Если rolling_summary_enabled и порог превышен — суммаризатор (по ветке)
    E0. Если inject_memory — блоки долговременной памяти и активных задач
    E. Собираем system prompt = base + память + задачи + summary(ветки) + facts(ветки)
    """
    settings = db.get_strategy_settings(chat_id)
    all_msgs = db.get_branch_history(chat_id, leaf_message_id)

    # B: Sticky Facts — extract before building prompt (anchor = текущий leaf)
    if settings.get("sticky_facts_enabled") and user_message and leaf_message_id:
        extract_and_apply_facts(client, chat_id, user_message, leaf_message_id)

    # C: Rolling Summary — summary, чей anchor на пути ветки
    summary_record = db.get_active_summary(chat_id, leaf_message_id)
    recent, window_start_id = get_recent_window(all_msgs)

    summary_text = maybe_summarize(
        client, model, chat_id,
        base_system_prompt, all_msgs, recent,
        window_start_id, summary_record,
        settings=settings,
    )

    # D: история ветки (Sliding Window убран в Фазе 5 — Rolling Summary его покрывает)
    messages = [{"role": m["role"], "content": m["content"]} for m in recent]

    # E0: Долговременная память + активные задачи (глобальные, между чатами).
    # Только для текстового пути (inject_memory=True). Память НЕ влияет на порог
    # суммаризации (maybe_summarize строит промпт без этих блоков).
    memory_block = tasks_block = None
    if inject_memory:
        notes = db.relevant_memory(user_message or "", limit=5)
        active_tasks = db.list_tasks(status="active")
        memory_block, tasks_block = build_memory_blocks(notes, active_tasks)

    # E: Build system prompt with branch-active facts
    facts = None
    if settings.get("sticky_facts_enabled"):
        facts_rows = db.get_active_facts(chat_id, leaf_message_id)
        if facts_rows:
            facts = facts_rows

    system_prompt = build_system_prompt(
        base_system_prompt, summary_text, facts,
        memory_block=memory_block, tasks_block=tasks_block,
    )

    return system_prompt, messages
