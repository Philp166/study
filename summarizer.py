"""
Rolling summary — сжатие истории диалога для экономии контекста.

Старые сообщения сжимаются через дешёвую модель (Haiku),
свежее окно (последние 10 сообщений) остаётся как есть.
"""

import anthropic

from agent import MODEL_CONTEXT_WINDOWS
from prompts.summarizer import SUMMARIZER_SYSTEM_PROMPT

SUMMARIZER_MODEL = "claude-haiku-4-5"
FRESH_WINDOW_SIZE = 10
CONTEXT_THRESHOLD = 0.35


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


def build_system_prompt(base_prompt: str, summary_text: str | None) -> str:
    """base_prompt + summary (если есть) в конце."""
    if not summary_text:
        return base_prompt
    return (
        base_prompt
        + "\n\n# Текущее краткое содержание разговора\n\n"
        + summary_text
    )


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
) -> bool:
    """True, если payload превышает 35 % окна модели."""
    window = MODEL_CONTEXT_WINDOWS.get(model, 200_000)
    threshold = int(window * CONTEXT_THRESHOLD)
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
        previous_summary if previous_summary else "(пусто — первое сжатие)",
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
) -> str | None:
    """Проверяет триггер и при необходимости запускает сжатие.

    Возвращает актуальный summary_text (новый или старый) или None.
    """
    import db

    summary_text = summary_record["summary_text"] if summary_record else None
    full_system = build_system_prompt(system_prompt, summary_text)

    if not should_summarize(client, model, full_system, recent):
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
