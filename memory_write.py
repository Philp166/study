"""
Путь ЗАПИСИ долговременной памяти (оркестратор).

Поток: content → построить graph_context (поиск по имени, для дедупа) → субагент-
экстрактор (Haiku) возвращает JSON → валидируем → применяем порог confidence →
пишем узлы/рёбра через db (валидатор пар страхует) → конфликты в supersedes.
Экстрактор сам в граф НЕ пишет.

Запуск только вручную (команда «запомни …», mode=on_request, порог ≥0.5) из
BackgroundTasks — диалог не блокируется. Любой сбой логируется и гасится.
"""

import json
import logging
import re

import db
import memory
import memory_schema
from prompts.memory_extractor import MEMORY_EXTRACTOR_SYSTEM_PROMPT

log = logging.getLogger("cos.memory.write")

EXTRACTOR_MODEL = "claude-haiku-4-5"
GRAPH_CONTEXT_LIMIT = 15

# Слово-команда «запомни/запомнить/запомните» — В ЛЮБОМ месте сообщения.
_CMD_RE = re.compile(r"\bзапомни(?:те|ть)?\b", re.IGNORECASE)
_EMPTY_POINTERS = {"", "это", "этот", "вот это", "это сообщение", "такое", "вот",
                   "запиши", "сохрани"}


# ── Команда «запомни …» ──

def parse_remember_command(text: str) -> tuple[bool, str | None]:
    """(is_command, content). Команда срабатывает, если слово «запомни» есть где угодно.

    content — остаток сообщения без слова-команды (факты обычно в том же
    сообщении: «… двое детей, запомни»). Если остаток пустой или это указатель
    («запомни это») → content=None, помним предыдущее сообщение ассистента.
    """
    if not text:
        return False, None
    m = _CMD_RE.search(text)
    if not m:
        return False, None
    rest = (text[:m.start()] + " " + text[m.end():]).strip(" ,.:;—-\n\t\r")
    if rest.lower() in _EMPTY_POINTERS or len(rest) < 3:
        return True, None
    return True, rest


# ── Граф-контекст для дедупа ──

def _build_graph_context(content: str) -> list[dict]:
    ctx = {n["id"]: n for n in memory._find_name_seeds(content)}
    items = list(ctx.values())[:GRAPH_CONTEXT_LIMIT]
    return [{"id": n["id"], "type": n["type"], "name": n["name"],
             "basis": n.get("basis", "")} for n in items]


# ── Вызов экстрактора ──

def _strip_markdown_json(text: str) -> str:
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _run_extractor(client, content, mode, graph_context) -> dict | None:
    user = (
        f"<CONTENT>\n{content}\n</CONTENT>\n\n"
        f"<MODE>\n{mode}\n</MODE>\n\n"
        f"<GRAPH_CONTEXT>\n{json.dumps(graph_context, ensure_ascii=False, indent=2)}\n</GRAPH_CONTEXT>"
    )
    try:
        resp = client.messages.create(
            model=EXTRACTOR_MODEL,
            max_tokens=2048,
            temperature=0,
            system=MEMORY_EXTRACTOR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text")
    except Exception:
        log.exception("memory extractor: вызов API упал")
        return None
    try:
        data = json.loads(_strip_markdown_json(raw))
    except (json.JSONDecodeError, ValueError):
        log.error("memory extractor: невалидный JSON. Ответ: %s", raw[:500])
        return None
    if not isinstance(data, dict):
        return None
    return data


# ── Применение JSON к графу ──

def _resolve(x, ref_to_id) -> int | None:
    if isinstance(x, int):
        return x if db.get_node(x) else None
    if isinstance(x, str):
        if x in ref_to_id:
            return ref_to_id[x]
        if x.isdigit():
            xi = int(x)
            return xi if db.get_node(xi) else None
    return None


def _apply(data, *, chat_id, message_id, mode, threshold) -> dict:
    ref_to_id: dict[str, int] = {}
    created_nodes = created_edges = conflicts = 0

    for n in data.get("nodes") or []:
        ntype, name = n.get("type"), n.get("name")
        if not ntype or not name:
            continue
        ok, _ = memory_schema.validate_node_type(ntype)
        if not ok:
            continue
        ref = n.get("ref")
        if n.get("id"):  # дедуп: переиспользуем существующий узел
            if ref:
                ref_to_id[ref] = n["id"]
            continue
        if float(n.get("confidence", 0)) < threshold:
            continue  # ниже порога — пропускаем
        nid = db.create_node(
            ntype, name, attributes=n.get("attributes") or {}, origin=mode,
            basis=n.get("basis", ""), confidence=float(n.get("confidence", 0)),
            source_chat_id=chat_id, source_message_id=message_id,
        )
        created_nodes += 1
        if ref:
            ref_to_id[ref] = nid

    # конфликты версий: new → supersedes → old, старый inactive
    for c in data.get("conflicts") or []:
        new_id = ref_to_id.get(c.get("new_ref"))
        old_id = c.get("supersedes_id")
        if not (new_id and isinstance(old_id, int) and db.get_node(old_id)):
            continue
        try:
            db.create_edge(new_id, old_id, "supersedes", origin=mode,
                           basis=c.get("reason", ""), confidence=1.0,
                           source_chat_id=chat_id, source_message_id=message_id)
            db.soft_delete_node(old_id)
            conflicts += 1
        except ValueError as ex:
            log.info("memory extractor: supersedes отклонён: %s", ex)

    for e in data.get("edges") or []:
        if float(e.get("confidence", 0)) < threshold:
            continue
        src = _resolve(e.get("src"), ref_to_id)
        dst = _resolve(e.get("dst"), ref_to_id)
        et = e.get("type")
        if not (src and dst and et):
            continue
        try:
            db.create_edge(src, dst, et, origin=mode, basis=e.get("basis", ""),
                           confidence=float(e.get("confidence", 0)),
                           source_chat_id=chat_id, source_message_id=message_id)
            created_edges += 1
        except ValueError as ex:
            log.info("memory extractor: ребро отклонено (%s): %s", et, ex)

    return {
        "nodes": created_nodes,
        "edges": created_edges,
        "conflicts": conflicts,
        "suggestions": len(data.get("suggestions") or []),
    }


# ── Точка входа (вызывается из BackgroundTasks) ──

def remember(client, content: str, *, chat_id: str | None = None,
             message_id: int | None = None, mode: str = "on_request") -> dict:
    """Разобрать content и записать в память. mode: on_request (≥0.5) / background (≥0.8)."""
    threshold = 0.5 if mode == "on_request" else 0.8
    try:
        if not content or not content.strip():
            return {"nodes": 0, "edges": 0, "conflicts": 0, "suggestions": 0}
        graph_context = _build_graph_context(content)
        data = _run_extractor(client, content, mode, graph_context)
        result = {"nodes": 0, "edges": 0, "conflicts": 0, "suggestions": 0}
        if data:
            result = _apply(data, chat_id=chat_id, message_id=message_id,
                            mode=mode, threshold=threshold)
        log.info("memory write (%s): %s", mode, result)
        return result
    except Exception:
        log.exception("memory write: оркестратор упал (content=%.80r)", content)
        return {"nodes": 0, "edges": 0, "conflicts": 0, "suggestions": 0}
