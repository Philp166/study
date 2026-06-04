"""
Путь ЗАПИСИ долговременной памяти (оркестратор).

Поток: content → построить graph_context (имя + вектор, для дедупа) → субагент-
экстрактор (Haiku) возвращает JSON → валидируем → применяем порог confidence →
пишем узлы/рёбра через db (валидатор пар страхует) → конфликты в supersedes →
сам content кладём в слой знания чанками. Экстрактор сам в граф НЕ пишет.

Фаза 2: запуск только вручную (команда «запомни …», mode=on_request, порог ≥0.5).
Фаза 4 (позже): автоматические чекпоинты, mode=background (≥0.8), очередь suggestions.

Запускается из BackgroundTasks — диалог не блокируется. Любой сбой логируется и
гасится, чтобы фон не уронил запрос.
"""

import json
import logging
import re

import db
import embeddings
import memory
import memory_schema
from prompts.memory_extractor import MEMORY_EXTRACTOR_SYSTEM_PROMPT

log = logging.getLogger("cos.memory.write")

EXTRACTOR_MODEL = "claude-haiku-4-5"
GRAPH_CONTEXT_LIMIT = 15
CHUNK_MAX_LEN = 900

_CMD_RE = re.compile(r"^\s*/?запомни(?:те|ть)?\b[\s:,\-—]*", re.IGNORECASE)
_EMPTY_POINTERS = {"", "это", "этот", "вот это", "это сообщение", "такое", "вот"}


# ── Команда «запомни …» ──

def parse_remember_command(text: str) -> tuple[bool, str | None]:
    """(is_command, content). content=None → «запомни это» (помним предыдущее сообщение)."""
    if not text:
        return False, None
    m = _CMD_RE.match(text)
    if not m:
        return False, None
    content = text[m.end():].strip()
    if content.lower() in _EMPTY_POINTERS:
        content = None
    return True, content


# ── Эмбеддинги узлов/чанков ──

def _node_embed_text(node_type: str, name: str, basis: str, attributes: dict) -> str:
    parts = [name]
    if basis:
        parts.append(basis)
    if attributes:
        parts.append(" ".join(f"{k}: {v}" for k, v in attributes.items()))
    return f"{node_type}: " + " | ".join(parts)


def _node_embedding(node_type, name, basis, attributes) -> bytes | None:
    if not memory._vector_enabled():
        return None
    vec = embeddings.embed_one(_node_embed_text(node_type, name, basis, attributes or {}))
    return embeddings.to_blob(vec) if vec else None


def index_node(node_id: int) -> None:
    """Пересчитать и сохранить эмбеддинг узла (для ручного API; в фоне)."""
    node = db.get_node(node_id)
    if not node:
        return
    blob = _node_embedding(node["type"], node["name"],
                           node.get("basis", ""), node.get("attributes") or {})
    if blob is not None:
        db.set_node_embedding(node_id, blob)


def _split_text(text: str) -> list[str]:
    text = (text or "").strip()
    if len(text) <= CHUNK_MAX_LEN:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > CHUNK_MAX_LEN:
            chunks.append(cur.strip())
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur.strip())
    return chunks or [text[:CHUNK_MAX_LEN]]


def _store_chunks(content, chat_id, message_id, origin) -> int:
    n = 0
    for piece in _split_text(content):
        vec = embeddings.embed_one(piece) if memory._vector_enabled() else None
        blob = embeddings.to_blob(vec) if vec else None
        db.create_chunk(piece, embedding=blob, origin=origin,
                        source_chat_id=chat_id, source_message_id=message_id)
        n += 1
    return n


# ── Граф-контекст для дедупа ──

def _build_graph_context(content: str) -> list[dict]:
    ctx: dict[int, dict] = {n["id"]: n for n in memory._find_name_seeds(content)}
    if memory._vector_enabled():
        qv = embeddings.embed_one(content)
        if qv is not None:
            for nid in memory._rank_ids(qv, db.get_node_vectors(), 10):
                if nid not in ctx:
                    n = db.get_node(nid)
                    if n:
                        ctx[nid] = n
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
            continue  # ниже порога — в фазе 4 уйдёт в suggestions; сейчас пропускаем
        attrs = n.get("attributes") or {}
        basis = n.get("basis", "")
        nid = db.create_node(
            ntype, name, attributes=attrs, origin=mode, basis=basis,
            confidence=float(n.get("confidence", 0)),
            source_chat_id=chat_id, source_message_id=message_id,
            embedding=_node_embedding(ntype, name, basis, attrs),
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
            return {"nodes": 0, "edges": 0, "conflicts": 0, "suggestions": 0, "chunks": 0}
        graph_context = _build_graph_context(content)
        data = _run_extractor(client, content, mode, graph_context)
        result = {"nodes": 0, "edges": 0, "conflicts": 0, "suggestions": 0}
        if data:
            result = _apply(data, chat_id=chat_id, message_id=message_id,
                            mode=mode, threshold=threshold)
        result["chunks"] = _store_chunks(content, chat_id, message_id, mode)
        log.info("memory write (%s): %s", mode, result)
        return result
    except Exception:
        log.exception("memory write: оркестратор упал (content=%.80r)", content)
        return {"nodes": 0, "edges": 0, "conflicts": 0, "suggestions": 0, "chunks": 0}
