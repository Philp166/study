"""
Путь ЧТЕНИЯ долговременной памяти.

Главный принцип: срез ищет КОД, не модель. Срез собирается из трёх источников и
кладётся в системный промпт маленьким блоком; граф целиком в контекст не попадает
НИКОГДА.

Источники среза:
1. Имя (фаза 1): узлы, чьё имя упомянуто в запросе — регистронезависимо и
   устойчиво к русским склонениям (матч по словам через длину общего префикса).
2. Вектор по узлам (фаза 2): семантически близкие узлы графа (top-k косинус).
3. Вектор по чанкам (фаза 2): слой знания — свободный текст, не лёгший в граф.

Векторная часть включается, только если поднялась локальная модель (embeddings)
И не выключена настройкой memory_vector_enabled. Если вектора нет — работает поиск
по имени, как в фазе 1 (мягкая деградация, чат не ломается).
"""

import logging
import re

import db
import embeddings

log = logging.getLogger("cos.memory")

MAX_SEED_NODES = 8           # «зацепок» по имени максимум
MAX_VEC_NODES = 6            # узлов из векторного поиска
MAX_VEC_CHUNKS = 4           # чанков знания из векторного поиска
VEC_MIN_SCORE = 0.30         # порог косинуса, ниже — шум, не берём
SAFETY_MAX_NODES = 24        # жёсткий потолок узлов ДО ужатия под бюджет
SAFETY_MAX_EDGES = 48
DEFAULT_TOKEN_BUDGET = 600   # дефолт бюджета среза в токенах (настройка перекрывает)

_MIN_FUZZY_LEN = 4
_MAX_DIVERGENCE = 3
_WORD_RE = re.compile(r"\w+", re.UNICODE)


# ── Сопоставление имён (склонения через общий префикс) ──

def _norm(word: str) -> str:
    return word.lower().replace("ё", "е")


def _words(text: str) -> list[str]:
    return [_norm(w) for w in _WORD_RE.findall(text or "")]


def _word_matches(a: str, b: str) -> bool:
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < _MIN_FUZZY_LEN:
        return False
    lcp = 0
    for ca, cb in zip(short, long):
        if ca != cb:
            break
        lcp += 1
    return lcp >= _MIN_FUZZY_LEN and lcp >= len(short) - _MAX_DIVERGENCE


def _name_matches_query(name: str, query_words: list[str]) -> bool:
    name_words = _words(name)
    if not name_words:
        return False
    for nw in name_words:
        if len(nw) < _MIN_FUZZY_LEN:
            if nw not in query_words:
                return False
        elif not any(_word_matches(nw, qw) for qw in query_words):
            return False
    return True


def _find_name_seeds(query_text: str) -> list[dict]:
    query_words = _words(query_text)
    if not query_words:
        return []
    seeds = [n for n in db.get_active_nodes()
             if _name_matches_query(n.get("name") or "", query_words)]
    seeds.sort(key=lambda n: len(n.get("name") or ""), reverse=True)
    return seeds[:MAX_SEED_NODES]


# ── Векторный поиск ──

def _vector_enabled() -> bool:
    return (db.get_setting("memory_vector_enabled", "true") != "false"
            and embeddings.is_available())


def _rank_ids(query_vec, vec_pairs, k: int) -> list[int]:
    items = [(pid, embeddings.from_blob(blob)) for pid, blob in vec_pairs]
    return [pid for pid, score in embeddings.top_k(query_vec, items, k)
            if score >= VEC_MIN_SCORE]


# ── Сборка среза ──

def retrieve(query_text: str) -> dict | None:
    """Срез-кандидат {nodes, edges, chunks} под запрос или None.

    Узлы упорядочены по приоритету (имя-зацепки → вектор → соседи на 1 хоп),
    чтобы ужатие под бюджет резало наименее важное с конца.
    """
    seeds = _find_name_seeds(query_text)
    seen = {s["id"] for s in seeds}

    query_vec = None
    if _vector_enabled():
        query_vec = embeddings.embed_one(query_text)

    # векторные узлы — как дополнительные зацепки
    if query_vec is not None:
        for nid in _rank_ids(query_vec, db.get_node_vectors(), MAX_VEC_NODES):
            if nid not in seen:
                n = db.get_node(nid)
                if n:
                    seeds.append(n)
                    seen.add(nid)

    nodes_ordered: list[dict] = list(seeds)
    edge_by_id: dict[int, dict] = {}
    for s in seeds:
        sub = db.get_subgraph(s["id"], hops=1)
        for n in sub["nodes"]:
            if n["id"] not in seen:
                nodes_ordered.append(n)
                seen.add(n["id"])
        for e in sub["edges"]:
            edge_by_id.setdefault(e["id"], e)
        if len(nodes_ordered) >= SAFETY_MAX_NODES:
            break

    nodes_ordered = nodes_ordered[:SAFETY_MAX_NODES]
    keep = {n["id"] for n in nodes_ordered}
    edges = [e for e in edge_by_id.values()
             if e["src_id"] in keep and e["dst_id"] in keep][:SAFETY_MAX_EDGES]

    # чанки знания
    chunks: list[dict] = []
    if query_vec is not None:
        for cid in _rank_ids(query_vec, db.get_chunk_vectors(), MAX_VEC_CHUNKS):
            ch = db.get_chunk(cid)
            if ch:
                chunks.append(ch)

    if not nodes_ordered and not chunks:
        return None
    return {"nodes": nodes_ordered, "edges": edges, "chunks": chunks}


def render_slice(slice_data: dict | None) -> str:
    """Компактный текстовый блок среза для системного промпта."""
    if not slice_data:
        return ""
    nodes = slice_data.get("nodes") or []
    edges = slice_data.get("edges") or []
    chunks = slice_data.get("chunks") or []
    if not nodes and not chunks:
        return ""

    name_by_id = {n["id"]: n.get("name", f"#{n['id']}") for n in nodes}
    lines: list[str] = []

    if nodes:
        lines.append("Узлы:")
        for n in nodes:
            basis = (n.get("basis") or "").strip()
            tail = f" — {basis}" if basis else ""
            lines.append(f"- [{n.get('type')}] {n.get('name')}{tail}")

    if edges:
        lines.append("Связи:")
        for e in edges:
            s = name_by_id.get(e["src_id"], f"#{e['src_id']}")
            d = name_by_id.get(e["dst_id"], f"#{e['dst_id']}")
            lines.append(f"- {s} —{e.get('type')}→ {d}")

    if chunks:
        lines.append("Знание:")
        for ch in chunks:
            t = (ch.get("text") or "").strip().replace("\n", " ")
            lines.append(f"- {t}")

    return "\n".join(lines)


# ── Ужатие под токен-бюджет ──

def _count_tokens(client, model: str, text: str) -> int:
    resp = client.beta.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return resp.input_tokens


def _fit_to_budget(client, model, slice_data, budget):
    """Режет срез под бюджет токенов: сперва чанки, затем соседние узлы — с конца
    (наименее важное). Бюджет считаем реальным API; чтобы не звать его на каждый
    элемент, меряем полный блок, прикидываем долю и доуточняем ≤4 шагами.
    """
    nodes = list(slice_data.get("nodes") or [])
    chunks = list(slice_data.get("chunks") or [])
    all_edges = slice_data.get("edges") or []

    def build():
        keep = {n["id"] for n in nodes}
        edges = [e for e in all_edges if e["src_id"] in keep and e["dst_id"] in keep]
        block = render_slice({"nodes": nodes, "edges": edges, "chunks": chunks})
        return block, edges

    block, edges = build()
    if not block:
        return "", 0, 0, 0, 0
    tokens = _count_tokens(client, model, block)
    if tokens <= budget:
        return block, tokens, len(nodes), len(edges), len(chunks)

    def drop_one():
        # сначала чанки, потом соседние узлы; минимум 1 узел оставляем
        if chunks:
            chunks.pop()
            return True
        if len(nodes) > 1:
            nodes.pop()
            return True
        return False

    # пропорциональная прикидка
    frac = budget / tokens if tokens else 1.0
    total = len(nodes) + len(chunks)
    to_drop = total - max(1, int(total * frac))
    while to_drop > 0 and drop_one():
        to_drop -= 1

    # доуточнение
    for _ in range(4):
        block, edges = build()
        tokens = _count_tokens(client, model, block) if block else 0
        if tokens <= budget or not (chunks or len(nodes) > 1):
            break
        drop_one()
    return block, tokens, len(nodes), len(edges), len(chunks)


def retrieve_block(client, model: str, query_text: str,
                   token_budget: int = DEFAULT_TOKEN_BUDGET) -> str:
    """Готовый текст блока памяти под запрос (или '' если ничего не нашлось).

    Логирует фактический размер инъекции — чтобы в логах Render было видно,
    сколько токенов память реально добавляет в контекст на каждом чтении.
    """
    slice_data = retrieve(query_text)
    if not slice_data:
        return ""

    block, tokens, n_nodes, n_edges, n_chunks = _fit_to_budget(
        client, model, slice_data, token_budget
    )
    q_preview = (query_text or "")[:60].replace("\n", " ")
    log.info(
        "memory read: nodes=%d edges=%d chunks=%d tokens=%d (budget=%d) vec=%s query=%r",
        n_nodes, n_edges, n_chunks, tokens, token_budget, _vector_enabled(), q_preview,
    )
    return block
