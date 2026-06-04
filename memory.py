"""
Путь ЧТЕНИЯ долговременной памяти.

Главный принцип: срез ищет КОД, не модель. По тексту запроса находим упомянутые
узлы, расширяем на 1 хоп по активным рёбрам, ужимаем под ТОКЕН-БЮДЖЕТ и
форматируем компактным текстовым блоком для системного промпта. Граф целиком
в контекст не попадает НИКОГДА.

Сопоставление имён (правка фазы 1):
- регистронезависимо и нечувствительно к ё/е;
- устойчиво к русским склонениям через матч по словам по длине общего префикса
  (LCP). Это сознательно НЕ полноценная лемматизация: имена собственными
  («Иван Петров») snowball-стеммер ломает непоследовательно, а pymorphy тянет
  словари и память (критично для Render free-tier). LCP-эвристика беззависима,
  и «Иван Петров» находится по «поговорил с Иваном Петровым», а «финансы» — по
  «что по финансам». Если позже понадобится строгая морфология — точечно
  подключим pymorphy3.

Фаза 2: к этой выборке добавится векторный top-k по слою знания.
"""

import logging
import re

import db

log = logging.getLogger("cos.memory")

MAX_SEED_NODES = 8            # сколько «зацепок» (упомянутых узлов) берём максимум
SAFETY_MAX_NODES = 24        # жёсткий потолок узлов ДО ужатия под бюджет
SAFETY_MAX_EDGES = 48        # жёсткий потолок рёбер ДО ужатия под бюджет
DEFAULT_TOKEN_BUDGET = 600   # дефолт бюджета среза в токенах (настройка перекрывает)

_MIN_FUZZY_LEN = 4           # слова короче матчим только точным совпадением
_MAX_DIVERGENCE = 3          # на сколько символов в хвосте словам можно расходиться

_WORD_RE = re.compile(r"\w+", re.UNICODE)


# ── Сопоставление имён ──

def _norm(word: str) -> str:
    return word.lower().replace("ё", "е")


def _words(text: str) -> list[str]:
    return [_norm(w) for w in _WORD_RE.findall(text or "")]


def _word_matches(a: str, b: str) -> bool:
    """Совпадают ли два слова с точностью до склонения (по общему префиксу)."""
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < _MIN_FUZZY_LEN:
        return False  # короткие слова / аббревиатуры — только точное совпадение
    lcp = 0
    for ca, cb in zip(short, long):
        if ca != cb:
            break
        lcp += 1
    # общий префикс длинный, а расходятся слова лишь в коротком хвосте основы
    return lcp >= _MIN_FUZZY_LEN and lcp >= len(short) - _MAX_DIVERGENCE


def _name_matches_query(name: str, query_words: list[str]) -> bool:
    """Имя узла «упомянуто» в запросе, если КАЖДОЕ его слово нашло пару в запросе."""
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


def _find_seeds(query_text: str) -> list[dict]:
    """Активные узлы, чьё имя упомянуто в тексте запроса (с учётом склонений)."""
    query_words = _words(query_text)
    if not query_words:
        return []
    seeds = [n for n in db.get_active_nodes()
             if _name_matches_query(n.get("name") or "", query_words)]
    # длиннее имя — специфичнее совпадение, меньше ложных срабатываний
    seeds.sort(key=lambda n: len(n.get("name") or ""), reverse=True)
    return seeds[:MAX_SEED_NODES]


# ── Сборка среза ──

def retrieve(query_text: str) -> dict | None:
    """Срез-кандидат {nodes, edges} под запрос (узлы в порядке приоритета) или None.

    Узлы упорядочены: сначала «зацепки» (совпавшие по имени), затем соседи на
    1 хоп. Это важно для ужатия под бюджет — режем с конца, наименее важное.
    """
    seeds = _find_seeds(query_text)
    if not seeds:
        return None

    nodes_ordered: list[dict] = []
    seen: set[int] = set()
    for s in seeds:
        if s["id"] not in seen:
            nodes_ordered.append(s)
            seen.add(s["id"])

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
    return {"nodes": nodes_ordered, "edges": edges}


def render_slice(slice_data: dict | None) -> str:
    """Компактный текстовый блок среза для системного промпта."""
    if not slice_data or not slice_data.get("nodes"):
        return ""
    name_by_id = {n["id"]: n.get("name", f"#{n['id']}") for n in slice_data["nodes"]}

    lines = ["Узлы:"]
    for n in slice_data["nodes"]:
        basis = (n.get("basis") or "").strip()
        tail = f" — {basis}" if basis else ""
        lines.append(f"- [{n.get('type')}] {n.get('name')}{tail}")

    if slice_data.get("edges"):
        lines.append("Связи:")
        for e in slice_data["edges"]:
            s = name_by_id.get(e["src_id"], f"#{e['src_id']}")
            d = name_by_id.get(e["dst_id"], f"#{e['dst_id']}")
            lines.append(f"- {s} —{e.get('type')}→ {d}")

    return "\n".join(lines)


# ── Ужатие под токен-бюджет ──

def _count_tokens(client, model: str, text: str) -> int:
    """Размер блока в токенах через тот же API, что и rolling summary."""
    resp = client.beta.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return resp.input_tokens


def _fit_to_budget(client, model, slice_data, budget):
    """Режет срез с конца (наименее важное) под бюджет токенов.

    Считаем токены реальным API. Чтобы не делать вызов на каждый узел, сначала
    меряем полный блок (1 вызов); если влез — готово. Если нет — прикидываем долю
    пропорционально и доуточняем не более нескольких шагов. Итого ≤ ~6 вызовов
    в худшем случае, обычно 1.
    """
    nodes = list(slice_data["nodes"])
    all_edges = slice_data["edges"]

    def build(node_list):
        keep = {n["id"] for n in node_list}
        edges = [e for e in all_edges if e["src_id"] in keep and e["dst_id"] in keep]
        return render_slice({"nodes": node_list, "edges": edges}), edges

    block, edges = build(nodes)
    if not block:
        return "", 0, 0, 0
    tokens = _count_tokens(client, model, block)
    if tokens <= budget:
        return block, tokens, len(nodes), len(edges)

    # пропорциональная прикидка, затем до 5 уточняющих шагов по одному узлу
    frac = budget / tokens if tokens else 1.0
    nodes = nodes[:max(1, int(len(nodes) * frac))]
    for _ in range(5):
        block, edges = build(nodes)
        tokens = _count_tokens(client, model, block) if block else 0
        if tokens <= budget or len(nodes) <= 1:
            break
        nodes.pop()
    return block, tokens, len(nodes), len(edges)


def retrieve_block(client, model: str, query_text: str,
                   token_budget: int = DEFAULT_TOKEN_BUDGET) -> str:
    """Готовый текст блока памяти под запрос (или '' если ничего не нашлось).

    Логирует фактический размер инъекции — чтобы в логах Render было видно,
    сколько токенов память реально добавляет в контекст на каждом чтении.
    """
    slice_data = retrieve(query_text)
    if not slice_data:
        return ""

    block, tokens, n_nodes, n_edges = _fit_to_budget(
        client, model, slice_data, token_budget
    )
    q_preview = (query_text or "")[:60].replace("\n", " ")
    log.info(
        "memory read: nodes=%d edges=%d tokens=%d (budget=%d) query=%r",
        n_nodes, n_edges, tokens, token_budget, q_preview,
    )
    return block
