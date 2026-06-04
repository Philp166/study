"""
Локальные эмбеддинги через fastembed (ONNX-рантайм, без torch — легко для Render).

Модель грузится ЛЕНИВО (при первом обращении) и кэшируется в синглтоне: на старте
процесса ничего не тянется, первый embed() оплачивает загрузку. Если модель не
поднялась (не хватило памяти на Render free-tier / нет файла / нет сети при первом
скачивании) — модуль деградирует МЯГКО: is_available() == False, embed() == None,
а долговременная память продолжает работать на поиске по имени из фазы 1. Так
векторный слой не может «уронить» рабочий чат.

Выбор модели: paraphrase-multilingual-MiniLM-L12-v2 (384-dim, ~0.22 ГБ) — самый
лёгкий из мультиязычных в fastembed с приличным русским. Остальные мультиязычные
(768/1024-dim, 1–2.2 ГБ) для Render 512 МБ не подходят.
"""

import logging
import struct

log = logging.getLogger("cos.embeddings")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384

_model = None
_load_failed = False


def _get_model():
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed:
        return None
    try:
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name=MODEL_NAME)
        log.info("embeddings: модель загружена (%s)", MODEL_NAME)
    except Exception:
        _load_failed = True
        log.exception("embeddings: не удалось загрузить модель — векторный слой ОТКЛЮЧЁН")
        return None
    return _model


def is_available() -> bool:
    """Готов ли векторный слой (модель поднялась)."""
    return _get_model() is not None


def embed(texts: list[str]) -> list[list[float]] | None:
    """Векторы для списка текстов. None — если модель недоступна или сбой."""
    if not texts:
        return []
    model = _get_model()
    if model is None:
        return None
    try:
        return [[float(x) for x in v] for v in model.embed(list(texts))]
    except Exception:
        log.exception("embeddings: сбой embed()")
        return None


def embed_one(text: str) -> list[float] | None:
    out = embed([text])
    return out[0] if out else None


# ── Сериализация вектора в BLOB (float32) и обратно ──

def to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def from_blob(blob: bytes | None) -> list[float] | None:
    if not blob:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ── Косинусная близость (чистый Python — без обязательной зависимости от numpy) ──

def cosine(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def top_k(query_vec: list[float], items: list[tuple], k: int) -> list[tuple]:
    """items: список (payload, vector). Возвращает [(payload, score), ...] top-k по косинусу."""
    scored = []
    for payload, vec in items:
        if vec is None:
            continue
        scored.append((payload, cosine(query_vec, vec)))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:k]
