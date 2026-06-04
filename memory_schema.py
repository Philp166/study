"""
Закрытая схема графа долговременной памяти + валидатор.

Это «предохранитель» от мусора: ни один путь записи (ручной API сейчас,
субагент-экстрактор позже) не может создать узел вне 10 разрешённых типов или
ребро вне 14 типов / с недопустимой парой (src_type, dst_type). Универсальная
связь relates_to / «связано_с» отсутствует by design — её нельзя создать,
потому что её нет в списке. Так мы защищаемся от «всё связано со всем».

Модуль НИЧЕГО не импортирует — его свободно зовут db.py, app.py и будущий
экстрактор без риска циклических импортов.
"""

# ── 10 типов узлов ──
NODE_TYPES = frozenset({
    "Person", "Org", "Project", "Task", "Decision",
    "Document", "Event", "Money", "Place", "Topic",
})

# references → любой тип цели.
ANY = NODE_TYPES

# Типы, у которых разрешён конфликт версий (supersedes). supersedes — всегда
# same-type: новая версия заменяет старую того же типа.
SUPERSEDABLE_TYPES = frozenset({"Decision", "Money", "Document"})

# depends_on — только между узлами одного типа из этого набора.
DEPENDS_TYPES = frozenset({"Task", "Project", "Decision"})

# ── 14 рёбер: (допустимые src, допустимые dst, требуется ли src_type == dst_type) ──
EDGE_RULES = {
    "works_at":        (frozenset({"Person"}),                          frozenset({"Org"}),                 False),
    "member_of":       (frozenset({"Person"}),                          frozenset({"Project"}),             False),
    "responsible_for": (frozenset({"Person"}),                          frozenset({"Project", "Task", "Money"}), False),
    "part_of":         (frozenset({"Task", "Money", "Event", "Project"}), frozenset({"Project"}),           False),
    "depends_on":      (DEPENDS_TYPES,                                  DEPENDS_TYPES,                       True),
    "allocated_to":    (frozenset({"Money"}),                           frozenset({"Project", "Task"}),     False),
    "affects":         (frozenset({"Decision"}),                        frozenset({"Money", "Project", "Task"}), False),
    "decided_in":      (frozenset({"Decision"}),                        frozenset({"Event"}),               False),
    "participates_in": (frozenset({"Person"}),                          frozenset({"Event"}),               False),
    "contracted_with": (frozenset({"Org", "Person"}),                   frozenset({"Org"}),                 False),
    "references":      (frozenset({"Document"}),                        ANY,                                False),
    "about":           (frozenset({"Document", "Task", "Project"}),     frozenset({"Topic"}),               False),
    "located_at":      (frozenset({"Event", "Org"}),                    frozenset({"Place"}),               False),
    "supersedes":      (SUPERSEDABLE_TYPES,                             SUPERSEDABLE_TYPES,                 True),
}

EDGE_TYPES = frozenset(EDGE_RULES.keys())


def validate_node_type(node_type: str) -> tuple[bool, str]:
    if node_type not in NODE_TYPES:
        return False, (
            f"Недопустимый тип узла '{node_type}'. "
            f"Разрешены только: {', '.join(sorted(NODE_TYPES))}."
        )
    return True, ""


def validate_edge(edge_type: str, src_type: str, dst_type: str) -> tuple[bool, str]:
    """Проверяет тип ребра и пару (src_type → dst_type). Возвращает (ok, причина)."""
    if edge_type not in EDGE_RULES:
        return False, (
            f"Недопустимый тип ребра '{edge_type}'. Разрешены только: "
            f"{', '.join(sorted(EDGE_TYPES))}. Универсальная связь запрещена — "
            f"если связь не ложится в список, ребро не создаётся."
        )
    if src_type not in NODE_TYPES:
        return False, f"Недопустимый тип источника '{src_type}'."
    if dst_type not in NODE_TYPES:
        return False, f"Недопустимый тип цели '{dst_type}'."

    allowed_src, allowed_dst, same_type = EDGE_RULES[edge_type]
    if src_type not in allowed_src:
        return False, (
            f"Ребро '{edge_type}' не выходит из '{src_type}'. "
            f"Допустимые источники: {', '.join(sorted(allowed_src))}."
        )
    if dst_type not in allowed_dst:
        return False, (
            f"Ребро '{edge_type}' не ведёт в '{dst_type}'. "
            f"Допустимые цели: {', '.join(sorted(allowed_dst))}."
        )
    if same_type and src_type != dst_type:
        return False, (
            f"Ребро '{edge_type}' связывает только узлы одного типа "
            f"(получили {src_type} → {dst_type})."
        )
    return True, ""


def schema_summary() -> dict:
    """Машиночитаемое описание схемы — для фронта (экран памяти) и для будущего
    промпта экстрактора."""
    return {
        "node_types": sorted(NODE_TYPES),
        "edge_types": {
            et: {
                "src": sorted(src),
                "dst": "ANY" if dst is ANY else sorted(dst),
                "same_type": same_type,
            }
            for et, (src, dst, same_type) in EDGE_RULES.items()
        },
    }
