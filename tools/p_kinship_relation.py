#!/usr/bin/env python3
"""
ЗАМЕР ПО РЕШЕНИЮ БОССА: можно ли менять ВЫБОР отношения экстрактором.

Вопрос не «какая глосса красивее», а ОДИН факт: собирается ли ПУТЬ загадки
невестка -> жена -> сын -> егор, если два ребра PART_OF заменить на
HAS_PROPERTY. Критерий нового отношения — ФУНКЦИЯ (путь), не глосса.
Если путь рвётся, замена недопустима независимо от того, насколько PART_OF
бессмысленен по смыслу.

Обход ОДИН И ТОТ ЖЕ для обоих миров: прямой шаг + обратный шаг (П2). Мир
строится руками, чтобы не зависеть от экстрактора: сравниваем отношения,
а не разбор.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.simulator import StaticKnowledge   # noqa: E402

# Ребра рассказа КАК ИХ пишет экстрактор сегодня (замер experiments/P2_AFTER.txt)
EDGES_AS_IS = [
    ("невестка", "IS_ROLE", "жена"),
    ("егор", "IS_ROLE", "сын"),
    ("жена", "PART_OF", "сын"),
    ("сын", "PART_OF", "зинаида"),
]
# То же, но владение родством выражено HAS_PROPERTY
EDGES_HAS_PROPERTY = [
    ("невестка", "IS_ROLE", "жена"),
    ("егор", "IS_ROLE", "сын"),
    ("жена", "HAS_PROPERTY", "сын"),
    ("сын", "HAS_PROPERTY", "зинаида"),
]


def knowledge(edges):
    """Мир из одних наблюдений — та же форма, что в контракте дыры F."""
    obs = {f"{h}|{r}|{t}": {"head": h, "relation": r, "tail": t}
           for h, r, t in edges}
    return StaticKnowledge({"observations": obs, "entities": {}})


def path(sk, start, goal, max_hops=4):
    """
    Тот же обход, что в П2: шаг вперёд И шаг назад (через referenced_by).
    Возвращает найденный путь списком (узел, отношение, узел) или None.
    """
    from collections import deque
    seen = {start}
    q = deque([(start, [])])
    while q:
        node, trail = q.popleft()
        if len(trail) >= max_hops:
            continue
        # прямой ход: по всем читаемым отношениям
        for rel in ("IS_ROLE", "PART_OF", "HAS_PROPERTY", "IS_A"):
            for t in sk.related(node, rel):
                step = (node, rel, t)
                if t == goal:
                    return trail + [step]
                if t not in seen:
                    seen.add(t)
                    q.append((t, trail + [step]))
        # обратный ход: кто ссылается на node
        for head, rel in sk.referenced_by(node):
            step = (node, rel, head)
            if head == goal:
                return trail + [step]
            if head not in seen:
                seen.add(head)
                q.append((head, trail + [step]))
    return None


def show(title, edges):
    sk = knowledge(edges)
    p = path(sk, "невестка", "егор")
    print(f"\n{title}")
    for h, r, t in edges:
        print(f"    {h} {r} {t}")
    if p is None:
        print("    ПУТЬ невестка -> егор: НЕТ ПУТИ")
    else:
        print("    ПУТЬ: " + " -> ".join(
            [p[0][0]] + [f"{r} {t}" for h, r, t in p]))
    return p


if __name__ == "__main__":
    a = show("МИР КАК ЕСТЬ (PART_OF):", EDGES_AS_IS)
    b = show("МИР С ЗАМЕНОЙ (HAS_PROPERTY):", EDGES_HAS_PROPERTY)
    print("\nВЫВОД ЗАМЕРА:")
    if a and b:
        print("    путь собирается в ОБОИХ мирах -> замена отношения "
              "функцию НЕ ломает, правку выбора экстрактора делать можно")
    elif a and not b:
        print("    с HAS_PROPERTY путь РВЁТСЯ -> замена недопустима, "
              "PART_OF остаётся, ложь документируется")
    else:
        print("    нечего сравнивать: путь не собрался даже в исходном мире "
              "(замер нерабочий, правку не делать)")
