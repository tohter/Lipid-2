#!/usr/bin/env python3
"""
tools/cortex_demo.py — самопроверка предиктивного цикла (переехала из ядра).

ПОЧЕМУ ОНА ЗДЕСЬ, А НЕ В core/cortex.py. Ядро не имеет права импортировать
lang/ — даже в демо, даже «совсем чуть-чуть» (закон C_core_reaches_lang = 0,
experiments/IR_BOUNDARY.md). Прежнее `python core/cortex.py` знало русский:
оно звало морфологию, чтобы разобрать фразы про кошку.

ВТОРАЯ ПРИЧИНА, ИЗМЕРЕННАЯ ПО ПУТИ. Первый переезд был половинчатый: тело
самопроверки оставили в ядре функцией run_demo, а язык вынесли. Трещотка
поймала это числом: R_core_ruslits вырос 169 -> 173, потому что строки «пить»
и «кошка» (сравнение `"пить" in graph.nodes`, запрос predict("кошка"))
оказались ВНУТРИ библиотечного пути — вне `__main__`, а значит не демо, а
русские ДАННЫЕ в ядре. Самопроверка, которой нужен язык, переедает целиком
или не переезжает вовсе: делить её на «умную часть в ядре» и «язык снаружи»
нельзя, иначе ядро тихо обрастает русским.

ВЫВОД ОБЯЗАН СОВПАДАТЬ ПОБАЙТНО с прежней самопроверкой ядра. Равенство
проверяет tests/test_segmenter_contract.py на эталоне, снятом ДО переноса, —
иначе «перенесли дословно» оставалось бы обещанием, а не фактом.
"""
from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cortex import PredictiveCortex
from core.graph import GraphMemory
from lang.ru.text import experience

# Узлы-свидетели: цикл читает граф, но создавать узлы не вправе. Именно это и
# проверяет вывод — «новых узлов: set()».
NODES = ("кошка", "собака", "вода", "пламя", "огонь", "подоконник",
         "спать", "молоко")

TEXT = ("Кошка сидит на подоконнике. Кошка пьёт молоко. "
        "Собака бегает во дворе. Кошка пьёт молоко.")


def main() -> dict:
    graph = GraphMemory()
    for w in NODES:
        graph.get_or_create_node(w)
    graph.sync_lemmas()
    loop = PredictiveCortex(graph, segmenter=experience)
    # ↓ ТЕЛО ПРЕЖНЕГО `if __name__ == "__main__"` из core/cortex.py, дословно.
    nodes_before = set(graph.nodes)
    loop.process_stream(TEXT, verbose=True)
    m = loop.metrics()
    print("\nПосле цикла узел «пить» создан?", "пить" in graph.nodes,
          "| новых узлов:", set(graph.nodes) - nodes_before)
    print("ожидаемое продолжение после «кошка»:", loop.predict("кошка"))
    loop.query("кошка")
    print("\n⚠  ТИКИ:", m["ticks"], "СТРОГАЯ ТОЧНОСТЬ:",
          f"{m['accuracy_strict']:.0%}", "ПОКРЫТИЕ:", f"{m['coverage']:.0%}",
          "ПЕРЕХОДОВ:", m["transitions"])
    return m


if __name__ == "__main__":
    main()
