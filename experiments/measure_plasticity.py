"""
Замер чисел пластичности предиктивного цикла (п.16 очереди).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph import GraphMemory
from core.cortex import PredictiveCortex, Hormones, Homeostasis
from core.transitions import TransitionTable


LESSON_TEXTS = [
    "Кошка спит на подоконнике. Кошка пьёт молоко. Кошка мяукает.",
    "Собака бегает во дворе. Собака лает на кошку. Собака ест мясо.",
    "Птица летит в небе. Птица поёт утром. Птица строит гнездо.",
]

TRANSFER_TEXT = "Кошка бегает во дворе. Собака спит на подоконнике. Птица пьёт воду."


@dataclass
class PlasticityConfig:
    delta_confirm: int = 1
    delta_surprise: int = 1
    dopamine_gain: int = 120
    norepinephrine_gain_hit: int = 40
    norepinephrine_gain_miss: int = 120
    norepinephrine_gain_new: int = 40
    cortisol_gain: int = 100
    predictability_gain: int = 15
    predictability_loss: int = 25
    hormone_decay_dopa: int = 40
    hormone_decay_norep: int = 60
    hormone_decay_cort: int = 30
    soft_overlap_threshold: int = 13


@dataclass
class SessionMetrics:
    lesson: int
    text_hash: str
    ticks: int
    hits: int
    misses: int
    no_expectation: int
    soft_overlap_hits: int
    surprises: int
    new_transitions: int
    accuracy_strict: float
    coverage: float
    accuracy_when_expected: float
    transitions_total: int
    predictability: int
    hormones: Dict[str, int]


def make_brain() -> tuple[GraphMemory, PredictiveCortex, Hormones, Homeostasis]:
    graph = GraphMemory()
    for w in ("кошка", "собака", "птица", "подоконник", "молоко", "двор",
              "небо", "утро", "гнездо", "мясо", "вода", "спать", "петь",
              "строить", "бегать", "лаять", "лететь", "пить", "мяукать", "есть"):
        graph.get_or_create_node(w)
    graph.sync_lemmas()
    
    hormones = Hormones()
    homeostasis = Homeostasis()
    transitions = TransitionTable()
    cortex = PredictiveCortex(graph, transitions=transitions, hormones=hormones)
    cortex.homeostasis = homeostasis
    return graph, cortex, hormones, homeostasis


def run_lesson(cortex: PredictiveCortex, text: str, verbose: bool = False) -> SessionMetrics:
    m = cortex.process_stream(text, verbose=verbose)
    return SessionMetrics(
        lesson=0, text_hash=str(hash(text)),
        ticks=m["ticks"], hits=m["hits"], misses=m["misses"],
        no_expectation=m["no_expectation"], soft_overlap_hits=m["soft_overlap_hits"],
        surprises=m["surprises"], new_transitions=m["new_transitions"],
        accuracy_strict=m["accuracy_strict"], coverage=m["coverage"],
        accuracy_when_expected=m["accuracy_when_expected"],
        transitions_total=m["transitions"], predictability=m["predictability"],
        hormones=m["hormones"],
    )


def measure_baseline(repetitions: int = 10) -> List[SessionMetrics]:
    graph, cortex, hormones, homeostasis = make_brain()
    all_metrics = []
    
    for lesson in range(repetitions):
        for text in LESSON_TEXTS:
            m = run_lesson(cortex, text)
            m.lesson = lesson
            all_metrics.append(m)
            if lesson == 0:
                print(f"  Урок 1: ticks={m.ticks}, acc={m.accuracy_strict:.1%}, cov={m.coverage:.1%}, trans={m.transitions_total}")
            elif lesson == repetitions - 1:
                print(f"  Урок {lesson+1}: ticks={m.ticks}, acc={m.accuracy_strict:.1%}, cov={m.coverage:.1%}, trans={m.transitions_total}")
    return all_metrics


def measure_transfer(cortex: PredictiveCortex) -> SessionMetrics:
    m = run_lesson(cortex, TRANSFER_TEXT)
    m.lesson = -1
    print(f"  Transfer: ticks={m.ticks}, acc={m.accuracy_strict:.1%}, cov={m.coverage:.1%}, soft={m.soft_overlap_hits}")
    return m


def main():
    print("=" * 70)
    print("ЗАМЕР ПЛАСТИЧНОСТИ ПРЕДИКТИВНОГО ЦИКЛА (п.16)")
    print("=" * 70)
    
    print("\n[1] Базовая линия (дефолтные параметры, 10 повторений)...")
    graph, cortex, hormones, homeostasis = make_brain()
    baseline = measure_baseline(10)
    
    print("\n[2] Перенос (новые комбинации известных слов)...")
    transfer = measure_transfer(cortex)
    
    print("\n[3] Анализ...")
    last = baseline[-1]
    first = baseline[0]
    
    print(f"\n  ИТОГИ БАЗОВОЙ ЛИНИИ (дефолт):")
    print(f"    Урок 1:   accuracy={first.accuracy_strict:.1%} coverage={first.coverage:.1%} trans={first.transitions_total} pred={first.predictability}")
    print(f"    Урок 10:  accuracy={last.accuracy_strict:.1%} coverage={last.coverage:.1%} trans={last.transitions_total} pred={last.predictability}")
    print(f"    Transfer: accuracy={transfer.accuracy_strict:.1%} coverage={transfer.coverage:.1%} soft_overlap={transfer.soft_overlap_hits}")
    
    acc_growth = last.accuracy_strict - first.accuracy_strict
    print(f"    Рост точности за 10 уроков: {acc_growth:+.1%}")
    print(f"    Переобучение (transfer < урок 10): {'ДА ⚠️' if transfer.accuracy_strict < last.accuracy_strict - 0.1 else 'НЕТ ✓'}")
    
    results = {
        "baseline": [asdict(m) for m in baseline],
        "transfer": asdict(transfer),
        "config": asdict(PlasticityConfig()),
        "summary": {
            "accuracy_growth": acc_growth,
            "final_accuracy": last.accuracy_strict,
            "final_coverage": last.coverage,
            "transfer_accuracy": transfer.accuracy_strict,
            "overfitting": transfer.accuracy_strict < last.accuracy_strict - 0.1,
        }
    }
    
    out_path = os.path.join(os.path.dirname(__file__), "plasticity_baseline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Результаты сохранены: {out_path}")
    
    print("\n[4] Рекомендации для контракта:")
    if last.accuracy_strict < 0.5:
        print("  - Точность низка: увеличить delta_confirm / уменьшить delta_surprise")
    if last.coverage < 0.5:
        print("  - Покрытие низко: нужен больше опыта или понизить SOFT_OVERLAP")
    if transfer.accuracy_strict < last.accuracy_strict - 0.1:
        print("  - Переобучение: добавить затухание переходов (decay) или лимит силы")
    if last.predictability < 100:
        print("  - Предсказуемость не насыщается: проверить predictability_gain/loss")
    
    return results


if __name__ == "__main__":
    main()
