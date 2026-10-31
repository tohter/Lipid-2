"""
Расширенный замер пластичности с вариацией параметров.
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
from lang.ru.text import experience as ru_experience  # IR A2.4: разбор приносят ядру


LESSON_TEXTS = [
    "Кошка спит на подоконнике. Кошка пьёт молоко. Кошка мяукает.",
    "Собака бегает во дворе. Собака лает на кошку. Собака ест мясо.",
    "Птица летит в небе. Птица поёт утром. Птица строит гнездо.",
]

TRANSFER_TEXT = "Кошка бегает во дворе. Собака спит на подоконнике. Птица пьёт воду."


@dataclass
class Config:
    delta_confirm: int = 1
    delta_surprise: int = 1
    soft_overlap: int = 13
    dopamine_gain_hit: int = 120
    norepinephrine_gain_no_exp: int = 40
    norepinephrine_gain_miss: int = 120
    norepinephrine_gain_new: int = 40
    cortisol_gain_miss: int = 100
    predictability_gain_hit: int = 15
    predictability_loss_miss: int = 25
    hormone_decay_dopa: int = 40
    hormone_decay_norep: int = 60
    hormone_decay_cort: int = 30


def make_brain(cfg: Config) -> tuple[GraphMemory, PredictiveCortex, Hormones, Homeostasis]:
    graph = GraphMemory()
    for w in ("кошка", "собака", "птица", "подоконник", "молоко", "двор",
              "небо", "утро", "гнездо", "мясо", "вода", "спать", "петь",
              "строить", "бегать", "лаять", "лететь", "пить", "мяукать", "есть"):
        graph.get_or_create_node(w)
    graph.sync_lemmas()
    
    hormones = Hormones()
    homeostasis = Homeostasis()
    transitions = TransitionTable()
    plasticity = PredictiveCortex.PlasticityConfig(
        delta_confirm=cfg.delta_confirm,
        delta_surprise=cfg.delta_surprise,
        dopamine_gain_hit=cfg.dopamine_gain_hit,
        norepinephrine_gain_no_exp=cfg.norepinephrine_gain_no_exp,
        norepinephrine_gain_miss=cfg.norepinephrine_gain_miss,
        norepinephrine_gain_new=cfg.norepinephrine_gain_new,
        cortisol_gain_miss=cfg.cortisol_gain_miss,
        predictability_gain_hit=cfg.predictability_gain_hit,
        predictability_loss_miss=cfg.predictability_loss_miss,
        hormone_decay_dopa=cfg.hormone_decay_dopa,
        hormone_decay_norep=cfg.hormone_decay_norep,
        hormone_decay_cort=cfg.hormone_decay_cort,
        soft_overlap_threshold=cfg.soft_overlap,
    )
    cortex = PredictiveCortex(graph, transitions=transitions, hormones=hormones, plasticity=plasticity, segmenter=ru_experience)
    cortex.homeostasis = homeostasis
    return graph, cortex, hormones, homeostasis


def run_lesson(cortex: PredictiveCortex, text: str) -> dict:
    m = cortex.process_stream(text, verbose=False)
    return m


def measure_config(cfg: Config, repetitions: int = 10) -> Dict[str, Any]:
    graph, cortex, hormones, homeostasis = make_brain(cfg)
    
    baseline = []
    for lesson in range(repetitions):
        for text in LESSON_TEXTS:
            m = run_lesson(cortex, text)
            m["lesson"] = lesson
            baseline.append(m)
    
    transfer = run_lesson(cortex, TRANSFER_TEXT)
    transfer["lesson"] = -1
    
    last = baseline[-1]
    first = baseline[0]
    acc_growth = last["accuracy_strict"] - first["accuracy_strict"]
    
    return {
        "config": asdict(cfg),
        "baseline": baseline,
        "transfer": transfer,
        "summary": {
            "accuracy_growth": acc_growth,
            "final_accuracy": last["accuracy_strict"],
            "final_coverage": last["coverage"],
            "transfer_accuracy": transfer["accuracy_strict"],
            "overfitting": transfer["accuracy_strict"] < last["accuracy_strict"] - 0.1,
            "final_predictability": last["predictability"],
        }
    }


def main():
    print("=" * 70)
    print("РАСШИРЕННЫЙ ЗАМЕР ПЛАСТИЧНОСТИ (п.16)")
    print("=" * 70)
    
    configs = [
        Config(),
        Config(delta_confirm=2, delta_surprise=1),
        Config(delta_confirm=3, delta_surprise=2),
        Config(delta_confirm=2, delta_surprise=1, soft_overlap=10),
        Config(delta_confirm=2, delta_surprise=1, soft_overlap=17),
        Config(delta_confirm=1, delta_surprise=0),
        Config(delta_confirm=2, delta_surprise=1, predictability_gain_hit=10, predictability_loss_miss=10),
        Config(delta_confirm=2, delta_surprise=1, hormone_decay_dopa=20, hormone_decay_norep=30, hormone_decay_cort=15),
    ]
    
    results = []
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] delta_c={cfg.delta_confirm}, delta_s={cfg.delta_surprise}, soft={cfg.soft_overlap}")
        r = measure_config(cfg, 10)
        results.append(r)
        s = r["summary"]
        print(f"  acc={s['final_accuracy']:.1%} cov={s['final_coverage']:.1%} "
              f"transfer={s['transfer_accuracy']:.1%} pred={s['final_predictability']} "
              f"growth={s['accuracy_growth']:+.1%} overfit={'OVERFIT' if s['overfitting'] else 'OK'}")
    
    out_path = os.path.join(os.path.dirname(__file__), "plasticity_sweep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nВсе результаты: {out_path}")
    
    print("\n" + "=" * 70)
    print("АНАЛИЗ")
    print("=" * 70)
    for r in results:
        cfg = r["config"]
        s = r["summary"]
        print(f"delta_c={cfg['delta_confirm']}, delta_s={cfg['delta_surprise']}, soft={cfg['soft_overlap']}: "
              f"acc={s['final_accuracy']:.1%} cov={s['final_coverage']:.1%} "
              f"trans={s['transfer_accuracy']:.1%} growth={s['accuracy_growth']:+.1%} "
              f"pred={s['final_predictability']} {'OVERFIT' if s['overfitting'] else ''}")


if __name__ == "__main__":
    main()
