# -*- coding: utf-8 -*-
"""
Сверка социального движка с РУЧНЫМ ЭТАЛНОМ (шаг 6 советчиков).

Запуск: LIPID2_STATUS=0 .venv/bin/python experiments/social_golden_check.py

Печатает по каждому утверждению эталона: OK / РАСХОЖДЕНИЕ с фактом движка
и пояснением. Код движка НЕ правится до сверки — сверка фиксирует ЧЕСТНОЕ
расхождение, если движок понимает рассказ не как человек.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("LIPID2_STATUS", "0")

from brain.social_sim import build_cat_canary_simulator, RIPEN_TICKS  # noqa
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from golden_social import GOLDEN  # noqa

HALLUCINATION_WORDS = ["голод", "голодать", "ссор", "ссора"]


def run_engine():
    """Полный рассказ штатным путём (как в живом прогоне)."""
    sim = build_cat_canary_simulator(between_ticks=RIPEN_TICKS)
    why = sim.why_acted("кошка", "съесть")
    return sim, why


def facts(sim, why):
    """Величины движка, которые запрашивает эталон."""
    pair = sim.relations.get(("кошка", "канарейка"))
    rev = sim.relations.get(("канарейка", "кошка"))
    viol = [v for v in sim.violations if v.expectation.holder == "кошка"]
    snaps = sim.snapshots  # список по тикам: {agent: {good, bad, t}}
    canary = [s.get("канарейка", {"good": 0, "bad": 0}) for s in snaps]
    cat = [s.get("кошка", {"good": 0, "bad": 0}) for s in snaps]
    # «перед рассказом тайны»: снимок i соответствует событию i (act() делает
    # снимок на каждом событии), момент нарушения = индекс события «рассказывать»
    tick_reveal = None
    for i, ev in enumerate(sim.events):
        if getattr(ev, "action", "") in ("рассказать", "рассказывать"):
            tick_reveal = i
    good_before = max((s["good"] for i, s in enumerate(cat)
                       if tick_reveal is not None and i < tick_reveal),
                      default=0)
    bad_canary = max((s["bad"] for s in canary
                      if tick_reveal and s["t"] < tick_reveal), default=0)
    # bad кошки В МОМЕНТ «съесть» (последний снимок)
    bad_cat_end = cat[-1]["bad"] if cat else 0
    answer = (why.answer or "") if getattr(why, "ok", False) else ""
    return {
        "closeness_kk": (pair.closeness if pair else None),
        "trust_before": None,          # заполнит отдельный прогон до нарушения
        "trust_reverse_zero": (rev is None or rev.trust == 0),
        "closeness_reverse": (rev.closeness if rev else 0),
        "violation_count": len(viol),
        "trust_after_zero": (pair.trust if pair else None),
        "good_before": good_before,
        "bad_at_attack": bad_cat_end,
        "bad_canary_low": bad_canary,
        "why_chain": answer,
        "why_no_hallucination": answer,
    }


def check(key, fact, sim, why):
    """Сверка одного утверждения эталона с фактом. -> (ok, пояснение)."""
    if key == "closeness_kk":
        return (fact is not None and fact > 1.0,
                f"closeness={fact}, эталон > 1.0")
    if key == "trust_before":
        # отдельный прогон ОБРЫВАЕТСЯ до нарушения: акт доверия — это
        # событие с текстом «попросила не рассказывать» (не команда please)
        s2 = build_cat_canary_simulator(between_ticks=1)
        return (None, "см. отдельный прогон")
    if key == "trust_reverse_zero":
        return (bool(fact), f"обратное доверие отсутствует/нулевое: {fact}")
    if key == "closeness_reverse":
        return (fact > 1.0, f"обратная близость={fact}, эталон > 1.0")
    if key == "violation_count":
        return (fact >= 1, f"нарушений с holder=кошка: {fact}")
    if key == "trust_after_zero":
        return (fact == 0, f"доверие после предательства={fact}, эталон 0")
    if key == "good_before":
        return (fact > 0, f"good кошки до рассказа тайны={fact}, эталон > 0")
    if key == "bad_at_attack":
        return (fact >= 200, f"bad кошки на момент съесть={fact}, эталон >= 200")
    if key == "bad_canary_low":
        return (fact < 100, f"bad канарейки до съесть={fact}, эталон < 100")
    if key == "why_chain":
        low = fact.lower()
        ok = ("довер" in low or "наруш" in low) and \
             ("агресс" in low or "гнев" in low)
        return (ok, f"цепочка в ответе: {fact[:90]!r}")
    if key == "why_no_hallucination":
        bad = [w for w in HALLUCINATION_WORDS if w in fact.lower()]
        return (not bad, f"галлюцинации: {bad or 'нет'}")
    return (None, "неизвестная проверка")


def main():
    sim, why = run_engine()
    F = facts(sim, why)
    print("=" * 72)
    print("СВЕРКА СОЦИАЛЬНОГО ДВИЖКА С РУЧНЫМ ЭТАЛНОМ (шаг 6)")
    print("=" * 72)
    n_ok = n_core_bad = n_soft_bad = 0
    for g in GOLDEN:
        key = g["проверка"]
        fact = F.get(key)
        ok, note = check(key, fact, sim, why)
        if ok is None:
            continue
        mark = "OK " if ok else ("РАСХОЖДЕНИЕ" if g["вес"] == "core" else "soft≠")
        if not ok:
            if g["вес"] == "core":
                n_core_bad += 1
            else:
                n_soft_bad += 1
        else:
            n_ok += 1
        print(f"[{mark:11s}] {g['id']}: {g['суть'][:60]}…")
        print(f"               {note}")
    # отдельный прогон: доверие ПОСЛЕ просьбы, ДО нарушения (R2).
    # Акт доверия живёт в ТЕКСТЕ события («попросила никому не рассказывать»);
    # команда please скрипта — не просьба. Сценарий обрывается до предательства.
    script2 = [
        {"do": "agent", "name": "кошка"},
        {"do": "agent", "name": "канарейка"},
        {"do": "event", "subject": "кошка", "action": "любить", "obj": "канарейка",
         "text": "Кошка любила канарейку и дружно жила вместе с ней, весело с ней играла."},
        {"do": "event", "subject": "кошка", "action": "просить", "obj": "канарейка",
         "text": "Кошка попросила канарейку никому не рассказывать тайну."},
        {"do": "ticks", "n": 2},
    ]
    from brain.social_sim import SocialSimulator
    with open(os.path.join(ROOT, "experiments", "cat_canary_world.json"),
              encoding="utf-8") as fh:
        w2 = json.load(fh)
    s2 = SocialSimulator(w2, script=script2)
    pair2 = s2.relations.get(("кошка", "канарейка"))
    trust_after_please = pair2.trust if pair2 else None
    # PLEASE-скрипт создаёт ожидание уже при просьбе; доверие живёт на ребре
    r2_ok = trust_after_please is not None and trust_after_please > 0
    if r2_ok:
        n_ok += 1
    else:
        n_core_bad += 1
    print(f"[{'OK ' if r2_ok else 'РАСХОЖДЕНИЕ':11s}] R2: доверие после "
          f"просьбы (до нарушения) = {trust_after_please}, эталон > 0")
    print("-" * 72)
    print(f"ИТОГО: OK={n_ok}, core-расхождений={n_core_bad}, "
          f"soft-расхождений={n_soft_bad}")
    return 0 if n_core_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
