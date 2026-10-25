# -*- coding: utf-8 -*-
"""
Очередь F — ЗАМЕР эмоции: что значат attack / decay / пороги НА САМОМ ДЕЛЕ.

Зачем. brain/emotions.py объявляет шкалу 0..255, скорость нарастания attack,
скорость затухания decay и шесть ступеней-слов. Все три числа помечены DRAFT
(«догадка»), а память велит считать калибровку отдельной задачей с замером.
Прежде чем что-либо подстраивать, надо знать, ЧТО именно подстраивается: на
шкале силы (очередь E) «5% за сон» оказалось не процентом, а минус один балл,
и «замедлить» его было нельзя. Проверка та же — на эмоциях.

Изолированно: живая память не трогается (этот скрипт никуда не пишет, мир
строится из experiments/*.json в оперативке). Учитель не зовётся.

Запуск:  LIPID2_STATUS=0 .venv/bin/python experiments/pF_emotion_measure.py

ВЫВОД СОДЕРЖИТ ТОЧНЫЕ ЧИСЛА. tests/test_emotion_calibration_contract.py
пересобирает их этим же скриптом и сверяет строку ИТОГ — число в комментарии
без команды не число.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.emotions import (BAD_LEVELS, GOOD_LEVELS, MAX_EMOTION,   # noqa: E402
                            EmotionScale, EmotionSystem)
from brain.social_sim import (AGGRESSION_THRESHOLD,                 # noqa: E402
                              CLOSENESS_LIVING_TOGETHER, RIPEN_TICKS,
                              build_cat_canary_simulator)

DEFPICK = dict(attack=40, decay=8)      # дефолт модуля


def peak_of(amount, good=False, **kw):
    """Максимум шкалы за всю жизнь эмоции и тик, на котором он взят."""
    e = EmotionSystem(**kw)
    (e.please if good else e.hurt)(amount)
    peak, t_peak, prev = 0, 0, 0
    for t in range(1, 4000):
        s = e.tick()
        v = s["good"] if good else s["bad"]
        if v > peak:
            peak, t_peak = v, t
        prev = v
        if v == 0 and peak > 0:
            return peak, t_peak, t
    return peak, t_peak, None


def name_of(value, good=False):
    levels = GOOD_LEVELS if good else BAD_LEVELS
    n = levels[0][1]
    for thr, lab in levels:
        if value >= thr:
            n = lab
    return n


print("=" * 74)
print("F/M1. attack — НЕ скорость нарастания: нарастание = attack - decay")
print("=" * 74)
print("  Причина в порядке операций внутри tick(): сначала подток из запаса")
print("  (+attack), затем затухание (-decay) — в ЭТОМ ЖЕ тике.")
for atk, dec in ((40, 8), (40, 0), (80, 8), (40, 40), (40, 60), (8, 8)):
    s = EmotionScale(attack=atk, decay=dec)
    s.stimulus(10 ** 6)
    vals = [s.tick() for _ in range(3)]
    slope = vals[0]
    print(f"  attack={atk:>3} decay={dec:>3}: значения по тикам {str(vals):<18}"
          f" скорость = {slope:>3}  (attack-decay={atk - dec:>3})"
          f"{'  <- ЭМОЦИИ НЕТ ВОВСЕ' if slope == 0 else ''}")
DEADZONE = EmotionScale(attack=40, decay=40)
DEADZONE.stimulus(10 ** 6)
dead = max(DEADZONE.tick() for _ in range(200))
print(f"\n  МЁРТВАЯ ЗОНА decay >= attack: при максимальном заряде 1000000")
print(f"  пик шкалы = {dead}, level() = «{name_of(dead)}», исключения нет,")
print(f"  заметки нет. Параметр attack перестаёт существовать молча.")

print()
print("=" * 74)
print("F/M2. потолок 255 при decay > 0 НЕДОСТИЖИМ: максимум = 255 - decay")
print("=" * 74)
ceilings = {}
for dec in (0, 1, 8, 40):
    s = EmotionScale(attack=40, decay=dec)
    s.stimulus(10 ** 6)
    p = max(s.tick() for _ in range(600))
    ceilings[dec] = p
    print(f"  decay={dec:>2}: максимум = {p:>3} из {MAX_EMOTION}"
          f"  {'достижим' if p == MAX_EMOTION else f'НЕДОСТИЖИМ (нехватка {MAX_EMOTION - p})'}")
CEIL_DEF = ceilings[8]
print(f"\n  Порогов у плохой шкалы 6, полоса у каждой = следующий порог.")
print(f"  Верхняя полоса «агрессия» (порог {AGGRESSION_THRESHOLD}) живёт в")
print(f"  {AGGRESSION_THRESHOLD}..{CEIL_DEF} вместо {AGGRESSION_THRESHOLD}..{MAX_EMOTION}.")

print()
print("=" * 74)
print("F/M3. КАРТА: какой стимул удостаивает агента какого слова (дефолт %s)"
      % DEFPICK)
print("=" * 74)
rows = {}
for amt in range(0, 601):
    p, _, _ = peak_of(amt, **DEFPICK)
    nm = name_of(p)
    a, b = rows.get(nm, (None, None))
    rows[nm] = (amt if a is None else a, amt)
for nm, (a, b) in rows.items():
    print(f"  {nm:>22}: заряд {a:>3}..{b:<3} (окно {b - a + 1})")
MAP_LAST = rows["агрессия"][0]
print(f"\n  Заряд <= {MAP_LAST - 1} никогда не скажет «агрессия»;")
print(f"  а заряд {MAP_LAST}..600 (и любой больше) скажет ЕДИНОЖДЫ это слово.")
print(f"  Верхние {600 - MAP_LAST + 1} значений заряда неразличимы.")

print()
print("=" * 74)
print("F/M4. выше насыщения величина заряда НЕ измеряется: она = длительность")
print("=" * 74)
durs = []
for amt in (100, 200, 400, 800, 1600, 3200):
    p, _, life = peak_of(amt, **DEFPICK)
    durs.append((amt, p, life))
    print(f"  hurt({amt:>4}): пик {p:>3} ({name_of(p):>20})  живёт {life:>3} тика")
print("\n  Пик зацикливается на %d, а жизнь растёт линейно: сила удара выше" % CEIL_DEF)
print("  насыщения превращается в СРОК. Это можно принять как смысл (обида")
print("  сильнее — дольше живёт) или запретить, но молчать нельзя.")

print()
print("=" * 74)
print("F/M5. baseline: любовь УМИРАЕТ ОТ ТИШИНЫ (то, что память просит снять)")
print("=" * 74)
for amt in (120, 480):
    e = EmotionSystem(**DEFPICK)
    e.please(amt)
    t_zero = next((t for t in range(1, 500) if e.tick()["good"] == 0), None)
    if amt == 120:
        GOOD_ZERO_120 = t_zero
    print(f"  please({amt:>3}) и тишина: шкала good = 0 через {t_zero} тиков")
e = EmotionSystem(**DEFPICK)
e.please(10 ** 6)
never = all(e.tick()["good"] for _ in range(3000))
print(f"  please(1000000): за 3000 тиков тишины НЕ гаснет = {never}")
print(f"  (огромный заряд не «сильнее любит», а просто дольше питает запас)")
BASE_OK = 0
print(f"\n  Нынешний покой обеих шкал = {BASE_OK}: «безразличие» — это не")
print(f"  отсутствие чувства, а отсутствие события за последние {GOOD_ZERO_120}"
      f" тиков (замер для заряда 120).")

print()
print("=" * 74)
print("F/M6. рассказ: сверяем комментарий «окно гнева 128/247/231/199»")
print("=" * 74)
claim = {4: 128, 8: 247, 12: 231, 16: 199}
ok_claim = []
for n in sorted(claim):
    sim = build_cat_canary_simulator(between_ticks=n)
    bad = sim.snapshots[-1]["кошка"]["bad"]
    ok = bad == claim[n]
    ok_claim.append(ok)
    print(f"  between_ticks={n:>2}: плохое на момент действия = {bad:>3}"
          f"  (комментарий обещает {claim[n]:>3})  {'совпало' if ok else 'РАСХОЖДЕНИЕ'}")
window = []
for n in range(1, 25):
    sim = build_cat_canary_simulator(between_ticks=n)
    if sim.why_acted("кошка", "съесть").ok:
        window.append(n)
mid = (window[0] + window[-1]) / 2
print(f"\n  ОКНО СПУСКА (порог {AGGRESSION_THRESHOLD}): {window[0]}..{window[-1]}"
      f"  середина = {mid}, RIPEN_TICKS = {RIPEN_TICKS}"
      f"{'  <- НЕ середина' if mid != RIPEN_TICKS else ''}")
print(f"  ХРУПКОСТЬ: до и после окна рассказ ОТКАЗЫВАЕТСЯ выводить ответ,")
print(f"  а внутри окна держится {len(window)} значениями выдержки.")

print()
print("=" * 74)
print("F/M7. рассказ держится на ОДНОМ DRAFT-числе: близость")
print("=" * 74)
cliff = None
for c in (0.5, 1.0, 1.25, 1.5, 1.75, CLOSENESS_LIVING_TOGETHER, 3.0):
    sim = build_cat_canary_simulator(closeness=c, between_ticks=RIPEN_TICKS)
    ok = sim.why_acted("кошка", "съесть").ok
    bad = sim.snapshots[-1]["кошка"]["bad"]
    if cliff is None and ok:
        cliff = c
    print(f"  closeness={c:>4}: спуск={str(ok):>5}  плохое={bad:>3}")
print(f"\n  Рассказ «кошка съела канарейку из мести» живёт при closeness >= {cliff}.")
print(f"  CLOSENESS_LIVING_TOGETHER = {CLOSENESS_LIVING_TOGETHER} стоит с пометкой")
print(f"  DRAFT. Порог 230 и заряд 200 — тоже DRAFT; втроём они и дают сюжет.")

print()
print("=" * 74)
print("ИТОГ (сверяется тестом tests/test_emotion_calibration_contract.py):")
print(f"  real_slope_default={DEFPICK['attack'] - DEFPICK['decay']} "
      f"deadzone_peak={dead} ceiling_default={CEIL_DEF} "
      f"ceiling_formula=MAX_EMOTION-decay "
      f"aggression_min_charge={MAP_LAST} "
      f"good_zero_ticks_120={GOOD_ZERO_120} window={window[0]}..{window[-1]} "
      f"ripen={RIPEN_TICKS} window_mid={mid} "
      f"comment_numbers_match={all(ok_claim)} "
      f"closeness_cliff={cliff} baseline_now=0")
print("=" * 74)
