#!/usr/bin/env python3
"""СКОЛЬКО СТОИТ долгая живая сессия: замер цены урока и сна, а не обещание.

Очередь memory.txt, п.D. Пункт открыт, потому что сквозной цикл на дистанции не
проверялся НИ РАЗУ, но начать его нельзя без ответа на вопрос «сколько это
времени». «Сотни уроков» без замера — это не план, а пожелание: я не знаю, вечер
это или сутки, и хозяин не знает.

ЧТО ЗДЕСЬ МЕРИЕТСЯ И ЧЕГО ЗДЕСЬ НЕТ.

  МЕРЯЕТСЯ: цена одного урока и одного сна на ПОЛНОМ мозге (Lipid2Brain:
  экстрактор + learner + предиктивный цикл + сон), цена записи файла мира, рост
  файла мира на уроках и поведение сил/весов на дистанции.

  НЕ МЕРИЕТСЯ КАНАЛ УЧИТЕЛЯ. Учитель (llama.cpp @8080) здесь ВЫКЛЮЧЕН
  намеренно: мерить канал моделью, которой нет, было бы враньём — замер выдал бы
  «канал работает, 0 обращений» о канале, который никто не звал. Поэтому числа
  этого скрипта — НИЖНЯЯ граница времени, а не план сессии. ЦЕНЫ ОДНОГО ОТВЕТА
  ЖИВОГО УЧИТЕЛЯ В ПРОЕКТЕ НЕ ИЗМЕРЕНО (проверено grep'ом по ARCHIVE_SESSIONS:
  там нет ни одной цены ответа; 180 с в teachers/teacher1.py — это ТАЙМАУТ
  ожидания ответа, LIPID2_TEACHER_TIMEOUT, а не стоимость ответа).
  Кто назовёт верхнюю границу сессии без этого замера — тот называет её глазами:
  чтобы оценка стала числом, скрипт надо прогнать с поднятой локалкой.

  НЕ МЕРИЕТСЯ GUI. Здесь тот же мозг, что под GUI, но без tk: окно не ускоряет и
  не замедляет цикл, а tk в замере только мешал бы (e2e_gui неимпортируем —
  см. п.F1).

ИЗОЛЯЦИЯ. Каталог памяти — временный (LIPID2_MEMORY_DIR сюда подставляется
самим скриптом, если переменная не задана). Живой data/memory/ НЕ ТРОГАЕТСЯ:
перед выходом скрипт сверяет его хэш и МОЛЧИТ, если совпал, и КРИЧИТ, если нет.
Тот же класс хазарда, что стоил проекту lipid_db.json (урок 52), здесь закрыт
до первого вызова, а не после потери.

  LIPID2_STATUS=0 .venv/bin/python tools/measure_long_session.py [--lessons N]
                                                                 [--sleep-every N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LIVE_WORLD = os.path.join(ROOT, "data", "memory", "lipid_world.json")
LIVE_DB = os.path.join(ROOT, "data", "memory", "lipid_db.json")

# Разные предложения, а не одно и то же: один и тот же текст на дистанции
# измерял бы кэш, а не рост ядра. 20 фраз -> цикл по кругу.
TEXTS = [
    "Кошка ловит мышь.", "Кошка спит на подоконнике.", "Вода гасит огонь.",
    "Птица летит в небе.", "Рыба живёт в воде.", "Мальчик читает книгу.",
    "Собака охраняет дом.", "Дерево растёт в лесу.", "Пчела собирает мёд.",
    "Зима приносит снег.", "Огонь даёт тепло.", "Ветер гонит облака.",
    "Кузнец кует гвозди.", "Мельница мелит зерно.", "Пчела делает мёд.",
    "Лиса живёт в норе.", "Волк охотится стаей.", "Пчеловод проверяет улей.",
    "Садовник поливает яблоню.", "Море качает лодку.",
]


def _sha(path: str) -> str:
    if not os.path.exists(path):
        return "<нет файла>"
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lessons", type=int, default=100)
    ap.add_argument("--sleep-every", type=int, default=20)
    ap.add_argument("--texts", default=None,
                    help="JSON-список фраз. Без него мир растёт из 8 фраз — и "
                         "вопрос «растёт ли ядро» таким прогоном НЕ закрывается")
    args = ap.parse_args()

    texts = TEXTS
    if args.texts:
        texts = json.load(open(args.texts, encoding="utf-8"))

    live_before = (_sha(LIVE_WORLD), _sha(LIVE_DB))
    tmp = os.environ.get("LIPID2_LONG_DIR") or tempfile.mkdtemp(prefix="l2_long_")
    os.makedirs(tmp, exist_ok=True)
    os.environ["LIPID2_MEMORY_DIR"] = tmp
    # Учитель ВЫКЛЮЧЕН: см. докстринг. Числа — нижняя граница.
    os.environ["LIPID2_LLM_HOST"] = ""

    from brain.lipid2_brain import Lipid2Brain
    from brain.sleep import has_meaningful_work, table_of_section

    t0 = time.time()
    brain = Lipid2Brain(memory_dir=tmp)
    boot = time.time() - t0

    lesson_times, sleep_times, snap, slept = [], [], [], []
    world_path = os.path.join(tmp, "lipid_world.json")

    for i in range(args.lessons):
        t = time.time()
        brain.learn_from_text(texts[i % len(texts)], source="measure",
                              allow_teacher=False)
        lesson_times.append(time.time() - t)
        # СНИМОК берётся ДО сна. Иначе таблица врёт: замер ровно после
        # затухания показывает переходов 0 при живом мире между снами (мой
        # первый прогон дал именно «0 переходов», и это был артефакт ритма,
        # а не результат).
        if args.sleep_every and (i + 1) % args.sleep_every == 0:
            # sleep_now возвращает {"report": SleepReport, "text": ..., "saved":
            # ...}. Отчёт — ОБЪЕКТ с полями created/raised/lowered/decayed, а не
            # словарь: rep.get(...) здесь молча вернул бы None и «сон никогда не
            # работает» уехало бы в отчёт как факт.
            w = (json.load(open(world_path, encoding="utf-8"))
                 if os.path.exists(world_path) else {})
            obs = (w.get("observations") or {})
            # ШКАЛА ДВУХ РАЗНЫХ ЧИСЕЛ, и путать их нельзя.
            #  * вес наблюдения (weight) — 0..255, им заведует сон;
            #  * сила перехода (strengths) — 4-битный счётчик 0..15.
            # Сила перехода живёт в "strengths" — СПИСОК рядом с "tails"
            # (строка таблицы: {head, tails[], strengths[]}), а НЕ в поле
            # "strength": мера по неверному ключу молча даёт пустой список, и
            # «сила неизвестна» уезжает в отчёт как факт. Замер 01.11.
            strengths = [o["weight"] for o in obs.values()
                         if isinstance(o.get("weight"), int)]
            # Переходы считаются МЕРИТОМ brain.sleep, а не своим ключом: секция
            # transitions устроена не так, как кажется по имени.
            n_trans = len(table_of_section(w.get("transitions") or {}))
            rows = ((w.get("transitions") or {}).get("rows")) or []
            strengths_t = [s for r in rows if isinstance(r, dict)
                           for s in (r.get("strengths") or [])
                           if isinstance(s, int)]
            snap.append({
                "lessons": i + 1,
                "entities": len(w.get("entities") or {}),
                "observations": len(obs),
                "transitions": n_trans,
                "world_kb": round(os.path.getsize(world_path) / 1024, 1)
                if os.path.exists(world_path) else 0,
                "strength_max": (max(strengths) if strengths else None),
                "tstr": (min(strengths_t), max(strengths_t)) if strengths_t
                else None,
            })
            # Сон — ПОСЛЕ снимка (см. выше). Цена и отчёт сна относятся к ЭТОМУ
            # сна и печатаются отдельным блоком ниже: в строку снимка они не
            # лезут, иначе колонка врёт на первом снимке, где спать ещё не
            # пришлось.
            s = time.time()
            out = brain.sleep_now()
            sleep_times.append(time.time() - s)
            rep = (out or {}).get("report")
            slept.append({"after": i + 1, "s": round(time.time() - s, 3),
                          "examined": getattr(rep, "examined", None),
                          "created": getattr(rep, "created", None),
                          "raised": getattr(rep, "raised", None),
                          "lowered": getattr(rep, "lowered", None),
                          "decayed": getattr(rep, "decayed", None),
                          "meaningful": (has_meaningful_work(rep)
                                         if rep is not None else None)})

    # ЭКСТРАПОЛЯЦИЯ НЕ СРЕДНИМ ПО ВСЕМ. Первый урок включает ОДНОРАЗОВУЮ
    # загрузку кодировщика (замер: 8.6 с против 0.055 с на медиане), и среднее
    # по всем урокам раздувает оценку на 600 уроках в ~14 раз. Разовое стоит
    # один раз, на дистанции его нет — значит переносится ТОЛЬКО цена урока без
    # прогрева, а старт печатается отдельной строкой (он и так напечатан выше).
    warm = lesson_times[0]
    steady = lesson_times[1:] or lesson_times
    med = sorted(steady)[len(steady) // 2]
    total = sum(lesson_times) + sum(sleep_times)
    print("=" * 68)
    print(f"ЗАМЕР ЦЕНЫ ДОЛГОЙ СЕССИИ (п.D), мозг ПОЛНЫЙ, Учитель ВЫКЛЮЧЕН")
    print("=" * 68)
    print(f"старт мозга:            {boot:6.2f} с")
    print(f"уроков прогнано:        {args.lessons}  "
          f"(уникальных фраз: {len(set(texts))})")
    print(f"прогрев (первый урок):  {warm:6.2f} с  — одноразово, в оценку не входит")
    print(f"урок без прогрева: медиана {med:5.3f} с  средняя "
          f"{sum(steady)/len(steady):5.3f} с  "
          f"макс {max(steady):5.3f} с")
    if sleep_times:
        print(f"сон:  медиана {sorted(sleep_times)[len(sleep_times)//2]:5.2f} с  "
              f"макс {max(sleep_times):5.2f} с  (каждые "
              f"{args.sleep_every} уроков)")
    print(f"ВСЕГО за прогон:        {total:6.1f} с")
    print("-" * 68)
    print("рост мира на дистанции (снимок ДО сна этого цикла):")
    print(f"{'уроков':>7} {'сущностей':>10} {'наблюд.':>8} {'переходов':>10} "
          f"{'КБ файла':>9} {'вес max 255':>12} {'сила 0..15':>12}")
    for s in snap:
        print(f"{s['lessons']:>7} {s['entities']:>10} {s['observations']:>8} "
              f"{s['transitions']:>10} {s['world_kb']:>9} "
              f"{str(s['strength_max']):>12} {str(s['tstr']):>12}")
    if slept:
        print("-")
        print("СНЫ по одному (цена и отчёт — ЭТОГО сна, не снятого мира):")
        print(f"{'после':>6} {'с':>6} {'examined':>9} {'создано':>8} "
              f"{'поднято':>8} {'опущено':>8} {'затухло':>8} {'смысл':>6}")
        for x in slept:
            print(f"{x['after']:>6} {x['s']:>6} {str(x['examined']):>9} "
                  f"{str(x['created']):>8} {str(x['raised']):>8} "
                  f"{str(x['lowered']):>8} {str(x['decayed']):>8} "
                  f"{str(x['meaningful']):>6}")
    print("-" * 68)
    # Перенос на объёмы, которые очередь называет «сотни уроков, десятки снов».
    per_lesson = sum(steady) / len(steady)
    per_sleep = (sum(sleep_times) / len(sleep_times)) if sleep_times else 0.0
    for n_lessons, n_sleeps in ((100, 5), (300, 15), (600, 30)):
        est = n_lessons * per_lesson + n_sleeps * per_sleep
        print(f"оценка {n_lessons:>4} уроков + {n_sleeps:>3} снов: "
              f"{est/60:6.1f} мин   ({est/3600:5.2f} ч)  — БЕз живого Учителя, "
              f"без прогрева {warm:.0f} с")
    print("=" * 68)

    live_after = (_sha(LIVE_WORLD), _sha(LIVE_DB))
    if live_after != live_before:
        print("!!! ЖИВАЯ ПАМЯТЬ ИЗМЕНЕНА — этотест не имеет права на это !!!")
        print("было:", live_before, "стало:", live_after)
        rc = 1
    else:
        print(f"живая память цела: {live_after[0]} / {live_after[1]}")
        rc = 0
    if not os.environ.get("LIPID2_LONG_KEEP"):
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("каталог прогона сохранён:", tmp)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
