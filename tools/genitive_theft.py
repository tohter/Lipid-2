#!/usr/bin/env python3
"""РОДИТЕЛЬНЫЙ ОПРЕДЕЛИТЕЛЬ ПРИ ПОДЛЕЖАЩЕМ — что он крадёт и где.

Зачем. Босс поставила между правилом шага (П4) и звеном 4 извлечение
родственных рёбер ИЗ РАССКАЗА. Прежде чем выбирать имя отношения и признак
различения (одушевлённость, урок 79), измерены три вещи, о которых я раньше
РАССУЖДАЛА, а не знала:

  1. ДВОЙНОЕ ХРАНИЛИЩЕ. Сказуемое настоящего времени кладёт связь в
     observations, прошедшего — в events. Замер, который смотрит в ОДНО
     хранилище, объявляет второе пустое «потерей всего прошедшего времени».
     Именно так я и получила ложную дыру на этой правке: «Жена Егора
     пожаловалась мужу» НЕ даёт ни одной ссылки в observations, но событие
     (жена, пожаловаться, егор, past) лежит рядом. Проверять оба.

  2. ВЫБОР ОТНОШЕНИЯ В РАССКАЗЕ НЕ НАСТУПАЕТ. PART_OF рождается только в
     бесглагольном члене (строка 10 core/extract.py), а сказуемое есть в
     каждом предложении рассказа. Спрашивать «PART_OF или HAS_PROPERTY» для
     рассказа не про что: пары (жена, егор) не возникает ни под каким именем.

  3. НАСТОЯЩИЙ ДЕФЕКТ — КРАЖА ОБЪЕКТА. Родительный определитель подлежащего
     занимает место прямого или косвенного дополнения глагола. Настоящий
    получатель
     действия исчезает из вывода совсем, а не переименовывается.

Разбор не трактуется: печатается то, что мир реально записал.

Изоляция: каждое предложение учится в СВОЙ /tmp-мир; живая память не
трогается.

    LIPID2_STATUS=0 .venv/bin/python tools/genitive_theft.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def store_of(sentence):
    """(ссылки observations, события events) мира, построенного на ОДНОМ
    предложении. Оба хранилища, иначе замер соврёт (см. пункт 1)."""
    os.environ["LIPID2_MEMORY_DIR"] = tempfile.mkdtemp(prefix="theft_")
    from brain.lipid2_brain import Lipid2Brain
    from core.worldfile import read_world
    b = Lipid2Brain(memory_dir=os.environ["LIPID2_MEMORY_DIR"])
    b.use_llm_teacher = False
    b.use_wiki_teacher = False
    b.ask_user_enabled = False
    b.learn_from_text(sentence, source="замер", allow_teacher=False)
    w = read_world(b.world_filepath)
    obs = w.get("observations") or {}
    links = sorted((o["head"], o["relation"], o["tail"])
                   for o in obs.values()
                   if isinstance(o, dict) and o.get("head"))
    events = [(e.get("subject"), e.get("action"), e.get("obj"))
              for e in (w.get("events") or []) if isinstance(e, dict)]
    return links, events


def show(title, sentences):
    print("\n" + title + "\n" + "-" * 74)
    empty_both = 0
    for s in sentences:
        links, events = store_of(s)
        if not links and not events:
            empty_both += 1
        got = events[0][2] if events else ""
        print(f"  {s}")
        print(f"      links : {links or 'нет'}")
        print(f"      events: {[(a, b, c) for a, b, c in events] or 'нет'}"
              f"   объект_события={got or '—'}")
    if empty_both:
        print(f"  !!! предложений, пустых В ОБОИХ хранилищах: {empty_both}")


def grid():
    """Решётка 2x2: включается ли кража одушевлённостью определителя или
    падежом объекта. Если ошибаются все четыре клетки — ни тот ни другой
    признак кражу не включают, она в самой конструкции."""
    print("\nРЕШЁТКА 2x2 (определитель одуш/неодуш x объект датив/винит.)\n"
          + "-" * 74)
    rows = [
        ("одуш", "датив", "Жена Егора улыбнулась мужу.", "муж"),
        ("одуш", "винит.", "Жена Егора увидела брата.", "брат"),
        ("неодуш", "датив", "Хозяйка дома улыбнулась соседке.", "соседка"),
        ("неодуш", "винит.", "Хозяйка дома увидела брата.", "брат"),
    ]
    bad = 0
    for kind, case, s, want in rows:
        _, events = store_of(s)
        got = events[0][2] if events else "НЕТ СОБЫТИЯ"
        ok = (got == want)
        bad += (not ok)
        print(f"  {kind:7} {case:7} {s:38} ждала={want:9} "
              f"получила={got:9} {'OK' if ok else 'КРАЖА'}")
    print(f"  ошиблись {bad} из {len(rows)} — кража НЕ зависит ни от "
          f"одушевлённости,\n  ни от падежа: она в самой конструкции")


def controls():
    """Контроль: без определителя объект правильный. Без этого строки выше
    неотличимы от сломанного разбора вообще."""
    print("\nКОНТРОЛЬ (тот же глагол БЕЗ родительного определителя)\n"
          + "-" * 74)
    for s, want in [("Жена пожаловалась мужу.", "муж"),
                    ("Мать позвонила сыну.", "сын"),
                    ("Хозяйка увидела брата.", "брат")]:
        _, events = store_of(s)
        got = events[0][2] if events else "НЕТ СОБЫТИЯ"
        print(f"  {s:38} ждала={want:9} получила={got:9} "
              f"{'OK' if got == want else 'ОШИБКА'}")


STORY = [
    "Зинаида гостит в семье своего сына Егора.",
    "Жена Егора пожаловалась мужу.",
    "Егор любит жену, но боится мамы.",
    "Он пообещал жене.",
]
CONTROL_SENT = [
    "Хвост собаки мокрый.",
    "Маша — жена сына Зинаиды.",
]

if __name__ == "__main__":
    print("=" * 74)
    print("РОДИТЕЛЬНЫЙ ОПРЕДЕЛИТЕЛЬ ПРИ ПОДЛЕЖАЩЕМ: двойное хранилище и "
          "кража объекта")
    print("=" * 74)
    show("РАССКАЗ (оба хранилища)", STORY)
    show("КОНТРОЛЬ: здесь сказуемого-действия нет, PART_OF рождается",
         CONTROL_SENT)
    grid()
    controls()
