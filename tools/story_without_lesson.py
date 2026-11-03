#!/usr/bin/env python3
"""ИСТОРИЯ БЕЗ УРОКА ОПРЕДЕЛЕНИЙ — что даёт один рассказ.

Замер по требованию босса перед звеном 4. Прежние прогоны строили мир из
рассказа И из урока («Маша — жена сына Зинаиды»). Надо знать, ГДЕ кончается
рассказ и НАЧИНАЕТСЯ урок: если связь «жена сына» заводится в мир только
уроком, то рассуждение «невестка = жена Егора» проверяло бы ПАМЯТЬ УРОКА, а не
способность связывать уже известное.

КАНАЛЫ ВЫКЛЮЧЕНЫ ФЛАЖКАМИ мозга (не подменой объектов): снятая галочка обязана
означать «канал не трогает НИЧЕГО» — этот смысл переключателей уже заперт
тестами. Числа обращений печатаются как ДОКАЗАТЕЛЬСТВО того, что учитель не
«молча не пришёл», а не приходил.

ОБХОД ИДЁТ ПО ВСЕМ ЧИТАЕМЫМ ОТНОШЕНИЯМ, а не по выбранному списку. Обход по
трём- четырём удобным именам дал бы «НЕТ ПУТИ» там, где ребро лежит под другим
именем, — и замер соврал бы об отсутствии данных.

Изоляция: мир пишется в /tmp; живая память не трогается (хэш сверяет хозяин).
Вывод — сырой дамп рёбер и отсутствующих пар, без толкования.

    LIPID2_STATUS=0 .venv/bin/python tools/story_without_lesson.py
"""
import json
import os
import sys
import tempfile
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STORY = ("Зинаида гостит в семье своего сына Егора. Она недолюбливает "
         "невестку и постоянно к ней придирается. Жена Егора пожаловалась "
         "мужу и попросила его поговорить с матерью, чтобы та не портила ей "
         "настроение. Егор любит жену, но боится мамы. Он пообещал жене, что "
         "поговорит с мамой, но обещание своё не выполнил. Жена затаила "
         "обиду.")

# Пары, без которых задача «кем приходятся невестка Зинаиды и жена Егора»
# нерешаема. Это НЕ проверка качества: это список того, на чём звену 4 нечего
# строить, если строк ниже окажется НЕТ РЕБРА.
PAIRS = [("жена", "егор"), ("жена", "сын"), ("сын", "зинаида"),
         ("мама", "егор"), ("мать", "егор"), ("жена", "зинаида"),
         ("невестка", "жена")]


def hr(t):
    print("\n" + t + "\n" + "-" * 70)


def path(sk, rels, start, goal, max_hops=4):
    """Обход вперёд и назад по ВСЕМ переданным отношениям (тот же обход, что
    в П2, только без списка «удобных» имён)."""
    seen = {start}
    q = deque([(start, [])])
    while q:
        node, trail = q.popleft()
        if len(trail) >= max_hops:
            continue
        for rel in rels:
            try:
                outs = sk.related(node, rel)
            except KeyError:
                outs = []
            for t in outs:
                step = (node, rel, t)
                if t == goal:
                    return trail + [step]
                if t not in seen:
                    seen.add(t)
                    q.append((t, trail + [step]))
            try:
                backs = sk.referenced_by(node, rel)
            except KeyError:
                backs = []
            for head, _ in backs:
                step = (node, f"<{rel}", head)
                if head == goal:
                    return trail + [step]
                if head not in seen:
                    seen.add(head)
                    q.append((head, trail + [step]))
    return None


def render(p):
    return "  ->  ".join(f"{h} -{r}-> {t}" for h, r, t in p) if p else "НЕТ ПУТИ"


def main():
    tmp = tempfile.mkdtemp(prefix="story_only_")
    os.environ["LIPID2_MEMORY_DIR"] = tmp
    from brain.lipid2_brain import Lipid2Brain
    from brain.simulator import StaticKnowledge
    from core.storage import RelationType
    from core.worldfile import read_world

    b = Lipid2Brain(memory_dir=tmp)
    b.use_llm_teacher = False
    b.use_wiki_teacher = False
    b.ask_user_enabled = False
    print(f"мир: {tmp}")
    print("учитель ЛЛМ: ВЫКЛ   вики: ВЫКЛ   спросить человека: ВЫКЛ")

    b.learn_from_text(STORY, source="rasskaz", allow_teacher=False)
    print(f"обращений к каналам: {b.channel_calls}")
    if any(b.channel_calls.values()):
        print("ВНИМАНИЕ: канал дёрнут при снятых флажках — замер нечест")

    world = read_world(b.world_filepath)
    obs = world.get("observations") or {}
    ents = world.get("entities") or {}
    edges = sorted((o["head"], o["relation"], o["tail"])
                   for o in obs.values()
                   if isinstance(o, dict) and o.get("head"))

    hr(f"РАССКАЗ ДАЛ: сущностей {len(ents)}, ссылок {len(edges)}")
    for h, r, t in edges:
        print(f"  {h:14} -{r:13}-> {t}")

    sk = StaticKnowledge(world)
    rels = sorted({x.name for x in RelationType})

    hr("ПУТЬ МЕЖДУ НУЖНЫМИ ПАРАМИ (в обе стороны, ЛЮБЫМ отношением)")
    for a, bb in PAIRS:
        p = path(sk, rels, a, bb, max_hops=1)
        print(f"  {a:10} {bb:10} : {render(p)}")

    hr("ЦЕПОЧКА невестка -> егор (ровно то, что должно вывести звено 4)")
    print(f"  {render(path(sk, rels, 'невестка', 'егор'))}")

    hr("ПРО КАЖДОЕ ИМЯ: что знает мир (наружу / кто ссылается)")
    for w in ("невестка", "жена", "сын", "зинаида", "егор", "мама", "семья"):
        if not sk.knows(w):
            print(f"  {w:10} — такому слову мир не знает")
            continue
        out = [f"{r}:{sk.related(w, r)}" for r in rels if sk.related(w, r)]
        print(f"  {w:10} наружу      : {'; '.join(out) or '—'}")
        print(f"  {'':10} ссылаются на: {sk.referenced_by(w) or '—'}")

    print(f"\nмир лежит здесь: {tmp}")
    print(f"хэш: {__import__('hashlib').sha256(open(b.world_filepath,'rb').read()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
