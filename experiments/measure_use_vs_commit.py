#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ШАГ Б: может ли ложь попасть в живую базу, став собственной уликой.

ВОПРОС ЭТОГО ФАЙЛА (шаг Б, контракт Жозефины «полезность != истинность»).
У кандидата есть право УЧАСТВОВАТЬ в рассуждении (_reasoning_view мозга),
но нет права лежать в lipid_world.json. Границу держат три замка
(brain/curiosity.py::promote): ответ найден, дофамин поднят, кандидат
зацепился за известное.

ПОДОЗРЕНИЕ. Все три замка отпирает ОДНО и то же событие: Thinker нашёл
ответ по миру, в который кандидат уже подмешан. Тогда «ответ найден»
порождается кандидатом, «дофамин» поднимается тем же вызовом relief(), а
«зацепка» у кандидата есть по построению (он опирается на известное слово).
Замки оказываются НЕ независимыми: ложь обеспечивает себе пропуск сама.

ЗАМЕР НИЧЕГО НЕ ЧИНИТ. Он печатает, доезжает ли выдуманная связь до живого
файла. Если доезжает — петля живая и закрывать её надо отдельным кирпичом.

ЗАМЕРЕННЫЙ ОТВЕТ (09.11, локалка выключена, учителя выключены):
  [1] кандидат ПОЛЕЗЕН: без него отказ, с ним ответ;
  [2] в живую базу он НЕ попадает — promote требует, чтобы зацепка была в
      ЖИВОЙ базе (knowledge() = StaticKnowledge(self._load_world()), а не
      в представлении для рассуждения), и живой файл остаётся нетронутым;
  [3] мерить происхождение нечего: записи нет.
То есть разделение USE_IN_REASONING / COMMIT_TO_WORLD работает именно так,
как задумано: кандидат имеет право быть доводом и не имеет права стать
фактом. Петли «ложь -> ответ -> relief -> запись» в текущем коде НЕТ.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lang.ru.lemma import install                    # noqa: E402
install()

from brain.lipid2_brain import Lipid2Brain           # noqa: E402
from brain.simulator import StaticKnowledge          # noqa: E402
from brain.thinker import Thinker                     # noqa: E402

ВОПРОС = "дождь может гасить огонь?"

# КАНДИДАТ. ЛОЖЕН по построению: мы его выдумали, никто не проверял.
ЛОЖНЫЙ_КАНДИДАТ = {"дождь": {"name": "дождь", "actions": ["гасить"]}}


def brain_in(dir_):
    b = Lipid2Brain(memory_dir=dir_)
    b.ask_user_enabled = False
    b.use_llm_teacher = False
    b.use_wiki_teacher = False
    b._say = lambda m, **kw: None
    return b


def known_world():
    """Мир про ВОДУ. Про дождь в нём нет НИ СТРОКИ.

    Собран по запертому примеру tests/test_ability_reasoning_contract.py:
    путь к ответу требует ребра «гасить OBJECT_OF огонь», без него Thinker
    обязан отказаться — и мерить было бы нечего.
    """
    def наблюдение(head, rel, tail):
        return {f"{head}|{rel}|{tail}": {
            "head": head, "relation": rel, "tail": tail, "source": "user",
            "lineage": "owner", "support": 1, "confidence": 0.9}}

    return {
        "entities": {"вода": {"name": "вода", "actions": ["гасить"]},
                     "гасить": {"name": "гасить", "hierarchy": ["действие"]},
                     "огонь": {"name": "огонь"}},
        "observations": {**наблюдение("вода", "CAN_DO", "гасить"),
                         **наблюдение("гасить", "OBJECT_OF", "огонь")},
        "unknown_words": {},
    }


def живой_файл(dir_):
    return json.load(open(os.path.join(dir_, "lipid_world.json"),
                          encoding="utf-8"))


def section(заголовок):
    print("\n" + заголовок)
    print("-" * len(заголовок))


def step1_usefulness():
    """Полезен ли кандидат: закрывает ли ОДИН он цель (полезность)."""
    section("[1] Полезность: закрывает ли кандидат цель ОДИН")
    with tempfile.TemporaryDirectory() as dir_:
        b = brain_in(dir_)
        b._save_world(known_world())
        без = Thinker(StaticKnowledge(b._load_world())).think(ВОПРОС)
        print(f"  без кандидата: ok={без.ok}")
        b.curiosity.stage(dict(ЛОЖНЫЙ_КАНДИДАТ), source="llm")
        print(f"  в карантине: {[i['name'] for i in b.curiosity.staged()]}")
        with_c = Thinker(StaticKnowledge(b._reasoning_view())).think(ВОПРОС)
        print(f"  с кандидатом: ok={with_c.ok}")
        print("  вывод:", "ПОЛЕЗЕН — ответ появляется только от него"
              if (with_c.ok and not без.ok) else
              "полезность НЕ доказана (оба прогона одинаковы) — мерить "
              "построение петли бессмысленно, чини сначала мерку")
        return bool(with_c.ok and not без.ok)


def step2_commit_leak():
    """Доезжает ли ВЫДУМКА до живого файла (COMMIT_TO_WORLD)."""
    section("[2] Доезжает ли выдумка до живого файла")
    with tempfile.TemporaryDirectory() as dir_:
        b = brain_in(dir_)
        b._save_world(known_world())
        b.curiosity.stage(dict(ЛОЖНЫЙ_КАНДИДАТ), source="llm")
        до = живой_файл(dir_)
        ответ = b.process_input(ВОПРОС)
        после = живой_файл(dir_)

        новые_сущ = set(после.get("entities", {})) - set(до.get("entities", {}))
        новые_связи = (set(после.get("observations", {}))
                       - set(до.get("observations", {})))
        print(f"  ответ: {str(ответ)[:120]}")
        print(f"  сущностей в файле: {len(до.get('entities', {}))} -> "
              f"{len(после.get('entities', {}))}")
        print(f"  вошло сущностей: {sorted(новые_сущ)}")
        print(f"  вошло связей:    {sorted(новые_связи)}")
        print(f"  в карантине осталось: "
              f"{[i['name'] for i in b.curiosity.staged()]}")
        print(f"  кортизол={b.hormone_gate.cortisol} "
              f"дофамин={b.hormone_gate.dopamine}")
        вошло = "дождь" in новые_сущ or any("дождь" in k for k in новые_связи)
        print("  вывод:", "ПЕТЛЯ ЖИВАЯ: выдумка записана в живую базу, "
              "свидетельством которой она же и была" if вошло else
              "ПЕТЛЯ НЕ ПРОХОДИТ: выдумка осталась в карантине")
        return вошло


def step3_provenance_of_written_claim():
    """Если связь всё-таки вошла — есть ли у неё честное имя основания."""
    section("[3] Происхождение вошедшей связи")
    with tempfile.TemporaryDirectory() as dir_:
        b = brain_in(dir_)
        b._save_world(known_world())
        b.curiosity.stage(dict(ЛОЖНЫЙ_КАНДИДАТ), source="llm")
        b.process_input(ВОПРОС)
        мир = живой_файл(dir_)
        связи = {k: v for k, v in мир.get("observations", {}).items()
                 if "дождь" in k}
        if not связи:
            print("  связей про дождь в файле НЕТ — мерить происхождение нечего")
            return
        from core.records import independent_bases, record_ground, record_weight
        for k, v in связи.items():
            print(f"  {k}: source={v.get('source')} lineage={v.get('lineage')!r} "
                  f"ground={record_ground(v)} bases={independent_bases(v)} "
                  f"вес={record_weight(v)}")


if __name__ == "__main__":
    print("=" * 72)
    print("ШАГ Б: USE_IN_REASONING против COMMIT_TO_WORLD")
    print("=" * 72)
    if step1_usefulness():
        вошло = step2_commit_leak()
        if вошло:
            step3_provenance_of_written_claim()
