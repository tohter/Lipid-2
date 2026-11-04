#!/usr/bin/env python3
"""ПРОХОДИТ ЛИ СТАТИКА ПРОШЕДШЕГО ВРЕМЕНИ В МИР — замер перед механизмом В.

Зачем. Босс перед механизмом В (родственные рёбра из УЧИТЕЛЬСКОГО определения,
решение от 03.11, оригинал в experiments/BOSS_PHASE3_ANSWER.txt) потребовала
один замер: «создай вручную одно ребро невестка IS_ROLE жена в прошедшем
времени и проверь, доходит ли оно до мира. Если не доходит — прежде чем
строить В, расширь правку с PART_OF на IS_ROLE».

Замер отвечает ТРИ вопроса, а не один, потому что одного ответа мало для
решения:

  1. ДОХОДИТ ЛИ IS_ROLE с меткой события. Ответ: НЕТ (`_apply_fact` выбрасывает
     любой is_event). Опасение босса подтверждено.
  2. ДОСТИЖИМО ЛИ ТАКОЕ СОСТОЯНИЕ ВОВСЕ. Если экстрактор не рождает
     `IS_ROLE`+`event` ни на одной фразе, расширение guard'а было бы
     НЕРАБОТАЮЩИМ условием — ровно тот класс, что мутация М3 фазы 2
     (урок 84: зелёный мутант бывает инертным условием, а не дырой).
  3. ГДЕ НАСТОЯЩАЯ ДЫРА. Проверка «был» как связки показала: именная часть
     после «был» уходит в INSTRUMENT, то есть прошедшее время теряет статику
     НЕ перекраской в событие, а НЕВЕРНЫМ ИМЕНЕМ отношения — и при этом
     predicate остаётся `fact`, то есть ЛОЖЬ доходит до мира.

Ручное ребро вставляется в разбор, как и велено указанием («созданное
вручную»), но прогон идёт через НАСТОЯЩИЙ `learn_from_text`, а не через
прямой вызов `_apply_fact`: первый мой вариант замер вызывал `_apply_fact` с
разными копиями словарей и показал «не доходит» по причине, не имеющей
отношения к делу.

Изоляция: мир пишется в /tmp; живая память не трогается.

    LIPID2_STATUS=0 .venv/bin/python tools/static_past_check.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.extract import Fact, SimpleExtractor  # noqa: E402


def hr(t):
    print("\n" + t + "\n" + "-" * 72)


def _mk(tmp, extractor=None, teacher=None):
    """Настоящий Learner над своим /tmp-файлом мира."""
    from brain.learner import Learner
    path = os.path.join(tmp, "world.json")

    def load_world():
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return {"entities": {}, "observations": {}, "unknown_words": {}}

    def save_world(w):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(w, fh, ensure_ascii=False)

    save_world(load_world())
    return Learner(load_world=load_world, save_world=save_world,
                   teacher=teacher, max_teacher_calls=3,
                   extractor=extractor), load_world


class _ManualEdge:
    """Настоящий экстрактор; в каждое предложение подмешивает ОДНО ручное
    ребро с требуемой меткой. Так выполняется слово указания «вручную»."""

    def __init__(self, head, rel, tail, predicate):
        self._e = SimpleExtractor()
        self.edge = (head, rel, tail, predicate)

    def set_known(self, k):
        self._e.set_known(k)

    def extract(self, s, **kw):
        return self._e.extract(s, **kw)

    def extract_text(self, text):
        sents = self._e.extract_text(text)
        head, rel, tail, predicate = self.edge
        for sf in sents:
            sf.facts.append(Fact(head=head, relation=rel, tail=tail,
                                 predicate=predicate, evidence=sf.sentence,
                                 confidence=0.9))
        return sents


SENTENCE = "Невестка жила с семьёй мужа."


def q1_reaches_world():
    hr("ВОПРОС 1. Ребро, созданное ВРУЧНУЮ, predicate=event — доходит до мира?")
    for rel in ("IS_ROLE", "PART_OF"):
        for predicate in ("event", "fact"):
            with tempfile.TemporaryDirectory() as tmp:
                lr, load = _mk(tmp, extractor=_ManualEdge("невестка", rel,
                                                          "жена", predicate))
                lr.learn_from_text(SENTENCE, allow_teacher=False)
                w = load()
                key = f"невестка|{rel}|жена"
                hit = key in w.get("observations", {})
                # Метки НЕПЕРЕСЕКАЮЩИЕСЯ: «НЕ ДОШЛО» содержит «ДОШЛО», и
                # любой поиск по положительному слову ловил бы отрицательные
                # ответы как положительные.
                print(f"  {rel:9} predicate={predicate:5} -> "
                      f"{'ЕСТЬ В МИРЕ' if hit else 'НЕТ В МИРЕ'}  "
                      f"(observations={len(w.get('observations', {}))}, "
                      f"events={len(w.get('events', []))})")


def q2_reachable():
    hr("ВОПРОС 2. Достижимо ли `IS_ROLE` с меткой события в разборe?")
    ex = SimpleExtractor()
    probes = [
        "Мальчик был сыном кузнеца.", "Вася был инженером.", "Мать была врачом.",
        "Она была матерью троих детей.", "Кузнец был отцом мальчика.",
        "Мальчик, который был пастухом, пришёл.",
        "Пастух, который был сыном мельника, пас коров.",
        "Мальчик, сын кузнеца, ушёл на войну.",
        "Зинаида, мать Егора, гостила в семье сына.",
        "Она стала женой моряка.", "Мальчик остался сиротой.",
        "Егор, муж невестки, уехал.",
    ]
    hit = 0
    for s in probes:
        r = ex.extract(s)
        bad = [f for f in r.facts if f.relation == "IS_ROLE" and f.is_event]
        hit += bool(bad)
        pairs = [(f.head, f.relation, f.tail, f.predicate) for f in r.facts]
        print(f"  {s!r:46} {pairs}")
    print(f"  IS_ROLE+event: {hit} из {len(probes)} прицельных фраз")

    n = ev = 0
    for fn in ("experiments/corpus_dev.json", "experiments/corpus_heldout.json"):
        d = json.load(open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), fn), encoding="utf-8"))
        cases = d if isinstance(d, list) else d.get("cases", d)
        for c in cases:
            s = c.get("text") or c.get("sentence")
            if not s:
                continue
            for f in ex.extract(s).facts:
                if f.relation == "IS_ROLE":
                    n += 1
                    ev += f.is_event
    print(f"  IS_ROLE в золотом корпусе: {n} фактов, из них с меткой "
          f"события: {ev}")


def q3_where_is_the_hole():
    hr("ВОПРОС 3. Куда на самом деле уходит статика прошедшего времени?")
    ex = SimpleExtractor()
    for s in ("Вася — инженер.", "Вася был инженером.",
              "Она — мать.", "Она была матерью.",
              "Мальчик — сын кузнеца.", "Мальчик был сыном кузнеца."):
        r = ex.extract(s)
        print(f"  {s!r:34} {[(f.head, f.relation, f.tail, f.predicate) for f in r.facts]}")


def q4_teacher_definition():
    hr("ВОПРОС 4. Что делает учитель с определением СЕЙЧАС (до механизма В)?")

    def teacher(lemma):
        if lemma == "невестка":
            return {"concept": "невестка", "definition": "жена сына",
                    "relates_to": ["сын", "семья"], "hierarchy": ["родственник"]}
        return None

    with tempfile.TemporaryDirectory() as tmp:
        lr, load = _mk(tmp, teacher=teacher)
        lr.learn_from_text("Невестка жила с семьёй мужа.")
        w = load()
        e = w["entities"].get("невестка", {})
        print("  определение в мире:", json.dumps(
            {k: e.get(k) for k in ("definition", "relates_to", "hierarchy")},
            ensure_ascii=False))
        obs = w.get("observations", {})
        print("  рёбра у невестки:", [k for k in obs if k.startswith("невестка")]
              or "НЕТ (определение лежит текстом, рёбер нет)")


def q5_definition_shape():
    hr("ВОПРОС 5. Что даёт разбор голого определения (правило В «сущность "
       "родительного» строится поверх этого)?")
    ex = SimpleExtractor()
    for s in ("жена сына", "жена сына Зинаиды", "мать отца", "носитель заряда",
              "большое животное", "брак с морем"):
        print(f"  {s!r:22} -> {[(f.head, f.relation, f.tail) for f in ex.extract(s).facts]}")


if __name__ == "__main__":
    q1_reaches_world()
    q2_reachable()
    q3_where_is_the_hole()
    q4_teacher_definition()
    q5_definition_shape()
    print()
