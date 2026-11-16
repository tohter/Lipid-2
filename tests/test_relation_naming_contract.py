#!/usr/bin/env python3
"""КОНТРАКТ ИМЕНОВАНИЯ СВЯЗЕЙ (решение хозяина 0.1/0.3).

Смысловое деление рёбер по типу отношения, а не по синтаксической ветке
(красный контракт до правки core/extract.py).

    PART_OF      = структурная часть целого      («хвост собаки»)
    OBJECT_OF    = предмет владения/кинетики     («кукла девочки», «играет с куклой»)
    RELATION_TO  = семиотическая связь/родство   («жена Егора», «разговаривает с куклой»)

ГЛАВНОЕ: «жена — НЕ часть мужа». Родственная пара (оба конца одушевлённые)
обязана давать RELATION_TO, а не PART_OF/OBJECT_OF. ЧАСТЬ остаётся PART_OF.

    LIPID2_STATUS=0 .venv/bin/python -m unittest tests.test_relation_naming_contract
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.extract import SimpleExtractor  # noqa: E402

_EX = SimpleExtractor()


def triples(sentence):
    return [(f.head, f.relation, f.tail) for f in _EX.extract(sentence).facts]


# РОДСТВЕННАЯ ПАРА ПРИ АЛЬТЕРНАТИВЕ: оба конца одушевлённые -> РЕЛЯЦИЯ,
# а не ЧАСТЬ. «жена PART_OF егор» было ложью (жена не часть мужа), теперь
# RELATION_TO. БЕЗ-альтернативный определитель («Сын Зинаиды играл во дворе»)
# живёт в запертой дыре «генитивный голод» и этим коммитом НЕ чинится.
KINSHIP = [
    ("Жена Егора пожаловалась мужу.", ("жена", "RELATION_TO", "егор")),
    ("Мать Егора позвонила сыну.", ("мать", "RELATION_TO", "егор")),
    ("Сестра Анны принесла воды.", ("сестра", "RELATION_TO", "анна")),
    ("Жена Егора увидела брата.", ("жена", "RELATION_TO", "егор")),
]

# ЧАСТЬ ЦЕЛОГО: неодушевлённый определитель -> PART_OF (не меняется).
PART = [
    ("Хвост собаки мокрый.", ("хвост", "PART_OF", "собака")),
    ("Крышка гроба тяжёлая.", ("крышка", "PART_OF", "гроб")),
    ("Хозяин сломал ножку стола.", ("ножка", "PART_OF", "стол")),
]

# ОТРИЦАТЕЛЬНЫЕ: родственная пара НЕ может быть ни PART_OF, ни OBJECT_OF.
# «жена — не часть Егора». Только с-альтернативные пары: в них определитель
# наверняка притяжатель, а не объект глагола.
NOT_PART_OF_KINSHIP = [
    "Жена Егора пожаловалась мужу.",
    "Сестра Анны принесла воды.",
]

# БЕЗ-АЛЬТЕРНАТИВНАЯ ДЫРА (не чинится этим коммитом, заперта как дыра).
# «Сын Зинаиды играл во дворе» — «Зинаиды» без соперника-дополнения
# неотличим от объекта глагола («Мать встретила сына» -> obj сын). Снять
# его = заменить правду ложью (урок TestOldLieKeptAndNamed). Дыра названа.
NO_RIVAL_KINSHIP = "Сын Зинаиды играл во дворе."


class TestKinshipIsRelation(unittest.TestCase):
    """Ядро решения хозяина: одушевлённая пара — RELATION_TO, а не PART_OF."""

    def test_animate_pair_is_relation_to(self):
        for sentence, want in KINSHIP:
            with self.subTest(sentence=sentence):
                self.assertIn(want, triples(sentence),
                              "%s: ребра %s нет; вывод: %s"
                              % (sentence, want, triples(sentence)))

    def test_kinship_never_part_of(self):
        for sentence in NOT_PART_OF_KINSHIP:
            with self.subTest(sentence=sentence):
                bad = [t for t in triples(sentence)
                       if t[0] == t[2] or t[1] == "PART_OF"]
                self.assertEqual([t for t in triples(sentence)
                                  if t[1] == "PART_OF"], [],
                                 "%s: одушевлённая пара осталась PART_OF: %s"
                                 % (sentence, triples(sentence)))

    def test_edge_points_from_modified_to_modifier(self):
        """Направление сохраняется: «жена» зависит от «Егора»."""
        for sentence, (head, rel, tail) in KINSHIP:
            with self.subTest(sentence=sentence):
                self.assertNotIn((tail, rel, head), triples(sentence),
                                 "%s: ребро развернуто" % sentence)


class TestPartStaysPartOf(unittest.TestCase):
    """Неодушевлённая часть тела/структуры не имеет права сдвинуться."""

    def test_inanimate_modifier_is_part_of(self):
        for sentence, want in PART:
            with self.subTest(sentence=sentence):
                self.assertIn(want, triples(sentence),
                              "%s: контроль %s потерян; вывод: %s"
                              % (sentence, want, triples(sentence)))


class TestKinshipRelationStatic(unittest.TestCase):
    """RELATION_TO, как и PART_OF, не бывает происшествием: «Жена Егора
    пожаловалась» не превращает брак в событие. Прошедшее время сказуемого
    не решает, чья жена."""

    def test_relation_is_static_in_past_tense(self):
        for sentence in ("Жена Егора пожаловалась мужу.",
                         "Сестра Анны принесла воды."):
            with self.subTest(sentence=sentence):
                ev = [f for f in _EX.extract(sentence).facts
                      if f.relation == "RELATION_TO"]
                self.assertTrue(ev, "%s: RELATION_TO нет" % sentence)
                self.assertTrue(all(f.predicate == "fact" for f in ev),
                                "%s: RELATION_TO помечен событием и не дойдёт "
                                "до мира: %s" % (sentence, ev))


class TestNoRivalKinshipHoleLocked(unittest.TestCase):
    """БЕЗ-альтернативная родня — ДЫРА, а не тихое место. «Сын Зинаиды
    играл во дворе» даёт «играть OBJECT_OF зинаида» — определитель без
    соперника-дополнения неотличим от объекта глагола («Мать встретила
    сына» -> obj сын). Снять его = заменить правду ложью. Пока дыра НЕ
    починена, она обязана давать прежнюю ложь, а не молчать или менять
    имя на RELATION_TO без отдельного решения."""

    def test_hole_is_named_and_unfixed(self):
        got = triples(NO_RIVAL_KINSHIP)
        self.assertIn(("играть", "OBJECT_OF", "зинаида"), got,
                      "без-альтернативная родня починена без замены дыры "
                      "(исчезла старая ложь): %s" % got)
        self.assertNotIn(("сын", "RELATION_TO", "зинаида"), got,
                         "без-альтернативная родня стала RELATION_TO, а "
                         "дыра не записана отдельным решением: %s" % got)


if __name__ == "__main__":
    unittest.main(verbosity=2)