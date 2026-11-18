#!/usr/bin/env python3
"""КОНТРАКТ: предлог «с/со» — только время честно, остальное отказ.

ЗАМЕР ДО ПРАВКИ (/tmp/prep_s.py, живой код; «с» вообще не было в
PREP_CASES, слово шло в голый падёжный зонд и врало):
  «чай с лимоном»          -> чай INSTRUMENT лимон   (ЛОЖЬ: чай не пьют
                              ЛИМОНОМ; лимон — ИНГРЕДИЕНТ/спутник)
  «кот гуляет с собакой»   -> гулять INSTRUMENT собака (ЛОЖЬ: кот гуляет
                              не С ПОМОЩЬЮ собаки — они гуляют ВМЕСТЕ)
  «яблоко упало с дерева»  -> упасть OBJECT_OF дерево (ЛОЖЬ: дерево не
                              объект падения, яблоко падает ИЗ/ОТ дерева)
  «звезда упала с неба»    -> упасть OBJECT_OF нёбо   (ЛОЖЬ, то же)
  «дом с крышей»           -> дом INSTRUMENT крыша    (ЛОЖЬ: дом не сделан
                              крышей; «с крышей» — признак/атрибут)

У «с» ЧЕТЫРЕ значения (совместность, атрибут/часть, источник/убывание,
время) и морфологически они НЕРАЗЛИЧИМЫ. Для совместности, атрибута и
источника отношений в реестре НЕТ; INSTRUMENT («чем?»), OBJECT_OF и
LOCATED_IN лгут на каждом из них (закон «ложная связь хуже отсутствующей»).

РЕШЕНИЕ (кирпичи «к», «по», «в» — тот же принцип: тип связи решает ТИП
ПРАВОГО СЛОВА):
  «с/со» + слово времени (с рассветом, с утра) -> CIRCUMSTANCE — честная
  правда («запели с рассветом» = когда? с рассветом);
  «с/со» + остальное -> ЧЕСТНЫЙ ОТКАЗ + note.

НАСТОЯЩИЙ ИНСТРУМЕНТ НЕ ЗАДЕТ: тв.п. БЕЗ предлога («кот открывает дверь
ключом») идёт своим зондом и остаётся INSTRUMENT.

ЗДЕСЬ НЕТ СОЗНАТЕЛЬНО: местоимения («со мной», «со всеми») — отдельная
гора; совместность («мать с ребёнком») ждёт новое отношение (WITH) —
решение хозяина.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LIPID2_STATUS", "0")

from core.extract import SimpleExtractor  # noqa: E402

_EX = SimpleExtractor()


def triples(sentence):
    return [(f.head, f.relation, f.tail)
            for f in _EX.extract(sentence).facts]


def notes(sentence):
    return _EX.extract(sentence).notes


class TestSCompanionshipRefusal(unittest.TestCase):
    """«с» + спутник/атрибут: INSTRUMENT лгал — теперь честный отказ."""

    def test_tea_is_not_drunk_with_lemon(self):
        t = triples("чай с лимоном")
        self.assertNotIn(("чай", "INSTRUMENT", "лимон"), t,
                         "чай пьют лимоном — прежняя ложь")

    def test_cat_walks_with_dog(self):
        t = triples("кот гуляет с собакой")
        self.assertNotIn(("гулять", "INSTRUMENT", "собака"), t,
                         "гуляют с помощью собаки — ложь")
        self.assertIn(("кот", "CAN_DO", "гулять"), t,
                      "основная связь потеряна")

    def test_house_with_roof(self):
        t = triples("дом с крышей")
        self.assertNotIn(("дом", "INSTRUMENT", "крыша"), t,
                         "дом сделан крышей — ложь")

    def test_refusal_is_not_silent(self):
        ns = notes("чай с лимоном")
        self.assertTrue(any("лимоном" in n for n in ns),
                        f"отказ молчалив (молчаливый ноль запрещён): {ns}")

    def test_dog_refusal_is_not_silent(self):
        ns = notes("кот гуляет с собакой")
        self.assertTrue(any("собакой" in n for n in ns),
                        f"отказ молчалив: {ns}")


class TestSSourceRefusal(unittest.TestCase):
    """«с» + род.п. (источник/убывание): OBJECT_OF лгал."""

    def test_apple_falls_from_tree(self):
        t = triples("яблоко упало с дерева")
        self.assertNotIn(("упасть", "OBJECT_OF", "дерево"), t,
                         "дерево — объект падения — ложь")
        self.assertNotIn(("упасть", "LOCATED_IN", "дерево"), t)
        self.assertIn(("яблоко", "CAN_DO", "упасть"), t,
                      "основная связь потеряна")

    def test_leaf_falls_from_roof(self):
        t = triples("лист упал с крыши")
        self.assertNotIn(("упасть", "OBJECT_OF", "крыша"), t)
        self.assertNotIn(("упасть", "LOCATED_IN", "крыша"), t)

    def test_source_refusal_is_not_silent(self):
        ns = notes("яблоко упало с дерева")
        self.assertTrue(any("дерева" in n for n in ns),
                        f"отказ молчалив: {ns}")


class TestSTimeIsCircumstance(unittest.TestCase):
    """«с/со» + слово времени = обстоятельство (честная правда)."""

    def test_sang_at_dawn(self):
        self.assertIn(("запеть", "CIRCUMSTANCE", "рассвет"),
                      triples("с рассветом птицы запели"))

    def test_raining_since_morning(self):
        self.assertIn(("лить", "CIRCUMSTANCE", "утро"),
                      triples("с утра льёт дождь"))


class TestSNoRegression(unittest.TestCase):
    """Настоящий инструмент и соседние предлоги не задеты."""

    def test_true_instrument_survives(self):
        t = triples("кот открывает дверь ключом")
        self.assertIn(("открывать", "INSTRUMENT", "ключ"), t,
                      "орудие в тв.п. без предлога сломано")

    def test_k_and_po_survive(self):
        self.assertIn(("письмо", "RECIPIENT", "брат"),
                      triples("письмо к брату"))
        self.assertIn(("болеть", "CIRCUMSTANCE", "утро"),
                      triples("щека болит по утрам"))

    def test_place_in_survives(self):
        self.assertIn(("сахар", "LOCATED_IN", "чай"),
                      triples("сахар в чае"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
