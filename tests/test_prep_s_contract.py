#!/usr/bin/env python3
"""КОНТРАКТ: предлог «с/со» — WITH (совместность), PATH (источник), время.

ИСТОРИЯ (три состояния одной строки):
  1) ДО кирпича «с»: «с» не было в PREP_CASES, слово проваливалось в голый
     падёжный зонд и ВРАЛО на всех значениях, кроме орудия:
       «чай с лимоном»         -> чай INSTRUMENT лимон    (чай пьют лимоном)
       «кот гуляет с собакой»  -> гулять INSTRUMENT собака (гуляют С ПОМОЩЬЮ собаки)
       «яблоко упало с дерева» -> упасть OBJECT_OF дерево  (дерево — объект падения)
  2) КИРПИЧ «с» (aa420be): все значения, кроме времени, — честный отказ
     + note: совместности и источника в реестре НЕ БЫЛО.
  3) РЕШЕНИЕ ХОЗЯИНА: ввести WITH и PATH. Отказ исчерпал себя — теперь
     честная связь вместо пропуска:
       «чай с лимоном»         -> чай WITH лимон
       «кот гуляет с собакой»  -> гулять WITH собака
       «яблоко упало с дерева» -> упасть PATH дерево (исходная точка)
     Время («с рассветом», «с утра») по-прежнему CIRCUMSTANCE — «когда?».

PATH ОТВЕЧАЕТ НА «ОТКУДА?», WITH НА «С КЕМ/С ЧЕМ?», CIRCUMSTANCE НА
«КОГДА?». Прежние лжи (INSTRUMENT для спутника, OBJECT_OF для источника)
заперты от возврата.

НАСТОЯЩИЙ ИНСТРУМЕНТ НЕ ЗАДЕТ: тв.п. БЕЗ предлога («кот открывает дверь
ключом») идёт своим зондом и остаётся INSTRUMENT. INSTRUMENT и WITH
разведены НАВЕК: «режет ножом» (орудие) против «режет с ножом» (спутник).

ЗДЕСЬ НЕТ СОЗНАТЕЛЬНО: местоимения («со мной», «со всеми») — отдельная
гора; атрибут с род.п. («дом с высокой крышей») — та же пара WITH, но
прилагательное требует согласования — покрыто кирпичом «к».
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


class TestSWith(unittest.TestCase):
    """«с» + тв.п. (не время) = совместность/атрибут (WITH), не орудие."""

    def test_tea_with_lemon(self):
        t = triples("чай с лимоном")
        self.assertIn(("чай", "WITH", "лимон"), t)
        self.assertNotIn(("чай", "INSTRUMENT", "лимон"), t,
                         "чай пьют лимоном — прежняя ложь вернулась")

    def test_cat_walks_with_dog(self):
        t = triples("кот гуляет с собакой")
        self.assertIn(("гулять", "WITH", "собака"), t)
        self.assertNotIn(("гулять", "INSTRUMENT", "собака"), t,
                         "гуляют с помощью собаки — прежняя ложь")
        self.assertIn(("кот", "CAN_DO", "гулять"), t,
                      "основная связь потеряна")

    def test_house_with_roof(self):
        t = triples("дом с крышей")
        self.assertIn(("дом", "WITH", "крыша"), t)
        self.assertNotIn(("дом", "INSTRUMENT", "крыша"), t)

    def test_pie_with_apples(self):
        self.assertIn(("пирог", "WITH", "яблоко"),
                      triples("пирог с яблоками"))

    def test_old_man_with_rod(self):
        t = triples("старик с удочкой сидел у реки")
        self.assertIn(("сидеть", "WITH", "удочка"), t)
        self.assertNotIn(("сидеть", "INSTRUMENT", "удочка"), t)


class TestSPath(unittest.TestCase):
    """«с» + род.п. = исходная точка (PATH), не объект."""

    def test_apple_falls_from_tree(self):
        t = triples("яблоко упало с дерева")
        self.assertIn(("упасть", "PATH", "дерево"), t)
        self.assertNotIn(("упасть", "OBJECT_OF", "дерево"), t,
                         "дерево — объект падения — прежняя ложь")
        self.assertNotIn(("упасть", "LOCATED_IN", "дерево"), t)
        self.assertIn(("яблоко", "CAN_DO", "упасть"), t,
                      "основная связь потеряна")

    def test_star_falls_from_sky(self):
        t = triples("звезда упала с неба")
        self.assertIn(("упасть", "PATH", "небо"), t)
        self.assertNotIn(("упасть", "OBJECT_OF", "небо"), t)

    def test_leaf_falls_from_roof(self):
        t = triples("лист упал с крыши")
        self.assertIn(("упасть", "PATH", "крыша"), t)
        self.assertNotIn(("упасть", "OBJECT_OF", "крыша"), t)


class TestSTimeIsCircumstance(unittest.TestCase):
    """«с/со» + слово времени = обстоятельство (честная правда)."""

    def test_sang_at_dawn(self):
        self.assertIn(("запеть", "CIRCUMSTANCE", "рассвет"),
                      triples("с рассветом птицы запели"))

    def test_raining_since_morning(self):
        self.assertIn(("лить", "CIRCUMSTANCE", "утро"),
                      triples("с утра льёт дождь"))

    def test_time_is_not_with(self):
        t = triples("с утра льёт дождь")
        self.assertNotIn(("лить", "WITH", "утро"), t,
                         "время записано спутником")


class TestSNoRegression(unittest.TestCase):
    """Настоящий инструмент и соседние предлоги не задеты."""

    def test_true_instrument_survives(self):
        t = triples("кот открывает дверь ключом")
        self.assertIn(("открывать", "INSTRUMENT", "ключ"), t,
                      "орудие в тв.п. без предлога сломано")

    def test_k_and_po_time_survive(self):
        self.assertIn(("письмо", "RECIPIENT", "брат"),
                      triples("письмо к брату"))
        self.assertIn(("болеть", "CIRCUMSTANCE", "утро"),
                      triples("щека болит по утрам"))

    def test_place_in_survives(self):
        self.assertIn(("сахар", "LOCATED_IN", "чай"),
                      triples("сахар в чае"))

    def test_companionship_verb_head(self):
        # спутник цепляется к БЛИЖАЙШЕМУ глаголу, а не к подлежащему:
        # «пошла с ребёнком» — шла вместе, не «мать WITH ребёнок»
        t = triples("мать с ребёнком пошла в поликлинику")
        self.assertIn(("пойти", "WITH", "ребёнок"), t)
        self.assertIn(("пойти", "LOCATED_IN", "поликлиника"), t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
