#!/usr/bin/env python3
"""КОНТРАКТ: «в/на» + слово времени = обстоятельство, а не место.

ЗАМЕР ДО ПРАВКИ (/tmp/prep_v.py, /tmp/prep_v2.py, живой код):
  «дождь пошёл в ночь»     -> пойти LOCATED_IN ночь   (ЛОЖЬ: дождь не
                              НАХОДИТСЯ в ночи — дождь пошёл В НОЧЬ, время)
  «снег выпал в январе»    -> выпасть LOCATED_IN январь (ЛОЖЬ: то же)
  «в пятницу была гроза»   -> гроза LOCATED_IN пятница (ЛОЖЬ)
  «на рассвете туман ушёл» -> уйти LOCATED_IN рассвет  (ЛОЖЬ)

Время с предлогом записывалось МЕСТОМ: граф утверждал, что ночь — контейнер,
в котором находится дождь. CIRCUMSTANCE в реестре существует («светит ->
днём») и отвечает на «когда?» — это чистая правда вместо лжи.

РЕШЕНИЕ (кирпич «по», тот же принцип): «в/во/на» + слово из TIME_LEMMAS ->
CIRCUMSTANCE. TIME_LEMMAS расширен месяцами, днями недели и частями суток,
чтобы правило их узнавало. «Среда» НЕ включена СОЗНАТЕЛЬНО: омоним «среда
обитания» при правиле времени дал бы ложь в обратную сторону.

МЕСТО НЕ ЗАДЕТО: «кошка спит в кресле», «сахар в чае», «мышь забралась в
нору» — честный LOCATED_IN сохранён (контейнер и направление остаются
местом; отвечают на «где?»/«куда?», а не «когда?»).
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


class TestVTimeIsCircumstance(unittest.TestCase):
    """«в/на» + время = CIRCUMSTANCE, время не место."""

    def test_rain_at_night(self):
        t = triples("дождь пошёл в ночь")
        self.assertIn(("пойти", "CIRCUMSTANCE", "ночь"), t)
        self.assertNotIn(("пойти", "LOCATED_IN", "ночь"), t,
                         "время записано местом — прежняя ложь")

    def test_snow_in_january(self):
        t = triples("снег выпал в январе")
        self.assertIn(("выпасть", "CIRCUMSTANCE", "январь"), t)
        self.assertNotIn(("выпасть", "LOCATED_IN", "январь"), t)

    def test_storm_on_friday(self):
        t = triples("в пятницу была гроза")
        self.assertIn(("гроза", "CIRCUMSTANCE", "пятница"), t)
        self.assertNotIn(("гроза", "LOCATED_IN", "пятница"), t)

    def test_fog_at_dawn(self):
        t = triples("на рассвете туман ушёл")
        self.assertIn(("уйти", "CIRCUMSTANCE", "рассвет"), t)
        self.assertNotIn(("уйти", "LOCATED_IN", "рассвет"), t)

    def test_train_into_night(self):
        t = triples("поезд уехал в ночь")
        self.assertIn(("уехать", "CIRCUMSTANCE", "ночь"), t)


class TestVPlaceSurvives(unittest.TestCase):
    """Место и направление — по-прежнему LOCATED_IN."""

    def test_cat_in_armchair(self):
        self.assertIn(("спать", "LOCATED_IN", "кресло"),
                      triples("кошка спит в кресле"))

    def test_sugar_in_tea(self):
        self.assertIn(("сахар", "LOCATED_IN", "чай"),
                      triples("сахар в чае"))

    def test_mouse_into_hole(self):
        t = triples("мышь забралась в нору")
        self.assertIn(("забраться", "LOCATED_IN", "нора"), t)

    def test_guest_to_city(self):
        t = triples("гость приехал в город")
        self.assertIn(("приехать", "LOCATED_IN", "город"), t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
