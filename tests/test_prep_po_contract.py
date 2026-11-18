#!/usr/bin/env python3
"""КОНТРАКТ: предлог «по» — время есть, темы нет.

ЗАМЕР ДО ПРАВКИ (инструмент /tmp/prep_po.py и /tmp/prep_po2.py, живой код):
  «поезд идёт по рельсам»  -> идти ABOUT рельс   (ЛОЖЬ: рельс — не тема
                              поездки, поездка идёт ВДОЛЬ рельсов)
  «щека болит по утрам»    -> болеть ABOUT утро  (ЛОЖЬ: боль не «про утро»,
                              боль происходит ПО УТРАМ — время)
  «звонок по телефону»     -> звонок ABOUT телефон (ЛОЖЬ: звонок не о телефоне,
                              звонок ЧЕРЕЗ телефон)
  «судья судит по закону»  -> судить ABOUT закон (ЛОЖЬ: основание, не тема)

У «по» ЧЕТЫРЕ значения (путь/поверхность, время, средство, основание) и они
МОрФОЛОГИЧЕСКИ НЕРАЗЛИЧИМЫ: один и тот же дат.п. с тем же предлогом.
Один предлог — одно значение нельзя выдумать: запись любой связи, кроме
времени, будет ложью для остальных значений (закон «ложная связь хуже
отсутствующей»).

РЕШЕНИЕ (кирпич «к», тот же принцип: тип связи решает ТИП ПРАВОГО СЛОВА):
  «по» + слово времени (утро/вечер/день...) -> CIRCUMSTANCE — отношение
  существует в реестре («светит -> днём»), это чистая правда;
  «по» + всё остальное -> ЧЕСТНЫЙ ОТКАЗ + note: пути, средства и основания
  в реестре нет, а LOCATED_IN для «звонок по телефону» утверждает, что
  звонок находится в телефоне, — тоже ложь.

ЗАПЕРТО ОТ МОЛЧАЛИВОГО НОЛЯ: отказ обязан оставить заметку, как у «к».

ЗДЕСЬ НЕТ СОЗНАТЕЛЬНО:
  «по» + одушевлённое («мать тоскует по сыну») — объект чувства, не место
  и не адресат в чистом виде; закрыт тем же отказом, отдельного решения
  хозяина ждёт;
  путь/поверхность («поезд идёт по рельсам») — ждёт базу знаний о
  сущностях (рельсы = поверхность) или новое отношение; пока честный ноль;
  «скучаю по маме», «мы едем по дороге» — местоимения, отдельная гора.
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


class TestPoTime(unittest.TestCase):
    """«по» + слово времени = обстоятельство (CIRCUMSTANCE)."""

    def test_ache_in_mornings(self):
        self.assertIn(("болеть", "CIRCUMSTANCE", "утро"),
                      triples("щека болит по утрам"))

    def test_read_in_evenings(self):
        # «по вечерам читаю» гибнет на местоимении? Нет: подлежащего нет
        # вовсе («читаю» = 1 лицо), фраза уйдёт в отказ целиком — берём
        # фразу с существительным.
        self.assertIn(("читать", "CIRCUMSTANCE", "вечер"),
                      triples("бабушка читает по вечерам"))

    def test_time_is_not_topic(self):
        t = triples("щека болит по утрам")
        self.assertNotIn(("болеть", "ABOUT", "утро"), t,
                         "время записано темой — прежняя ложь")


class TestPoNonTimeHonestRefusal(unittest.TestCase):
    """«по» + не время: отказ обязан быть ЧЕСТНЫМ и НАЗВАННЫМ."""

    def test_rails_are_not_topic(self):
        t = triples("поезд идёт по рельсам")
        self.assertNotIn(("идти", "ABOUT", "рельс"), t)
        self.assertNotIn(("идти", "LOCATED_IN", "рельс"), t)
        self.assertIn(("поезд", "CAN_DO", "идти"), t,
                      "основная связь потеряна")

    def test_phone_is_not_topic(self):
        t = triples("звонок по телефону")
        self.assertNotIn(("звонок", "ABOUT", "телефон"), t)
        self.assertNotIn(("звонок", "LOCATED_IN", "телефон"), t)

    def test_refusal_is_not_silent(self):
        ns = notes("поезд идёт по рельсам")
        self.assertTrue(any("рельсам" in n for n in ns),
                        f"отказ молчалив (молчаливый ноль запрещён): {ns}")

    def test_phone_refusal_is_not_silent(self):
        ns = notes("звонок по телефону")
        self.assertTrue(any("телефону" in n for n in ns),
                        f"отказ молчалив: {ns}")


class TestPoNoRegression(unittest.TestCase):
    """Правка не задела соседей: «о/про/для» (настоящая тема) и время в тв.п."""

    def test_real_about_survives(self):
        self.assertIn(("книга", "ABOUT", "город"),
                      triples("книга о городе"))

    def test_dative_time_instrumental_survives(self):
        # «Солнце светит днём» — время в тв.п., не через «по»
        self.assertIn(("светить", "CIRCUMSTANCE", "день"),
                      triples("солнце светит днём"))

    def test_recipient_k_still_works(self):
        self.assertIn(("письмо", "RECIPIENT", "брат"),
                      triples("письмо к брату"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
