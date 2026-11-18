#!/usr/bin/env python3
"""КОНТРАКТ: предлог «по» — PATH (путь/колея) и время (CIRCUMSTANCE).

ИСТОРИЯ (три состояния одной строки):
  1) ДО кирпича «по»: «поезд идёт по рельсам» -> идти ABOUT рельс (ложь
     темы: рельс — не О ЧЁМ поездка), «щека болит по утрам» -> болеть
     ABOUT утро (боль не ПРО утро).
  2) КИРПИЧ «по» (73c2d8a): время -> CIRCUMSTANCE, остальное — честный
     отказ + note: пути в реестре НЕ БЫЛО.
  3) РЕШЕНИЕ ХОЗЯИНА: ввести PATH. Отказ исчерпал себя — теперь честная
     связь вместо пропуска:
       «поезд идёт по рельсам» -> идти PATH рельс
       «дети бегают по двору»  -> бегать PATH двор
     Время по-прежнему CIRCUMSTANCE («щека болит по утрам» = когда?).

PATH ОТВЕЧАЕТ НА «ГДЕ КУРСИРОВАЛО ДВИЖЕНИЕ?» — поверхность («по рельсам»,
«по морю», «по щеке»), среда («по двору») и канал/маршрут («звонок по
телефону», «судить по закону», «занятия по расписанию»: закон и расписание
здесь — колея, по которой идёт действие). Позиция консервативна: лучше
одна широкая колея, чем четыре выдуманных смысла — различать подвиды
(поверхность/канал) позже, если графу понадобится.

ABOUT (тема) здесь заперт НАВЕК: «по» не делает рельс темой поездки.

ЗДЕСЬ НЕТ СОЗНАТЕЛЬНО: местоимения («мы едем по дороге») — отдельная
гора; «скучаю по маме» (объект чувства, одушевлённое) — сейчас PATH
(«колея тоски»), отдельное решение хозяина; распределительное «по»
(«дать по яблоку») — редкий падеж, не заперт.
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


class TestPoPath(unittest.TestCase):
    """«по» + не-время = путь/колея (PATH), не тема."""

    def test_train_on_rails(self):
        t = triples("поезд идёт по рельсам")
        self.assertIn(("идти", "PATH", "рельс"), t)
        self.assertNotIn(("идти", "ABOUT", "рельс"), t,
                         "рельс стал темой поездки — прежняя ложь")
        self.assertNotIn(("идти", "LOCATED_IN", "рельс"), t)
        self.assertIn(("поезд", "CAN_DO", "идти"), t,
                      "основная связь потеряна")

    def test_children_run_in_yard(self):
        t = triples("дети бегают по двору")
        self.assertIn(("бегать", "PATH", "двор"), t)
        self.assertNotIn(("бегать", "ABOUT", "двор"), t)

    def test_tears_down_cheek(self):
        self.assertIn(("течь", "PATH", "щека"),
                      triples("слёзы текут по щеке"))

    def test_ship_on_sea(self):
        self.assertIn(("плыть", "PATH", "море"),
                      triples("корабль плывёт по морю"))

    def test_mother_along_street(self):
        self.assertIn(("идти", "PATH", "улица"),
                      triples("мать идёт по улице"))

    def test_channel_reading(self):
        # канал/маршрут: та же колея, не тема
        self.assertIn(("звонок", "PATH", "телефон"),
                      triples("звонок по телефону"))
        self.assertIn(("судить", "PATH", "закон"),
                      triples("судья судит по закону"))


class TestPoTime(unittest.TestCase):
    """«по» + слово времени = обстоятельство (CIRCUMSTANCE)."""

    def test_ache_in_mornings(self):
        t = triples("щека болит по утрам")
        self.assertIn(("болеть", "CIRCUMSTANCE", "утро"), t)
        self.assertNotIn(("болеть", "PATH", "утро"), t,
                         "время записано колеей")
        self.assertNotIn(("болеть", "ABOUT", "утро"), t,
                         "время записано темой — прежняя ложь")

    def test_grandmother_reads_in_evenings(self):
        self.assertIn(("читать", "CIRCUMSTANCE", "вечер"),
                      triples("бабушка читает по вечерам"))


class TestPoNoRegression(unittest.TestCase):
    """Соседние кирпичи не задеты."""

    def test_real_about_survives(self):
        self.assertIn(("книга", "ABOUT", "город"),
                      triples("книга о городе"))

    def test_dative_time_instrumental_survives(self):
        self.assertIn(("светить", "CIRCUMSTANCE", "день"),
                      triples("солнце светит днём"))

    def test_recipient_k_still_works(self):
        self.assertIn(("письмо", "RECIPIENT", "брат"),
                      triples("письмо к брату"))

    def test_with_s_survives(self):
        self.assertIn(("чай", "WITH", "лимон"),
                      triples("чай с лимоном"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
