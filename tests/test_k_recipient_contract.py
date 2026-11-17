#!/usr/bin/env python3
"""КОНТРАКТ: предлог «к/ко» — адресат, а не место.

ЗАМЕР ДО ПРАВКИ (инструмент /tmp/prep_k_measure.py, живой код):
  «письмо к брату»            -> письмо LOCATED_IN брат   (ЛОЖЬ: письмо не
                                  находится в брате, письмо АДРЕСОВАНО брату)
  «мать придирается к сыну»   -> придираться LOCATED_IN сын (ЛОЖЬ)
  «врач идёт к больному»      -> врач HAS_PROPERTY больной (ЛОЖЬ: прилага-
                                  тельное с предлогом — предложная группа
                                  при глаголе, а не свойство врача)
  «врач подошёл к окну»       -> связь с «окну» потеряна МОЛЧА
  «ветер стих к вечеру»       -> ветер LOCATED_IN вечер (ЛОЖЬ, время)

РЕШЕНИЕ ПО МАТРИЦЕ (хозяин: тип связи решают ТИПЫ СУЩНОСТЕЙ ПАРЫ, а не
фиксированная строка в PREP_CASES):
  «к» + живое существо (animacy=anim) -> RECIPIENT (адресат есть в реестре);
  «к» + предмет («к окну») или время («к вечеру») -> ЧЕСТНЫЙ ОТКАЗ + note:
  отношений направления и времени в реестре нет, RECIPIENT для них — ложь
  (закон «ложная связь хуже отсутствующей»).
  Прилагательное С ПРЕДЛОГОМ свойством не становится: свойство — это
  «какой?» и оно без предлога («врач идёт к больному» не делает врача
  больным). Предложное прилагательное идёт в падёжный разбор.

ЗАПЕРТО ОТ МОЛЧАЛИВОГО НОЛЯ: отказ «к» + неодушевлённое обязан оставить
заметку (note), иначе потеря связи выглядит как её отсутствие в тексте.

ЗДЕСЬ НЕТ СОЗНАТЕЛЬНО:
  «к» + местоимение («я иду к сестре») — отдельная гора (местоимения);
  глаголы движения с направлением («пойти к победе» как INTENDS) — нет
  отношения в реестре, отдельное решение хозяина;
  RECIPIENT в мире (learner стирает падежную роль в OBJECT_OF) — отдельная
  мера role_loss, не трогаем.
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


class TestKRecipient(unittest.TestCase):
    """«к» + живое существо = адресат (RECIPIENT), а не место."""

    def test_letter_to_brother(self):
        self.assertIn(("письмо", "RECIPIENT", "брат"),
                      triples("письмо к брату"))

    def test_naggle_at_son(self):
        self.assertIn(("придираться", "RECIPIENT", "сын"),
                      triples("мать придирается к сыну"))

    def test_recipient_not_location(self):
        t = triples("письмо к брату")
        self.assertNotIn(("письмо", "LOCATED_IN", "брат"), t,
                         "адресат записан местом — прежняя ложь вернулась")


class TestKInanimateHonestRefusal(unittest.TestCase):
    """«к» + предмет/время: отказ обязан быть ЧЕСТНЫМ и НАЗВАННЫМ."""

    def test_window_is_not_recipient(self):
        t = triples("врач подошёл к окну")
        self.assertNotIn(("подойти", "RECIPIENT", "окно"), t)
        self.assertNotIn(("подойти", "LOCATED_IN", "окно"), t,
                         "направление записано местом — выдумка")

    def test_evening_is_not_location(self):
        t = triples("ветер стих к вечеру")
        self.assertNotIn(("стихнуть", "LOCATED_IN", "вечер"), t)
        self.assertNotIn(("ветер", "LOCATED_IN", "вечер"), t,
                         "время записано местом — прежняя ложь")

    def test_refusal_is_not_silent(self):
        ns = notes("врач подошёл к окну")
        self.assertTrue(any("окну" in n for n in ns),
                        f"отказ молчалив (молчаливый ноль запрещён): {ns}")

    def test_evening_refusal_is_not_silent(self):
        ns = notes("ветер стих к вечеру")
        self.assertTrue(any("вечеру" in n for n in ns),
                        f"отказ молчалив: {ns}")


class TestKAdjectiveWithPreposition(unittest.TestCase):
    """Прилагательное С предлогом — НЕ свойство подлежащего."""

    def test_doctor_is_not_sick(self):
        t = triples("врач идёт к больному")
        self.assertNotIn(("врач", "HAS_PROPERTY", "больной"), t,
                         "«к больному» сделало врача больным — ложь")
        self.assertIn(("врач", "CAN_DO", "идти"), t,
                      "основная связь потеряна")

    def test_plain_adjective_still_works(self):
        t = triples("старый дом стоит у дороги")
        self.assertIn(("дом", "HAS_PROPERTY", "старый"), t,
                      "обычное свойство сломано правкой «к»")


class TestKNoRegression(unittest.TestCase):
    """Правка не задала соседей: дат.п. без предлога и «по»."""

    def test_dative_without_preposition_still_recipient(self):
        # «Витя дал кошке корм»: кошка — получатель без предлога
        t = triples("Витя дал кошке корм")
        self.assertIn(("дать", "RECIPIENT", "кошка"), t,
                      "дат.п. получателя сломан")

    def test_ko_form_honest_refusal(self):
        # «ко» — та же логика, что «к»: форма перед сочетанием согласных
        self.assertIn(("письмо", "RECIPIENT", "врач"),
                      triples("письмо ко врачу"))
        t = triples("подойти ко входу")
        self.assertNotIn(("подойти", "RECIPIENT", "вход"), t,
                         "«ко» + предмет стало адресатом")


if __name__ == "__main__":
    unittest.main(verbosity=2)
