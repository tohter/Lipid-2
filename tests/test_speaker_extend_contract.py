# -*- coding: utf-8 -*-
"""
КОНТРАКТ: ГОВОРИЛКА-КОНСТРУКТОР, расширение на примеры 2/3/4
(кирпич №1-доп по плану советчиков).

ДЫРА
----
speaker.py закрывает только пример 1 («фонарь освещал аптеку на ночной
улице»). Остальные три словесных набора из постановки хозяина движок собрать
не умеет — нет:
  * cause-обстоятельства «от + род.» («от ужаса», «от приятной музыки»);
  * обстоятельства «при виде + род.» («при виде чудовища»);
  * обстоятельства «по + дат.» («по коже»);
  * present-tense глагола («бегут», «шатает») и безличного инфинитива
    («дрожать»);
  * possessive-оборота в родительном («твоего дома»).

ЧТО ЗАКРЕПЛЕНО (выход — точные строки из постановки хозяина)
------------------------------------------------------------
  2) {predicate: дрожать, cause: ужас, view: чудовище} ->
        «дрожать от ужаса при виде чудовища»   (безличный инфинитив ведёт)
  3) {predicate: бежать, subject: мурашка(pl), tense: present,
      cause: музыка+приятный, along: кожа} ->
        «от приятной музыки по коже бегут мурашки»
  4) {predicate: шатать, subject: ветер, tense: present,
      object: труба + possessor: дом(твой)} ->
        «ветер шатает трубу твоего дома»

   Пример 1 не ломается (старый контракт остаётся зелёным): порядок
   [субъект][глагол][объект][место] без cause не трогаем.

МУТАЦИИ, на которых контракт обязан покраснеть:
  * cause-обстоятельство не ставит «от» или не ставит род. («ужас» -> «от
    ужаса»);
  * прилагательное причины не согласуется с существительным в род./ж.р./ед.ч.
    («от приятный музыки»);
  * present-tense не ставит 3-е лицо по числу субъекта («мурашка бежим»);
  * possessive-оборот не ставит посессора и его прилагательное в род.
    («трубу твой дом»).
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lang.ru.speaker import speaker_sentence  # noqa: E402


def frames():
    """Рукописные ролевые структуры расширения (без думалки)."""
    return {
        # пример 2: безличный инфинитив, cause + «при виде»
        "ужас": {
            "predicate": "дрожать",
            "cause": {"lemma": "ужас"},
            "view": {"lemma": "чудовище"},
        },
        # пример 3: present, cause с прилагательным, «по коже»
        "мурашки": {
            "predicate": "бежать",
            "predicate_tense": "present",
            "subject": "мурашка",
            "subject_number": "pl",
            "cause": {"lemma": "музыка", "adjective": "приятный"},
            "along": {"lemma": "кожа"},
        },
        # пример 4: present, объект с possessor-оборотом
        "ветер": {
            "predicate": "шатать",
            "predicate_tense": "present",
            "subject": "ветер",
            "object": {
                "lemma": "труба",
                "possessor": {"lemma": "дом", "possessive": "твой"},
            },
        },
    }


class SpeakerExtendContract(unittest.TestCase):

    def test_infinitive_lead_with_cause_and_view(self):
        # 2: безличная конструкция — глагол-инфинитив ведёт
        s = speaker_sentence(frames()["ужас"])
        self.assertEqual(s, "дрожать от ужаса при виде чудовища")

    def test_cause_uses_genitive_preposition_ot(self):
        # cause -> «от» + род. («ужас» -> «ужаса»)
        s = speaker_sentence(frames()["ужас"])
        self.assertIn("от ужаса", s)
        self.assertNotIn("от ужас ", s)

    def test_view_construction_privid_gen(self):
        # «при виде» + род. («чудовище» -> «чудовища»)
        s = speaker_sentence(frames()["ужас"])
        self.assertIn("при виде чудовища", s)
        self.assertNotIn("при виде чудовище", s)

    def test_present_cause_along_subject_verb_order(self):
        # 3: [причина][по+дат.][субъект][глагол-наст.]
        s = speaker_sentence(frames()["мурашки"])
        self.assertEqual(s, "от приятной музыки по коже бегут мурашки")

    def test_cause_adjective_agrees_in_gen_fem(self):
        # «музыка» + прил. «приятный» -> «от приятной музыки»
        s = speaker_sentence(frames()["мурашки"])
        self.assertIn("от приятной музыки", s)
        self.assertNotIn("приятный музыки", s)

    def test_present_verb_agrees_in_person_number(self):
        # «мурашки» (3-е лицо, мн.) -> «бегут», а не «бежим»/«бежишь»
        s = speaker_sentence(frames()["мурашки"])
        self.assertIn("бегут", s)
        self.assertNotIn("бежим", s)
        self.assertNotIn("бежит", s)

    def test_along_uses_dative_preposition_po(self):
        # «по коже» (по + дат.)
        s = speaker_sentence(frames()["мурашки"])
        self.assertIn("по коже", s)

    def test_present_verb_agrees_with_singular_subject(self):
        # 4: «ветер шатает трубу твоего дома»
        s = speaker_sentence(frames()["ветер"])
        self.assertEqual(s, "ветер шатает трубу твоего дома")

    def test_possessive_genitive_oborot(self):
        # «дом» -> «дома», притяж. «твой» -> «твоего», род. ед.ч.
        s = speaker_sentence(frames()["ветер"])
        self.assertIn("твоего дома", s)
        self.assertNotIn("твой дом", s)

    def test_object_still_in_accusative(self):
        # «труба» -> «трубу» (винительный), несмотря на possessor
        s = speaker_sentence(frames()["ветер"])
        self.assertIn("трубу", s)
        self.assertNotIn("труба ", s)


if __name__ == "__main__":
    unittest.main()