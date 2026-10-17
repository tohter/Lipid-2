"""
tests/test_question_contract.py — ЗАМОК РАЗБОРА ВОПРОСА (шаг C из п.29).

Проверяется ЗВЕНО между свободным вопросом и симулятором:
  * роли (тип/субъект/действие/объект) берутся из ПАДЕЖА, не из позиции;
  * вопрос направляется НУЖНОМУ симулятору;
  * непонятый вопрос — ГРОМКИЙ отказ с указанием, чего не хватило;
  * отвечает СИМУЛЯТОР, а не разборщик (учителя не спрашиваем никогда).
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from brain.question import answer, parse_question, WHY, WHAT, UNKNOWN
from brain.simulator import SimulationResult, build_fire_rain_simulator
from brain.social_sim import build_cat_canary_simulator


class TestQuestionParse(unittest.TestCase):

    def test_why_stop_roles(self):
        q = parse_question("Почему прекратился пожар?")
        self.assertTrue(q.ok)
        self.assertEqual(q.qtype, WHY)
        self.assertEqual(q.subject, "пожар")
        self.assertEqual(q.action, "прекратиться")

    def test_why_agent_roles(self):
        q = parse_question("Почему кошка съела канарейку?")
        self.assertTrue(q.ok)
        self.assertEqual(q.subject, "кошка")
        self.assertEqual(q.action, "съесть")
        self.assertEqual(q.obj, "канарейка")

    def test_what_type(self):
        q = parse_question("Что такое доверие?")
        self.assertTrue(q.ok)
        self.assertEqual(q.qtype, WHAT)
        self.assertEqual(q.subject, "доверие")

    def test_unknown_type_loud(self):
        q = parse_question("Как дела?")
        self.assertFalse(q.ok)
        self.assertEqual(q.qtype, UNKNOWN)
        self.assertTrue(q.missing)

    def test_empty_loud(self):
        q = parse_question("")
        self.assertFalse(q.ok)
        self.assertTrue(q.missing)


class TestDispatch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sims = {"fire": build_fire_rain_simulator(),
                    "soc": build_cat_canary_simulator()}

    def test_fire_question_routed(self):
        res = answer("Почему прекратился пожар?", self.sims)
        self.assertIsInstance(res, SimulationResult)
        self.assertTrue(res.ok, res.as_text())
        self.assertIn("вода", res.answer)

    def test_soc_question_routed(self):
        res = answer("Почему кошка съела канарейку?", self.sims)
        self.assertTrue(res.ok, res.as_text())
        self.assertIn("доверие", res.answer)

    def test_unknown_question_refused(self):
        res = answer("Как дела?", self.sims)
        self.assertFalse(res.ok)
        self.assertTrue(res.missing)

    def test_missing_simulator_loud(self):
        res = answer("Почему кошка съела канарейку?", {"fire": self.sims["fire"]})
        self.assertFalse(res.ok)
        self.assertIn("soc", " ".join(res.missing))


if __name__ == "__main__":
    unittest.main(verbosity=2)