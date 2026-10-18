"""
tests/test_expect_contract.py — ЗАМОК ОЖИДАНИЯ ИЗ ТЕКСТА (штрих A, задача 34).

Раньше ожидание («кошка ждёт, что канарейка НЕ расскажет») задавалось
ДИРЕКТИВОЙ {"do":"expect"} в файле рассказа, то есть рукой человека.
Тест требует, чтобы ожидание ВЫВОДИЛОСЬ из фразы рассказа, и что
распознанное не превращается в выдумку: если части не хватает — ожидание
НЕ создаётся, а причина видна.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXP = os.path.join(ROOT, "experiments")

from core.extract import SimpleExtractor
from brain.learner import Learner
from brain.simulator import Event
from brain.social_sim import (Expectation, SocialSimulator,
                              build_cat_canary_simulator, verbs_related)

REQUEST = "Кошка попросила канарейку никому не рассказывать тайну."

class TestExpectFromText(unittest.TestCase):
    """Ожидание выводится из фразы просьбы, а не из директивы."""

    def setUp(self):
        self.ex = SimpleExtractor()

    def test_request_yields_expectation(self):
        sf = self.ex.extract(REQUEST, order=1)
        self.assertEqual(len(sf.expectations), 1, "просьба обязана дать ожидание")
        e = sf.expectations[0]
        self.assertEqual(e.holder, "кошка")
        self.assertEqual(e.target, "канарейка")
        self.assertEqual(e.action, "рассказывать")
        self.assertEqual(e.obj, "тайна")
        self.assertTrue(e.negated, "частица «не» обязана сделать ожидание запретом")
        self.assertEqual(e.order, 1)

    def test_forbid_verb_negates_without_ne(self):
        """«Запретил разговаривать» — отрицание в самом глаголе, «не» нет."""
        sf = self.ex.extract("Учитель запретил ученику разговаривать.")
        self.assertEqual(len(sf.expectations), 1)
        self.assertTrue(sf.expectations[0].negated)
        self.assertEqual(sf.expectations[0].target, "ученик")

    def test_dative_addressee(self):
        """«Велела сыну» — адресат в ДАТЕЛЬНОМ, не в винительном."""
        sf = self.ex.extract("Мать велела сыну вернуться домой.")
        self.assertEqual(len(sf.expectations), 1)
        self.assertEqual(sf.expectations[0].holder, "мать")
        self.assertEqual(sf.expectations[0].target, "сын")
        self.assertEqual(sf.expectations[0].action, "вернуться")
        self.assertFalse(sf.expectations[0].negated)

    def test_no_infinitive_no_expectation(self):
        """Просьба без действия: чего ждёт кошка — неизвестно, выдумывать нельзя."""
        sf = self.ex.extract("Кошка попросила канарейку.")
        self.assertFalse(sf.expectations)
        self.assertTrue(any("инфинитив" in n for n in sf.notes),
                        "причина отказа обязана быть названа")

    def test_plain_event_no_expectation(self):
        sf = self.ex.extract("Кошка любила канарейку.")
        self.assertFalse(sf.expectations)


def _learner(tmp, max_calls=0):
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
                   max_teacher_calls=max_calls), load_world


class TestExpectLearnerAndSim(unittest.TestCase):
    """Ожидание проходит весь путь: текст -> мир -> симулятор."""

    TEXT = (REQUEST + " Канарейка рассказала тайну. Кошка съела канарейку.")

    def test_learner_records_expectation(self):
        with tempfile.TemporaryDirectory() as tmp:
            lr, load_world = _learner(tmp)
            rep = lr.learn_from_text(self.TEXT)
            self.assertEqual(rep.expectations_new, 1)
            w = load_world()
            self.assertEqual(len(w["expectations"]), 1)
            self.assertEqual(w["expectations"][0]["holder"], "кошка")

    def test_relearn_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            lr, load_world = _learner(tmp)
            lr.learn_from_text(self.TEXT)
            lr.learn_from_text(self.TEXT)
            self.assertEqual(len(load_world()["expectations"]), 1)

    def test_simulator_reads_expectation_from_world(self):
        """Симулятор берёт ожидание ИЗ МИРА — директивы expect нет."""
        with tempfile.TemporaryDirectory() as tmp:
            lr, load_world = _learner(tmp)
            lr.learn_from_text(self.TEXT)
            sim = SocialSimulator(load_world())
            self.assertEqual(len(sim.expectations), 1)
            self.assertEqual(sim.expectations[0].target, "канарейка")


class TestAspectGap(unittest.TestCase):
    """
    Видовой разрыв: инфинитив просьбы (несов. вид) и сказуемое рассказа
    (сов. вид) — РАЗНЫЕ леммы. Эвристика по общей основе замерена и
    отвергнута ('повторять' -> 'втор'). Тест держит и порог, и то, что
    нерешённый случай НЕ молчит.
    """

    def test_verbs_related_accepts_aspect_pairs(self):
        for a, b in (("рассказывать", "рассказать"),
                     ("повторять", "повторить"),
                     ("открывать", "открыть")):
            self.assertTrue(verbs_related(a, b), f"{a}/{b} — видовая пара")

    def test_verbs_related_rejects_weak_stems(self):
        # ЗАМЕР: общий префикс короче порога -> связь НЕ выдумывается.
        for a, b in (("нападать", "напасть"), ("убивать", "убить"),
                     ("есть", "съесть"), ("брать", "спрятать")):
            self.assertFalse(verbs_related(a, b), f"{a}/{b} не пара по основе")

    def test_violation_found_via_object_not_verb(self):
        """Нарушение найдено по ОБЪЕКТУ, хотя глаголы разных видов."""
        exp = Expectation(holder="кошка", target="канарейка",
                          action="рассказывать", obj="тайна", negated=True)
        ev = Event(subject="канарейка", action="рассказать", obj="тайна")
        self.assertTrue(exp.violated_by(ev))

    def test_object_alone_is_not_enough(self):
        """Тот же объект, ДРУГОЕ действие — нарушения нет."""
        exp = Expectation(holder="кошка", target="канарейка",
                          action="рассказывать", obj="тайна", negated=True)
        ev = Event(subject="канарейка", action="прятать", obj="тайна")
        self.assertFalse(exp.violated_by(ev),
                         "объект без похожей основы глагола — ложь")

    def test_loud_when_aspect_gap_unresolved(self):
        """Если нарушение НЕ найдено молча — обязаны быть предупреждение."""
        sim = SocialSimulator({"entities": {}, "observations": {}})
        sim.agent("кошка")
        sim.expect("кошка", "канарейка", "рассказывать", negated=True)
        err = io.StringIO()
        with redirect_stderr(err):
            sim.act("канарейка", "спрятать", obj="мяч")
        self.assertFalse(sim.violations)
        self.assertIn("НЕ сопоставлено", err.getvalue(),
                      "нерешённый случай не должен молчать")


class TestStoryNeedsNoDirective(unittest.TestCase):
    """В рассказе нет ни одной директивы expect — ожидание выводится из фраз."""

    def test_no_expect_directive_in_story(self):
        with open(os.path.join(EXP, "cat_canary_story.json"),
                  encoding="utf-8") as fh:
            script = json.load(fh)["script"]
        kinds = [d.get("do") for d in script]
        self.assertNotIn("expect", kinds,
                         "директива вернулась: ожидание снова задаётся рукой")

    def test_answer_unchanged_without_directive(self):
        """Эталон задачи 31 держится без директивы expect."""
        sim = build_cat_canary_simulator()
        self.assertEqual(len(sim.expectations), 1)
        res = sim.why_acted("кошка", "съесть")
        self.assertTrue(res.ok, res.as_text())
        self.assertIn("доверие", res.answer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
