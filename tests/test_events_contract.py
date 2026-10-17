"""
tests/test_events_contract.py — ЗАМОК РАЗДЕЛЕНИЯ СТАТИКИ И СОБЫТИЙ (задача 32).

Проверяется ГРАНИЦА, а не удобство:
  * «кошки едят птиц» (наст. время) — ФАКТ, идёт в онтологию;
  * «кошка съела канарейку» (прош. время) — СОБЫТИЕ, идёт в секцию events;
  * событие НЕ попадает в онтологию (иначе один случай становится истиной);
  * события хранятся с ПОРЯДКОМ (время) и читаются симулятором;
  * рассказ целиком восстанавливается из данных, а не из кода.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from brain.learner import Learner
from brain.simulator import Simulator
from brain.social_sim import build_cat_canary_simulator
from core.extract import SimpleExtractor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")


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
    lr = Learner(load_world=load_world, save_world=save_world,
                 max_teacher_calls=max_calls)
    return lr, load_world


class TestFactEventBoundary(unittest.TestCase):
    """Грамматическая граница: настоящее время — факт, прошедшее — событие."""

    def test_present_tense_is_fact(self):
        ex = SimpleExtractor()
        sf = ex.extract("Кошки едят птиц.")
        self.assertFalse(sf.events, "настоящее время не должно быть событием")
        self.assertTrue(sf.facts)
        self.assertTrue(all(not f.is_event for f in sf.facts))

    def test_past_tense_is_event(self):
        ex = SimpleExtractor()
        sf = ex.extract("Кошка съела канарейку.", order=3)
        self.assertEqual(len(sf.events), 1, "прошедшее время — это событие")
        ev = sf.events[0]
        self.assertEqual(ev.subject, "кошка")
        self.assertEqual(ev.action, "съесть")
        self.assertEqual(ev.tense, "past")
        self.assertEqual(ev.order, 3)

    def test_past_tense_facts_marked_event(self):
        """Факты предложения-события помечены predicate='event'."""
        ex = SimpleExtractor()
        sf = ex.extract("Кошка напала на канарейку.")
        self.assertTrue(sf.facts)
        for f in sf.facts:
            self.assertTrue(f.is_event, f"{f} не помечен событием")

    def test_copula_is_not_event(self):
        """«Вася — инженер»: связка, а не происшествие."""
        ex = SimpleExtractor()
        sf = ex.extract("Вася — инженер.")
        self.assertFalse(sf.events)


class TestLearnerWritesEventsOnce(unittest.TestCase):
    """Learner пишет события в секцию events, а не в онтологию."""

    def test_event_goes_to_events_not_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            lr, load_world = _learner(tmp)
            lr.learn_from_text("Кошка съела канарейку.")
            w = load_world()
            self.assertEqual(len(w.get("events", [])), 1)
            self.assertEqual(w["events"][0]["action"], "съесть")
            self.assertFalse(w.get("observations"),
                             "событие не должно попасть в онтологию")

    def test_static_fact_goes_to_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            lr, load_world = _learner(tmp)
            lr.learn_from_text("Кошки едят птиц.")
            w = load_world()
            self.assertFalse(w.get("events"), "факт не должен стать событием")
            self.assertTrue(w.get("observations"), "факт должен быть в онтологии")

    def test_relearn_does_not_duplicate_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            lr, load_world = _learner(tmp)
            lr.learn_from_text("Кошка съела канарейку.")
            lr.learn_from_text("Кошка съела канарейку.")
            w = load_world()
            self.assertEqual(len(w["events"]), 1, "повторное чтение удвоило событие")

    def test_report_counts_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            lr, _ = _learner(tmp)
            rep = lr.learn_from_text(
                "Кошка разозлилась. Кошка напала на канарейку.")
            self.assertEqual(rep.events_new, 2)


class TestStoryLivesInData(unittest.TestCase):
    """Рассказ — в данных, а не в Python."""

    def test_fire_story_loaded_from_file(self):
        with open(os.path.join(EXP, "fire_rain_world.json"), encoding="utf-8") as fh:
            world = json.load(fh)
        sim = Simulator(world)
        self.assertFalse(sim.events, "статическая база не должна нести событий")
        sim.load_story(os.path.join(EXP, "fire_rain_story.json"))
        self.assertEqual(len(sim.events), 3)
        self.assertTrue(sim.why_stopped("пожар").ok)

    def test_static_world_has_no_events(self):
        for name in ("fire_rain_world.json", "cat_canary_world.json"):
            with open(os.path.join(EXP, name), encoding="utf-8") as fh:
                w = json.load(fh)
            self.assertNotIn("events", w, f"{name}: события попали в статику")

    def test_cat_canary_story_from_data(self):
        sim = build_cat_canary_simulator()
        self.assertEqual(len(sim.events), 4)
        self.assertEqual(sim.events[-1].action, "съесть")
        self.assertTrue(sim.why_acted("кошка", "съесть").ok)

    def test_simulator_reads_learned_events(self):
        """Симулятор читает события, записанные ОБУЧЕНИЕМ."""
        with tempfile.TemporaryDirectory() as tmp:
            lr, load_world = _learner(tmp)
            lr.learn_from_text("Кошка разозлилась. Кошка напала на канарейку.")
            sim = Simulator(load_world())
            n = sim.load_events(load_world())
            self.assertEqual(n, 2)
            self.assertEqual(sim.events[1].action, "напасть")


if __name__ == "__main__":
    unittest.main(verbosity=2)