"""
tests/test_schemes_contract.py — СХЕМЫ ВЫВОДА КАК ДАННЫЕ (задача 31).

ЧТО ЭТОТ ТЕСТ ЗАПИРАЕТ
----------------------
До этой задачи путь вывода был ЗАШИТ В PYTHON: в brain/simulator.py::why_stopped
и brain/social_sim.py::why_acted цепочки рёбер стояли циклами, а порог агрессии
(230) — константой модуля. Пока так, учить модель нечему (схеме негде лежать)
и в C переносить нечего (переносится интерпретатор, а не частный обход).

Тесты ниже требуют:
  1. Схемы ЧИТАЮТСЯ из данных мира (секция "schemes"), а не берутся из кода.
  2. Симуляторы дают ТОТ ЖЕ ответ, что давал зашитый код (сверка с эталоном).
  3. Порог и модификаторы ответа живут В СХЕМЕ: их правка в JSON меняет вывод
     БЕЗ правки Python — это и есть доказательство «данные, не код».
  4. Отсутствие схемы/ребра/числа даёт ГРОМКИЙ отказ с указанием чего нет.
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("LIPID2_STATUS", "0")

from core.schemes import load_schemes, SchemeInterpreter
from brain.simulator import StaticKnowledge, build_fire_rain_simulator
from brain.social_sim import build_cat_canary_simulator, AGGRESSION_THRESHOLD

FIRE = os.path.join(ROOT, "experiments", "fire_rain_world.json")
CAT = os.path.join(ROOT, "experiments", "cat_canary_world.json")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class TestSchemesLiveInData(unittest.TestCase):
    """Схемы — в файле мира, читаются обобщённым загрузчиком."""

    def test_fire_world_has_scheme(self):
        s = load_schemes(load(FIRE))
        self.assertIn("process_stopped", s)
        self.assertTrue(s["process_stopped"].steps)

    def test_cat_world_has_scheme(self):
        s = load_schemes(load(CAT))
        self.assertIn("social_betrayal", s)
        self.assertTrue(s["social_betrayal"].steps)

    def test_no_schemes_section_is_not_a_crash(self):
        """Мир без секции "schemes" — это пустой словарь, а не падение."""
        self.assertEqual(load_schemes({}), {})
        self.assertEqual(load_schemes({"entities": {}}), {})

    def test_threshold_lives_in_scheme_not_in_python(self):
        """
        Число порога обязано лежать в ДАННЫХ. Здесь проверяем, что схема его
        несёт — и заодно что константа модуля с ним СОГЛАСОВАНА (зеркало).
        Если кто-то поменяет одно без другого — тест упадёт, и это честно.
        """
        s = load_schemes(load(CAT))["social_betrayal"]
        thr = [st for st in s.steps if st.kind == "threshold"]
        self.assertEqual(len(thr), 1, "в схеме должен быть ровно один порог")
        self.assertEqual(thr[0].scale, "bad")
        self.assertEqual(thr[0].value, AGGRESSION_THRESHOLD,
                         "константа модуля разошлась со схемой — правь оба места")


class TestDerivesTheSameAnswer(unittest.TestCase):
    """Схемный вывод совпадает с эталоном, который давал зашитый код."""

    def test_fire_answer_is_same(self):
        res = build_fire_rain_simulator().why_stopped("пожар")
        self.assertTrue(res.ok, res.as_text())
        self.assertEqual(res.answer, "потому что вода гасит огонь")

    def test_flame_answer_is_same(self):
        from brain.simulator import Simulator
        world = load(FIRE)
        sim = Simulator(world)
        sim.add_event("пожар", "начаться")
        sim.add_event("дождь", "лить")
        sim.add_event("пламя", "остановиться")
        res = sim.why_stopped("пламя")
        self.assertTrue(res.ok, res.as_text())
        self.assertIn("вода гасит огонь", res.answer)

    def test_cat_answer_is_same(self):
        res = build_cat_canary_simulator().why_acted("кошка", "съесть")
        self.assertTrue(res.ok, res.as_text())
        self.assertIn("нарушил(а) доверие", res.answer)

    def test_chain_is_presented(self):
        """Ответ предъявляется ВМЕСТЕ с путём — иначе это подстановка."""
        res = build_fire_rain_simulator().why_stopped("пожар")
        joined = " | ".join(res.chain)
        for link in ("HAS_PART", "IS_A", "CAN_DO", "OBJECT_OF"):
            self.assertIn(link, joined, f"в пути нет звена «{link}»")


class TestSchemeIsDataNotCode(unittest.TestCase):
    """
    ГЛАВНОЕ ДОКАЗАТЕЛЬСТВО: правка СХЕМЫ в JSON меняет вывод без правки Python.
    """

    def test_threshold_value_comes_from_scheme(self):
        """Понизим порог в данных — «рано» перестанет быть отказом."""
        sim = build_cat_canary_simulator(between_ticks=1)   # гнев не дозрел
        self.assertFalse(sim.why_acted("кошка", "съесть").ok)
        # Правим ПОРОГ В ДАННЫХ (то же, что правка JSON):
        sim.schemes["social_betrayal"].steps[-1].value = 0
        self.assertTrue(sim.why_acted("кошка", "съесть").ok,
                        "порог из схемы не применился — значит он в коде")

    def test_answer_template_comes_from_scheme(self):
        """Смена шаблона ответа в данных меняет ответ без правки Python."""
        sim = build_fire_rain_simulator()
        sim.schemes["process_stopped"].answer = "ПОТОМУ ЧТО {agents} {verbs:fin} {targets}!"
        res = sim.why_stopped("пожар")
        self.assertEqual(res.answer, "ПОТОМУ ЧТО вода гасит огонь!")

    def test_relation_names_come_from_scheme(self):
        """Рёбра пути заданы данными: подмена ребра меняет путь."""
        sim = build_fire_rain_simulator()
        steps = sim.schemes["process_stopped"].steps
        self.assertEqual([st.relation for st in steps],
                         ["HAS_PART", "IS_A", "CAN_DO", "OBJECT_OF"])


class TestLoudRefusal(unittest.TestCase):
    """Нет схемы / ребра / числа — ГРОМКО, с указанием чего именно нет."""

    def test_missing_scheme_is_loud(self):
        sim = build_fire_rain_simulator()
        res = sim.why_stopped("пожар", scheme_name="не_существует")
        self.assertFalse(res.ok)
        self.assertIn("схемы", " ".join(res.missing))

    def test_missing_edge_is_loud(self):
        """Уберём OBJECT_OF из статики — схема обязана сказать, какого ребра нет."""
        world = load(FIRE)
        world["observations"].pop("гасить|OBJECT_OF|огонь")
        from brain.simulator import Simulator
        sim = Simulator(world)
        sim.add_event("пожар", "начаться")
        sim.add_event("дождь", "лить")
        sim.add_event("пожар", "остановиться")
        res = sim.why_stopped("пожар")
        self.assertFalse(res.ok)
        self.assertIn("OBJECT_OF", " ".join(res.missing))

    def test_threshold_below_is_loud_and_mentions_numbers(self):
        sim = build_cat_canary_simulator(between_ticks=1)
        res = sim.why_acted("кошка", "съесть")
        self.assertFalse(res.ok)
        joined = " ".join(res.missing)
        self.assertIn("порог", joined)
        self.assertIn("230", joined)


if __name__ == "__main__":
    unittest.main()