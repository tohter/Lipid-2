"""
Контракт СИМУЛЯТОРА (brain/simulator.py) — механизм вывода «почему».

Что тут проверяется и почему это замки
--------------------------------------
Хозяин поставил задачу прямо: по трём предложениям («начался пожар», «полил
дождь», «пожар остановился») ответить на вопрос «почему прекратился пожар?»
ОТВЕТОМ «потому что вода гасит огонь» — при этом так, чтобы ответ был ВЫВЕДЕН,
а не подставлен.

Главная опасность (и то, что тесты обязаны ловить): подсунуть готовую строку
«потому что вода гасит огонь» легко, и такой симулятор будет «работать» ровно
на одном примере. Замки ниже требуют, чтобы вывод ШЁЛ ПО ДАННЫМ:

  1. Симулятор сам проходит мост «дождь IS_A вода» (без него ответа нет).
  2. Симулятор сам находит закон «вода CAN_DO гасить / гасить OBJECT_OF огонь».
  3. Симулятор сам убеждается, что гаснет именно то, что внутри пожара
     («пожар HAS_PART огонь»).
  4. Если любого звена нет — симулятор ГОВОРИТ, какого, а не угадывает
     (тихая деградация = баг).
  5. Симулятор только ЧИТАЕТ статику — мир до и после побайтово равен.

Отдельно — разделение статики и динамики: события рассказа живут в симуляторе
(вручную), а не в базе знаний.
"""
import copy
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("LIPID2_STATUS", "0")

from brain.simulator import Simulator, StaticKnowledge, build_fire_rain_simulator

DB = os.path.join(ROOT, "experiments", "fire_rain_world.json")


def load_world():
    with open(DB, encoding="utf-8") as fh:
        return json.load(fh)


def make_story(world=None, events=None):
    sim = Simulator(world if world is not None else load_world())
    for ev in (events or [("пожар", "начаться"), ("дождь", "лить"),
                          ("пожар", "остановиться")]):
        sim.add_event(*ev)
    return sim



class TestSimulatorDerivesAnswer(unittest.TestCase):
    """Ответ обязан быть ВЫВЕДЕН из данных, а не подставлен."""

    def test_answers_the_question(self):
        res = make_story().why_stopped("пожар")
        self.assertTrue(res.ok, res.as_text())
        self.assertIn("потому что вода гасит огонь", res.answer)

    def test_chain_shows_bridge_and_law(self):
        """Ответ обязан предъявить ПУТЬ, иначе «вывел» недоказуемо."""
        res = make_story().why_stopped("пожар")
        joined = " | ".join(res.chain)
        self.assertIn("дождь", joined)
        self.assertIn("вода", joined, "в цепочке нет моста дождь->вода")
        self.assertIn("гасить", joined, "в цепочке нет закона вода гасит")
        self.assertIn("огонь", joined, "в цепочке нет цели: огонь")

    def test_answer_is_built_from_rule(self):
        """Строка ответа = собранная из agent+action+target, не константа."""
        res = make_story().why_stopped("пожар")
        self.assertEqual(res.answer, "потому что вода гасит огонь")


class TestSimulatorSaysWhatIsMissing(unittest.TestCase):
    """Нет звена — симулятор НАЗЫВАЕТ его, а не молчит и не выдумывает."""

    def test_without_bridge_no_answer_and_missing_is_named(self):
        world = load_world()
        # Мост «дождь IS_A вода» живёт в ДВУХ местах (типизированное
        # наблюдение и поле hierarchy) — иначе ЗАМЕР врёт: тест «убрал мост»
        # прошёл бы при живом мосте во втором поле. Убираем ОБА.
        world["observations"].pop("дождь|IS_A|вода", None)
        world["entities"]["дождь"]["hierarchy"] = []
        res = make_story(world).why_stopped("пожар")
        self.assertFalse(res.ok, "ответ при отсутствии моста означает угадайку")
        self.assertTrue(res.missing, "не сказано, чего не хватает")
        self.assertIn("дождь", " ".join(res.missing))

    def test_without_law_no_answer(self):
        world = load_world()
        # Закон тоже может дублироваться полем actions — чистим оба.
        world["observations"].pop("вода|CAN_DO|гасить", None)
        world["entities"]["вода"]["actions"] = []
        res = make_story(world).why_stopped("пожар")
        self.assertFalse(res.ok)

    def test_without_part_no_answer(self):
        world = load_world()
        del world["observations"]["пожар|HAS_PART|огонь"]
        world["entities"]["пожар"]["hierarchy"] = ["бедствие"]
        res = make_story(world).why_stopped("пожар")
        self.assertFalse(res.ok, "без знания, что в пожаре горит огонь, "
                                 "причина не выводится")

    def test_no_stop_event(self):
        sim = make_story(events=[("пожар", "начаться"), ("дождь", "лить")])
        res = sim.why_stopped("пожар")
        self.assertFalse(res.ok)
        self.assertIn("прекратился", " ".join(res.missing))

    def test_no_event_before_stop(self):
        sim = make_story(events=[("пожар", "остановиться")])
        res = sim.why_stopped("пожар")
        self.assertFalse(res.ok)
        self.assertTrue(res.missing)


class TestSimulatorBridgesByIsA(unittest.TestCase):
    """Закон «гасить огонь» применим и к пламени: пламя IS_A огонь."""

    def test_flame_stopped_by_rain(self):
        sim = make_story(events=[("пожар", "начаться"), ("дождь", "лить"),
                                 ("пламя", "остановиться")])
        res = sim.why_stopped("пламя")
        self.assertTrue(res.ok, res.as_text())
        self.assertIn("вода гасит огонь", res.answer)


class TestSimulatorOnlyReads(unittest.TestCase):
    """У симулятора нет прав записи: статика до и после совпадает."""

    def test_world_unchanged(self):
        world = load_world()
        before = copy.deepcopy(world)
        sim = make_story(world)
        sim.why_stopped("пожар")
        self.assertEqual(world, before, "симулятор изменил базу знаний")

    def test_events_live_only_in_simulator(self):
        """События — динамика: в статической базе их нет и быть не должно."""
        world = load_world()
        sim = make_story(world)
        self.assertTrue(sim.events)
        self.assertNotIn("events", world)


class TestSimulatorDemoBuilds(unittest.TestCase):
    """Демонстрационный сценарий собирается и отвечает (дым-тест)."""

    def test_demo(self):
        sim = build_fire_rain_simulator()
        self.assertEqual(len(sim.events), 3)
        self.assertTrue(sim.why_stopped("пожар").ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
