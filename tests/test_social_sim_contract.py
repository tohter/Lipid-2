"""
Контракт СОЦИАЛЬНОГО симулятора (brain/social_sim.py + brain/emotions.py).

Что тут заперто и почему
------------------------
Рассказ: кошка и канарейка жили вместе, кошка любила канарейку, попросила не
рассказывать тайну, канарейка рассказала, кошка съела канарейку.

Соблазн — подсунуть готовую строку «потому что канарейка предала». Такой
симулятор «работал» бы на одном примере и врал бы на всех остальных. Замки
ниже требуют ЧЕСТНОГО вывода по данным:

  1. Вывод идёт по цепи: ОЖИДАНИЕ -> НАРУШЕНИЕ -> предательство -> обида ->
     гнев -> агрессия, и цепь предъявляется.
  2. Действие оправдано ТОЛЬКО если «плохая» шкала ПЕРЕШЛА ПОРОГ к моменту
     действия. Замерено: 4 тика -> 128 (мало), 8 -> 247 (пик), 12 -> 231,
     16 -> 199 (уже прошло). Отсюда — ОКНО, и оно заперто: рано/поздно
     симулятор ОТКАЗЫВАЕТСЯ выводить, а не подгоняет.
  3. Эмоции — ДВЕ независимые шкалы: любовь и обида живут одновременно.
  4. Нет звена (ожидания/нарушения/агента/действия) — сказано, какого.
  5. Симулятор только ЧИТАЕТ статику; тесты не пишут в живую память.

Мутация (проверена отдельно, командой ниже): замена вывода на константу
краснит п.1, п.2 и п.4 — то есть тесты ловят подстановку, а не проверяют её.
"""
import copy
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("LIPID2_STATUS", "0")

from brain.emotions import EmotionSystem, MAX_EMOTION
from brain.social_sim import (AGGRESSION_THRESHOLD, RIPEN_TICKS,
                              SocialSimulator, build_cat_canary_simulator)

DB = os.path.join(ROOT, "experiments", "cat_canary_world.json")


def load_world():
    with open(DB, encoding="utf-8") as fh:
        return json.load(fh)


class TestEmotionModel(unittest.TestCase):
    """Слой эмоций: две шкалы, нарастание, затухание, наложение."""

    def test_rise_takes_time_by_attack(self):
        """Стимул сильнее attack -> нарастание занимает НЕСКОЛЬКО тиков."""
        e = EmotionSystem(attack=40, decay=0)
        e.hurt(120)
        e.tick()
        self.assertEqual(e.bad.value, 40, "за первый тик в шкалу идёт ровно attack")
        e.tick()
        self.assertEqual(e.bad.value, 80)
        e.tick()
        self.assertEqual(e.bad.value, 120)

    def test_decay_without_events(self):
        """Пустые тики гасят шкалу — эмоция проходит сама."""
        e = EmotionSystem(attack=40, decay=10)
        e.hurt(40)
        e.tick()
        self.assertEqual(e.bad.value, 30)     # 40 влилось, 10 утекло
        e.tick()
        self.assertEqual(e.bad.value, 20)

    def test_ceiling(self):
        """Шкала не превышает 255 даже при огромном заряде."""
        e = EmotionSystem(attack=100, decay=0)
        e.hurt(10000)
        e.tick_n(5)
        self.assertEqual(e.bad.value, MAX_EMOTION)

    def test_two_scales_coexist(self):
        """ЛЮБОВЬ И НЕНАВИСТЬ ОДНОВРЕМЕННО — то, ради чего две шкалы."""
        e = EmotionSystem(attack=40, decay=0)
        e.please(120, closeness=1.0)
        e.hurt(120, closeness=1.0)
        e.tick_n(3)
        self.assertGreater(e.good.value, 0)
        self.assertGreater(e.bad.value, 0)

    def test_overlay_not_reset(self):
        """Новая обида ложится НА остаток, а не с нуля (инерция)."""
        e = EmotionSystem(attack=40, decay=1)
        e.hurt(40)
        e.tick_n(2)
        before = e.bad.value
        e.hurt(40)
        e.tick()
        self.assertGreater(e.bad.value, before)

    def test_closeness_multiplies(self):
        """
        Гадость от близкого ранит сильнее: closeness — множитель заряда.

        ЗАМЕР, который дал этот тест: при attack=100 за ОДИН тик в шкалу
        вливается ровно 100, сколько бы заряда ни лежало в запасе. Поэтому
        эффект близости виден НЕ с первого тика — надо дать запасу перетечь
        (иначе тест ловил бы не множитель, а потолок attack — ложный замер).
        """
        e1 = EmotionSystem(attack=100, decay=0)
        e2 = EmotionSystem(attack=100, decay=0)
        e1.hurt(100, closeness=1.0)
        e2.hurt(100, closeness=2.0)
        e1.tick_n(3)
        e2.tick_n(3)
        self.assertGreater(e2.bad.value, e1.bad.value,
                           "близость должна дать больший заряд к одному тику")


class TestDerivesTheAnswer(unittest.TestCase):
    """Ответ ВЫВЕДЕН по данным и предъявляет путь."""

    def setUp(self):
        self.sim = build_cat_canary_simulator()
        self.res = self.sim.why_acted("кошка", "съесть")

    def test_ok(self):
        self.assertTrue(self.res.ok, self.res.as_text())

    def test_answer_mentions_trust_and_anger(self):
        self.assertIn("доверие", self.res.answer)
        self.assertIn("агресс", self.res.answer)

    def test_chain_shows_expectation_and_violation(self):
        joined = " | ".join(self.res.chain)
        self.assertIn("ожидание", joined)
        self.assertIn("нарушение", joined)

    def test_chain_shows_static_causes_chain(self):
        joined = " | ".join(self.res.chain)
        for link in ("предательство", "обида", "гнев", "агрессия", "нападение"):
            self.assertIn(link, joined, f"в цепи нет звена «{link}»")

    def test_chain_shows_numbers(self):
        # Порог теперь в СХЕМЕ (данные), и интерпретатор называет шкалу и числа.
        joined = " | ".join(self.res.chain)
        self.assertIn("порог", joined)
        self.assertIn("230", joined)

    def test_action_connected_to_behavior_by_is_a(self):
        joined = " | ".join(self.res.chain)
        self.assertIn("нападение", joined)


class TestNumbersMustSupportAction(unittest.TestCase):
    """
    ОКНО ГНЕВА (заперто по замеру). Действие оправдано, только если «плохая»
    шкала перешла порог К МОМЕНТУ действия Caliente. Рано — гнев не дозрел;
    поздно — уже прошёл. В обоих случаях симулятор ОТКАЗЫВАЕТСЯ выводить.
    """

    def test_in_window_is_ok(self):
        sim = build_cat_canary_simulator(between_ticks=RIPEN_TICKS)
        self.assertTrue(sim.why_acted("кошка", "съесть").ok)
        self.assertGreaterEqual(sim.agents["кошка"].bad.value,
                                AGGRESSION_THRESHOLD)

    def test_too_early_is_refused(self):
        sim = build_cat_canary_simulator(between_ticks=1)
        res = sim.why_acted("кошка", "съесть")
        self.assertFalse(res.ok, "рано: гнев не дозрел, вывода быть не должно")
        self.assertIn("порог", " ".join(res.missing))

    def test_too_late_is_refused(self):
        sim = build_cat_canary_simulator(between_ticks=30)
        res = sim.why_acted("кошка", "съесть")
        self.assertFalse(res.ok, "поздно: гнев прошёл, вывода быть не должно")
        self.assertIn("порог", " ".join(res.missing))


class TestSaysWhatIsMissing(unittest.TestCase):
    """Нет звена — симулятор НАЗЫВАЕТ его."""

    def test_no_such_action(self):
        sim = build_cat_canary_simulator()
        res = sim.why_acted("кошка", "летать")
        self.assertFalse(res.ok)
        self.assertIn("нет события", " ".join(res.missing))

    def test_no_expectation(self):
        """Есть действие и эмоции, но никто ничего не ждал — объяснить нечем."""
        sim = SocialSimulator(load_world())
        sim.agent("кошка")
        sim.act("кошка", "съесть", obj="канарейка")
        res = sim.why_acted("кошка", "съесть")
        self.assertFalse(res.ok)
        self.assertIn("ожидания", " ".join(res.missing))

    def test_no_violation(self):
        """Ожидание есть, но его не нарушали — объяснять нечего."""
        sim = SocialSimulator(load_world())
        sim.agent("кошка")
        sim.agent("канарейка")
        sim.act("кошка", "просить", obj="канарейка")
        sim.expect("кошка", "канарейка", "рассказывать", negated=True)
        sim.act("кошка", "съесть", obj="канарейка")
        res = sim.why_acted("кошка", "съесть")
        self.assertFalse(res.ok)
        self.assertIn("наруш", " ".join(res.missing))

    def test_no_emotions(self):
        """Агент без эмоций: объяснять нечем — эмоции это числа."""
        sim = SocialSimulator(load_world())
        sim.act("кошка", "съесть", obj="канарейка")
        res = sim.why_acted("кошка", "съесть")
        self.assertFalse(res.ok)


class TestSocialSimulatorOnlyReads(unittest.TestCase):
    """Симулятор не пишет в статику — она же источник истины обучения."""

    def test_world_unchanged(self):
        world = load_world()
        before = copy.deepcopy(world)
        sim = build_cat_canary_simulator()
        sim.why_acted("кошка", "съесть")
        self.assertEqual(world, before, "симулятор изменил базу знаний")

    def test_emotions_not_in_world(self):
        """Эмоции — числа симулятора, в статике их нет и быть не должно."""
        world = load_world()
        sim = build_cat_canary_simulator()
        self.assertTrue(sim.agents)
        blob = json.dumps(world, ensure_ascii=False)
        self.assertNotIn("bad", blob.lower().split("evidence")[0])
        self.assertNotIn("emotion", blob.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
