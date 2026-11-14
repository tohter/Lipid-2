# -*- coding: utf-8 -*-
"""
Контракт ВЕРТИКАЛЬНОГО СРЕЗА: близость/доверие ВЫВОДЯТСЯ из текста.

Что тут заперто и почему
------------------------
Советчики (consult/близость_консультация_вывод.md) и хозяин утвердили: близость
между участниками — попарное отношение с ПРОВЕНАНСОМ, оно живёт ТОЛЬКО в
симуляторе (не в базе знаний), выводится из текста словарём-признаков +
правилом-суммой, а НЕ задаётся глобальной константой. Доверие — ОТДЕЛЬНОЕ
направленное отношение: его создаёт АКТ доверия (просьба «не рассказывать»),
а не совместная жизнь.

Соблазн — спрятать близость в константу (как было: CLOSENESS_LIVING_TOGETHER)
и отвечать «кошка близко к канарейке» без свидетельства в тексте. Замки
ниже требуют ЧЕСТНОГО ВЫВОДА:

  1. Близость выводится из ПРИЗНАКОВ текста («жить»/«дружно»/«весело»/
     «играть») и несёт провенанс «вывод по правилу».
  2. НЕСКОЛЬКО согласованных признаков дают БОЛЬШУЮ близость, чем один
     (страховка от ложного завышения порога — формула босса).
  3. Действие без социальных признаков («сожрала») НЕ создаёт ребра близости.
  4. Доверие ≠ близость: просьба о тайне даёт доверие, но сама по себе
     совместная жизнь близости доверия НЕ даёт (первое предложение — это
     близость, не доверие).
  5. Ничего не пишется в базу знаний: рёбра живут в симуляторе.
  6. Полный рассказ: близость ВЫВЕДЕНА из текста (не задана) и заряд обиды
     считается через неё, а не через глобальную константу.
"""
import copy
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("LIPID2_STATUS", "0")

from brain.social_sim import (CLOSENESS_LIVING_TOGETHER, RIPEN_TICKS,
                              SocialSimulator, build_cat_canary_simulator)

DB = os.path.join(ROOT, "experiments", "cat_canary_world.json")


def load_world():
    with open(DB, encoding="utf-8") as fh:
        return json.load(fh)


def make_pair(first_text, second_text=None):
    """Собрать симулятор из двух событий-фраз и вернуть (sim, relation)."""
    script = [
        {"do": "agent", "name": "кошка"},
        {"do": "agent", "name": "канарейка"},
        {"do": "event", "subject": "кошка", "action": "жить",
         "obj": "канарейка", "text": first_text},
    ]
    if second_text:
        script.append({"do": "event", "subject": "кошка", "action": "просить",
                       "obj": "канарейка", "text": second_text})
    sim = SocialSimulator(load_world(), script=script)
    return sim, sim.relations.get(("кошка", "канарейка"))


def make_trust_event(verb_text, action="доверить"):
    """Собрать симулятор из ОДНОГО события-акта доверия.

    Вернуть (sim, relation): проверяет, что глагол доверия («доверила»,
    «доверилась», «вверила») из ТЕКСТА даёт ребро доверия кошка->канарейка,
    не опираясь на просьбу-ожидание. Явный `action` — только для директивы;
    настоящий глагол берётся из фразы.
    """
    script = [
        {"do": "agent", "name": "кошка"},
        {"do": "agent", "name": "канарейка"},
        {"do": "event", "subject": "кошка", "action": action,
         "obj": "канарейка", "text": verb_text},
    ]
    sim = SocialSimulator(load_world(), script=script)
    return sim, sim.relations.get(("кошка", "канарейка"))


class TestClosenessDerivedFromText(unittest.TestCase):
    """Близость выводится из признаков текста с провенансом, не дана рукой."""

    def test_love_word_gives_low_closeness(self):
        """«любила» — слабый признак близости, а не вывод из константы."""
        sim, r = make_pair("Кошка любила канарейку.")
        self.assertIsNotNone(r)
        self.assertGreater(r.closeness, 0.0)
        self.assertLess(r.closeness, 1.0,
                        "один слабый признак не должен задирать близость")
        self.assertEqual(r.provenance, "вывод по правилу",
                         "близость обязана нести провенанс, а не молчать")
        self.assertIn("любить", r.sources)

    def test_many_consistent_features_give_more(self):
        """Несколько согласованных признаков дают БОЛЬШУЮ близость."""
        sim, r_weak = make_pair("Кошка любила канарейку.")
        sim2, r_strong = make_pair(
            "Кошка дружно живёт с канарейкой и весело с ней играет.")
        self.assertGreater(r_strong.closeness, r_weak.closeness,
                           "«дружно живёт + весело играет» ближе, чем просто «любила»")
        self.assertGreater(r_strong.closeness, 1.0,
                           "совместная жизнь + положительная манера — это высокая близость")

    def test_sources_carry_feature_lemmas(self):
        """Провенанс держит ПРИЗНАКИ-источники (что именно дало близость)."""
        sim, r = make_pair(
            "Кошка дружно живёт с канарейкой и весело с ней играет.")
        self.assertIn("жить", r.sources)
        self.assertIn("играть", r.sources)
        self.assertIn("весело", r.sources)

    def test_aggression_word_creates_no_edge(self):
        """«сожрала» — не социальный признак; ребра близости быть не должно."""
        sim, r = make_pair("Кошка сожрала канарейку.")
        self.assertIsNone(r, "без признаков близости ребро не выводится")
        self.assertEqual(len(sim.relations), 0)


class TestTrustIsSeparateFromCloseness(unittest.TestCase):
    """Доверие — ОТДЕЛЬНОЕ направленное отношение, создаёт его АКТ доверия."""

    def test_act_of_trust_creates_trust(self):
        """Просьба «не рассказывать тайну» — акт доверия, даёт доверие."""
        sim, r = make_pair(
            "Кошка дружно живёт с канарейкой и весело с ней играет.",
            "Кошка попросила канарейку никому не рассказывать тайну.")
        self.assertIsNotNone(r)
        self.assertGreaterEqual(r.trust, 1.0,
                                "просьба о тайне создаёт доверие holder->target")

    def test_cohabitation_alone_gives_no_trust(self):
        """Совместная жизнь даёт БЛИЗОСТЬ, но НЕ доверие (советчики)."""
        sim, r = make_pair(
            "Кошка дружно живёт с канарейкой и весело с ней играет.")
        self.assertIsNotNone(r)
        self.assertGreater(r.closeness, 0.0, "близость от совместной жизни есть")
        self.assertEqual(r.trust, 0.0,
                         "доверие из совместной жизни НЕ выводится — это не доверие")

    def test_trust_directional(self):
        """Доверие направленное: кошка->канарейка, но не наоборот."""
        sim, _ = make_pair(
            "Кошка дружно живёт с канарейкой и весело с ней играет.",
            "Кошка попросила канарейку никому не рассказывать тайну.")
        # акт доверия задаёт кошка->канарейка
        self.assertIn(("кошка", "канарейка"), sim.relations)
        # обратное направление близости может не существовать отдельно,
        # но доверие в обратную сторону не выведено из просьбы кошки
        back = sim.relations.get(("канарейка", "кошка"))
        if back is not None:
            self.assertEqual(back.trust, 0.0,
                             "просьба кошки не создаёт доверие канарейка->кошка")

    def test_trust_act_verb_doverila_creates_trust(self):
        """«доверила тайну» — АКТ доверия сам по себе, без просьбы-ожидания."""
        sim, r = make_trust_event("Кошка доверила канарейке свою тайну.")
        self.assertIsNotNone(r, "глагол «доверила» должен дать ребро доверия")
        self.assertGreaterEqual(r.trust, 1.0,
                                "«доверила тайну» — акт доверия кошка->канарейка")
        self.assertEqual(r.closeness, 0.0,
                         "акт доверия НЕ поднимает близость (близость != доверие)")

    def test_trust_act_verb_doverilas_creates_trust(self):
        """«доверилась канарейке» — акт доверия (рефлексив), даёт доверие."""
        sim, r = make_trust_event("Кошка доверилась канарейке.", action="довериться")
        self.assertIsNotNone(r)
        self.assertGreaterEqual(r.trust, 1.0,
                                "«доверилась» — акт доверия кошка->канарейка")

    def test_trust_act_verb_vverila_creates_trust(self):
        """«вверила свой секрет» — акт доверия, даёт доверие."""
        sim, r = make_trust_event("Кошка вверила канарейке свой секрет.",
                                  action="вверить")
        self.assertIsNotNone(r)
        self.assertGreaterEqual(r.trust, 1.0,
                                "«вверила секрет» — акт доверия кошка->канарейка")

    def test_trust_act_verb_rasskazat_creates_no_trust(self):
        """«рассказала» НЕ акт доверия сам по себе (неоднозначно, шаг 5).

        Рассказ может быть и актом доверия, и предательством (если было
        ожидание «не рассказывать»). Поэтому «рассказала тайну» без контекста
        доверия НЕ даёт — это дыра 4, отдельный шаг.
        """
        sim, r = make_trust_event("Кошка рассказала канарейке тайну.",
                                  action="рассказать")
        self.assertIsNone(r, "«рассказала» без контекста не создаёт доверия")

    def test_trust_act_verb_razboltala_creates_no_trust(self):
        """«разболтала» — предательство, НЕ акт доверия."""
        sim, r = make_trust_event("Канарейка разболтала секрет.",
                                  action="разболтать")
        self.assertIsNone(r, "«разболтала» — не акт доверия")


class TestStoryUsesDerivedCloseness(unittest.TestCase):
    """Полный рассказ: близость ВЫВЕДЕНА из текста и двигает заряд обиды."""

    def test_story_derives_closeness_not_constant(self):
        """Близость в рассказе выведена из первой фразы, а не задана рукой."""
        sim = build_cat_canary_simulator()
        r = sim.relations.get(("кошка", "канарейка"))
        self.assertIsNotNone(r, "первая фраза должна вывести попарную близость")
        self.assertEqual(r.provenance, "вывод по правилу")
        self.assertGreater(r.closeness, 1.0,
                           "рассказ про близкую кошку и канарейку даёт высокую близость")

    def test_violation_charge_uses_derived_closeness(self):
        """Заряд обиды = base_charge x ВЫВЕДЕННАЯ близость, не константа."""
        sim = build_cat_canary_simulator(between_ticks=RIPEN_TICKS)
        r = sim.relations.get(("кошка", "канарейка"))
        self.assertIsNotNone(r)
        # заряд обиды обязан равняться 200 x выведенная близость
        viol = next(v for v in sim.violations
                    if v.expectation.holder == "кошка")
        self.assertAlmostEqual(viol.applied, 200.0 * r.closeness, places=6,
                               msg="применённый заряд обязан считать через выведенную близость")


class TestNoDbWrites(unittest.TestCase):
    """Попарные рёбра живут в симуляторе, в базу знаний НЕ пишутся."""

    def test_world_unchanged_after_derivation(self):
        world = load_world()
        before = copy.deepcopy(world)
        sim, _ = make_pair(
            "Кошка дружно живёт с канарейкой и весело с ней играет.",
            "Кошка попросила канарейку никому не рассказывать тайну.")
        sim.why_acted("кошка", "съесть")
        self.assertEqual(world, before,
                         "вывод близости/доверия не должен трогать базу знаний")

    def test_relations_not_in_world_file(self):
        """Поля relations/close/trust не должны попасть в статику."""
        blob = json.dumps(load_world(), ensure_ascii=False)
        self.assertNotIn("closeness", blob)
        self.assertNotIn("trust", blob)


if __name__ == "__main__":
    unittest.main(verbosity=2)