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


def make_betrayal_story(first_text, trust_act=True, betrayal=True):
    """Полный рассказ: близость + акт доверия (+ разглашение) + кара.

    Возвращает (sim, why_acted, violations[0] if any).
    """
    script = [
        {"do": "agent", "name": "кошка"},
        {"do": "agent", "name": "канарейка"},
        {"do": "please", "name": "кошка", "amount": 120},
        {"do": "event", "subject": "кошка", "action": "жить",
         "obj": "канарейка", "text": first_text},
    ]
    if trust_act:
        script.append({"do": "event", "subject": "кошка",
                       "action": "доверить", "obj": "канарейка",
                       "text": "Кошка доверила канарейке свою тайну."})
    script.append({"do": "ticks", "n": 2})
    if betrayal:
        script.append({"do": "event", "subject": "канарейка",
                       "action": "рассказать", "obj": "тайна",
                       "text": "Канарейка рассказала тайну."})
    else:
        script.append({"do": "event", "subject": "канарейка",
                       "action": "спать", "obj": "гнездо",
                       "text": "Канарейка спала в гнезде."})
    script += [
        {"do": "ticks", "n": 8},
        {"do": "event", "subject": "кошка", "action": "съесть",
         "obj": "канарейка", "text": "Кошка съела канарейку."},
    ]
    sim = SocialSimulator(load_world(), script=script)
    why = sim.why_acted("кошка", "съесть")
    viol = sim.violations[0] if sim.violations else None
    return sim, why, viol


class TestTrustIsViolableBetrayal(unittest.TestCase):
    """ШАГ 5: доверие НАРУШАЕМОЕ — разглашение тайны после акта = предательство.

    Акт доверия («доверила тайну») создаёт ОЖИДАНИЕ «не рассказывать тайну».
    Позже «рассказала тайну» = НАРУШЕНИЕ (предательство), а не пустое событие.
    Без акта доверия то же разглашение — НЕ предательство (нет ожидания).
    """

    def test_trust_act_creates_expectation_not_to_tell(self):
        """Акт доверия создаёт ожидание «не рассказывать тайну»."""
        sim, _, _ = make_betrayal_story(
            "Кошка дружно жила с канарейкой и весело с ней играла.",
            trust_act=True, betrayal=True)
        exps = [(e.holder, e.target, e.action, e.obj, e.negated)
                for e in sim.expectations]
        self.assertIn(("кошка", "канарейка", "рассказывать", "тайна", True),
                      exps, "акт доверия должен дать ожидание «не рассказывать тайну»")

    def test_betrayal_after_trust_is_violation(self):
        """Разглашение тайны после акта доверия = НАРУШЕНИЕ."""
        sim, why, viol = make_betrayal_story(
            "Кошка дружно жила с канарейкой и весело с ней играла.",
            trust_act=True, betrayal=True)
        self.assertIsNotNone(viol, "разглашение после доверия — предательство")
        self.assertEqual(viol.expectation.holder, "кошка",
                         "обижается тот, кто доверил тайну")
        self.assertEqual(viol.expectation.target, "канарейка",
                         "нарушила канарейка, что доверили тайну")

    def test_betrayal_charge_uses_derived_closeness(self):
        """Заряд предательства = base_charge x ВЫВЕДЕННАЯ близость."""
        sim, why, viol = make_betrayal_story(
            "Кошка дружно жила с канарейкой и весело с ней играла.",
            trust_act=True, betrayal=True)
        pair = sim.relations.get(("кошка", "канарейка"))
        self.assertAlmostEqual(viol.applied, 200.0 * pair.closeness, places=4,
                               msg="предательство ранит через близость к нарушителю")

    def test_betrayal_without_trust_act_is_not_violation(self):
        """Контроль: разглашение БЕЗ акта доверия — НЕ предательство."""
        sim, why, viol = make_betrayal_story(
            "Кошка дружно жила с канарейкой и весело с ней играла.",
            trust_act=False, betrayal=True)
        self.assertIsNone(viol,
                          "без доверия нет ожидания — разглашение не предательство")

    def test_no_disclosure_after_trust_is_no_violation(self):
        """Контроль: доверие есть, но тайну НЕ разгласили — НЕ предательство."""
        sim, why, viol = make_betrayal_story(
            "Кошка дружно жила с канарейкой и весело с ней играла.",
            trust_act=True, betrayal=False)
        self.assertIsNone(viol,
                          "без разглашения нет нарушения ожидания")

    def test_betrayal_gives_aggression_with_high_closeness(self):
        """Высокая близость + предательство -> заряд выше порога -> агрессия."""
        sim, why, viol = make_betrayal_story(
            "Кошка дружно жила с канарейкой и весело с ней играла.",
            trust_act=True, betrayal=True)
        self.assertTrue(why.ok,
                        "близость 1.7 x 200 = 340 > 294 -> агрессия")

    def test_betrayal_weak_closeness_no_aggression(self):
        """Слабая близость + предательство -> заряд мал -> НЕ агрессия.

        Близость — множитель, не причина (советчики): предательство есть
        (нарушение), но без близости кошке почти всё равно.
        """
        sim, why, viol = make_betrayal_story(
            "Кошка любила канарейку.", trust_act=True, betrayal=True)
        self.assertIsNotNone(viol, "предательство распознано даже при слабой близости")
        self.assertFalse(why.ok,
                         "близость 0.3 x 200 = 60 < 294 -> нет агрессии")


class TestTrustFromBaseMarker(unittest.TestCase):
    """Дорога А: доверие читает confidential-слова ИЗ БАЗЫ (маркер на сущности).

    Раньше слова-носители тайны жили в коде (_SECRET_WORDS). Теперь движок
    сканирует базу: у сущности «секрет» маркер semantic:confidential, её
    синонимы (тайна, конфиденциальная информация, для служебного пользования)
    — источник слов. Расширение — правкой базы, не кода (советчики + 3-й
    эксперт: маркер = индекс, а не команда).
    """

    def test_confidential_words_read_from_base(self):
        """Слова-носители confidential приходят из сущности «секрет» (база), а не кода."""
        sim = SocialSimulator(load_world())
        words = sim._confidential_words()
        self.assertIn("тайна", words, "синоним «тайна» сущности «секрет» должен быть в пуле")
        self.assertIn("информация", words,
                      "синоним «конфиденциальная информация» даёт значимые слова")
        self.assertIn("секрет", words, "название сущности — тоже слово-носитель")

    def test_confidential_marker_controls_trigger(self):
        """Маркер в базе решает: убрать его — confidential-пул пустеет."""
        w = load_world()
        ent = w["entities"]["секрет"]
        had = [p for p in ent.get("properties", []) if p == "semantic:confidential"]
        ent["properties"] = [p for p in ent.get("properties", [])
                              if p != "semantic:confidential"]
        sim = SocialSimulator(w)
        self.assertEqual(sim._confidential_words(), set(),
                         "без маркера в базе confidential-слов нет")
        ent["properties"] += had

    def test_betrayal_drops_trust_to_zero(self):
        """ПРЕДАТЕЛЬСТВО рушит доверие: trust падает до 0 (главная дыра Дороги А)."""
        sim, why, viol = make_betrayal_story(
            "Кошка дружно жила с канарейкой и весело с ней играла.",
            trust_act=True, betrayal=True)
        pair = sim.relations.get(("кошка", "канарейка"))
        self.assertIsNotNone(pair, "ребро кошка->канарейка должно существовать")
        self.assertAlmostEqual(pair.trust, 0.0, places=4,
                               msg="разглашение тайны должно УРОНИТЬ доверие до 0")

    def test_trust_survives_without_betrayal(self):
        """Контроль: без разглашения доверие НЕ падает."""
        sim, why, viol = make_betrayal_story(
            "Кошка дружно жила с канарейкой и весело с ней играла.",
            trust_act=True, betrayal=False)
        pair = sim.relations.get(("кошка", "канарейка"))
        self.assertIsNotNone(pair, "ребро должно быть (акт доверия был)")
        self.assertGreater(pair.trust, 0.0,
                           "без предательства доверие держится")

    def test_betrayal_verb_razboltala_is_violation(self):
        """Глагол разглашения «разболтала» = нарушение ожидания (был пробел)."""
        script = [
            {"do": "agent", "name": "кошка"},
            {"do": "agent", "name": "канарейка"},
            {"do": "please", "name": "кошка", "amount": 120},
            {"do": "event", "subject": "кошка", "action": "доверить",
             "obj": "канарейка", "text": "Кошка доверила канарейке свою тайну."},
            {"do": "ticks", "n": 2},
            {"do": "event", "subject": "канарейка", "action": "разболтать",
             "obj": "тайна", "text": "Канарейка разболтала тайну на всю улицу."},
        ]
        sim = SocialSimulator(load_world(), script=script)
        self.assertEqual(len(sim.violations), 1,
                         "«разболтала тайну» после «доверила тайну» — предательство")
        pair = sim.relations.get(("кошка", "канарейка"))
        self.assertAlmostEqual(pair.trust, 0.0, places=4,
                               msg="и разболтала рушит доверие до 0")


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