# -*- coding: utf-8 -*-
"""Ответ обязан называть свою достоверность: гипотеза — вслух, знание — прямо.

ЗАЧЕМ (слова Хозяина 08.11, той же серией, что кирпичи «свидетельство = связь»
и «пересказ не подтверждает»). Хозяин велел отличать фазы на числах:

    «ребёнок всю жизнь бросал кирпичи — они падают с глухим стуком; он бросил
     мяч, он отскочил — ребёнок испугался; бросил несколько раз и убедился —
     это переходит в фазу знания, событие предсказуемо, кортизол падает,
     дофамин растёт».

ЧТО ЗАМЕРЕНО 08.11 ДО ПРАВКИ (живой прогон, два мира, различающиеся ОДНИМ
числом support у тех же двух рёбер):

    support=1:  «вода может гасить огонь: вода гасить огонь»
                кортизол 20, дофамин 255, открытых целей 0
    support=3:  «вода может гасить огонь: вода гасить огонь»
                кортизол 20, дофамин 255, открытых целей 0

ОДИНАКОВО байт в байт. То есть фаза «испугался» и фаза «знаю» до сих пор
дают ОДНО число и ОДНУ фразу, а число достоверности, которое мы честно
научились накапливать (core/records.py), до ответа НЕ доходит: его пишет
learner, складывает add_witness, но НЕ ЧИТАЕТ никто, кто отвечает. Накопить
и не показать — значит накопить впустую.

ЧТО ЗАКРЕПЛЯЕТСЯ:

  * у найденной связи есть число ОТДЕЛЬНЫХ свидетельств weakest звена пути.
    Внутри одного объяснения берётся МИНИМУМ: цепочка убедительна ровно
    настолько, насколько её самое неуверенное ребро. Между альтернативными
    объяснениями берётся МАКСИМУМ: достаточно одного надёжного пути, чтобы
    ответить уверенно (второе, шаткое, объяснение не обязано тянуть ответ
    вниз — оно и не нужно, когда есть первое).
  * ответ по ОДНОМУ свидетельству обязан быть назван предположением. Молча
    выдать однократное наблюдение за знание — та же ложь, что и раньше,
    только с правильно посчитанным числом в файле.
  * Связь, у которой свидетельств вообще не записано (ребро из полей
    сущности: hierarchy / actions), — НЕ «ноль подтверждений» и не «три»:
    это НЕИЗВЕСТНО, и неизвестное трактуется как НЕ проверенное. Различать
    «не записано» и «записан ноль» здесь нельзя: число молчит, а ответ
    обязан говорить правду.

ЧЕГО ЭТОТ КОНТРАКТ СОЗНАТЕЛЬНО НЕ КАСАЕТСЯ. Гормоны (кортизол/дофамин) по
гипотезе пока успокаиваются как по знанию — это отдельный шаг, у него свой
контракт и свой риск: он трогает уже запертые тесты HormoneGate. Здесь
заперто ТОЛЬКО слово: система обязана называть свою неуверенность, когда она
есть. Числа на выходе — следующий кирпич, а не этот, чтобы падение одного
не пряталось под правкой другого.

ЧТО УМРЁТ ПРИ МУТАЦИЯХ: число свидетельств не считается (всегда максимум);
считается максимумом внутри пути; рёбра из полей сущности считаются
подтверждёнными; фраза перестает называть предположение; предположение
называется при подтверждённой связи.
"""

import json
import os
import shutil
import tempfile
import unittest

from core.records import fact_weight
from lang.ru.ability import render_verdict
from lang.ru.lemma import install as _install_lemma_service

_install_lemma_service()

from brain.simulator import StaticKnowledge       # noqa: E402
from brain.thinker import Thinker                 # noqa: E402


def world_of(support: int) -> dict:
    """Мир, где вода гасит огонь. Различие между мирами — ОДНО число."""
    obs = {}
    for head, rel, tail in (("вода", "CAN_DO", "гасить"),
                            ("гасить", "OBJECT_OF", "огонь")):
        obs[f"{head}|{rel}|{tail}"] = {
            "head": head, "relation": rel, "tail": tail,
            "confidence": 0.9, "negated": False, "support": support,
            "weight": fact_weight(0.9, False, rel, support),
            "evidence": "наблюдение", "source": "text",
            "provenance": [{"source": "wiki", "evidence": "статья", "at": 1},
                           {"source": "text", "evidence": "рассказ", "at": 2}]
            [: max(1, support)],
        }
    return {"entities": {"вода": {}, "огонь": {}},
            "observations": obs, "unknown_words": {}}


class TestAnswerNamesItsConfidence(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _think(self, support):
        return Thinker(StaticKnowledge(world_of(support))).think_link(
            "вода", "гасить", "огонь")

    # ── 1. число есть и оно честное ─────────────────────────────
    def test_1_thinker_reports_witness_count(self):
        """Одно свидетельство против трёх — разные числа, а не одно."""
        once = self._think(1)
        thrice = self._think(3)
        self.assertTrue(once.ok and thrice.ok,
                        "путь должен находиться в обоих мирах, иначе сравнивать нечего")
        self.assertEqual(1, once.certainty,
                         "путь из одной связки показан не одним свидетельством")
        self.assertEqual(3, thrice.certainty,
                         "подтверждённый путь показан неподтверждённым")

    def test_2_weakest_edge_decides(self):
        """Убедительность пути задаёт САМОЕ СЛАБОЕ звено."""
        world = world_of(3)
        # одно из двух рёбер разваливаем до единственного наблюдения
        world["observations"]["гасить|OBJECT_OF|огонь"]["support"] = 1
        world["observations"]["гасить|OBJECT_OF|огонь"]["provenance"] = [
            {"source": "text", "evidence": "один раз", "at": 1}]
        v = Thinker(StaticKnowledge(world)).think_link("вода", "гасить",
                                                       "огонь")
        self.assertTrue(v.ok, render_verdict(v))
        self.assertEqual(1, v.certainty,
                         "шаткое звено не повлияло на уверенность ответа — "
                         "цепочка убедительнее своего слабого места быть не может")

    def test_3_edge_without_observations_is_not_proof(self):
        """Ребро из поля сущности — НЕ подтверждение, а неизвестность.

        Замер: у «огонь --IS_A--> пламя», записанного только в hierarchy,
        наблюдений нет вовсе. Считать это «подтверждено» нельзя, как нельзя
        и врать про «ноль раз видели»: мы просто не измеряли.
        """
        world = {"entities": {"огонь": {"hierarchy": ["пламя"]}},
                 "observations": {}, "unknown_words": {}}
        knower = StaticKnowledge(world)
        self.assertIn("пламя", knower.related("огонь", "IS_A"),
                      "связь из поля сущности обязана читаться (она знание)")
        self.assertEqual(0, knower.witnesses("огонь", "IS_A", "пламя"),
                         "несуществующее наблюдение засчитано как подтверждение")

    # ── 2. слово ────────────────────────────────────────────────
    def test_4_single_witness_is_called_a_guess(self):
        """Ответ по одному наблюдению обязан быть назван предположением."""
        text = render_verdict(self._think(1))
        self.assertTrue(("предполож" in text) or ("пока" in text),
                        f"однократное наблюдение выдано как знание: «{text}» — "
                        f"Хозяин требовал отличать испуг от знания")

    def test_5_confirmed_link_is_not_a_guess(self):
        """Три отдельных свидетельства не имеют права притворяться догадкой."""
        text = render_verdict(self._think(3))
        self.assertNotIn("предполож", text,
                         f"проверенную связь назвали догадкой: «{text}» — "
                         f"опыт перестал что-либо значить")

    def test_6_refusal_does_not_claim_confidence(self):
        """Отказ не обязан оправдываться достоверностью: у него нет пути."""
        world = world_of(3)
        world["observations"] = {}
        v = Thinker(StaticKnowledge(world)).think_link("вода", "гасить",
                                                       "огонь")
        self.assertFalse(v.ok)
        self.assertEqual(0, v.certainty,
                         "отказ выглядит как уверенный ответ с пустым путём")
        self.assertNotIn("предполож", render_verdict(v))


class TestConfidenceIsNotInvented(unittest.TestCase):
    """Число обязано приходить из данных, а не из головы рассуждающего."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_7_count_comes_from_observations(self):
        """Поддержка записи и число ответа — ОДНО и то же число."""
        for support in (1, 2, 5):
            world = world_of(support)
            knower = StaticKnowledge(world)
            self.assertEqual(support,
                             knower.witnesses("вода", "CAN_DO", "гасить"),
                             f"поддержка {support} не дошла до читателя")

    def test_8_unknown_pair_is_zero_not_one(self):
        """Связи нет -> ноль. Ноль ≠ «связь есть, но её не подтверждали»."""
        knower = StaticKnowledge(world_of(2))
        self.assertEqual(0, knower.witnesses("вода", "CAN_DO", "летать"))


if __name__ == "__main__":
    unittest.main()
