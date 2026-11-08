# -*- coding: utf-8 -*-
"""Свидетельство обязано быть СВЯЗЬЮ с достоверностью, а не счётчиком слова.

ЗАЧЕМ ЭТОТ КОНТРАКТ (слова Хозяина, 08.11). Хозяин запретил устраивать
пожизненное ребро по одному факту и велел мерить достоверность опытом:
«всю жизнь бросал кирпичи — они падали с глухим стуком; первый раз бросил
мяч — он отскочил, ребёнок испугался; бросил несколько раз и убедился — это
переходит в фазу знания, событие предсказуемо, кортизол падает, дофамин
растёт». То есть: связь рождается ГИПОТЕЗОЙ с низкой достоверностью, похожие
свидетельства её повышают, и только тогда она становится знанием.

ЧТО ЗАМЕРЕНО 08.11 в живом коде, прежде чем что-либо менять.

  1) ЗАПИСЬ ЛЮБОПЫТСТВА ТЕРЯЕТ САМУ СВЯЗЬ.
     brain/lipid2_brain.py::_record_observation пишет ключ f"{concept}|curiosity"
     и три поля: confidence, support, provenance. Полей head / relation / tail
     НЕТ. А brain/simulator.py::StaticKnowledge собирает граф рассуждения
     ИСКЛЮЧИТЕЛЬНО из записей, где head и tail есть. Замер:

         _record_observation("мяч", {"hierarchy": ["игрушка"],
                                     "actions": ["катиться"]}, conf=0.9)
         -> StaticKnowledge.knows("мяч")      = False
         -> StaticKnowledge.related("мяч","IS_A")   = []
         -> StaticKnowledge.related("мяч","CAN_DO") = []

     Знание, добытое любопытством, НЕВИДИМО тому, кто рассуждает. Это и есть
     настоящая причина, по которой лестница стоит на 1/7: выучить слово
     можно, а связать его — нечем.

  2) ДОСТОВЕРНОСТЬ РАСТЁТ ОТ ПОСТОРОННИХ СВИДЕТЕЛЬСТВ.
     Ключ — одно слово, поэтому ДВА РАЗНЫХ факта про мяч сваливаются в одну
     запись. Замер: «мяч — игрушка» (conf 0.4) и «мяч катится» (conf 0.9)
     дали ОДНУ запись support=2, confidence=0.9. Полтора чужих факта сделали
     второй факт «подтверждённым дважды». Это ровно то, что Хозяин запретил:
     уверенность, нажитая не своим опытом.

  3) вес наблюдения для этой записи вообще НЕ ЗАПИСАН (weight=None), хотя
     формула в ядре существует и умеет считать (тогда это был
     core.records.fact_weight; с 08.11 ночь вес записи порождает сила
     основания — core.records.weight_from_ground, см. test_6).

ЧТО ЗАКРЕПЛЯЕТСЯ:

  * свидетельство любопытства записывается КАК СВЯЗЬ: head/relation/tail,
    ключ f"{head}|{relation}|{tail}" — тот же формат, что у обучающего из
    рассказов (brain/learner.py), а не свой собственный;
  * повторенный ТОТ ЖЕ факт повышает support; ДРУГОЙ факт про то же слово —
    отдельная запись, и чужой повтор её не греет;
  * число достоверности (weight) лежит в записи и растёт от подтверждения
    СВОЕЙ связью;
  * рассуждение обязано видеть выученное: после записи связь находится через
    StaticKnowledge, иначе учиться бессмысленно.

ЗАЧЕМ ЗДЕСЬ ТЕСТ НА НЕУТВЕРЖДЕНИЕ ПО ОДНОМУ ФАКТУ. Единичное свидетельство
имеет право предложить связь, но не право быть пиком уверенности: после
подтверждений вес обязан стать ВЫШЕ. Иначе «один раз увидел» и «убедился»
дали бы одно число, и различить гипотезу со знанием было бы нечем.

ЧТО УМРЁТ ПРИ МУТАЦИЯХ (проверено): ключ снова из одного слова; поля
head/relation/tail не пишутся; weight не считается; support растёт от любой
записи про то же слово.
"""

import json
import os
import shutil
import tempfile
import unittest

from core.records import observation_weight
from lang.ru.lemma import install as _install_lemma_service

_install_lemma_service()

from brain.lipid2_brain import Lipid2Brain        # noqa: E402
from brain.simulator import StaticKnowledge       # noqa: E402


def make_brain(dirpath):
    world = {"entities": {}, "observations": {}, "unknown_words": {}}
    with open(os.path.join(dirpath, "lipid_world.json"), "w",
              encoding="utf-8") as fh:
        json.dump(world, fh, ensure_ascii=False)
    brain = Lipid2Brain(memory_dir=dirpath)
    brain.ask_user_enabled = False
    brain.use_llm_teacher = False
    brain.use_wiki_teacher = False
    return brain


def read_world(dirpath):
    with open(os.path.join(dirpath, "lipid_world.json"), encoding="utf-8") as fh:
        return json.load(fh)


class TestEvidenceIsALink(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.brain = make_brain(self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    # ── 1. запись обязана НЕСТИ связь ───────────────────────────────
    def test_1_evidence_carries_head_relation_tail(self):
        """Без head/relation/tail наблюдение невидимо тому, кто рассуждает."""
        self.brain._record_observation("мяч", {"hierarchy": ["игрушка"]},
                                       confidence=0.4, evidence="мяч игрушка",
                                       source="llm")
        obs = read_world(self.dir)["observations"]
        self.assertEqual(1, len(obs), "свидетельство не записано")
        rec = list(obs.values())[0]
        for field in ("head", "relation", "tail"):
            self.assertIn(field, rec,
                          f"свидетельство любопытства не несёт {field} — "
                          f"рассуждение его не увидит (замер: knows=False)")
        self.assertEqual("мяч", rec["head"])
        self.assertEqual("игрушка", rec["tail"])

    def test_2_reasoning_sees_what_curiosity_learned(self):
        """Выученное должно УЧАСТВОВАТЬ в ответе, иначе учиться незачем."""
        self.brain._record_observation("мяч",
                                       {"hierarchy": ["игрушка"],
                                        "actions": ["катиться"]},
                                       confidence=0.9, evidence="видела много",
                                       source="user")
        knower = StaticKnowledge(read_world(self.dir))
        self.assertTrue(knower.knows("мяч"),
                        "узел выучен, но рассуждению о нём ничего не известно")
        self.assertIn("игрушка", knower.related("мяч", "IS_A"))
        self.assertIn("катиться", knower.related("мяч", "CAN_DO"))

    def test_3_two_facts_make_two_links(self):
        """Два разных факта про одно слово — ДВЕ связи, а не один счётчик."""
        self.brain._record_observation("мяч", {"hierarchy": ["игрушка"],
                                              "actions": ["катиться"]},
                                       confidence=0.5, evidence="два факта",
                                       source="llm")
        obs = read_world(self.dir)["observations"]
        self.assertEqual(2, len(obs),
                         "разные факты свалены в одну запись — достоверность "
                         "одного будет расти от другого (замер: support=2)")
        for rec in obs.values():
            self.assertEqual(1, int(rec.get("support", 0)),
                             "ни то, ни другое не подтверждено дважды")

    # ── 2. достоверность растёт СВОИМ повтором ──────────────────────
    def test_4_second_named_source_raises_support(self):
        """Подтверждение видно числом — но засчитывается НАЗВАННОЕ.

        ПЕРЕПИСАНО 08.11 вместе с законом lineage
        (tests/test_lineage_contract.py). Смысл теста НЕ изменён: связь
        обязана становиться убедительнее от второго опыта, и это обязано
        быть видно по support. Изменилось основание зачёта. Прежняя версия
        подавала тот же текст вторым каналом без документа и получала
        support=2 — ровно та дыра, которую Жозефина предсказала, а замер
        подтвердил: пересказ отличается от оригинала ВСЕГДА, поэтому
        «другой канал/другая фраза» — самый дешёвый способ надуть
        достоверность.

        Первая ветка (замеренная) остаётся запертой: та же статья,
        названная другим каналом, НЕ подтверждение. Вторая — настоящее
        независимое основание.
        """
        self.brain._record_observation("мяч", {"hierarchy": ["игрушка"],
                                              "title": "Мяч"},
                                       confidence=0.4, evidence="первый раз",
                                       source="wiki")
        self.brain._record_observation("мяч", {"hierarchy": ["игрушка"],
                                              "title": "Мяч"},
                                       confidence=0.4, evidence="второй раз",
                                       source="llm")
        obs = read_world(self.dir)["observations"]
        self.assertEqual(1, len(obs), "повтор того же факта раздвоился")
        rec = list(obs.values())[0]
        self.assertEqual(1, int(rec["support"]),
                         "одна и та же статья, названная двумя каналами, "
                         "согрела связь дважды")
        self.brain._record_observation("мяч", {"hierarchy": ["игрушка"],
                                              "title": "Игрушки двора"},
                                       confidence=0.4, evidence="третья фраза",
                                       source="wiki")
        rec = list(read_world(self.dir)["observations"].values())[0]
        self.assertEqual(2, int(rec["support"]),
                         "повторённый факт не стал убедительнее")

    def test_5_foreign_fact_does_not_warm_this_one(self):
        """Чужое свидетельство про то же слово НЕ греет эту связь."""
        self.brain._record_observation("мяч", {"hierarchy": ["игрушка"]},
                                       confidence=0.4, evidence="факт А",
                                       source="llm")
        self.brain._record_observation("мяч", {"actions": ["катиться"]},
                                       confidence=0.9, evidence="факт Б",
                                       source="wiki")
        obs = read_world(self.dir)["observations"]
        hier = [r for r in obs.values() if r.get("relation") == "IS_A"]
        self.assertEqual(1, len(hier))
        self.assertEqual(1, int(hier[0]["support"]),
                         "достоверность «мяч — игрушка» выросла оттого, что "
                         "кто-то видел, как мяч катится")
        self.assertAlmostEqual(
            0.4, float(hier[0]["confidence"]), places=6,
            msg="чужая уверенность перетекла в эту связь")

    # ── 3. вес лежит В записи и различает гипотезу и знание ─────────
    def test_6_weight_follows_ground_not_self_report(self):
        """Вес = СИЛА ОСНОВАНИЯ, а не то, насколько уверенно канал высказался.

        ЗАКОН ИЗМЕНЁН ОТКРЫТО (шаг 2.2/2.3 босса, 08.11 ночь). Прежняя
        редакция этого теста запирала fact_weight(conf, ...), то есть
        90 + 130*самоотчёт. ЗАМЕР, из-за которого формула снята
        (experiments/measure_weight_authority.py):
          * пять каналов с ОДНИМ основанием давали 161..220 — разница 59
            точек там, где основание ОДНО;
          * модель без всякого основания с самоотчётом 1.00 (220) перекрыла
            ДВЕ названные книги (171);
          * живой мир: три связи с НУЛЁМ названных оснований весили
            120..210 из 255.
        Проверка была НЕ лишней, а неверной: она сверяла запись с формулой,
        но не спрашивала, ОТ ЧЕГО формула. Спрошено здесь.
        """
        from core.records import (GROUND_TESTIMONY, ground_of,
                                  weight_from_ground)
        self.brain._record_observation("мяч", {"hierarchy": ["игрушка"]},
                                       confidence=0.5, evidence="один раз",
                                       source="llm")
        once = list(read_world(self.dir)["observations"].values())[0]
        self.assertIn("weight", once,
                      "в записи нет числа достоверности (замер: weight=None) — "
                      "отличить гипотезу от знания нечем")
        self.assertEqual("ground-v2", once.get("weight_law"),
                         "число лежит в данных без имени закона: читатель не "
                         "может отличить вес от самоотчёта и обязан его "
                         "пересчитать")
        w_once = observation_weight(once)
        # СВЕРКА ИДЁТ С ТЕМИ ОСНОВАНИЯМИ, что лежат в записи, а не с 1 «на
        # глазок»: у этой связи основания нет вовсе (канал llm, документ не
        # назван), и вес 20 — честное «назвать нечем». Прежняя сверка с
        # единицей проходила только потому, что support завышал число.
        from core.records import independent_bases
        self.assertEqual(
            weight_from_ground(ground_of("llm"), independent_bases(once),
                               False), w_once,
            "вес в записи выводится не из силы основания")
        self.assertLess(w_once, 150,
                        "одинокий чужой пересказ перешагнул порог диагноза "
                        "«онтология без опыта»: диагноз начал бы срабатывать "
                        "на громкости, а не на пустом опыте")
        self.assertEqual(GROUND_TESTIMONY, once.get("ground"))

        self.brain._record_observation("мяч", {"hierarchy": ["игрушка"]},
                                       confidence=0.5, evidence="ещё раз",
                                       source="user")
        thrice = list(read_world(self.dir)["observations"].values())[0]
        w_owner = observation_weight(thrice)
        self.assertGreater(w_owner, w_once,
                           "убеждённость и единичное наблюдение дают ОДНО "
                           "число: фазы «испугался» и «знаю» неразличимы")
        self.assertEqual(weight_from_ground(ground_of("user"), 1, False),
                         w_owner,
                         "слово Хозяина не получило своей силы основания")
        self.assertLess(w_once, 255,
                        "первое же свидетельство уже на пике шкалы")


if __name__ == "__main__":
    unittest.main()
