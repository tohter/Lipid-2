"""
ОПРЕДЕЛЕНИЕ НЕ ТЕРЯЕТСЯ, КОГДА У СУЩНОСТИ ЕСТЬ РОДИТЕЛЬ.

ЗАЧЕМ ЭТОТ ФАЙЛ. Замер 04.11 (владельцу показан в переписке): запись с
готовым определением перестаёт быть ответом, как только у неё появляется
parent_concept:

    def без предка                -> "Бык — самец крупного рогатого скота."
    def + предок + delta          -> "Бык — это скот со свойствами: самец."
    def + предок + delta ПУСТО    -> None -> повторный ПЛАТНЫЙ зов учителя
    def + предок, delta НЕТ       -> None -> то же
    def + предок + hierarchy      -> None -> то же

Причина в brain/lipid2_brain.py: когда цепь предков есть, base_entity =
РОДИТЕЛЬ (find_in_memory строит базу из chain[-1]), а собственная запись
лексемы уезжает в derived_entity. Ответ же читает definition только у
base_entity; производную ветку он смотрит лишь при НЕПУСТОМ delta_properties.
Знание лежит в файле — мозг его не видит и идёт платить LLM за копию
имеющегося. Это ровно та форма записи, которую учитель выдаёт по своему
промпту (derived_entity с parent_concept), то есть дыра бьёт по штатному
каналу, а не по экзотике.

ОСОБЕННАЯ ОПАСНОСТЬ: совпадает с запертым test_definition_curiosity_contract
по симптому («ответа нет -> идём учиться»), но противоположна по причине:
там узел ПРАВДА пустой и любопытство обязано работать; здесь узел ПОЛНЫЙ и
любопытство — растрата. Поэтому в файле есть тест-живость: пустой узел обязан
по-прежнему звать любопытство (мутация «всегда молчи» не пройдёт).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def world(entities: dict) -> dict:
    return {"entities": entities, "observations": {}}


def write_world(dirpath: str, entities: dict) -> None:
    with open(os.path.join(dirpath, "lipid_world.json"), "w",
              encoding="utf-8") as fh:
        json.dump(world(entities), fh, ensure_ascii=False)


class TestDefinitionSurvivesParent(unittest.TestCase):
    """Своё определение первично, предок — это база для НАСЛЕДОВАНИЯ,
    а не подмена ответа."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="l2dpar_")
        os.environ["LIPID2_STATUS"] = "0"

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _brain(self, entities: dict):
        write_world(self.dir, entities)
        from brain.lipid2_brain import Lipid2Brain
        b = Lipid2Brain(memory_dir=self.dir)
        b.use_llm_teacher = False
        b.use_wiki_teacher = False
        b.ask_user_enabled = False
        return b

    def test_definition_survives_with_delta(self):
        """Штатная учителевская форма (def + предок + дельта) обязана
        оставаться ответом — и не только дельтой: определение в ней есть."""
        b = self._brain({
            "бык": {"concept": "бык", "definition": "самец крупного скота",
                    "parent_concept": "скот", "delta_properties": ["самец"]},
            "скот": {"concept": "скот", "hierarchy": ["животные"],
                     "definition": "домашние животные"},
        })
        data = b.find_in_memory("бык")
        self.assertIsNotNone(data)
        out = b._definition_answer("бык", data)
        self.assertIsNotNone(out, "ответ потерялся при штатной записи учителя")

    def test_definition_survives_without_delta(self):
        """ДЕЛЬТА ПУСТА — определение обязано дойти. Это и есть измеренная
        дыра: до правки возвращалось None, и _handle_definition звал учителя
        заново ради знания, которое лежит в файле."""
        for delta in ([], None):
            with self.subTest(delta=delta):
                ent = {"concept": "бык",
                       "definition": "самец крупного рогатого скота",
                       "parent_concept": "скот"}
                if delta is not None:
                    ent["delta_properties"] = delta
                b = self._brain({
                    "бык": ent,
                    "скот": {"concept": "скот", "hierarchy": ["животные"]},
                })
                data = b.find_in_memory("бык")
                out = b._definition_answer("бык", data)
                self.assertIsNotNone(
                    out, "пустая дельта обнулила наличие определения")
                self.assertIn("самец крупного рогатого скота", out,
                              f"своё определение проигнорировано: {out!r}")

    def test_definition_survives_with_hierarchy_only(self):
        """Ни дельты, ни parent, ОСТАЛАСЬ hierarchy-цепь — та же форма
        провала (замер): None. Определение обязано пережить её."""
        b = self._brain({
            "бык": {"concept": "бык",
                    "definition": "самец крупного рогатого скота",
                    "parent_concept": "скот", "hierarchy": ["скот"]},
            "скот": {"concept": "скот", "hierarchy": ["животные"]},
        })
        data = b.find_in_memory("бык")
        out = b._definition_answer("бык", data)
        self.assertIsNotNone(out)
        self.assertIn("самец крупного рогатого скота", out)

    def test_no_teacher_call_for_known_definition(self):
        """СКВОЗНАЯ проверка цены: «что такое бык» на полной записи НЕ
        обязан порождать любопытство. Ловит мутацию «чиним _definition_answer,
        а _handle_definition всё равно зовёт учёбу»."""
        calls = []
        b = self._brain({
            "бык": {"concept": "бык",
                    "definition": "самец крупного рогатого скота",
                    "parent_concept": "скот"},
            "скот": {"concept": "скот", "hierarchy": ["животные"]},
        })
        orig = b._trigger_curiosity_and_learn

        def spy(concept):
            calls.append(concept)
            return orig(concept)

        b._trigger_curiosity_and_learn = spy
        out = b.process_input("что такое бык")
        self.assertEqual(
            calls, [],
            f"знание было в файле, а любопытство вызвано {len(calls)} раз: "
            f"растрата канала знаний. Ответ: {out!r}")
        self.assertIn("самец крупного рогатого скота", out)

    def test_living_empty_node_still_triggers_curiosity(self):
        """ЖИВОСТЬ ЗАМКА: пустой узел (только действие из наблюдения) обязан
        по-прежнему звать любопытство — заперто в
        test_definition_curiosity_contract; здесь проверяем, что починка
        не заклеила симптом ценой симптома."""
        b = self._brain({
            "пчела": {"concept": "пчела", "actions": ["опылять"]},
        })
        data = b.find_in_memory("пчела")
        self.assertIsNone(b._definition_answer("пчела", data),
                          "пустой узел внезапно стал ответом — любопытство "
                          "перестанет учиться на узлах из наблюдений")


if __name__ == "__main__":
    unittest.main()
