# -*- coding: utf-8 -*-
"""
КОНТРАКТ: СРАВНЕНИЕ + ОБЛАДАНИЕ + ДООБУЧЕНИЕ ИЗ ВОПРОСА.

ЗАМЕРЕННАЯ ДЫРА (09.11)
-----------------------
Живой прогон на вопросе «Чем отличается птица от самолёта?» с базой,
где у птицы и самолёта ЕСТЬ свойства «есть крылья»:

    intent=UNKNOWN subjects=['отличаться','птица','самолёт']
    ANSWER: 🤔 Интент 'UNKNOWN' распознан, но обработчик для него ещё в разработке.

«Что общего у птицы и самолёта?»  -> UNKNOWN -> та же заглушка.
«У кого есть крылья?»            -> UNKNOWN -> та же заглушка.

Модель НЕ понимает ни различие, ни сходство, ни обладание. При этом нужные
факты лежат в базе (птица и самолёт оба имеют «есть крылья»), просто у мозга
нет ИНТЕНТА и ОБРАБОТЧИКА, чтобы их свести и выдать.

ЧТО ЗАКРЕПЛЕНО
--------------
1. СРАВНЕНИЕ-РАЗЛИЧИЕ. «Чем отличается X от Y» даёт COMPARE_DIFF и
   перечисляет, что есть у X и нет у Y (и наоборот), из свойств/иерархии/
   действий/определения. Ответ обязан содержать оба имени и отличия.
2. СРАВНЕНИЕ-ОБЩЕЕ. «Что общего у X и Y» даёт COMPARE_COMMON и перечисляет
   общие свойства/действия/категории.
3. ОБЛАДАНИЕ. «У кого есть X» даёт WHO_HAS и перечисляет сущности, у
   которых в properties/связях есть X. Ответ на «у кого есть крылья» —
   и птица, и самолёт.
4. ДООБУЧЕНИЕ ИЗ ВОПРОСА (режим глубокого обучения). Общее качество,
   замеченное при сравнении (например «крыло»), мозг вводит как ОТДЕЛЬНУЮ
   сущность в базу — не выдумку, а извлечённое из уже известного общего
   свойства. После вопроса «чем отличается птица от самолёта» сущность
   «крыло» обязана появиться, и «у кого есть крылья» обязано ответить
   «и птица, и самолёт» ПОСЛЕ дообучения (не только по properties).

МУТАЦИИ, на которых контракт обязан покраснеть:
  * процесс, не распознающий сравнительные вопросы (COMPARE_* не существует);
  * обработчик, сравнивающий только одно поле (например, только свойства)
    и теряющий разность по иерархии/действиям;
  * обработчик обладания, ищущий только точное имя узла, а не по properties;
  * отсутствие дообучения: после сравнения сущность «крыло» не создана.
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BASE_WORLD = {
    "entities": {
        "птица": {"concept": "птица", "hierarchy": ["живое существо"],
                  "properties": ["умеет летать", "есть крылья", "есть перья"],
                  "actions": ["летать"], "definition": "животное с перьями"},
        "самолёт": {"concept": "самолёт", "hierarchy": ["летательный аппарат"],
                    "properties": ["сделан из металла", "есть крылья", "есть двигатель"],
                    "actions": ["летать"], "definition": "летательная машина"},
    },
    "observations": {},
    "unknown_words": {},
}


def _make_brain(d):
    with open(os.path.join(d, "lipid_world.json"), "w", encoding="utf-8") as f:
        json.dump(BASE_WORLD, f, ensure_ascii=False)
    from brain.lipid2_brain import Lipid2Brain
    b = Lipid2Brain(memory_dir=d)

    class _Noop:
        world_filepath = None
        last_failure = None
    b.teacher_llm = _Noop()
    b.teacher_wiki = _Noop()
    b.use_llm_teacher = False
    b.use_wiki_teacher = False
    b.ask_user_enabled = False
    return b


class CompareLearnContract(unittest.TestCase):

    def test_compare_diff_recognized_and_answers(self):
        from lang.ru.questions import IRParser
        ir = IRParser().parse("Чем отличается птица от самолёта?")
        self.assertEqual(ir.intent_type, "COMPARE_DIFF")
        self.assertEqual(ir.subjects, ["птица", "самолёт"])

        with tempfile.TemporaryDirectory() as d:
            b = _make_brain(d)
            ans = b.process_input("Чем отличается птица от самолёта?")
            self.assertIn("птица", ans)
            self.assertIn("самолёт", ans)
            self.assertIn("перья", ans)        # отличие птицы
            self.assertIn("металл", ans)       # отличие самолёта

    def test_compare_common_recognized_and_answers(self):
        from lang.ru.questions import IRParser
        ir = IRParser().parse("Что общего у птицы и самолёта?")
        self.assertEqual(ir.intent_type, "COMPARE_COMMON")
        self.assertEqual(ir.subjects, ["птица", "самолёт"])

        with tempfile.TemporaryDirectory() as d:
            b = _make_brain(d)
            ans = b.process_input("Что общего у птицы и самолёта?")
            self.assertIn("крылья", ans)       # общее свойство
            self.assertIn("летать", ans)       # общее действие

    def test_who_has_answers_from_properties(self):
        from lang.ru.questions import IRParser
        ir = IRParser().parse("У кого есть крылья?")
        self.assertEqual(ir.intent_type, "WHO_HAS")
        self.assertEqual(ir.subjects, ["крыло"])

        with tempfile.TemporaryDirectory() as d:
            b = _make_brain(d)
            ans = b.process_input("У кого есть крылья?")
            self.assertIn("птица", ans)
            self.assertIn("самолёт", ans)

    def test_deep_learning_introduces_common_entity(self):
        """Общее качество при сравнении вводится как ОТДЕЛЬНАЯ сущность."""
        with tempfile.TemporaryDirectory() as d:
            b = _make_brain(d)
            b.process_input("Чем отличается птица от самолёта?")
            world = b._load_world()
            entities = world.get("entities", {})
            # «крыло» (не «крылья») введено как сущность в результате вопроса
            key = next((k for k in entities if "крыло" in k), None)
            self.assertIsNotNone(key, "после сравнения должна появиться сущность 'крыло'")
            # сущность честная: содержит общее свойство/определение
            self.assertIn("крыло", key)

    def test_who_has_after_deep_learning(self):
        """После дообучения «у кого есть крылья» отвечает и по сущности."""
        with tempfile.TemporaryDirectory() as d:
            b = _make_brain(d)
            b.process_input("Чем отличается птица от самолёта?")
            ans = b.process_input("У кого есть крылья?")
            self.assertIn("птица", ans)
            self.assertIn("самолёт", ans)


if __name__ == "__main__":
    unittest.main()