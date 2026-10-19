"""
КОНТРАКТ: пустой узел НЕ есть знание, а отказ обязан называть известное.

Откуда файл. Сквозной GUI-прогон (tools/e2e_gui.py) печатал «выучил и
вспомнил», когда НЕ БЫЛО ВЫУЧЕНО НИЧЕГО:

  1. «Пчела опыляет цветок на лугу» создаёт узел «пчела» ИЗ НАБЛЮДЕНИЯ
     (пчела|CAN_DO|опылять). find_in_memory на такой узел возвращает запись,
     и _handle_definition считал вопрос закрытым: любопытство (три канала)
     не вызывалось НИ РАЗУ — ЗАМЕРЕНО: вопросов пользователю 0, в файле одна
     сущность («опылять»), «пчела/цветок/луга» остались not_asked.
  2. Ответ на этом пустом узле был «О 'пчела' в памяти есть только само
     слово — знания нет», то есть врал в обе стороны: знания про действие
     действительно нет, но «знания нет» отрицает и записанное действие.

Проверяется здесь:
  * на пустом узле любопытство ВЫЗЫВАЕТСЯ (а не молча возвращается пустота);
  * если любопытство ничего не дало — отказ называет известное из опыта
    действие словом, а не утверждает «знания нет»;
  * пустое определение («только само слово») не считается ответом.

Мутации, которые обязан поймать этот файл:
  * _handle_definition без обращения к curiosity при пустом узле;
  * _definition_hollow без перечисления действий (возврат старой фразы).
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class TestEmptyNodeIsNotKnowledge(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="l2def_")
        os.environ["LIPID2_MEMORY_DIR"] = self.dir
        os.environ["LIPID2_STATUS"] = "0"

    def tearDown(self):
        os.environ.pop("LIPID2_MEMORY_DIR", None)
        shutil.rmtree(self.dir, ignore_errors=True)

    def _brain(self):
        from brain.lipid2_brain import Lipid2Brain
        b = Lipid2Brain(memory_dir=self.dir)
        # Все каналы выключены: тест про ПУСК любопытства, а не про поиск.
        b.use_llm_teacher = b.use_wiki_teacher = b.ask_user_enabled = False
        return b

    def _world(self):
        path = os.path.join(self.dir, "lipid_world.json")
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    # ── 1. любопытство обязано просыпаться на пустом узле ──
    def test_curiosity_fires_on_observation_only_node(self):
        """Узел из наблюдения (слово + действие) НЕ закрывает вопрос «что такое X».

        ЗАМЕРЕНО до правки: questions=0, сущностей в файле 1 («опылять»),
        «пчела» в unknown_words со статусом not_asked — то есть система
        отказала, не спросив НИ ОДНОГО канала.
        """
        b = self._brain()
        b.learn_from_text("Пчела опыляет цветок на лугу.", source="t",
                          allow_teacher=False)
        # Узел есть — но пустой: ни определения, ни категорий, ни свойств.
        hit = b.find_in_memory("пчела")
        self.assertIsNotNone(hit, "узел из наблюдения обязан находиться")
        ent = hit.get("base_entity") or {}
        self.assertFalse(ent.get("definition"))
        self.assertFalse(ent.get("hierarchy"))
        self.assertFalse(ent.get("properties"))

        calls = []
        b._trigger_curiosity_and_learn = (
            lambda c: calls.append(c) or None)
        answer = b._handle_definition("пчела")
        self.assertEqual(calls, ["пчела"],
                         "на пустом узле любопытство не вызвано — система "
                         "отказала, не спросив ни одного канала")
        self.assertIsInstance(answer, str)

    def test_curiosity_result_is_used_as_answer(self):
        """Если канал ДАЛ знание, отвечать обязано им, а не прежней пустотой."""
        b = self._brain()
        b.learn_from_text("Пчела опыляет цветок на лугу.", source="t",
                          allow_teacher=False)
        b._trigger_curiosity_and_learn = lambda c: {
            "matched_concept": "пчела",
            "base_entity": {"concept": "пчела", "kind": "entity",
                            "hierarchy": ["насекомое"], "properties": [],
                            "definition": "", "actions": ["опылять"]},
            "derived_entity": {},
        }
        answer = b._handle_definition("пчела")
        self.assertIn("насекомое", answer,
                      f"знание канала проигнорировано: {answer!r}")

    # ── 2. отказ обязан быть честным про известное из опыта ──
    def test_refusal_names_action_known_from_experience(self):
        """«Знания нет» при известном действии — вранье. Действие обязано называться."""
        b = self._brain()
        b.learn_from_text("Пчела опыляет цветок на лугу.", source="t",
                          allow_teacher=False)
        b._trigger_curiosity_and_learn = lambda c: None
        answer = b._handle_definition("пчела")
        self.assertIn("опылять", answer,
                      f"опытное действие скрыто отказом: {answer!r}")
        self.assertIn("не знаю", answer.lower(),
                      "отказ не назван словом «не знаю»")
        self.assertNotIn("в памяти есть только само слово", answer,
                         "отказ отрицает действие, которое в памяти есть")

    def test_refusal_without_any_data_still_says_nothing_found(self):
        """Про совершенную пустоту — прежняя честная формулировка (без выдумок)."""
        b = self._brain()
        b._trigger_curiosity_and_learn = lambda c: None
        answer = b._handle_definition("нюхор")
        self.assertIsInstance(answer, str)
        self.assertTrue(("искал" in answer) or ("не дали" in answer),
                        f"отказ по пустому слову не назван поиском: {answer!r}")

    # ── 3. пустое определение не считается ответом ──
    def test_hollow_definition_does_not_short_circuit_search(self):
        """
        _definition_answer обязан вернуть None, когда отвечать нечем: по этому
        None вызывающий отличает «знать нечего» (звать curiosity) от
        «знать есть чего» (отвечать). Молчаливая строчка-заглушка здесь и
        была источником лжи.
        """
        b = self._brain()
        hollow = {"matched_concept": "зюзя",
                  "base_entity": {"concept": "зюзя", "kind": "entity",
                                  "hierarchy": [], "properties": [],
                                  "definition": "", "actions": []},
                  "derived_entity": {}}
        self.assertIsNone(b._definition_answer("зюзя", hollow),
                          "пустая запись выдала себя за ответ")
        filled = json.loads(json.dumps(hollow))
        filled["base_entity"]["hierarchy"] = ["сосуд"]
        self.assertIsNotNone(b._definition_answer("зюзя", filled))


class TestE2EAssertionIsNotSubstring(unittest.TestCase):
    """
    Проверка «выучил» по подстроке ответа провалилась потому, что ОТКАЗ
    содержит спрашиваемое слово. Здесь закрепляется сам принцип: отказ и
    знание различаются признаками отказа, а не наличием слова в тексте.
    """
    REFUSAL = ("[Lipid-2]: О 'пчела' в памяти есть только само слово — "
               "знания нет. Спроси ещё раз с включённым Учителем "
               "или подскажи сам.")
    KNOWLEDGE = "[Lipid-2]: Пчела — летаемое насекомое; опыляет цветы."
    MARKERS = ("знания нет", "не знаю", "спроси ещё раз",
               "источники знаний не дали")

    def test_refusal_contains_the_asked_word(self):
        """Почему подстрока не работает: в отказе спрашиваемое слово ЕСТЬ."""
        self.assertIn("пчел", self.REFUSAL.lower())

    def test_refusal_detected_in_refusal_only(self):
        self.assertTrue(any(m in self.REFUSAL.lower() for m in self.MARKERS))
        self.assertFalse(any(m in self.KNOWLEDGE.lower() for m in self.MARKERS),
                         "признак отказа сработал на осмысленном ответе — "
                         "проверка снова станет дырявой")


if __name__ == "__main__":
    unittest.main()
