"""
Интент ACTION («что делает X») обязан ОТВЕЧАТЬ, а не извиняться.

ЗАМЕРЕНО в живом окне: после выученного урока «Вода гасит пламя.» на вопрос
«Что делает вода?» мозг отвечал
    «Интент 'ACTION' распознан, но обработчик для него ещё в разработке»
— при том что знание LEЖАЛО в файле вторым ключом наблюдений
    «вода|CAN_DO|гасить»
и второго пласта кода для ответа не требовалось: действие — это хвост той же
связи. Заглушка на месте обработчика означала, что система молчит именно там,
где она единственна: она умеет связать, но не умеет сказать.

Учитель — заглушка, llama.cpp не трогается.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORLD = {
    "entities": {
        "вода": {"concept": "вода", "hierarchy": ["жидкости"], "properties": []},
        "гасить": {"concept": "гасить", "kind": "action",
                   "hierarchy": ["действие"], "action_profile": [20, 50, 10],
                   "participants": {"agent": ["вода"], "object": ["пламя"]}},
        "камень": {"concept": "камень", "hierarchy": ["минералы"], "properties": []},
    },
    "observations": {
        "вода|CAN_DO|гасить": {
            "head": "вода", "relation": "CAN_DO", "tail": "гасить",
            "confidence": 0.85, "negated": False, "evidence": "Вода гасит пламя",
            "source": "урок", "support": 1},
        # Отрицание НЕ обязано выдаваться за умение.
        "камень|CAN_DO|летать": {
            "head": "камень", "relation": "CAN_DO", "tail": "летать",
            "confidence": 0.5, "negated": True, "evidence": "камень не летает",
            "source": "урок", "support": 1},
    },
    "unknown_words": {},
}


class StubTeacher:
    calls = 0

    def run_agent(self, **kw):
        type(self).calls += 1
        return None

    def fetch_concept(self, concept):
        type(self).calls += 1
        return None


class TestActionIntent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from brain.lipid2_brain import Lipid2Brain
        cls.tmp = tempfile.mkdtemp(prefix="l2action-")
        with open(os.path.join(cls.tmp, "lipid_world.json"), "w",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(WORLD, ensure_ascii=False))
        cls.brain = Lipid2Brain(memory_dir=cls.tmp)
        cls.stub = StubTeacher()
        cls.brain.teacher_llm = cls.stub
        cls.brain.teacher_wiki = cls.stub
        cls.brain.ask_user_enabled = False

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_action_intent_is_not_a_stub_anymore(self):
        ans = self.brain.process_input("Что делает вода?")
        self.assertNotIn("в разработке", ans,
                         f"ACTION остался без обработчика: {ans}")
        self.assertIn("гасить", ans, f"умение не названо: {ans}")

    def test_action_answer_comes_from_observations_only(self):
        """Ответ строится по факту связи; LLM по пустякам дёргать нельзя."""
        before = StubTeacher.calls
        self.brain.process_input("Что умеет вода?")
        self.assertEqual(StubTeacher.calls, before,
                         "обработчик ACTION пошёл к Учителю вместо своей памяти")

    def test_negated_ability_is_not_announced_as_ability(self):
        ans = self.brain.process_input("Что делает камень?")
        self.assertNotIn("летать", ans,
                         f"отрицание выдано за умение: {ans}")

    def test_reference_numbers_are_untouched_by_the_naming_dispute(self):
        """
        Тройки справочника обязаны быть НЕПРИКОСНОВЕННЫМИ, whatever подписи.

        Произошло то, что обязан фиксироваться тестом, а не обсуждением: в
        проекте две расшифровки одних и тех же трёх байт. Таблица была
        написана рядом с одной подписью, пользователь постановил ДРУГУЮ —
        [Кинетика, Стоимость сигнала, Метаболизм]. Ни одна тройка при этом не
        переставлена и не исправлена «под смысл»: числа — договорённость, а
        не моё впечатление о том, сколько стоит сон.

        Проверяется именно это: побайтовое совпадение с эталоном. Если кто-то
        (в том числе я) решит «починить числа под правильную расшифровку», тест
        встанет и потребует замера, а не догадки.
        """
        from core.actions import ACTION_PROFILES, profile_of
        reference = {"гасить": [20, 50, 10], "летать": [235, 70, 5],
                  "спать": [0, 200, 0], "мяукать": [10, 60, 255],
                  "бегать": [220, 60, 10]}
        for verb, prof in reference.items():
            with self.subTest(verb=verb):
                self.assertEqual(profile_of(verb), prof,
                                 "число справочника изменено — это договор")
        self.assertEqual(len(ACTION_PROFILES), 25,
                         "состав справочника изменился без явного решения")


if __name__ == "__main__":
    unittest.main(verbosity=2)
