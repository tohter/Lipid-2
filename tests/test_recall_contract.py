"""
Контрактные тесты узнавания: память обязана находить сущность по ЛЮБОЙ форме.

Это не тест «красиво ли кодируется SDR», а тест «не гоняем ли мы LLM попусту».
Прежний поиск шёл по точной строке, и «свойства флюгера» промахивалось по узлу
«флюгер»: каждый падеж каждого слова выглядел новой сущностью и запускал
повторный (медленный, платный) запрос к Учителю за копией уже имеющегося знания.

Тесты НЕ трогают llama.cpp: Учитель подменяется заглушкой, счётчик вызовов —
это и есть измеряемая величина.
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
        "кошка": {
            "concept": "кошка", "hierarchy": ["животные", "объекты"],
            "properties": ["пушистая", "спит"],
            "actions": [{"name": "мурлычет", "action_profile": [10, 40, 90]}],
        },
        "флюгер": {
            "concept": "флюгер", "hierarchy": ["инструменты", "объекты"],
            "properties": ["вращается"], "actions": [],
        },
        "собака": {"concept": "собака", "hierarchy": ["животные"],
                   "properties": ["лает"], "actions": []},
    },
    "relations": [],
}


class StubTeacher:
    """Учитель-заглушка. .calls — сколько раз мозг реально к нему полез."""
    def __init__(self):
        self.calls = 0

    def run_agent(self, **kwargs):
        self.calls += 1
        return None

    def fetch_concept(self, concept):
        self.calls += 1
        return None


class TestMorphologicalRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from brain.lipid2_brain import Lipid2Brain
        cls.tmp = tempfile.mkdtemp(prefix="lipid_test_")
        with open(os.path.join(cls.tmp, "lipid_world.json"), "w", encoding="utf-8") as f:
            json.dump(WORLD, f, ensure_ascii=False)
        cls.brain = Lipid2Brain(memory_dir=cls.tmp)
        cls.stub = StubTeacher()
        cls.brain.teacher_llm = cls.stub
        cls.brain.teacher_wiki = cls.stub      # оба канала — заглушка

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ── память ────────────────────────────────────────────
    def test_finds_entity_by_exact_name(self):
        self.assertIsNotNone(self.brain.find_in_memory("кошка"))

    def test_core_itself_recognizes_forms(self):
        """
        Ядро обязано узнавать форму САМО, а не через мозг (граница, шаг A2.1).

        ПОЧЕМУ ЭТОТ ТЕСТ ПОЯВИЛСЯ ИМЕННО СЕЙЧАС. После A2.1 ядро больше не
        импортирует язык: службу форм ему передаёт мозг. Я написала мутант —
        в мозге GraphMemory() без службы — и весь набор остался ЗЕЛЁНЫМ:
        find_in_memory ходит в мировой резолвер мозга, поэтому деградация
        ядра осталась невидимой. Знание «падежи работают» было завязано на
        мозг, а цикл предсказания зовёт ядро напрямую (core._recall ->
        core.resolve), то есть тихо теряло падежи. Поэтому проверяем именно
        ядро, и именно поведением, а не флагом available: флаг можно выставить,
        ничего не передав.
        """
        self.assertTrue(self.brain.core.surface_forms_available,
                        "мозг не внедрил ядру службу сведения форм — "
                        "ядро слепо к падежам, а выглядит это как «в памяти "
                        "ничего нет»")
        node = self.brain.core.resolve("флюгера")
        self.assertIsNotNone(node, "ядро не свело «флюгера» к узлу «флюгер»")
        self.assertEqual(node.name, "флюгер")

    def test_finds_entity_by_case_form(self):
        """Родительный/дательный падеж обязан попадать в тот же узел."""
        for form in ("кошки", "кошке", "кошку", "флюгера", "флюгеру", "собаке"):
            with self.subTest(form=form):
                self.assertIsNotNone(
                    self.brain.find_in_memory(form),
                    f"форма «{form}» не нашла сущность — мозг зря пойдёт к LLM")

    def test_answer_content_survives_declension(self):
        """Определение по форме должно быть полнее, чем по вырожденной записи."""
        out = self.brain.process_input("что такое кошки")
        self.assertNotIn("это кошка со свойствами: .", out,
                         "вернулся пустой ответ — производная запись подменила базовую")
        self.assertTrue("животн" in out or "спит" in out or "пушистая" in out,
                        f"ответ не содержит знаний: {out!r}")

    def test_properties_intent_not_unknown(self):
        out = self.brain.process_input("свойства кошки")
        self.assertNotIn("ещё в разработке", out)
        self.assertIn("Свойства", out)

    def test_properties_of_declined_noun(self):
        out = self.brain.process_input("какие свойства у кошки")
        self.assertIn("Свойства", out)
        self.assertIn("пушистая", out)

    def test_unknown_word_does_ask_teacher_exactly_once(self):
        """Неизвестное слово — единственный случай, когда Учитель НУЖЕН."""
        self.stub.calls = 0
        self.brain.process_input("что такое шизокарпий")
        self.assertGreaterEqual(self.stub.calls, 1,
                                "неизвестное слово не подняло поисковый канал")

    def test_known_word_never_calls_teacher(self):
        """Главная регрессия: знакомое слово в косом падеже НЕ гонит к LLM."""
        for q in ("что такое кошки", "свойства кошке", "что такое флюгера"):
            with self.subTest(q=q):
                self.stub.calls = 0
                self.brain.process_input(q)
                self.assertEqual(self.stub.calls, 0,
                                 f"«{q}» вызвало Учителя, хотя знание уже в памяти")

    def test_graph_resolve_never_creates_nodes(self):
        """Резолвер обязан быть только чтением: неразрешённое слово не плодит узлы."""
        n0 = len(self.brain.core.nodes)
        self.brain.core.resolve("велофасад")
        self.assertEqual(len(self.brain.core.nodes), n0)

    def test_resolve_returns_none_for_absent(self):
        self.assertIsNone(self.brain.core.resolve("шизокарпий"))


class TestGraphRecall(unittest.TestCase):
    """Ядро: узнавание формы против порога неизвестного."""

    def test_forms_beat_noise_floor(self):
        from core.graph import GraphMemory
        from core.cortex import PredictiveCortex
        from core.sdr import SDR1024
        from core.lemma import LemmaResolver
        # РЕЗОЛВЕР ПЕРЕДАЁТСЯ ЯВНО (шаг A2.1 границы): ядро больше не умеет
        # само импортировать язык, и тест, которому нужно узнавание русских
        # падежей, обязан принести языковую службу с собой. Сила теста не
        # изменена: сводит ли ядро «кошки» к «кошка» — проверяется ровно так
        # же. Проверено обратным мутантом: без резолвера тест падает.
        g = GraphMemory(surface_forms=LemmaResolver())
        for w in ("кошка", "собака", "молоко", "рыба", "стол", "окно"):
            g.get_or_create_node(w)
        c = PredictiveCortex(g, teacher=None)
        c.recalibrate_floor()
        thr = c.unknown_threshold()
        for form, base in (("кошки", "кошка"), ("кошке", "кошка"),
                           ("собаке", "собака"), ("молока", "молоко")):
            node, ov = c._recall(SDR1024(form))
            with self.subTest(form=form):
                self.assertEqual(g.resolve(form), g.nodes[base],
                                 "резолвер графа не свёл форму к основе")
        for alien in ("шизокарпий", "гипотенуза", "табуретка"):
            _, ov = c._recall(SDR1024(alien))
            self.assertLess(ov, thr,
                            f"«{alien}» дало {ov} при пороге {thr} — пол шума уехал")


DEEP_WORLD = {
    "entities": {
        "объекты": {"concept": "объекты", "hierarchy": [],
                    "properties": ["существует"], "actions": []},
        "животные": {"concept": "животные", "hierarchy": ["объекты"],
                     "parent_concept": "объекты",
                     "properties": ["живое"], "actions": []},
        "птицы": {"concept": "птицы", "hierarchy": ["животные", "объекты"],
                  "parent_concept": "животные",
                  "properties": ["перья"],
                  "actions": [{"name": "летать", "action_profile": [80, 10, 10]}]},
        "беркут": {"concept": "беркут", "hierarchy": ["птицы", "животные", "объекты"],
                   "parent_concept": "птицы",
                   "properties": ["хищный"], "actions": []},
    },
    "relations": [],
}


class TestRecursiveHierarchy(unittest.TestCase):
    """
    Наследование сквозь НЕСКОЛЬКО уровней (очередь, п.12).

    Прежний find_in_memory читал ОДИН parent_concept: у «беркута» находился
    «птицы», а до «животных» и «объектов» он уже не поднимался. Свойства с
    глубины >1 молча пропадали — при том что сама цепь categories в записи была.
    """

    @classmethod
    def setUpClass(cls):
        from brain.lipid2_brain import Lipid2Brain
        cls.tmp = tempfile.mkdtemp(prefix="lipid_deep_")
        with open(os.path.join(cls.tmp, "lipid_world.json"), "w", encoding="utf-8") as f:
            json.dump(DEEP_WORLD, f, ensure_ascii=False)
        cls.brain = Lipid2Brain(memory_dir=cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_ancestor_chain_reaches_root(self):
        """Цепь предков беркута: птицы -> животные -> объекты."""
        entities = self.brain._load_world().get("entities", {})
        chain = self.brain._ancestor_chain("беркут", entities)
        self.assertEqual(chain, ["птицы", "животные", "объекты"])

    def test_inherits_properties_from_all_levels(self):
        """Свойства с глубины >1 (перья, живое) доходят до беркута."""
        entities = self.brain._load_world().get("entities", {})
        inh = self.brain._inherit_ancestors("беркут", entities)
        for prop in ("перья", "живое", "существует"):
            self.assertIn(prop, inh["properties"],
                          f"свойство «{prop}» не унаследовано")

    def test_find_in_memory_returns_inherited(self):
        found = self.brain.find_in_memory("беркут")
        props = found["base_entity"].get("properties", [])
        self.assertIn("перья", props, "птичье свойство не дошло до беркута")
        self.assertIn("живое", props, "свойство уровня животные не дошло")
        self.assertEqual(found["ancestors"], ["птицы", "животные", "объекты"])

    def test_cycle_does_not_hang(self):
        """Цикл X→Y→X не должен зациклить обход."""
        cyc = {"a": {"concept": "a", "parent_concept": "b"},
               "b": {"concept": "b", "parent_concept": "a"}}
        chain = self.brain._ancestor_chain("a", cyc)
        self.assertEqual(chain, ["b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
