"""
КОНТРАКТ П1a: УЧИТЕЛЬ НЕ ИМЕЕТ ПРАВА ПИСАТЬ В ЯДРО РЕБРО БЕЗ ИМЕНИ КАНАЛА.

Почему это дыра, а не косметика. В мире рассказа и в мире, где ответы
придумала локальная модель, ребро графа выглядит ОДИНАКОВО:
    невестка --IS_A--> сущность   (weight=200)
Отличить «я прочитала» от «мне сказала модель» по графу нельзя. А решение
«можно ли на знание опираться» принимается именно по этому признаку:
придуманное определение правдоподобно ровно настолько, чтобы его не
заподозрили. Ядро — единственный слой, который живёт между прогонами, и если
оно не помнит канал, знание становится безымянным навсегда.

ЗАМЕРЕНО ДО ПРАВКИ (experiments/P1_GRAPH_SOURCE.txt):

1) РЕБРО УМЕЕТ НОСИТЬ ИСТОЧНИК. Node.connect_to(..., source="") пишет
   out_synapses_typed[цель]["source"] (core/graph.py:94-103), и сериализатор
   его сохраняет (core/storage.py, блок «ПРОИСХОЖДНИЕ ЗНАНИЯ»). То есть поле
   есть, дорога до диска есть.

2) integrate_knowledge ЭТОТ ПОЛЕ НЕ ПЕРЕДАЁТ. Сигнатура
   integrate_knowledge(knowledge: dict, weight=255); внутри три вызова
   connect_to БЕЗ аргумента source. Источник теряется здесь.

3) ВЫЗОВОВ ОДИН, не пять (grep по дереву без тестов): brain/learner.py:1253.
   Поэтому правка — не миграция. Первый аргумент свободный словарь, так что
   СИГНАТУРУ МЕНЯТЬ НЕ НУЖНО: источник приходит тем же словарём по ключу
   "source". Это важно для запрета босса ломать вызывающих.

4) УЗЕЛ ТОЖЕ МОЛЧИТ. get_or_create_node принимает source и evidence, но
   integrate_knowledge их не передаёт — узел, созданный учителем, записывается
   с source="".

Граница файла: ДОСТАВКА источника от учителя до ребра и узла ядра. Никаких
решений о том, какое отношение выбрать для родства (это writer-дыра, см.
experiments/F2_WRITER.txt), и никаких изменений скаляра learned_from (это П1b,
решение босса не получено).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph import GraphMemory            # noqa: E402
from core.storage import RelationType         # noqa: E402


class TestTeacherSourceReachesGraph(unittest.TestCase):
    """Источник обязан дойти от словаря учителя до ребра ядра."""

    def setUp(self):
        self.g = GraphMemory()

    def _synapses(self, node_name):
        node = self.g.nodes.get(node_name)
        self.assertIsNotNone(node, f"узел {node_name} не создан")
        return node.out_synapses_typed

    def test_category_edge_carries_source(self):
        """Ребро IS_A от учителя помнит канал."""
        self.g.integrate_knowledge({"concept": "невестка",
                                    "category": "родственник",
                                    "source": "teacher:llm"})
        syn = self._synapses("невестка").get("родственник")
        self.assertIsNotNone(syn, "ребро IS_A не создано")
        self.assertEqual(syn.get("source"), "teacher:llm",
                         "ребро IS_A записано без имени канала — знание "
                         "учителя неотличимо от знания из рассказа")

    def test_property_edge_carries_source(self):
        self.g.integrate_knowledge({"concept": "невестка",
                                    "properties": ["молодая"],
                                    "source": "teacher:llm"})
        syn = self._synapses("невестка").get("молодая")
        self.assertIsNotNone(syn, "ребро HAS_PROPERTY не создано")
        self.assertEqual(syn.get("source"), "teacher:llm")

    def test_action_edge_carries_source(self):
        self.g.integrate_knowledge({"concept": "невестка",
                                    "actions": ["гостить"],
                                    "source": "teacher:llm"})
        syn = self._synapses("невестка").get("гостить")
        self.assertIsNotNone(syn, "ребро CAN_DO не создано")
        self.assertEqual(syn.get("source"), "teacher:llm")

    def test_created_node_carries_source(self):
        """Сам узел, созданный учителем, тоже помнит канал."""
        self.g.integrate_knowledge({"concept": "сноха",
                                    "category": "родственник",
                                    "source": "teacher:wiki"})
        self.assertEqual(self.g.nodes["сноха"].source, "teacher:wiki",
                         "узел создан без source — get_or_create_node умеет "
                         "его принимать, integrate_knowledge не передаёт")

    def test_no_source_stays_silent_not_fake(self):
        """БЕЗ источника поле остаётся пустым: молчание честнее выдумки.

        Если бы при отсутствии source писалось «неизвестно» или «text», прибор
        начал бы приписывать канал. Пустая строка читается как «не поименован»,
        и именно так её трактует сериализатор.
        """
        self.g.integrate_knowledge({"concept": "флюгер",
                                    "category": "прибор"})
        syn = self._synapses("флюгер").get("прибор")
        self.assertIsNotNone(syn)
        self.assertNotIn("source", syn,
                         "источник выдуман там, где его не передавали")
        self.assertEqual(self.g.nodes["флюгер"].source, "")


class TestSignatureIntegrity(unittest.TestCase):
    """Правка не имеет права сломать вызывающих (запрет босса)."""

    def test_old_style_call_still_works(self):
        """Вызов БЕЗ ключа source обязан работать ровно как раньше.

        Единственный внешний вызователь — brain/learner.py:1253. Если правка
        сделает ключ обязательным, она сломает его. Поэтому проверяю старый
        стиль вызова напрямую, а не только новый.
        """
        g = GraphMemory()
        g.integrate_knowledge({"concept": "кошка", "category": "животное",
                               "properties": ["пушистая"]}, weight=200)
        syn = g.nodes["кошка"].out_synapses_typed
        self.assertIn("животное", syn)
        self.assertEqual(syn["животное"]["rel_type"], RelationType.IS_A)
        self.assertEqual(syn["животное"]["weight"], 200)

    def test_positional_weight_argument_kept(self):
        """weight передаётся ПОЗИЦИОННО (learner вызывает weight=200 именовано,
        но core/graph.py::-selftest позиционно). Порядок аргументов менять
        нельзя."""
        g = GraphMemory()
        g.integrate_knowledge({"concept": "стол", "category": "мебель"}, 130)
        self.assertEqual(g.nodes["стол"].out_synapses_typed["мебель"]["weight"],
                         130)


class TestLearnerPassesSource(unittest.TestCase):
    """Цепочка учитель -> learner -> ядро замкнута, а не оборвана на середине.

    Замкнуть только ядро — значит получить правильно устроенный приёмник без
    передатчика: integrate_knowledge примет source, а learner так и не пришлёт.
    Тогда граф по-прежнему безымянный, но тесты ядра зелёные. Поэтому тест
    идёт сквозь learner.
    """

    def test_learner_sends_source_to_core(self):
        from brain.learner import Learner, LearningReport

        sent = []

        class CoreSpy:
            def integrate_knowledge(self, knowledge, weight=255):
                sent.append(dict(knowledge))

        lr = Learner(load_world=lambda: {"entities": {}, "observations": {},
                                         "unknown_words": {}},
                     save_world=lambda w: None,
                     core=CoreSpy())
        entities: dict = {}
        lr._store_entity(entities, "невестка",
                         {"definition": "жена сына"},
                         source="llm", evidence="урок от модели",
                         rep=LearningReport())
        self.assertEqual(len(sent), 1, "ядро не вызвано")
        self.assertIn("source", sent[0],
                      "learner не передаёт источник в ядро: поле в графе есть, "
                      "данные есть, связь между ними не замкнута")
        self.assertIn("llm", str(sent[0]["source"]),
                      f"имя канала испорчено: {sent[0]['source']!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
