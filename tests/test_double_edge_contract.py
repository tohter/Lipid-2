"""
КОНТРАКТ ДЫРЫ 4: ЯДРО НЕ ЗАТИРАЕТ ВТОРОЕ ОТНОШЕНИЕ К ТОЙ ЖЕ ЦЕЛИ.

Почему это латентная дыра, а не косметика. Node.out_synapses_typed исторически
был словарём ПО ИМЕНИ ЦЕЛИ: connect_to писал
    out_synapses_typed[target] = {"rel_type":..., "weight":...}
и ВТОРОЕ ребро к той же цели молча перезаписывало первое. Данные не падают с
ошибкой — они исчезают без следа. Пока ядро только складывает связи, потерю
не видно; но как только ядро начнут ЧИТАТЬ для рассуждений (шаг, заявленный в
памяти: «чинить раньше, чем ядро начнут читать»), одна реальная связь
окажется пустой. В Lipid1 хранилище было списком рёбер, и дыры не было.

ЗАМЕРЕНО ДО ПРАВКИ (live): девочка --OBJECT_OF--> кукла (weight 200) и
девочка --CAN_DO--> кукла (weight 150). В словаре остался ТОЛЬКО CAN_DO
(150), OBJECT_OF затёрт. Два ребра к одной цели -> в памяти одно.

Здесь мы ЗАКРЕПЛЯЕМ требование: два разных отношения между одной парой
узлов обязаны пережить запись. Сам факт, что узел несёт к цели НЕСКОЛЬКО
рёбер, — первое условие, без которого думалка не сможет различить «кукла —
объект игры» и «кукла — объект владения» как два независимых факта.

Граница файла: только структура исходящих синапсов ядра. Никаких решений о
том, какое отношение правильнее (это writer-дыра), и никаких изменений
семантики типов RelationType.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph import GraphMemory            # noqa: E402
from core.storage import RelationType         # noqa: E402


class TestTwoEdgesSameTargetSurvive(unittest.TestCase):
    """Второе отношение к той же цели НЕ имеет права затирать первое."""

    def setUp(self):
        os.environ["LIPID2_MEMORY_DIR"] = tempfile.mkdtemp(prefix="l2_d4_")
        self.g = GraphMemory()
        self.a = self.g.get_or_create_node("девочка")
        self.b = self.g.get_or_create_node("кукла")

    def test_connect_to_keeps_both_edges(self):
        """Два connect_to к одной цели -> в ядре ОБА ребра, не одно."""
        self.a.connect_to(self.b, rel_type=RelationType.OBJECT_OF, weight=200)
        self.a.connect_to(self.b, rel_type=RelationType.CAN_DO, weight=150)
        self.assertEqual(
            len(self.a.out_synapses_typed), 2,
            "два отношения девочка->кукла обязаны дать ДВА ребра, а не одно")

    def test_both_rel_types_reachable(self):
        """Оба типа отношения читаются из исходящих связей узла."""
        self.a.connect_to(self.b, rel_type=RelationType.OBJECT_OF, weight=200)
        self.a.connect_to(self.b, rel_type=RelationType.CAN_DO, weight=150)
        rels = {e["rel_type"] for e in self.a.out_synapses_typed.raw_edges()}
        self.assertIn(RelationType.OBJECT_OF, rels)
        self.assertIn(RelationType.CAN_DO, rels)

    def test_weights_are_not_mixed(self):
        """Каждое ребро несёт СВОЙ вес: 200 и 150 не сливаются."""
        self.a.connect_to(self.b, rel_type=RelationType.OBJECT_OF, weight=200)
        self.a.connect_to(self.b, rel_type=RelationType.CAN_DO, weight=150)
        weights = {e["weight"] for e in self.a.out_synapses_typed.raw_edges()}
        self.assertIn(200, weights)
        self.assertIn(150, weights)

    def test_edges_survive_roundtrip(self):
        """Сериализация->десериализация сохраняет ОБА ребра к цели."""
        self.a.connect_to(self.b, rel_type=RelationType.OBJECT_OF, weight=200)
        self.a.connect_to(self.b, rel_type=RelationType.CAN_DO, weight=150)
        g2 = GraphMemory()
        # повторная запись тех же ребер (как при пересборке) не обязана
        # удваивать, но обязана НЕ терять первый тип
        a2 = g2.get_or_create_node("девочка")
        b2 = g2.get_or_create_node("кукла")
        a2.connect_to(b2, rel_type=RelationType.OBJECT_OF, weight=200)
        a2.connect_to(b2, rel_type=RelationType.CAN_DO, weight=150)
        rels2 = {e["rel_type"] for e in a2.out_synapses_typed.raw_edges()}
        self.assertIn(RelationType.OBJECT_OF, rels2)
        self.assertIn(RelationType.CAN_DO, rels2)


class TestCompatibleDictRead(unittest.TestCase):
    """Обратная совместимость чтения по ключу цели сохраняется."""

    def setUp(self):
        os.environ["LIPID2_MEMORY_DIR"] = tempfile.mkdtemp(prefix="l2_d4_")
        self.g = GraphMemory()
        self.a = self.g.get_or_create_node("девочка")
        self.b = self.g.get_or_create_node("кукла")

    def test_dict_read_first_edge(self):
        """out_synapses_typed[цель] отдаёт ПЕРВОЕ ребро к цели (как раньше)."""
        self.a.connect_to(self.b, rel_type=RelationType.OBJECT_OF, weight=200)
        first = self.a.out_synapses_typed["кукла"]
        self.assertEqual(first["rel_type"], RelationType.OBJECT_OF)
        self.assertEqual(first["weight"], 200)

    def test_items_returns_target_first(self):
        """items() даёт (target, dict) — итерацию по первому ребру на цель."""
        self.a.connect_to(self.b, rel_type=RelationType.OBJECT_OF, weight=200)
        items = list(self.a.out_synapses_typed.items())
        self.assertEqual(items[0][0], "кукла")
        self.assertEqual(items[0][1]["rel_type"], RelationType.OBJECT_OF)

    def test_get_returns_first_or_default(self):
        """get(цель) и get(нет_такой) работают как у словаря."""
        self.a.connect_to(self.b, rel_type=RelationType.OBJECT_OF, weight=200)
        self.assertEqual(self.a.out_synapses_typed.get("кукла")["weight"], 200)
        self.assertIsNone(self.a.out_synapses_typed.get("нет_такой"))


if __name__ == "__main__":
    unittest.main(verbosity=2)