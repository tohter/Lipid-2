import json
import os
import sys
from typing import Dict, List, Tuple, Optional

# Позволяет запускать модуль как скрипт (python3 core/graph.py), а не только как пакет.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sdr import SDR1024, prewarm
from core.index import SdrIndex
from core.lemma import LemmaResolver
from core.storage import StorageManager, NodeType, RelationType

class Node:
    """Узел графа памяти с поддержкой типов узлов и синонимов."""
    def __init__(self, name: str, node_type: int = NodeType.ENTITY, canonical_id: int = 0):
        self.id = 0
        self.name = name.lower().strip()
        self.node_type = node_type
        self.canonical_id = canonical_id
        self.sdr = SDR1024(self.name)
        # Профиль действий для ACTION узлов: [kinetic, metabolic, semiotic] (0..255)
        self.action_profile = [0, 0, 0]
        # Синапсы: {target_name: {"rel_type": int, "weight": int}}
        self.out_synapses_typed: Dict[str, dict] = {}

    def connect_to(self, target_node: 'Node', rel_type: int, weight: int = 255):
        """Создает типерированную направленную связь к target_node."""
        clamped_weight = max(0, min(255, weight))
        self.out_synapses_typed[target_node.name] = {
            "rel_type": rel_type,
            "weight": clamped_weight
        }

    def __repr__(self):
        return f"<Node '{self.name}' type={self.node_type} synapses={len(self.out_synapses_typed)}>"


class GraphMemory:
    """
    Целочисленное графовое ядро памяти (Липид-2) с поддержкой:
    - Синонимов (SYNONYM_OF)
    - Иерархического наследования (IS_A)
    - Профилей действий (Kinetic, Metabolic, Semiotic)
    - Сохранения и сжатия в бинарник
    """
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.storage = StorageManager()
        # SDR-индекс: аргмакс перекрытия матрицей, а не Python-циклом по узлам.
        self.index = SdrIndex()
        # Резолвер поверхностных форм («флюгера» -> «флюгер»). Работает по
        # float-эмбеддингу, а НЕ по перекрытию SDR: на 20 битах морфология не
        # различима (замеры — experiments/RESULTS.md, раздел 6).
        self.lemmas = LemmaResolver()
        self._lemmas_attached = False

    def get_or_create_node(self, name: str, node_type: int = NodeType.ENTITY) -> Node:
        """Возвращает существующий узел или создает новый."""
        clean_name = name.lower().strip()
        if clean_name not in self.nodes:
            self.nodes[clean_name] = Node(clean_name, node_type=node_type)
            self.nodes[clean_name].id = len(self.nodes)
            self.index.add(clean_name, self.nodes[clean_name].sdr)
            if self._lemmas_attached:
                self.lemmas.add(clean_name)
        return self.nodes[clean_name]

    # ── узнавание формы слова ──────────────────────────────
    def sync_lemmas(self):
        """
        Индексирует все имена узлов для сведения форм. Вызывать пачкой —
        кодирование имён дорогое, по одному его гонять нельзя.
        """
        if not self.nodes:
            return 0
        self.lemmas.attach(list(self.nodes.keys()))
        self._lemmas_attached = True
        return len(self.lemmas)

    def resolve(self, word: str) -> Optional[Node]:
        """
        Узел по слову ЛЮБОЙ формы: сначала точное имя, потом морфологическое
        сведения. НИКОГДА не создаёт узел.

        Почему это отдельная операция, а не побочный эффект get_or_create_node:
        сведения формы по косинусу 0.85 обратимо, пока это только ОТВЕТ ПОИСКА.
        Если бы узел схлопывался в момент создания, «выражать» и «выразить»
        слились бы в графе без следа и без возможности разобрать ошибку. Здесь
        же узел остаётся на месте — ошибку можно увидеть и откатить.
        """
        word = word.lower().strip()
        if not word:
            return None
        node = self.nodes.get(word)
        if node is not None:
            return node
        if not self._lemmas_attached:
            # Ленивая инициализация: молча возвращать None было бы ловушкой —
            # заблывает один вызов sync_lemmas(), и вся система «вдруг» перестаёт
            # понимать падежи, без ошибки и без объяснения.
            self.sync_lemmas()
        name = self.lemmas.resolve_name(word)
        return self.nodes.get(name) if name else None

    def resolve_name(self, word: str) -> Optional[str]:
        node = self.resolve(word)
        return node.name if node else None

    def prewarm_sdr(self):
        """
        Пачечное кодирование всех имён узлов. Без этого загрузка графа на 10^4
        узлов — это 10^4 отдельных вызовов эмбеддинг-модели вместо одного.
        """
        if not self.nodes:
            return 0
        n = prewarm(list(self.nodes.keys()))
        if n:
            print(f"[Graph]: SDR прогрев {n} имён пачкой.")
        return n

    def add_synonym(self, word: str, canonical_word: str):
        """
        Привязывает узел-синоним к каноническому концепту.
        Пример: add_synonym("гиппопотам", "бегемот")
        """
        canon_node = self.get_or_create_node(canonical_word, NodeType.ENTITY)
        syn_node = self.get_or_create_node(word, NodeType.ENTITY)
        
        syn_node.canonical_id = canon_node.id
        syn_node.connect_to(canon_node, rel_type=RelationType.SYNONYM_OF, weight=255)
        print(f"[Graph]: Узел '{word}' зарегистрирован как СИНОНИМ для '{canonical_word}'.")

    def integrate_knowledge(self, knowledge: dict, weight: int = 255):
        """Интегрирует данные от Учителя с расстановкой типов связей."""
        if not knowledge or "concept" not in knowledge:
            return

        main_concept = knowledge["concept"]
        root_node = self.get_or_create_node(main_concept, NodeType.ENTITY)

        # 1. Категория (IS_A)
        if "category" in knowledge and knowledge["category"]:
            cat_node = self.get_or_create_node(knowledge["category"], NodeType.ENTITY)
            root_node.connect_to(cat_node, rel_type=RelationType.IS_A, weight=weight)

        # 2. Свойства (HAS_PROPERTY)
        for prop in knowledge.get("properties", []):
            prop_node = self.get_or_create_node(prop, NodeType.PROPERTY)
            root_node.connect_to(prop_node, rel_type=RelationType.HAS_PROPERTY, weight=weight)

        # 3. Действия (CAN_DO)
        for act in knowledge.get("actions", []):
            act_node = self.get_or_create_node(act, NodeType.ACTION)
            root_node.connect_to(act_node, rel_type=RelationType.CAN_DO, weight=weight)

        print(f"[Graph]: Интегрирован '{main_concept}' | Связей: {len(root_node.out_synapses_typed)}")

    def resolve_canonical_node(self, concept_name: str) -> Node:
        """Проваливается сквозь синонимы к каноническому узлу."""
        node = self.resolve(concept_name)
        if not node:
            return None

        # Проверяем связь SYNONYM_OF
        for target, syn_info in node.out_synapses_typed.items():
            if syn_info["rel_type"] == RelationType.SYNONYM_OF:
                print(f"[Graph]: ПЕРЕНАПРАВЛЕНИЕ СИНОНИМА: '{node.name}' -> '{target}'")
                return self.nodes[target]
        return node

    def recall_by_sdr(self, input_sdr: SDR1024, top_k: int = 3) -> List[Tuple[str, int, float]]:
        """Поиск по SDR перекрытию (через индекс)."""
        return [(name, ov, ov / float(SDR1024.TARGET_ON_BITS))
                for name, ov in self.index.topk(input_sdr, top_k) if ov > 0]

    def save_to_disk(self):
        """Сохраняет JSON и автоматически компилирует в .lipid бинарник."""
        self.storage.save_json(self.nodes)
        self.storage.compile_to_binary()

    def print_node_associations(self, concept_name: str):
        """Печать дерева связей с расшифровкой типов."""
        node = self.resolve_canonical_node(concept_name)
        if not node:
            print(f"[Graph]: Узел '{concept_name}' не найден.")
            return

        rel_names = {0: "IS_A", 1: "HAS_PROPERTY", 2: "CAN_DO", 3: "SYNONYM_OF", 4: "CAUSES"}

        print(f"\nАссоциации канонического узла [{node.name.upper()}]:")
        for target, syn_info in node.out_synapses_typed.items():
            r_name = rel_names.get(syn_info["rel_type"], "LINK")
            w = syn_info["weight"]
            print(f"   └── --({r_name}: {w})--> [{target}]")


if __name__ == "__main__":
    print("--- ТЕСТИРОВАНИЕ РАБОТЫ ГРАФА И СИНОНИМОВ (ЛИПИД-2) ---\n")

    g = GraphMemory()

    # Имитируем добавление знаний
    g.integrate_knowledge({
        "concept": "бегемот",
        "category": "млекопитающее",
        "properties": ["огромный", "тяжелый"],
        "actions": ["плавать"]
    })

    # Добавляем синоним
    g.add_synonym("гиппопотам", "бегемот")

    # Проверяем резолв: при запросе гиппопотама выведутся связи бегемота
    g.print_node_associations("гиппопотам")

    # Сохраняем и компилируем
    g.save_to_disk()