"""
core/adapter.py — Адаптер: формат Учителя (human-readable JSON) -> формат Storage (бинарный граф).

Проблема: Учитель (teachers/teacher1.py, teachers/teacher_wiki.py) пишет в
lipid_world.json структуру для ЧТЕНИЯ ЧЕЛОВЕКОМ:
    {
      "entities": {
        "кошка": {"hierarchy": [...], "properties": [...], "actions": [...], ...}
      },
      "relations": [],
      "observations": {...},
      "unknown_words": {...}
    }

А StorageManager.compile_to_binary() ожидает:
    {
      "meta": {...},
      "nodes": [
        {"id": 1, "label": "...", "node_type": 0, "sdr_bits": [...],
         "action_profile": [K,M,S], "synapses": [...], "source": "...", "evidence": "..."}
      ]
    }

Этот модуль транслирует формат Учителя в формат Storage, генерируя SDR1024
для каждого концепта и создавая типизированные синапсы:
  - hierarchy -> IS_A
  - properties -> HAS_PROPERTY
  - actions -> CAN_DO (с action_profile)
  - derived_entity parent_concept -> IS_A (или SYNONYM_OF)
"""
from __future__ import annotations

import os
import sys
import json
from typing import Dict, List, Any, Optional, Tuple

# Путь к корню проекта
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.encoder import get_encoder
from core.bitsstore import pack_bits
from core.actions import ACTION_PROFILES, profile_of, is_known_action
from core.storage import NodeType, RelationType


class TeacherToStorageAdapter:
    """
    Конвертирует lipid_world.json (формат Учителя) в формат, подходящий
    для StorageManager.compile_to_binary().
    """

    def __init__(self, world_path: str = None):
        if world_path is None:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            mem_dir = os.path.join(root, "data", "memory")
            world_path = os.path.join(mem_dir, "lipid_world.json")
        self.world_path = world_path
        self.encoder = get_encoder()
        self._entity_id_map: Dict[str, int] = {}  # label -> id (1-based)

    def load_world(self) -> Dict[str, Any]:
        """Загружает lipid_world.json."""
        if not os.path.exists(self.world_path):
            raise FileNotFoundError(f"World file not found: {self.world_path}")
        with open(self.world_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _generate_sdr_bits(self, label: str) -> List[int]:
        """Генерирует 16 uint64 SDR биты для метки через энкодер."""
        try:
            sdr_array = self.encoder.encode(label)  # numpy array (1024,) uint8
            # pack_bits ожидает список bool/int, возвращает bytes (128 байт = 16 uint64)
            packed = pack_bits(sdr_array.astype(bool).tolist())
            # Распаковываем bytes обратно в 16 uint64 для JSON сериализации
            import struct
            return list(struct.unpack("<16Q", packed))
        except Exception as e:
            print(f"[Adapter]: Ошибка генерации SDR для '{label}': {e}", file=sys.stderr)
            return [0] * 16

    def _get_action_profile(self, action_name: str) -> List[int]:
        """Получает профиль действия [K,M,S] из справочника."""
        prof = profile_of(action_name)
        if prof:
            return prof
        # Неизвестное действие — нулевой профиль (будет помечено для проверки)
        return [0, 0, 0]

    def _build_entity_node(self, label: str, entity_data: Dict[str, Any],
                           all_entities: Dict[str, Any]) -> Dict[str, Any]:
        """Строит один узел в формате Storage из записи Учителя."""
        node_id = self._entity_id_map[label]

        # Определяем node_type: ENTITY по умолчанию
        node_type = NodeType.ENTITY

        # SDR биты
        sdr_bits = self._generate_sdr_bits(label)

        # Профиль действий для узла-сущности (обычно нулевой, заполняется в ACTION узлах)
        action_profile = [0, 0, 0]

        # Синапсы
        synapses = []

        # 1. Иерархия -> IS_A
        hierarchy = entity_data.get("hierarchy", [])
        for parent in hierarchy:
            if parent and parent != label:
                parent_id = self._entity_id_map.get(parent.lower().strip())
                if parent_id:
                    synapses.append({
                        "target_label": parent,
                        "rel_type": RelationType.IS_A,
                        "weight": 255
                    })

        # 2. Свойства -> HAS_PROPERTY
        properties = entity_data.get("properties", [])
        for prop in properties:
            if prop:
                prop_label = prop.lower().strip()
                # Создаём/используем узел свойства (PROPERTY type)
                prop_id = self._entity_id_map.get(prop_label)
                if not prop_id:
                    # Свойство пока не в мапе — зарезервируем ID позже
                    prop_id = len(self._entity_id_map) + 1
                    self._entity_id_map[prop_label] = prop_id
                synapses.append({
                    "target_label": prop_label,
                    "rel_type": RelationType.HAS_PROPERTY,
                    "weight": 255
                })

        # 3. Действия -> CAN_DO (через узел ACTION)
        actions = entity_data.get("actions", [])
        for action in actions:
            if not action:
                continue
            if isinstance(action, dict):
                action_name = action.get("name", "").lower().strip()
                # Можно использовать action_profile из ответа Учителя если есть
                action_profile_data = action.get("action_profile")
            else:
                action_name = str(action).lower().strip()
                action_profile_data = None

            if not action_name:
                continue

            # Получаем профиль действия
            if action_profile_data and isinstance(action_profile_data, list) and len(action_profile_data) == 3:
                act_prof = action_profile_data
            else:
                act_prof = self._get_action_profile(action_name)

            # Создаём/используем узел действия (ACTION type)
            action_id = self._entity_id_map.get(action_name)
            if not action_id:
                action_id = len(self._entity_id_map) + 1
                self._entity_id_map[action_name] = action_id

            synapses.append({
                "target_label": action_name,
                "rel_type": RelationType.CAN_DO,
                "weight": 255
            })

            # Сохраняем профиль действия в метаданных узла действия (будет выделен отдельно)
            # Мы запомним это для создания отдельного ACTION узла

        # Provenance
        source = entity_data.get("source", "")
        evidence = entity_data.get("evidence", "")
        definition = entity_data.get("definition", "")

        # Собираем evidence из definition и source
        evidence_parts = []
        if definition:
            evidence_parts.append(definition[:200])
        if source:
            evidence_parts.append(f"source:{source}")
        evidence_str = " | ".join(evidence_parts) if evidence_parts else ""

        return {
            "id": node_id,
            "label": label,
            "node_type": int(node_type),
            "canonical_id": 0,
            "sdr_bits": sdr_bits,
            "action_profile": action_profile,
            "synapses": synapses,
            "source": source,
            "evidence": evidence_str
        }

    def _build_property_node(self, label: str) -> Dict[str, Any]:
        """Строит узел свойства (PROPERTY type)."""
        node_id = self._entity_id_map[label]
        sdr_bits = self._generate_sdr_bits(label)
        return {
            "id": node_id,
            "label": label,
            "node_type": int(NodeType.PROPERTY),
            "canonical_id": 0,
            "sdr_bits": sdr_bits,
            "action_profile": [0, 0, 0],
            "synapses": [],
            "source": "",
            "evidence": ""
        }

    def _build_action_node(self, label: str) -> Dict[str, Any]:
        """Строит узел действия (ACTION type) с профилем."""
        node_id = self._entity_id_map[label]
        sdr_bits = self._generate_sdr_bits(label)
        action_profile = self._get_action_profile(label)
        return {
            "id": node_id,
            "label": label,
            "node_type": int(NodeType.ACTION),
            "canonical_id": 0,
            "sdr_bits": sdr_bits,
            "action_profile": action_profile,
            "synapses": [],
            "source": "",
            "evidence": ""
        }

    def convert(self) -> Dict[str, Any]:
        """
        Основной метод конвертации.
        Возвращает dict в формате StorageManager (meta + nodes[]).
        """
        world = self.load_world()
        entities = world.get("entities", {})

        # Шаг 1: Собираем все уникальные метки и назначаем ID
        # Приоритет: entities -> properties -> actions
        all_labels = set()

        # Сначала все сущности
        for label in entities:
            all_labels.add(label.lower().strip())

        # Свойства и действия из entities
        for entity_data in entities.values():
            for prop in entity_data.get("properties", []):
                if prop:
                    all_labels.add(prop.lower().strip())
            for action in entity_data.get("actions", []):
                if isinstance(action, dict):
                    name = action.get("name", "")
                else:
                    name = str(action)
                if name:
                    all_labels.add(name.lower().strip())

        # derived_entity -> parent_concept
        for entity_data in entities.values():
            parent = entity_data.get("parent_concept")
            if parent and parent != entity_data.get("concept"):
                all_labels.add(parent.lower().strip())

        # Назначаем ID (1-based)
        self._entity_id_map = {label: idx + 1 for idx, label in enumerate(sorted(all_labels))}

        # Шаг 2: Строим узлы
        nodes = []

        # 2a. Сущности (ENTITY)
        for label, entity_data in entities.items():
            label_clean = label.lower().strip()
            node = self._build_entity_node(label_clean, entity_data, entities)
            nodes.append(node)

        # 2b. Свойства (PROPERTY) — те, что не являются сущностями
        entity_labels = set(e.lower().strip() for e in entities)
        for label in all_labels:
            if label not in entity_labels and label not in [n["label"] for n in nodes]:
                # Проверяем, является ли это действием
                if is_known_action(label) or label in ACTION_PROFILES:
                    continue  # будет создано в 2c
                node = self._build_property_node(label)
                nodes.append(node)

        # 2c. Действия (ACTION)
        for label in all_labels:
            if is_known_action(label) or label in ACTION_PROFILES:
                node = self._build_action_node(label)
                nodes.append(node)

        # Шаг 3: Формируем итоговый JSON
        export_data = {
            "meta": {
                "version": "2.1",
                "project": "Lipid-2",
                "total_nodes": len(nodes),
                "sdr_encoder": self.encoder.name,
                "sdr_encoder_config": self.encoder.config_hash(),
            },
            "nodes": nodes
        }

        return export_data

    def save_for_storage(self, output_path: str = None) -> bool:
        """Сохраняет сконвертированный граф в JSON для StorageManager."""
        if output_path is None:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            mem_dir = os.path.join(root, "data", "memory")
            output_path = os.path.join(mem_dir, "lipid_db.json")

        export_data = self.convert()

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            print(f"[Adapter]: Граф сконвертирован и сохранён в '{output_path}' ({len(export_data['nodes'])} узлов)")
            return True
        except Exception as e:
            print(f"[Adapter Error]: Не удалось сохранить: {e}", file=sys.stderr)
            return False


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Конвертация lipid_world.json -> lipid_db.json для Storage")
    parser.add_argument("--input", help="Путь к lipid_world.json")
    parser.add_argument("--output", help="Путь для lipid_db.json")
    args = parser.parse_args()

    adapter = TeacherToStorageAdapter(args.input)
    success = adapter.save_for_storage(args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()