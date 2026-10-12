import json
import struct
import os
from enum import IntEnum
from typing import Dict, List, Optional
from core.sdr import SDR1024
from core.encoder import get_encoder
from core.records import merge_entity   # noqa: F401 — правило слияния ОДНО на все каналы
from core.worldfile import atomic_write_json

# ==============================================================================
# ENUM-КОДЫ ДЛЯ МАКСИМАЛЬНОЙ ЭКОНОМИИ ПАМЯТИ
# ==============================================================================
class NodeType(IntEnum):
    ENTITY = 0     # Сущность (кот, мячик, бегемот)
    PROPERTY = 1   # Свойство (пушистый, упругий)
    ACTION = 2     # Действие / Глагол (прыгать, мяукать)

class RelationType(IntEnum):
    IS_A = 0          # Категория / Иерархия (кошка -> кошачьи)
    HAS_PROPERTY = 1  # Свойство (кошка -> пушистая)
    CAN_DO = 2        # Доступное действие (кошка -> мяукать)
    SYNONYM_OF = 3    # Алиас / Синоним (гиппопотам -> бегемот)
    CAUSES = 4        # Причинно-следственная связь (упасть -> разбить)
    # ---- извлечено из предложения (core/extract.py) ----
    # Индекс 3 занимает SYNONYM_OF, поэтому новые отношения идут с 5: старые
    # файлы и сохранённые веса не сдвигаются ни на единицу.
    IS_ROLE = 5       # «Вася — это инженер» (совпадение с IS_A разрешено)
    LOCATED_IN = 6    # «кошка на подоконнике» (падеж/предлог, не IS_A)
    OBJECT_OF = 7     # объект действия: «съела -> рыбу»
    PART_OF = 8       # составь/принадлежность: «хвост -> собака»
    INSTRUMENT = 9    # Тв.п. орудие: «открывает -> мышь»
    RECIPIENT = 10    # Дат.п. одуш. получатель: «дал -> кошке»
    PURPOSE = 11      # инфинитив цели: «пришёл -> увидеть»
    ABOUT = 12        # «для/про/о»: «корм для кошки»
    HAS_PART = 13     # владение/часть: «у кошки есть хвост»
    CIRCUMSTANCE = 14 # обстоятельство времени: «светит -> днём»
    CONDITION = 15    # условие/мера: «кипит при -> градусе»
    VALUE = 16        # величина/мера: «весит -> килограмм»

# ==============================================================================
# СТРУКТУРА ХРАНИЛИЩА (STORAGE MANAGER)
# ==============================================================================
class StorageManager:
    def __init__(self, json_path: str = None, binary_path: str = None):
        # По умолчанию храним всё в data/memory/, чтобы не сорить в корень проекта.
        if json_path is None or binary_path is None:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            mem_dir = os.path.join(root, "data", "memory")
            os.makedirs(mem_dir, exist_ok=True)
            if json_path is None:
                json_path = os.path.join(mem_dir, "lipid_db.json")
            if binary_path is None:
                binary_path = os.path.join(mem_dir, "lipid_db.lipid")
        self.json_path = json_path
        self.binary_path = binary_path

    def save_json(self, nodes: dict) -> bool:
        """
        Сохраняет граф памяти в читаемый JSON-формат.
        """
        enc = get_encoder()
        export_data = {
            "meta": {
                "version": "2.1",
                "project": "Lipid-2",
                "total_nodes": len(nodes),
                # SDR сохраняется как 16 uint64. Эти биты осмысленны ТОЛЬКО
                # вместе с кодировщиком, который их породил. Без метки граф,
                # собранный буквенным хэшем, молча дочитался бы эмбеддинговым
                # — и все веса синапсов оказались бы привязаны к чужим векторам.
                "sdr_encoder": enc.name,
                "sdr_encoder_config": enc.config_hash(),
            },
            "nodes": []
        }

        for name, node in nodes.items():
            # Извлекаем сырые биты SDR для компактного сохранения
            sdr_raw = node.sdr.bits if hasattr(node, 'sdr') else [0]*16
            
            node_data = {
                "id": getattr(node, 'id', 0),
                "label": node.name,
                "node_type": getattr(node, 'node_type', NodeType.ENTITY),
                "canonical_id": getattr(node, 'canonical_id', 0),
                "sdr_bits": sdr_raw,
                "action_profile": getattr(node, 'action_profile', [0, 0, 0]),
                "synapses": []
            }
            # ПРОИСХОЖДНИЕ ЗНАНИЯ. Знание из рассказа и знание из Википелии
            # равноценны только пока их не пришлось проверить. Без source
            # через месяц невозможно отличить факт от правдоподобной выдумки
            # локалки, а именно это решает, можно ли на знание опираться.
            source = getattr(node, 'source', '')
            if source:
                node_data["source"] = source
            evidence = getattr(node, 'evidence', '')
            if evidence:
                node_data["evidence"] = evidence

            # Сохраняем исходящие синапсы
            if hasattr(node, 'out_synapses_typed'):
                for target_name, syn_info in node.out_synapses_typed.items():
                    syn = {
                        "target_label": target_name,
                        "rel_type": syn_info["rel_type"],
                        "weight": syn_info["weight"]
                    }
                    if syn_info.get("source"):
                        syn["source"] = syn_info["source"]
                    node_data["synapses"].append(syn)
            else:
                # Фоллбэк для старых синапсов без типа
                for target_name, weight in node.out_synapses.items():
                    node_data["synapses"].append({
                        "target_label": target_name,
                        "rel_type": RelationType.HAS_PROPERTY,
                        "weight": weight
                    })

            export_data["nodes"].append(node_data)

        try:
            # атомарно: оборванный экспорт хуже, чем «не экспортировано»
            atomic_write_json(self.json_path, export_data)
            print(f"[Storage]: Граф успешно сохранен в JSON: '{self.json_path}' ({len(nodes)} узлов)")
            return True
        except Exception as e:
            print(f"[Storage Error]: Не удалось сохранить JSON: {e}")
            return False

    def load_json(self) -> Optional[dict]:
        """
        Загружает граф из JSON-файла.
        """
        if not os.path.exists(self.json_path):
            print(f"[Storage]: Файл '{self.json_path}' не найден. Начинаем с чистой памяти.")
            return None

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[Storage]: Успешно загружен JSON '{self.json_path}' (Узлов: {data['meta']['total_nodes']})")
            meta = data.get("meta", {})
            saved_cfg = meta.get("sdr_encoder_config")
            cur = get_encoder()
            if saved_cfg and saved_cfg != cur.config_hash():
                print("⚠️  [Storage]: ВНИМАНИЕ: файл сохранён ДРУГИМ SDR-кодировщиком "
                      f"({meta.get('sdr_encoder')}/{saved_cfg}), чем сейчас "
                      f"({cur.name}/{cur.config_hash}). Сохранённые 1024-битные "
                      "векторы несовместимы — узнавание будет неверным. "
                      "Пересоберите граф или верните LIPID2_SDR_BACKEND прежний.")
            return data
        except Exception as e:
            print(f"[Storage Error]: Ошибка загрузки JSON: {e}")
            return None

    def compile_to_binary(self) -> bool:
        """
        Компилирует JSON-базу в сверхкомпактный бинарник (.lipid).
        
        Бинарный формат записи узла:
        - Header: "LPD2" (4 байта)
        - Node:
            uint32 id
            uint8  node_type
            uint32 canonical_id
            uint64[16] sdr_bits (128 байт)
            uint8[3] action_profile (kinetic, metabolic, semiotic)
            uint16 num_synapses
            -- Synapses array:
                uint32 target_id
                uint8  rel_type
                uint8  weight
        """
        raw_json = self.load_json()
        if not raw_json:
            print("[Storage Error]: Нет данных для компиляции.")
            return False

        label_to_id = {node["label"]: idx + 1 for idx, node in enumerate(raw_json["nodes"])}

        try:
            with open(self.binary_path, "wb") as f:
                # Заголовок магии "LPD2"
                f.write(b"LPD2")
                
                # Записываем количество узлов (uint32)
                f.write(struct.pack("<I", len(raw_json["nodes"])))

                for idx, node in enumerate(raw_json["nodes"]):
                    node_id = label_to_id[node["label"]]
                    node_type = int(node.get("node_type", NodeType.ENTITY))
                    
                    canon_label = node.get("canonical_label", "")
                    canonical_id = label_to_id.get(canon_label, 0)

                    sdr_bits = node.get("sdr_bits", [0]*16)
                    act_prof = node.get("action_profile", [0, 0, 0])
                    synapses = node.get("synapses", [])

                    # Упаковка базовых полей узла
                    # < I B I (id, node_type, canonical_id)
                    f.write(struct.pack("<IBI", node_id, node_type, canonical_id))
                    
                    # Упаковка 16x uint64 SDR бит (128 байт)
                    f.write(struct.pack("<16Q", *sdr_bits))
                    
                    # Упаковка профиля действий 3x uint8 (kinetic, metabolic, semiotic)
                    f.write(struct.pack("<3B", *act_prof))

                    # Упаковка количества синапсов (uint16)
                    f.write(struct.pack("<H", len(synapses)))

                    # Упаковка каждого синапса: target_id (uint32), rel_type (uint8), weight (uint8)
                    for syn in synapses:
                        target_id = label_to_id.get(syn["target_label"], 0)
                        rel_type = int(syn.get("rel_type", RelationType.HAS_PROPERTY))
                        weight = int(syn.get("weight", 255))
                        f.write(struct.pack("<IBB", target_id, rel_type, weight))

            size_bytes = os.path.getsize(self.binary_path)
            print(f"[Storage]: БАЗА УСПЕШНО СКОМПИЛИРОВАНА В БИНАРНИК!")
            print(f" -> Файл: '{self.binary_path}' | Размер: {size_bytes} байт ({size_bytes / 1024:.2f} KB)")
            return True

        except Exception as e:
            print(f"[Storage Error]: Сбой компиляции в бинарник: {e}")
            return False


# ==============================================================================
# ТЕСТИРОВАНИЕ КОМПИЛЯТОРА
# ==============================================================================
if __name__ == "__main__":
    print("--- ТЕСТИРОВАНИЕ STORAGE MANAGER & СКОМПИЛИРОВАННОГО .LIPID ---\n")

    storage = StorageManager()

    # Имитация тестового графа для проверки компилятора
    class DummyNode:
        def __init__(self, name, n_type=NodeType.ENTITY):
            self.id = 1
            self.name = name
            self.node_type = n_type
            self.canonical_id = 0
            self.sdr = SDR1024(name)
            self.action_profile = [0, 0, 0]
            self.out_synapses_typed = {}

    node_cat = DummyNode("кошка", NodeType.ENTITY)
    node_hip = DummyNode("гиппопотам", NodeType.ENTITY)
    node_beg = DummyNode("бегемот", NodeType.ENTITY)

    # Синонимическая связь: гиппопотам -> бегемот
    node_hip.out_synapses_typed["бегемот"] = {"rel_type": RelationType.SYNONYM_OF, "weight": 255}
    node_cat.out_synapses_typed["кошачьи"] = {"rel_type": RelationType.IS_A, "weight": 255}

    test_nodes = {
        "кошка": node_cat,
        "бегемот": node_beg,
        "гиппопотам": node_hip
    }

    print("1. Сохраняем граф в JSON...")
    storage.save_json(test_nodes)

    print("\n2. Компилируем JSON в бинарный файл .lipid...")
    storage.compile_to_binary()