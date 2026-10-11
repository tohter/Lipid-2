import os
import json
import re
from typing import Dict, Any, Optional

# Импортируем вынесенный IR-парсер
from brain.ir import IRParser
# Импортируем Учителей (3 канала знаний)
from teachers.teacher1 import AgentTeacherV2       # Канал 1: локальная LLM (Gemma/llama.cpp)
from teachers.teacher_wiki import WikiTeacher      # Канал 2: интернет (Википедия)
# Импортируем нейроморфное ядро (SDR / STDP / Гормоны)
from core.cortex import Hormones
from core.graph import GraphMemory
from core.storage import NodeType, RelationType


class Lipid2Brain:
    """
    Когнитивный оркестратор Lipid-2.

    Связывает воедино:
      - IR-парсер интентов (текст -> IR -> текст)
      - нейроморфное ядро (SDR + STDP + гормоны) — реально участвует в обучении
      - единую память Вселенной (lipid_world.json)
      - 3 канала самообучения при встрече с неизвестным:
          1. Локальный Учитель (LLM)  -> teachers.teacher1
          2. Интернет (Википедия)     -> teachers.teacher_wiki
          3. Вопрос пользователю      -> через GUI (колбэк)
    """

    def __init__(self, memory_dir: str = None):
        if memory_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.memory_dir = os.path.join(project_root, "data", "memory")
        else:
            self.memory_dir = memory_dir

        os.makedirs(self.memory_dir, exist_ok=True)
        self.world_filepath = os.path.join(self.memory_dir, "lipid_world.json")

        # Модули ввода
        self.ir_parser = IRParser()

        # Три канала знаний (host локалки настраиваемый, по умолчанию 8080)
        llm_host = os.environ.get("LIPID2_LLM_HOST", "http://localhost:8080")
        self.teacher_llm = AgentTeacherV2(host=llm_host, memory_dir=self.memory_dir)
        self.teacher_wiki = WikiTeacher(memory_dir=self.memory_dir)
        self.ask_user_callback = None      # колбэк: (вопрос) -> ответ пользователя, ставится из GUI

        # Нейроморфное ядро (SDR + STDP + гормоны)
        self.core = GraphMemory()
        self.hormones = Hormones()

        # Загружаем существующие знания мира в граф ядра
        self._sync_world_to_core()

    # ── СИНХРОНИЗАЦИЯ МИРА И ЯДРА ──────────────────────────

    def _sync_world_to_core(self):
        """Переносит сущности из lipid_world.json в нейроморфный граф ядра."""
        world = self._load_world()
        count = 0
        for concept, data in world.get("entities", {}).items():
            self._integrate_entity_into_core(concept, data)
            count += 1
        if count:
            print(f"[Lipid-2] Ядро загружено: {count} сущностей из Вселенной.")

    def _integrate_entity_into_core(self, concept: str, data: dict):
        """Записывает одну сущность из мира в граф ядра (SDR-узел + синапсы)."""
        if not concept:
            return
        root = self.core.get_or_create_node(concept, NodeType.ENTITY)

        # Иерархия (IS_A) — связываем концепт с ближайшей родительской категорией
        hierarchy = data.get("hierarchy", [])
        if hierarchy:
            parent = self.core.get_or_create_node(hierarchy[0], NodeType.ENTITY)
            root.connect_to(parent, rel_type=RelationType.IS_A, weight=180)

        # Свойства (HAS_PROPERTY)
        for prop in data.get("properties", []):
            pnode = self.core.get_or_create_node(prop, NodeType.PROPERTY)
            root.connect_to(pnode, rel_type=RelationType.HAS_PROPERTY, weight=160)

        # Действия (CAN_DO) + профиль [K,M,S] на узле действия
        for act in data.get("actions", []):
            aname = act["name"] if isinstance(act, dict) else str(act)
            anode = self.core.get_or_create_node(aname, NodeType.ACTION)
            if isinstance(act, dict) and "action_profile" in act:
                anode.action_profile = list(act["action_profile"])
            root.connect_to(anode, rel_type=RelationType.CAN_DO, weight=150)

        # Дельта-наследование (производный узел с родителем)
        parent_concept = data.get("parent_concept")
        if parent_concept:
            parent = self.core.get_or_create_node(parent_concept, NodeType.ENTITY)
            root.connect_to(parent, rel_type=RelationType.IS_A, weight=180)

    # ── РАБОТА С ВСЕЛЕННОЙ ─────────────────────────────────

    def _load_world(self) -> Dict[str, Any]:
        """Загружает текущую оперативную память Вселенной с диска."""
        if os.path.exists(self.world_filepath):
            try:
                with open(self.world_filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"entities": {}, "relations": []}

    def _save_world(self, world: dict):
        with open(self.world_filepath, "w", encoding="utf-8") as f:
            json.dump(world, f, ensure_ascii=False, indent=2)

    def find_in_memory(self, concept: str) -> Optional[Dict[str, Any]]:
        """Быстрый поиск сущности и её родительских связей в Графе Вселенной."""
        concept_clean = concept.lower().strip()
        world = self._load_world()
        entities = world.get("entities", {})

        if concept_clean in entities:
            entity_data = entities[concept_clean]
            if "parent_concept" in entity_data:
                parent_data = entities.get(entity_data["parent_concept"], {})
                return {"base_entity": parent_data, "derived_entity": entity_data}
            else:
                return {"base_entity": entity_data, "derived_entity": {}}
        return None

    # ── ТРЁХКАНАЛЬНОЕ САМООБУЧЕНИЕ ─────────────────────────

    def _trigger_curiosity_and_learn(self, concept: str) -> Optional[Dict[str, Any]]:
        """
        Спайк Норадреналина: реакция на неизвестное слово.
        Пробует источники по порядку: 1) локалка, 2) интернет, 3) пользователь.
        Первый успешный источник даёт знания, которые записываются в мир и ядро.
        """
        print(f"\n⚡ [НОРАДРЕНАЛИН] Внимание! Неизвестная сущность '{concept}'!")
        print(f"🧠 [Lipid-2]: Мой граф не содержит '{concept}'. Ищу знание...")

        self.hormones.norepinephrine = 255
        self.hormones.dopamine = 10

        learned_data = None

        # ── КАНАЛ 1: Локальный Учитель (LLM) ──
        print("   [Канал 1] Локальный Учитель (Gemma)...")
        try:
            self.teacher_llm.run_agent(base_concept=concept, derived_concept=concept,
                                       filename="lipid_world.json")
            learned_data = self.find_in_memory(concept)
            if learned_data:
                print("   \u2714 Знание получено от локальной LLM.")
        except Exception as e:
            print(f"   \u2716 Локалка недоступна: {e}")
            learned_data = None

        # ── КАНАЛ 2: Интернет (Википедия) ──
        if not learned_data:
            print("   [Канал 2] Интернет (Википедия)...")
            try:
                wiki = self.teacher_wiki.fetch_concept(concept)
                if wiki and wiki.get("base_entity", {}).get("concept"):
                    if self.teacher_wiki.write_to_world(wiki):
                        learned_data = self.find_in_memory(concept)
                if learned_data:
                    print("   \u2714 Знание получено из интернета.")
            except Exception as e:
                print(f"   \u2716 Интернет недоступен: {e}")
                learned_data = None

        # ── КАНАЛ 3: Вопрос пользователю ──
        if not learned_data:
            print("   [Канал 3] Вопрос пользователю...")
            if self.ask_user_callback is not None:
                user_answer = self.ask_user_callback(concept)
                if user_answer:
                    learned_data = self._learn_from_user(concept, user_answer)
                    if learned_data:
                        print("   \u2714 Знание получено от пользователя.")
                else:
                    print("   \u2716 Пользователь не дал знания.")
            else:
                print("   \u2716 Нет колбэка к пользователю (GUI не подключён).")

        # ── Результат ──
        if learned_data:
            self.hormones.norepinephrine = 20
            self.hormones.dopamine = 255
            print(f"🎉 [ДОФАМИН] Успех! Граф пополнен сущностью '{concept}'.\n")
            # Интегрируем новые знания в нейроморфное ядро
            self._integrate_learned_data(learned_data)
        else:
            self.hormones.cortisol = min(255, self.hormones.cortisol + 120)
            print(f"😔 [КОРТИЗОЛ] Все источники исчерпаны. Сущность '{concept}' осталась неизвестной.\n")

        return learned_data

    def _learn_from_user(self, concept: str, answer: str) -> Optional[Dict[str, Any]]:
        """
        Заворачивает ответ пользователя в онтологию (грубая, но рабочая).
        Пользователь может написать: "это животное; пушистый; прыгает"
        Разделяем по ';' / ',' — первый элемент = категория, остальные = свойства.
        """
        answer_clean = answer.strip().strip(".,;").strip()
        if not answer_clean:
            return None

        parts = [p.strip() for p in re.split(r"[;,]", answer_clean) if p.strip()]
        if not parts:
            return None

        # Убираем "это/он/она/оно" в начале первого элемента
        first = parts[0].lower().strip()
        for w in ("это ", "это", "он ", "она ", "оно ", "этот ", "эта "):
            if first.startswith(w):
                parts[0] = parts[0][len(w):].strip()
                break
        if parts[0].lower() in ("это", "он", "она", "оно"):
            parts = parts[1:]
        if not parts:
            return None

        category = parts[0]
        properties = parts[1:]

        data = {
            "base_entity": {
                "concept": concept,
                "hierarchy": [category],
                "properties": properties,
                "actions": [],
            },
            "derived_entity": {},
        }

        world = self._load_world()
        world["entities"][concept.lower().strip()] = data["base_entity"]
        self._save_world(world)
        return data

    def _integrate_learned_data(self, learned_data: dict):
        """Записывает вновь полученные знания (из любого источника) в ядро."""
        base = learned_data.get("base_entity", {})
        concept = base.get("concept", "").lower().strip()
        if not concept:
            return
        self._integrate_entity_into_core(concept, base)

        derived = learned_data.get("derived_entity", {})
        if derived and derived.get("concept"):
            self._integrate_entity_into_core(derived["concept"].lower().strip(), derived)

    # ── ГЛАВНАЯ ТОЧКА ВХОДА ────────────────────────────────

    def process_input(self, text: str) -> str:
        """
        Главная точка входа обработки текста:
        1. IR-парсер вычленяет интент и сущности.
        2. Мозг запрашивает память или активирует обучение.
        3. Формируется осмысленный ответ.
        """
        # 1. Вызываем отдельный IR-модуль
        ir_intent = self.ir_parser.parse_query(text)

        if not ir_intent or not ir_intent.subjects:
            return "🤯 [Lipid-2]: Не удалось распознать интент или ключевые сущности в фразе."

        intent_type = ir_intent.intent_type
        subject = ir_intent.subjects[0]

        # 2. Маршрутизация по типам интентов
        if intent_type == "DEFINITION":
            return self._handle_definition(subject)
        elif intent_type == "PROPERTIES":
            return self._handle_properties(subject)
        elif intent_type == "INTERACTION":
            return self._handle_interaction(ir_intent.subjects)
        else:
            return f"🤔 [Lipid-2]: Интент '{intent_type}' распознан, но обработчик для него ещё в разработке."

    # ── ОБРАБОТЧИКИ ────────────────────────────────────────

    def _handle_definition(self, concept: str) -> str:
        data = self.find_in_memory(concept)
        if not data:
            data = self._trigger_curiosity_and_learn(concept)
            if not data:
                return f"[Lipid-2]: Я искал значение '{concept}', но все источники знаний не дали ответа."

        base_ent = data.get("base_entity", {})
        derived_ent = data.get("derived_entity", {})

        if derived_ent.get("concept") == concept:
            parent = derived_ent.get("parent_concept", "базовый вид")
            delta = ", ".join(derived_ent.get("delta_properties", []))
            return f"[Lipid-2]: {concept.capitalize()} — это {parent} со свойствами: {delta}."

        if base_ent.get("concept") == concept:
            hierarchy = base_ent.get("hierarchy", [])
            if hierarchy:
                if len(hierarchy) == 1:
                    # одиночная категория (например, от пользователя)
                    return f"[Lipid-2]: {concept.capitalize()} — это {hierarchy[0]}."
                # последний элемент иерархии = самый общий корень
                return f"[Lipid-2]: {concept.capitalize()} — это {hierarchy[-1]}, категория: {' -> '.join(hierarchy)}."

            # из интернета: есть 'title' и 'definition'
            definition = base_ent.get("definition", "")
            if definition:
                return f"[Lipid-2]: {concept.capitalize()} — {definition}."

        return f"[Lipid-2]: Недостаточно онтологических данных для определения '{concept}'."

    def _handle_properties(self, concept: str) -> str:
        data = self.find_in_memory(concept)
        if not data:
            data = self._trigger_curiosity_and_learn(concept)
            if not data:
                return f"[Lipid-2]: Не могу определить свойства '{concept}'."

        base_ent = data.get("base_entity", {})
        derived_ent = data.get("derived_entity", {})

        props = []
        if derived_ent.get("concept") == concept:
            props = base_ent.get("properties", []) + derived_ent.get("delta_properties", [])
        elif base_ent.get("concept") == concept:
            props = base_ent.get("properties", [])

        props_str = ", ".join(props) if props else "нет данных"
        return f"[Lipid-2]: Свойства сущности '{concept}': [{props_str}]."

    def _handle_interaction(self, subjects: list) -> str:
        """Обработка взаимодействия двух и более сущностей."""
        subj_str = " и ".join(subjects)
        return f"[Lipid-2]: Запущен анализ взаимодействия между [{subj_str}]. (Векторный просчет [K, M, S] симулятора)."

    # ── ДОСТУП ДЛЯ GUI / ХОРМОНЫ ──────────────────────────

    @property
    def noradrenaline(self):
        return self.hormones.norepinephrine

    @property
    def dopamine(self):
        return self.hormones.dopamine