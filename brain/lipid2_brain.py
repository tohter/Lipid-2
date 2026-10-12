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
from brain.learner import Learner
from core.sdr import prewarm, encoder_info
from core.lemma import LemmaResolver
from core.logchannel import status as _status, request as _ask
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
        # Резолвер форм по именам ВСЕЛЕННОЙ (мир шире ядра: там есть сущности,
        # ещё не интегрированные в граф).
        self.lemmas_world = LemmaResolver()

        # Загружаем существующие знания мира в граф ядра
        self._sync_world_to_core()
        # Индекс форм строится ОДИН раз после загрузки графа, а не на запрос:
        # кодирование имён — единственная дорогая операция в резолвере.
        self.core.sync_lemmas()

        # ОБУЧЕНИЕ ИЗ ТЕКСТА. Отдельный модуль, а не метод мозга: у него своя
        # ответственность (что извлечь и как записать, не испортив известное),
        # и зависимости он получает функциями — иначе тесты таскали бы
        # llama.cpp и 230 МБ эмбеддингов.
        self.learner = Learner(
            load_world=self._load_world,
            save_world=self._save_world,
            core=self.core,
            known_forms=self._resolve_known_form,
            teacher=self._teacher_entity_for_learner,
        )
        # Связи, выученные из рассказов, живут в lipid_world.json (наблюдения),
        # а не в памяти процесса. Без этого шага обучение длилось бы «до
        # закрытия окна»: граф собирается из JSON и связей в нём нет.
        n = self.learner.replay_observations(self._load_world())
        if n:
            print(f"[Brain]: восстановлено из наблюдений: {n} связей")
            self.core.sync_lemmas()

    def _resolve_known_form(self, word: str):
        """Форма слова -> имя известной сущности (ядро, затем Вселенная)."""
        hit = self.core.resolve_name(word)
        if hit:
            return hit
        return self.lemmas_world.resolve_name(word)

    def _say(self, message: str) -> None:
        """
        Шаг работы мозга в общий канал статуса (core.logchannel).

        В stdout это шло и раньше, но GUI stdout НЕ читает: пока Учитель
        перебирает шаги, окно стоит мёртвым 30-60 с, и это выглядит зависанием.
        """
        _status(message)

    def _teacher_entity_for_learner(self, lemma: str):
        """
        Плоская сущность для Learner из любого канала знаний.
        Учитель говорит «не знаю» иначе, чем падает, и это различие важно:
        провал — повод заметить проблему, «не знаю» — нормальный ответ.
        """
        found = self.find_in_memory(lemma)
        if found:
            ent = found.get("derived_entity") or found.get("base_entity") or {}
            if ent:
                return self._flatten_entity(ent)
        data = self._trigger_curiosity_and_learn(lemma)
        if not data:
            return None
        ent = data.get("derived_entity") or data.get("base_entity") or {}
        return self._flatten_entity(ent) if ent else None

    @staticmethod
    def _flatten_entity(ent: dict) -> dict:
        """
        ЗАПИСЬ УЧИТЕЛЯ -> ПЛОСКАЯ Сущность.
        Формат lipid_world.json дублируется в teachers/teacher1.py
        (base/derived + delta_properties), teachers/teacher_wiki.py (свойства
        вида 'текст (N)'), _learn_from_user и core/graph.py. Читать только
        hierarchy/properties — значит тихо терять delta_properties и полные
        тексты Вики. Одна нормализация на все четыре формата.
        """
        import re as _re
        props = list(ent.get("properties") or [])
        for extra in (ent.get("delta_properties") or []):
            if extra and extra not in props:
                props.append(extra)
        props = [_re.sub(r"\s*\(\d+\)$", "", str(p)).strip() for p in props
                 if str(p).strip()]
        if ent.get("summary") and ent["summary"] not in props:
            props.append(ent["summary"])
        acts = []
        for a in (ent.get("actions") or []):
            name = a if isinstance(a, str) else (a or {}).get("name", "")
            if name:
                acts.append(name)
        hierarchy = list(ent.get("hierarchy") or [])
        parent = (ent.get("parent_concept") or "").lower().strip()
        if parent and parent != ent.get("concept", "").lower().strip() \
                and parent not in hierarchy:
            hierarchy.insert(0, parent)
        return {"hierarchy": hierarchy, "properties": props, "actions": acts}

    # ── СИНХРОНИЗАЦИЯ МИРА И ЯДРА ──────────────────────────

    def _sync_world_to_core(self):
        """Переносит сущности из lipid_world.json в нейроморфный граф ядра."""
        world = self._load_world()
        entities = world.get("entities", {})

        # Сначала имена пачкой в SDR-кэш: без этого 10^4 узлов = 10^4 обращений
        # к эмбеддинг-модели по одному слову.
        names = set(entities.keys())
        for data in entities.values():
            names.update(data.get("hierarchy", []) or [])
            names.update(data.get("properties", []) or [])
            for act in data.get("actions", []) or []:
                if isinstance(act, dict):
                    names.add(act.get("name", ""))
                elif act:
                    names.add(str(act))
            if data.get("parent_concept"):
                names.add(data["parent_concept"])
        names = sorted(n for n in names if isinstance(n, str) and n)
        prewarm(names)
        # Мир шире ядра: в нём есть сущности, ещё не ставшие узлами. Резолвер
        # мира нужен, чтобы «свойства флюгера» нашло сущность «флюгер», даже
        # если узел ещё не интегрирован.
        self.lemmas_world.attach(sorted(entities.keys()))

        count = 0
        for concept, data in entities.items():
            self._integrate_entity_into_core(concept, data)
            count += 1
        if count:
            enc_name, semantic, cfg = encoder_info()
            print(f"[Lipid-2] Ядро загружено: {count} сущностей из Вселенной. "
                  f"SDR-кодировщик: {enc_name} (семантический={semantic}, cfg={cfg}).")

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
        """
        Быстрый поиск сущности и её родительских связей в Графе Вселенной.

        Ищем НЕ только по точной строке. Раньше «свойства флюгера» промахивалось
        по узлу «флюгер» — сравнение было по точному ключу, хотя семантический
        кодировщик эти формы не различает. Следствие: каждый падеж каждого слова
        выглядел как новая сущность и гнал повторный (платный, медленный) запрос
        к LLM за копией уже имеющегося знания.

        Порядок: точное имя → морфологическое сведение через ядро. Создание узлов
        здесь не производится: промах резолвера обратим, слияние узлов — нет.
        """
        concept_clean = concept.lower().strip()
        world = self._load_world()
        entities = world.get("entities", {})

        if concept_clean not in entities:
            # 1) форма слова, которую ядро уже знает
            hit = self.core.resolve_name(concept_clean)
            if hit and hit in entities:
                concept_clean = hit
            else:
                # 2) форма другого слова ВСЕЛЕННОЙ, которую ядро ещё не видело
                hit = self.lemmas_world.resolve_name(concept_clean)
                if hit and hit in entities:
                    concept_clean = hit

        if concept_clean not in entities:
            return None

        entity_data = entities[concept_clean]
        if "parent_concept" in entity_data:
            parent_data = entities.get(entity_data["parent_concept"], {})
            found = {"base_entity": parent_data, "derived_entity": entity_data}
        else:
            found = {"base_entity": entity_data, "derived_entity": {}}
        # КАНОННОЕ имя, под которым знание реально лежит. Обработчики обязаны
        # сверяться с НИМ, а не со словом из запроса: иначе «что такое кошки»
        # находит запись «кошка», потом сравнивает "кошка" == "кошки", не
        # совпадает — и отвечает «недостаточно данных», хотя данные только что
        # достал. Именно эта несостыковка и выдавала пустые определения.
        found["matched_concept"] = concept_clean
        return found

    # ── ТРЁХКАНАЛЬНОЕ САМООБУЧЕНИЕ ─────────────────────────

    def _trigger_curiosity_and_learn(self, concept: str) -> Optional[Dict[str, Any]]:
        """
        Спайк Норадреналина: реакция на неизвестное слово.
        Пробует источники по порядку: 1) локалка, 2) интернет, 3) пользователь.
        Первый успешный источник даёт знания, которые записываются в мир и ядро.
        """
        print(f"\n⚡ [НОРАДРЕНАЛИН] Внимание! Неизвестная сущность '{concept}'!")
        print(f"🧠 [Lipid-2]: Мой граф не содержит '{concept}'. Ищу знание...")
        self._say(f"не знаю «{concept}» — ищу знание в трёх каналах")

        self.hormones.norepinephrine = 255
        self.hormones.dopamine = 10

        learned_data = None

        # ── КАНАЛ 1: Локальный Учитель (LLM) ──
        print("   [Канал 1] Локальный Учитель (Gemma)...")
        self._say(f"канал 1: спрашиваю локальную модель про «{concept}»")
        try:
            self.teacher_llm.run_agent(base_concept=concept, derived_concept=concept,
                                       filename="lipid_world.json")
            learned_data = self.find_in_memory(concept)
            self._say(f"канал 1: {'нашёл' if learned_data else 'НЕ нашёл'} «{concept}»")
            if learned_data:
                print("   \u2714 Знание получено от локальной LLM.")
        except Exception as e:
            print(f"   \u2716 Локалка недоступна: {e}")
            learned_data = None

        # ── КАНАЛ 2: Интернет (Википедия) ──
        if not learned_data:
            print("   [Канал 2] Интернет (Википедия)...")
            self._say(f"канал 1 пуст — ищу в интернете «{concept}»")
            try:
                wiki = self.teacher_wiki.fetch_concept(concept)
                if wiki and wiki.get("base_entity", {}).get("concept"):
                    if self.teacher_wiki.write_to_world(wiki):
                        learned_data = self.find_in_memory(concept)
                self._say(f"канал 2: {'нашёл' if learned_data else 'НЕ нашёл'} "
                          f"«{concept}» в Википедии")
                if learned_data:
                    print("   \u2714 Знание получено из интернета.")
            except Exception as e:
                print(f"   \u2716 Интернет недоступен: {e}")
                learned_data = None

        # ── КАНАЛ 3: Вопрос пользователю ──
        if not learned_data:
            print("   [Канал 3] Вопрос пользователю...")
            self._say(f"каналы знания пусты — спрашиваю у пользователя «{concept}»")
            # ДВА пути спросить: явный колбэк (исторический API) и общий
            # обратный канал core.logchannel. Второй нужен, чтобы вопрос дошёл
            # до пользователя даже там, где колбэк забыли подключить, — иначе
            # «спросить пользователя» остаётся обработчиком на бумаге.
            user_answer = None
            if self.ask_user_callback is not None:
                user_answer = self.ask_user_callback(concept)
            if not user_answer:
                user_answer = _ask(f"Что такое «{concept}»? "
                                   "Напиши коротко: категория; свойства; действия.")
            if user_answer:
                learned_data = self._learn_from_user(concept, user_answer)
                if learned_data:
                    print("   \u2714 Знание получено от пользователя.")
                    self._say(f"записано с твоих слов: «{concept}»")
            else:
                print("   \u2716 Пользователь не дал знания.")

        # ── Результат ──
        if learned_data:
            self.hormones.norepinephrine = 20
            self.hormones.dopamine = 255
            self._say(f"выучено: «{concept}» вошло в граф")
            print(f"🎉 [ДОФАМИН] Успех! Граф пополнен сущностью '{concept}'.\n")
            # Интегрируем новые знания в нейроморфное ядро
            self._integrate_learned_data(learned_data)
        else:
            self.hormones.cortisol = min(255, self.hormones.cortisol + 120)
            self._say(f"⚠️ «{concept}» осталось неизвестным: все каналы пусты")
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
        key = concept.lower().strip()
        world["entities"][key] = data["base_entity"]
        self._save_world(world)
        data["matched_concept"] = key
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

    # ── ОБУЧЕНИЕ ───────────────────────────────────────────

    def learn_from_text(self, text: str, source: str = "text",
                        allow_teacher: bool = True):
        """
        Learning mode: текст -> знания в графе + измеримый отчёт.
        Отчёт обязателен: без чисел нельзя сказать, стала ли система умнее
        после правки правил, а «выглядит работающим» — не измерение.
        """
        rep = self.learner.learn_from_text(text, source=source,
                                           allow_teacher=allow_teacher)
        # Учились — значит мир изменился: резолвер форм Вселенной обязан
        # увидеть новые имена, иначе следующее предложение спросит LLM про
        # уже выученное слово.
        self.lemmas_world.attach(sorted(self._load_world().get("entities", {})))
        if rep.learned_concepts:
            self.hormones.dopamine = 255
        if rep.unknown:
            self.hormones.cortisol = min(255, self.hormones.cortisol + 40)
        return rep

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

        # Дальше работаем с КАНОННИКОМ, а не с формой из запроса.
        concept = data.get("matched_concept", concept)
        base_ent = data.get("base_entity", {})
        derived_ent = data.get("derived_entity", {})

        # degenerate derived: LLM очень любит на вопрос «что такое X» возвращать
        # derived_concept == base_concept. Тогда производная запись несёт ТОЛЬКО
        # обрывок (parent_rel, пустой delta_properties), а настоящая иерархия
        # лежит в base. Прежний порядок «сначала derived» выдавал «Флюгер — это
        # флюгер со свойствами: .» — то есть честный провал, замаскированный под
        # ответ. Производную ветку смотрим ТОЛЬКО когда это действительно
        # производная запись (другое слово либо непустая дельта).
        real_derived = (derived_ent.get("concept") == concept
                        and base_ent.get("concept") != concept)

        if real_derived:
            parent = derived_ent.get("parent_concept") or base_ent.get("concept")                 or "базовый вид"
            delta = ", ".join(derived_ent.get("delta_properties", []))
            if delta:
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

        concept = data.get("matched_concept", concept)
        base_ent = data.get("base_entity", {})
        derived_ent = data.get("derived_entity", {})

        props = list(base_ent.get("properties", []) or [])
        if derived_ent.get("concept") == concept:
            props += [p for p in derived_ent.get("delta_properties", []) or []
                      if p not in props]

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