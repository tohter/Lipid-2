import os
import json
import re
import time
from typing import Dict, Any, Optional

# Импортируем вынесенный IR-парсер
from brain.ir import IRParser
# Импортируем Учителей (3 канала знаний)
from teachers.teacher1 import AgentTeacherV2       # Канал 1: локальная LLM (Gemma/llama.cpp)
from teachers.teacher_wiki import WikiTeacher      # Канал 2: интернет (Википедия)
# Импортируем нейроморфное ядро (SDR / STDP / Гормоны)
from core.cortex import Hormones
from core.graph import GraphMemory
from brain.learner import Learner, TeacherDisabled, TeacherUnavailable
from core.sdr import prewarm, encoder_info
from core.lemma import LemmaResolver
from core.logchannel import status as _status, request as _ask
from core.records import (action_name, meaningful_categories, merge_entity,
                          normalize_actions, same_word, strip_self_name)
from core.worldfile import default_memory_dir, read_world, write_world
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
        self.memory_dir, self.world_filepath = default_memory_dir(memory_dir)

        # Модули ввода
        self.ir_parser = IRParser()

        # Три канала знаний (host локалки настраиваемый, по умолчанию 8080)
        llm_host = os.environ.get("LIPID2_LLM_HOST", "http://localhost:8080")
        # Переключатели каналов. Смысл не в экономии, а в ЧЕСТНОСТИ: снятая
        # галочка обязана означать «этот канал не трогает НИЧЕГО», включая путь
        # ответа на вопрос. До этого флаг «спрашивать Учителя» действовал только
        # на обучение, и ответ на незнакомое слово всё равно уходил в Gemma на
        # ~70 с (ЗАМЕРЕНО на GUI-прогоне: 70 с и «ответ без вызова инструмента»).
        self.use_llm_teacher = True
        self.use_wiki_teacher = True
        self.ask_user_enabled = True
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
            # Знания нет, НО каналы могли отказать механически (llama.cpp лёг,
            # HTTP 500, ответ обрезан). Молча вернуть None — значит дать
            # Learner-у право записать asked_empty: «мы искали, там пусто».
            # Исключение долетает до Learner (он его ловит) и превращается в
            # not_asked — слово возвращается в очередь и будет спрошено позже.
            if kinds := getattr(self, "last_learning_failures", None):
                # Тип исключения выбирает ВЫЗЫВАЮЩИЙ, а не мы: «не спрашивали,
                # потому что канал выключен» и «не спросили, потому что сервер
                # лёг» ведут к РАЗНОЙ судьбе слова (см. brain.learner).
                why = "; ".join(m for _k, m in kinds)
                if all(k == "disabled" for k, _m in kinds):
                    raise TeacherDisabled(why)
                raise TeacherUnavailable(why)
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
            # ПРОФИЛЬ [K, M, S] терять нельзя: это единственный носитель
            # физического смысла действия для SDR-ядра. Раньше тут оставалось
            # только имя, и ядро получало «действие есть, какой оно природы —
            # неизвестно». Нормализация по имени — в core.records.
            name = action_name(a)
            if not name:
                continue
            if isinstance(a, dict):
                acts.append({**a, "name": name})
            else:
                acts.append(name)
        acts = normalize_actions(acts)
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
        """
        Загружает текущую оперативную память Вселенной с диска.

        Чтение вынесено в core.worldfile: раньше «битый JSON» и «файла нет»
        давали одинаковый результат — пустой мир. При последующем merge
        («перечитать и слить») это СТИРАЕТ Вселенную, а не дополняет её.
        Теперь битый файл сохраняется рядом и событие громко пишется в stderr.
        """
        return read_world(self.world_filepath)

    def _save_world(self, world: dict):
        """
        Атомарная запись. Открытие на "w" уссекает файл сразу: закрытие окна
        во время обучения оставляло на диске оборванный JSON, то есть потерю
        всей памяти из-за одного нажатия.
        """
        write_world(self.world_filepath, world)

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
        found = None
        if "parent_concept" in entity_data:
            parent_data = entities.get(entity_data["parent_concept"], {})
            found = {"base_entity": parent_data, "derived_entity": entity_data}
        else:
            found = {"base_entity": entity_data, "derived_entity": {}}
        # Плоский срез ЗАПИСИ (kind, participants, action_profile). Ответам о
        # глаголах не хватает именно его: под видом base_entity приходит
        # родитЕЛЬСкая запись, и «что такое гасить» отвечал бы по родителю.
        found["subject"] = entity_data
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
        self.last_learning_failures = []
        # Механические отказы каналов («сервер лёг», «HTTP 500», «обрезано по
        # токенам») И ВЫКЛЮЧЕННЫЕ КАНАЛЫ. И то и другое означает одно: знание
        # НЕ ДОБЫТО НЕ ПОТОМУ, ЧТО ЕГО НЕТ. Их НЕЛЬЗЯ смешивать с «канал
        # работал и не нашёл»: во втором случае слово справедливо получает
        # asked_empty и больше не спрашивается, в первом — обязано остаться в
        # очереди уроков. ЗАМЕРЕНО: при трёх выключенных каналах все слова
        # урока получали asked_empty, то есть снятая галочка превращалась в
        # пожизненный запрет учить слово.
        failures = []

        # ── КАНАЛ 1: Локальный Учитель (LLM) ──
        if not self.use_llm_teacher:
            self._say("канал 1 ВЫКЛЮЧЕН (переключатель «спрашивать Учителя») — "
                      "LLM не вызывается")
            failures.append(("disabled", "канал 1 (локальная модель) выключен"))
        else:
            print("   [Канал 1] Локальный Учитель (Gemma)...")
            self._say(f"канал 1: спрашиваю локальную модель про «{concept}»")
            try:
                got = self.teacher_llm.run_agent(
                    base_concept=concept, derived_concept=concept,
                    filename="lipid_world.json")
                # Учитель с этих пор ВОЗВРАЩАЕТ записанное. Читать мир с диска
                # вместо этого означало бы принять за успех канала запись,
                # которую оставил другой канал или прошлый урок.
                learned_data = got if got else None
                failure = getattr(self.teacher_llm, "last_failure", None)
                if learned_data:
                    self._say(f"канал 1: нашёл «{concept}»")
                    print("   \u2714 Знание получено от локальной LLM.")
                elif failure:
                    failures.append(("broken", f"канал 1: {failure}"))
                    self._say(f"канал 1: ОТКАЗАЛ ({failure}) — это НЕ «не нашли»")
                else:
                    self._say(f"канал 1: НЕ нашёл «{concept}»")
            except Exception as e:
                failures.append(("broken",
                                 f"канал 1: {type(e).__name__}: {str(e)[:120]}"))
                print(f"   \u2716 Локалка недоступна: {e}")
                self._say(f"канал 1: УПАЛ ({type(e).__name__}: {str(e)[:120]})")
                learned_data = None

        # ── КАНАЛ 2: Интернет (Википедия) ──
        if not learned_data and not self.use_wiki_teacher:
            self._say("канал 2 ВЫКЛЮЧЕН — интернет не опрашивается")
            failures.append(("disabled", "канал 2 (Википедия) выключен"))
        if not learned_data and self.use_wiki_teacher:
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
                failures.append(("broken",
                                 f"канал 2: {type(e).__name__}: {str(e)[:120]}"))
                print(f"   \u2716 Интернет недоступен: {e}")
                learned_data = None

        # ── КАНАЛ 3: Вопрос пользователю ──
        if not learned_data and not self.ask_user_enabled:
            self._say("спросить пользователя запрещено (переключатель снят) — "
                      "знание осталось ненайденным")
            failures.append(("disabled", "канал 3 (пользователь) выключен"))
        if not learned_data and self.ask_user_enabled:
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
        # Отказ каналов публикуется ДЛЯ ВСЕХ вызовов (успех тоже его сбрасывает):
        # решение «спрашивали или нет» принимает вызывающий по ПОСЛЕДНЕМУ сеансу.
        self.last_learning_failures = list(failures)
        if learned_data:
            self.hormones.norepinephrine = 20
            self.hormones.dopamine = 255
            self._forget_unknown(concept, learned_data)
            self._say(f"выучено: «{concept}» вошло в граф")
            print(f"🎉 [ДОФАМИН] Успех! Граф пополнен сущностью '{concept}'.\n")
            # Интегрируем новые знания в нейроморфное ядро
            self._integrate_learned_data(learned_data)
        else:
            self.hormones.cortisol = min(255, self.hormones.cortisol + 120)
            if failures:
                # «Каналы пусты» и «каналы не работали» — РАЗНЫЕ новости.
                # Первая значит «мы искали и не нашли», вторая — «мы не искали».
                self._say(f"⚠️ «{concept}»: знание НЕ ДОБЫТО — "
                          + "; ".join(m for _k, m in failures))
            else:
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

        # «Пчела — летаемое насекомое» — полный ответ, но его первая половина
        # повторяет заголовок. Без среза категория остаётся всей фразой, и
        # ответ склеивается из двух имён: ЗАМЕРЕНО в GUI-прогоне —
        # «Пчела — это Пчела — летаемое насекомое».
        category = strip_self_name(concept, parts[0])
        properties = [strip_self_name(concept, p) for p in parts[1:]]
        properties = [p for p in properties if p]

        data = {
            "base_entity": {
                "concept": concept,
                "hierarchy": [category] if category else [],
                "definition": strip_self_name(concept, answer_clean),
                "properties": properties,
                "actions": [],
            },
            "derived_entity": {},
        }

        world = self._load_world()
        key = concept.lower().strip()
        record = dict(data["base_entity"])
        record["provenance"] = [{"source": "user", "evidence": answer[:200],
                                 "at": int(time.time())}]
        record["learned_from"] = "user"
        # merge_entity, а не присваивание: ответ пользователя беднее статьи,
        # и прямое присваивание стирало ранее выученные свойства того же слова
        # (то же правило, что для Википедии и Учителя — см. core/records.py).
        world["entities"][key] = merge_entity(world["entities"].get(key), record)
        world["unknown_words"].pop(key, None)
        self._save_world(world)
        data["matched_concept"] = key
        return data

    def _forget_unknown(self, concept: str, learned_data: dict) -> None:
        """
        Слово получило значение — запись в unknown_words обязана исчезнуть.

        unknown_words — это список НЕЗНАНИЯ. Пока слово стоит там и в
        entities, файл утверждает про одно и то же слово «не знаю» и «вот что
        оно такое», а панель памяти показывает «сущностей 6 · неизвестных 8»
        там, где неизвестных должно быть меньше. Удаляются и формы слова: они
        попали в список тем же поиском, который их теперь находит.
        """
        keys = {concept.lower().strip()}
        base = (learned_data or {}).get("base_entity") or {}
        if base.get("concept"):
            keys.add(str(base["concept"]).lower().strip())
        matched = (learned_data or {}).get("matched_concept")
        if matched:
            keys.add(str(matched).lower().strip())
        try:
            world = self._load_world()
            unknowns = world.setdefault("unknown_words", {})
            doomed = [k for k in unknowns
                      if k in keys or same_word(k, concept)
                      or any(same_word(k, o) for o in keys)]
            for k in doomed:
                del unknowns[k]
            if doomed:
                self._save_world(world)
                self._say(f"убрано из неизвестных: {', '.join(doomed)}")
        except Exception as exc:
            # Не критично для знания, но и не «тихо»: чистка не выполнена.
            self._say(f"⚠️ unknown_words не почистился: {type(exc).__name__}: {exc}")

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

    _DEF_MAX_CHARS = 320

    @staticmethod
    def _definition_phrase(concept: str, definition: str) -> str:
        """
        Сжать определение Википедии в одну фразу, которую можно ПРОЧИТАТЬ.

        ЗАМЕРЕНО на живом определении «экосистема»: дословный перенос первого
        предложения статьи давал
            «Экосистема — Экосистема, или экологическая система (от др.-греч.
             οἶκος — жилище, местопребывание и σύστημα — «целое, составленное
             из частей; соединение») — основная природная единица...»
        То есть ядро повторяло название дважды и вставляло этимологическую
        справку посреди ответа. Это не «красивее сделать»: в таком виде определение
        не читается в окне вообще, а полезная часть тонет на 200-м символе.

        Правила (каждое — по находке, не по вкусу):
          1. выкинуть этимологическую скобку «(от ...)/(...греч...)/(...лат...)»;
          2. если определение НАЧИНАЕТСЯ с самого концепта (в любой форме) —
             отрезать всё до первого « — », иначе ответ = «X — X, ...»;
             отрезаем только если остаток самодостаточен (>= 15 символов,
             иначе есть риск оставить «вот так» вместо определения);
             порог 40 был первой догадкой и ломал короткий ответ
             «Флюгер — прибор для измерения направления ветра»: остаток 39
             символов не проходил, и двойное название возвращалось;
          3. обрезать по границе слова на _DEF_MAX_CHARS с многоточием.
        """
        text = " ".join((definition or "").split())
        if not text:
            return "."
        # 1. этимология
        text = re.sub(r"\(\s*(?:от|из|букв.)[^)]{0,200}\)\s*", " ", text)
        text = re.sub(r"\(\s*(?:др.-греч|лат|греч|нем|англ|фр)[^)]{0,200}\)", " ", text,
                      flags=re.I)
        text = " ".join(text.split())

        # 2. убрать самоповтор в начале — общее правило из core/records.py,
        # то же самое чистит и ответ пользователя (иначе «X — это X, ...»
        # возвращалось с другой стороны: от пользователя, а не из Вики).
        text = strip_self_name(concept, text, min_rest=15)

        # 3. обрезка по границе слова
        if len(text) > Lipid2Brain._DEF_MAX_CHARS:
            head = text[:Lipid2Brain._DEF_MAX_CHARS]
            cut = head.rfind(" ")
            if cut > 120:
                head = head[:cut]
            text = head.rstrip(" ,;:—-") + "…"
        return text if text.endswith((".", "!", "?", "…")) else text + "."

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

        if base_ent.get("kind") == "action":
            return self._describe_action(concept, base_ent)

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
            hierarchy = meaningful_categories(concept, base_ent.get("hierarchy", []))
            definition = str(base_ent.get("definition", "") or "").strip()

            # Приоритет определения над категорией — ЗАМЕРЕНО, а не «архитектурно»:
            # в lipid_world.json из 6 сущностей 4 имели в иерархии форму самого
            # себя ('экосистема' -> 'экосистемы'), потому что категории Вики
            # включают множество того же понятия. Из-за этого на уже скачанную
            # статью окно выдавало «Экосистема — это экосистемы»: ответ-обманка,
            # скрывающий, что у нас в файле лежит готовое определение.
            if definition:
                return (f"[Lipid-2]: {concept.capitalize()} — "
                        + self._definition_phrase(concept, definition))

            if hierarchy:
                if len(hierarchy) == 1:
                    # одиночная категория (например, от пользователя)
                    return f"[Lipid-2]: {concept.capitalize()} — это {hierarchy[0]}."
                # КАТЕГОРИИ Вики ничем не упорядочены, «самый общий корень»
                # из них не выбирается: h[-1] оказался ближайшим к истине
                # только случайно. Берём первую — она ближе всего к предмету.
                return (f"[Lipid-2]: {concept.capitalize()} — это {hierarchy[0]}"
                        + (f" (категории: {' -> '.join(hierarchy)})."
                           if len(hierarchy) > 1 else "."))

            props = base_ent.get("properties", []) or []
            if props:
                return (f"[Lipid-2]: Определения у '{concept}' нет, но известно "
                        f"о свойствах: {', '.join(props)}.")

            # Ни определения, ни категорий, ни свойств. Это НЕ «данных мало»,
            # это «данных нет»: раньше сюда попадали редко, потому что
            # тавтологическая иерархия делала пустоту неотличимой от знания.
            return (f"[Lipid-2]: О '{concept}' в памяти есть только само слово — "
                    f"знания нет. Спроси ещё раз с включённым Учителем "
                    f"или подскажи сам.")

    def _handle_properties(self, concept: str) -> str:
        data = self.find_in_memory(concept)
        if not data:
            data = self._trigger_curiosity_and_learn(concept)
            if not data:
                return f"[Lipid-2]: Не могу определить свойства '{concept}'."

        concept = data.get("matched_concept", concept)
        base_ent = data.get("base_entity", {})
        derived_ent = data.get("derived_entity", {})

        subj = data.get("subject")
        if subj is not None:
            if isinstance(subj, dict) and subj.get("kind") == "action":
                return self._describe_action(concept, subj, properties=True)
        elif base_ent.get("kind") == "action":
            # Спрашивают ГЛАГОЛ. У действия свойств нет по устройству, зато
            # есть участники. Прежний ответ «свойств не записано — только
            # факт, что такое слово существует» был ЛОЖЬЮ: узел знает, что
            # это действие, кто его совершает и над чем, — то есть запись
            # объявлялась пустой там, где знание есть.
            return self._describe_action(concept, base_ent, properties=True)

        props = list(base_ent.get("properties", []) or [])
        if derived_ent.get("concept") == concept:
            props += [p for p in derived_ent.get("delta_properties", []) or []
                      if p not in props]

        props_str = ", ".join(props) if props else "нет данных"
        if not props:
            # «Свойства: [нет данных]» на пустой записи читается как ответ.
            return (f"[Lipid-2]: О '{concept}' свойств не записано — только "
                    f"факт, что такое слово существует.")
        return f"[Lipid-2]: Свойства сущности '{concept}': [{props_str}]."

    @staticmethod
    def _describe_action(concept: str, ent: dict, properties: bool = False) -> str:
        """
        Ответ про ГЛАГОЛ — по-русски и по существу, а не «сущность не найдена».

        Действие описывается НЕ родом и иерархией, а участниками: кто может,
        над чем, где. Если спросили «какие свойства», так и говорим — свойств
        у действия нет, зато есть вот эти связи. Врать про пустоту нельзя:
        запись заполнена, просто у неё ДРУГАЯ размерность.
        """
        part = ent.get("participants") or {}
        agents = [a for a in (part.get("agent") or []) if a]
        objects = [o for o in (part.get("object") or []) if o]
        who = ", ".join(agents[:4]) if agents else "не записано, кто"
        what = ", ".join(objects[:4]) if objects else "не записано, над чем"
        head = (f"[Lipid-2]: «{concept}» — это действие. Совершают: {who}. "
                f"Относится к: {what}.")
        if properties:
            head += " Свойств у действия нет — есть участники и профиль."
        prof = ent.get("action_profile")
        if prof:
            head += f" Профиль K/M/S: {prof}."
        else:
            head += " Профиля K/M/S нет — предиктивный цикл это действие ещё не различает."
        return head

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