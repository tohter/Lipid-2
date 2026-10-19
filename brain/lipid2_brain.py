import os
import json
import re
import time
from typing import Dict, Any, List, Optional

# Импортируем вынесенный IR-парсер
from brain.ir import IRParser
# Импортируем Учителей (3 канала знаний)
from teachers.teacher1 import AgentTeacherV2       # Канал 1: локальная LLM (Gemma/llama.cpp)
from teachers.teacher_wiki import WikiTeacher      # Канал 2: интернет (Википедия)
# Импортируем нейроморфное ядро (SDR / STDP / Гормоны)
from core.cortex import Hormones, PredictiveCortex
from core.graph import GraphMemory
from brain.learner import Learner, TeacherDisabled, TeacherUnavailable
from core.transitions import (TransitionTable, merge_json,
                                  table_from_world)
from core.sdr import prewarm, encoder_info
from core.lemma import LemmaResolver
from core.logchannel import status as _status, request as _ask
from core.records import (action_name, meaningful_categories, merge_entity,
                          normalize_actions, observation_weight, same_word,
                          strip_self_name)
from core.worldfile import default_memory_dir, read_world, write_world, world_transaction
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

    # Уверенность по каналам (решено хозяином 14.10.26):
    #   user   = 1.00 — человек отвечает прямо
    #   wiki   = 0.85 — проверяемая энциклопедия
    #   llm    = 0.75 — локальная модель, может выдумать
    #   text   = 0.60 — извлечено из рассказа без учителя
    CHANNEL_CONFIDENCE = {"user": 1.0, "wiki": 0.85, "llm": 0.75, "text": 0.6}

    def __init__(self, memory_dir: str = None):
        self.memory_dir, self.world_filepath = default_memory_dir(memory_dir)
        # Кэш мира (очередь, п.13): мир + сигнатура файла, к которой он привязан.
        self._world_cache: Optional[Dict[str, Any]] = None
        self._world_sig = None

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

        # СЧЁТЧИКИ ОБРАЩЕНИЙ К КАНАЛАМ (сбрасываются на каждый сеанс обучения).
        # Зачем: без них утверждение «знание пришло из Gemma» НЕЧЕМ ПРОВЕРИТЬ
        # машиной. ЗАМЕРЕНО: сквозной e2e рапортовал «выучил и вспомнил» даже
        # при ВЫКЛЮЧЕННОМ LLM — цикл проверялся, канал нет. Число обращений и
        # есть доказательство, что канал работали, а не что тест подставил
        # ответ.
        self.channel_calls = {"llm": 0, "wiki": 0, "user": 0}
        self.channel_found = {"llm": 0, "wiki": 0, "user": 0}

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

        # ── ПРЕДИКТИВНЫЙ ЦИКЛ (core/cortex.py) ──────────────────────────
        # Тот же принцип, что у переключателей каналов: выключенный цикл не
        # должен трогать НИЧЕГО, включая цифры отчёта. Молча работающий
        # «экспериментальный модуль» в GUI — способ получить числа, которые
        # никто не просил и никто не проверяет.
        self.learn_experience = True
        # Таблица переходов — ЕДИНСТВЕННАЯ запись, которую цикл имеет право
        # создавать; она же и всё его долговременное состояние. До этого опыт
        # писался в веса синапсов и терялся при перезапуске (замер — в
        # docstring core/cortex.py, пункты D и E).
        self.transitions = table_from_world(self._load_world())
        # Гормоны ПЕРЕДАЮТСЯ, а не создаются заново: иначе GUI рисует шкалу
        # мозга, а цикл живёт со своей, и «дофамин 255» в окне ничего не
        # говорит о том, что происходило в цикле.
        self.cortex = PredictiveCortex(self.core,
                                      transitions=self.transitions,
                                      hormones=self.hormones)
        # ── СЧЁТЧИК СНА (очередь п.23-доработка) ───────────────────────
        # Считать надо ЗДЕСЬ, а не внутри sleep_now: вопрос «пора ли спать»
        # задаёт вызывающий, а не сам сон.
        # Считаются ТИКИ ЦИКЛА с прошлого сна, а не сила переходов — почему,
        # разобрано в brain.sleep.sleep_due (порог по силе закрыл бы дверь:
        # опыт доращивает предсказание, а сон этот опыт точит).
        self.ticks_since_sleep = 0
        self.sleep_enabled = True
        self.sleep_auto = True

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
            # При загрузке берём прovенанс из записи, если он там есть.
            # record=False: наблюдения УЖЕ в мире — перезапись не нужна.
            self._integrate_entity_into_core(
                concept, data,
                confidence=data.get("confidence"),
                evidence=data.get("definition") or data.get("evidence"),
                source=data.get("source") or (data.get("learned_from") and
                                               f"teacher:{data['learned_from']}"),
                record=False)
            count += 1
        if count:
            enc_name, semantic, cfg = encoder_info()
            print(f"[Lipid-2] Ядро загружено: {count} сущностей из Вселенной. "
                  f"SDR-кодировщик: {enc_name} (семантический={semantic}, cfg={cfg}).")

    def _integrate_entity_into_core(self, concept: str, data: dict, confidence: float = None, evidence: str = None, source: str = None, record: bool = True):
        """Записывает одну сущность из мира в граф ядра (SDR-узел + синапсы).
        
        confidence: уверенность 0..1 (LLM=0.75, Wiki=0.85, User=1.0)
        evidence: текст-основание (фрагмент ответа)
        source: источник ("teacher:llm", "teacher:wiki", "user", "text:<tag>")
        record: писать ли наблюдение в observations (False при загрузке мира)
        """
        if not concept:
            return
        root = self.core.get_or_create_node(concept, NodeType.ENTITY)

        # Уверенность по умолчанию для обратной совместимости
        if confidence is None:
            confidence = 0.5
        # Переводим уверенность в вес синапса (как в learner.py)
        # CAN_DO/HAS_PROPERTY: 90 + 130*conf (уверенное — до 220)
        # IS_A (иерархия): 100 + 100*conf (уверенное — до 200)
        weight_prop = int(90 + 130 * confidence)
        weight_hier = int(100 + 100 * confidence)
        weight_action = int(80 + 120 * confidence)

        # Provenance для узла
        if source:
            root.source = source
        if evidence:
            root.evidence = evidence[:200]

        # Иерархия (IS_A) — связываем концепт с ближайшей родительской категорией
        hierarchy = data.get("hierarchy", [])
        if hierarchy:
            parent = self.core.get_or_create_node(hierarchy[0], NodeType.ENTITY)
            root.connect_to(parent, rel_type=RelationType.IS_A, weight=weight_hier, source=source)

        # Свойства (HAS_PROPERTY)
        for prop in data.get("properties", []):
            pnode = self.core.get_or_create_node(prop, NodeType.PROPERTY)
            root.connect_to(pnode, rel_type=RelationType.HAS_PROPERTY, weight=weight_prop, source=source)

        # Действия (CAN_DO) + профиль [K,M,S] на узле действия
        for act in data.get("actions", []):
            aname = act["name"] if isinstance(act, dict) else str(act)
            anode = self.core.get_or_create_node(aname, NodeType.ACTION)
            if isinstance(act, dict) and "action_profile" in act:
                anode.action_profile = list(act["action_profile"])
            root.connect_to(anode, rel_type=RelationType.CAN_DO, weight=weight_action, source=source)

        # Дельта-наследование (производный узел с родителем)
        parent_concept = data.get("parent_concept")
        if parent_concept:
            parent = self.core.get_or_create_node(parent_concept, NodeType.ENTITY)
            root.connect_to(parent, rel_type=RelationType.IS_A, weight=weight_hier, source=source)

        # Записываем наблюдение в observations (как в Learner)
        if record:
            self._record_observation(concept, data, confidence, evidence, source)

    def _record_observation(self, concept: str, data: dict, confidence: float,
                            evidence: str, source: str):
        """Записывает наблюдение (evidence + confidence) в world["observations"].

        Наблюдения — это то, ЧТО МЫ ВИДЕЛИ и насколько были уверены. Они
        отличаются от entities (что мы ЗНАЕМ). Без наблюдений нельзя отличить
        факт из Вики (одно свидетельство) от факта, подтверждённого трижды.
        """
        key = f"{concept}|curiosity"
        now = int(time.time())
        prov_entry = {"source": source or "unknown",
                      "evidence": (evidence or "")[:200],
                      "at": now}
        # Транзакция: чтение и запись под ОДНИМ замком. Иначе фоновое обучение
        # и Учитель-агент могли пересечься между чтением и записью — и один
        # снимок молча стёр правки другого.
        with world_transaction(self.world_filepath) as world:
            observations = world.setdefault("observations", {})
            obs = observations.get(key)
            if obs:
                # Повторное свидетельство: support растёт, confidence берём максимум
                cur_conf = max(float(obs.get("confidence", 0)), confidence)
                supports = int(obs.get("support", 1)) + 1
                prov = obs.setdefault("provenance", [])
                prov.append(prov_entry)
                observations[key] = {"confidence": cur_conf, "support": supports,
                                     "provenance": prov}
            else:
                observations[key] = {"confidence": confidence, "support": 1,
                                     "provenance": [prov_entry]}

    # ── РАБОТА С ВСЕЛЕННОЙ ─────────────────────────────────

    def _load_world(self) -> Dict[str, Any]:
        """
        Загружает текущую оперативную память Вселенной с диска.

        Чтение вынесено в core.worldfile: раньше «битый JSON» и «файла нет»
        давали одинаковый результат — пустой мир. При последующем merge
        («перечитать и слить») это СТИРАЕТ Вселенную, а не дополняет её.
        Теперь битый файл сохраняется рядом и событие громко пишется в stderr.

        КЭШ (очередь, п.13). При 500+ сущностях разбор JSON на каждый запрос —
        50–200 мс на вопрос, и «что такое кошка» заметно тормозит. Кэш привязан
        к СИГНАТУРЕ файла (mtime_ns + size): чужая запись (Учитель в другом
        потоке) меняет сигнатуру — и следующее чтение перечитает диск. Без
        сигнатуры кэш означал бы «не вижу чужих правок».
        """
        try:
            st = os.stat(self.world_filepath)
            sig = (st.st_mtime_ns, st.st_size)
        except OSError:
            # Файла ещё нет — кэш невалиден, читаем как обычно
            self._world_cache = None
            self._world_sig = None
            return read_world(self.world_filepath)
        if self._world_cache is not None and self._world_sig == sig:
            return self._world_cache
        world = read_world(self.world_filepath)
        self._world_cache = world
        self._world_sig = sig
        return world

    def _save_world(self, world: dict):
        """
        Атомарная запись. Открытие на "w" уссекает файл сразу: закрытие окна
        во время обучения оставляло на диске оборванный JSON, то есть потерю
        всей памяти из-за одного нажатия.

        После записи кэш обновляется СРАЗУ: иначе следующий _load_world()
        сравнил бы сигнатуру, увидел свежий mtime и перечитал файл — то есть
        запись обесценила бы кэш (и мы платили бы за разбор дважды).
        """
        write_world(self.world_filepath, world)
        try:
            st = os.stat(self.world_filepath)
            self._world_cache = world
            self._world_sig = (st.st_mtime_ns, st.st_size)
        except OSError:
            self._world_cache = None
            self._world_sig = None

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
            # Проверка: есть ли узел в графе (из observations), но нет в entities
            graph_node = self.core.resolve_name(concept_clean)
            if graph_node and graph_node in self.core.nodes:
                return self._build_from_observations(concept_clean, world)
            return None

        entity_data = entities[concept_clean]
        # РЕКУРСИВНЫЙ ОБХОД ИЕРАРХИИ (очередь, п.12).
        # Раньше читался ОДИН уровень parent_concept: у «беркута» брался «орёл»,
        # а «орёл» уже не поднимался к «птице» — и свойства с глубины >1 молча
        # пропадали. Теперь базой становится БЛИЖАЙШИЙ ПРЕДОК с записью, а
        # унаследованные свойства/действия и полная цепь категорий собираются
        # до КОРНЯ (с защитой от циклов X→Y→X).
        chain = self._ancestor_chain(concept_clean, entities)
        base_data = entities.get(chain[-1], {}) if chain else entity_data
        found = None
        if chain:
            found = {"base_entity": base_data, "derived_entity": entity_data}
        else:
            found = {"base_entity": entity_data, "derived_entity": {}}
        # Наследование на глубину >1: свойства и действия предков ДОБАВЛЯЮТСЯ
        # к своим (не заменяют), а цепь категорий — склейка hierarchy по всей
        # линии. Без этого «беркут» не знал бы, что он птица.
        inherited = self._inherit_ancestors(concept_clean, entities)
        if inherited:
            merged_base = dict(base_data)
            props = list(merged_base.get("properties") or [])
            for p in inherited.get("properties", []):
                if p not in props:
                    props.append(p)
            merged_base["properties"] = props
            acts = list(merged_base.get("actions") or [])
            acts.extend(inherited.get("actions", []))
            merged_base["actions"] = acts
            hier = list(merged_base.get("hierarchy") or [])
            for h in inherited.get("hierarchy", []):
                if h not in hier:
                    hier.append(h)
            merged_base["hierarchy"] = hier
            found["base_entity"] = merged_base
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
        found["ancestors"] = chain
        return found

    def _ancestor_chain(self, concept: str, entities: dict) -> list:
        """Цепочка предков от НЕПОСРЕДСТВЕННОГО к КОРНЮ (без самого concept).

        Идёт по parent_concept, с защитой от циклов и от регистра ключей
        (в world ключи иногда хранятся в другом регистре, чем ссылка).
        Возвращает [] если предков нет.
        """
        def _lookup(name: str):
            if name in entities:
                return name
            low = str(name).lower().strip()
            for key in entities:
                if str(key).lower().strip() == low:
                    return key
            return None

        chain = []
        seen = {str(concept).lower().strip()}
        cur = entities.get(concept) or {}
        while True:
            parent = str(cur.get("parent_concept") or "").lower().strip()
            if not parent or parent in seen:
                break
            key = _lookup(parent)
            if key is None:
                break
            seen.add(parent)
            chain.append(key)
            cur = entities[key]
        return chain

    def _inherit_ancestors(self, concept: str, entities: dict) -> dict:
        """Собирает унаследованное с ВСЕЙ линии предков (глубина >1).

        Свойства/действия объединяются без дублей, hierarchy — склейка всех
        цепей. Возвращает {} если наследовать нечего.
        """
        chain = self._ancestor_chain(concept, entities)
        if not chain:
            return {}
        props, acts, hier = [], [], []
        for key in chain:
            flat = self._flatten_entity(entities.get(key) or {})
            for p in flat.get("properties", []):
                if p not in props:
                    props.append(p)
            for a in flat.get("actions", []):
                name = action_name(a) or (a if isinstance(a, str) else "")
                if not name:
                    continue
                if not any((action_name(x) or x) == name for x in acts):
                    acts.append(a)
            for h in flat.get("hierarchy", []):
                if h not in hier:
                    hier.append(h)
        return {"properties": props, "actions": acts, "hierarchy": hier}

    def _build_from_observations(self, concept: str, world: dict) -> Optional[Dict[str, Any]]:
        """
        Собирает псевдо-сущность из наблюдений (observations), когда в entities
        записи нет, а в графе узел есть (восстановлен через replay_observations).
        Это позволяет отвечать на вопросы про знания, выученные ТОЛЬКО из текстов,
        без участия Учителя.
        """
        obs = world.get("observations", {})
        # Ищем наблюдения где concept — head (субъект)
        actions = []
        properties = []
        for key, v in obs.items():
            if v.get("negated"):
                continue
            try:
                head, rel, tail = key.split("|", 2)
            except ValueError:
                continue
            if head != concept:
                continue
            if rel == "CAN_DO":
                actions.append(tail)
            elif rel == "HAS_PROPERTY":
                properties.append(tail)
        # Если нет наблюдений вообще — ничего не возвращаем
        if not actions and not properties:
            return None
        # Строим псевдо-entity как в answer_what_does
        pseudo_entity = {
            "concept": concept,
            "kind": "entity",
            "properties": properties,
            "actions": actions,
            "hierarchy": [],
            "definition": "",
            "source": "observation",
        }
        return {
            "base_entity": pseudo_entity,
            "derived_entity": {},
            "subject": pseudo_entity,
            "matched_concept": concept,
            "from_observations": True,
        }

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
                self.channel_calls["llm"] += 1
                got = self.teacher_llm.run_agent(
                    base_concept=concept, derived_concept=concept,
                    filename="lipid_world.json")
                # Учитель с этих пор ВОЗВРАЩАЕТ записанное. Читать мир с диска
                # вместо этого означало бы принять за успех канала запись,
                # которую оставил другой канал или прошлый урок.
                learned_data = got if got else None
                failure = getattr(self.teacher_llm, "last_failure", None)
                if learned_data:
                    self.channel_found["llm"] += 1
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
                self.channel_calls["wiki"] += 1
                wiki = self.teacher_wiki.fetch_concept(concept)
                if wiki and wiki.get("base_entity", {}).get("concept"):
                    if self.teacher_wiki.write_to_world(wiki):
                        learned_data = self.find_in_memory(concept)
                        if learned_data:
                            self.channel_found["wiki"] += 1
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
            self.channel_calls["user"] += 1
            if self.ask_user_callback is not None:
                user_answer = self.ask_user_callback(concept)
            if not user_answer:
                user_answer = _ask(f"Что такое «{concept}»? "
                                   "Напиши коротко: категория; свойства; действия.")
            if user_answer:
                learned_data = self._learn_from_user(concept, user_answer)
                if learned_data:
                    self.channel_found["user"] += 1
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

        key = concept.lower().strip()
        record = dict(data["base_entity"])
        record["provenance"] = [{"source": "user", "evidence": answer[:200],
                                 "at": int(time.time())}]
        record["learned_from"] = "user"
        # merge_entity, а не присваивание: ответ пользователя беднее статьи,
        # и прямое присваивание стирало ранее выученные свойства того же слова
        # (то же правило, что для Википедии и Учителя — см. core/records.py).
        # Транзакция защищает от гонки с фоновым обучением.
        with world_transaction(self.world_filepath) as world:
            world["entities"][key] = merge_entity(world["entities"].get(key), record)
            world["unknown_words"].pop(key, None)
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
            doomed = []
            with world_transaction(self.world_filepath) as world:
                unknowns = world.setdefault("unknown_words", {})
                doomed = [k for k in unknowns
                          if k in keys or same_word(k, concept)
                          or any(same_word(k, o) for o in keys)]
                for k in doomed:
                    del unknowns[k]
            if doomed:
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
        # Определяем канал и уверенность (единая логика для всех источников)
        conf, evidence, src = self._channel_provenance(base)
        self._integrate_entity_into_core(concept, base, confidence=conf,
                                         evidence=evidence, source=src)

        derived = learned_data.get("derived_entity", {})
        if derived and derived.get("concept"):
            conf, evidence, src = self._channel_provenance(derived)
            self._integrate_entity_into_core(derived["concept"].lower().strip(),
                                             derived, confidence=conf,
                                             evidence=evidence, source=src)

    def _channel_provenance(self, data: dict):
        """Возвращает (confidence, evidence, source) по каналу-источнику.

        Правило одно на все каналы — отсюда и название. Раньше три пути
        обучения писали разные веса (160 без confidence, 180, 150), и один
        и тот же факт весил по-разному в зависимости от того, какой канал
        его добыл. Теперь канал даёт только уверенность, а вес считает
        _integrate_entity_into_core.
        """
        raw_src = str(data.get("source") or data.get("learned_from") or "llm").lower()
        # Нормализуем имена каналов
        if raw_src in ("wikipedia", "wiki", "teacher:wiki"):
            channel = "wiki"
        elif raw_src in ("user", "teacher:user", "пользователь"):
            channel = "user"
        elif raw_src in ("text", "teacher:text"):
            channel = "text"
        else:
            channel = "llm"
        conf = float(data.get("confidence") or self.CHANNEL_CONFIDENCE.get(channel, 0.6))
        evidence = data.get("definition") or data.get("evidence") or ""
        return conf, evidence, f"teacher:{channel}"

    # ── ОБУЧЕНИЕ ───────────────────────────────────────────

    def learn_from_text(self, text: str, source: str = "text",
                        allow_teacher: bool = True):
        """
        Learning mode: текст -> знания в графе + измеримый отчёт.
        Отчёт обязателен: без чисел нельзя сказать, стала ли система умнее
        после правки правил, а «выглядит работающим» — не измерение.
        """
        # Сеанс обучения начинается — счётчики каналов обнуляются, иначе
        # «нашёл 2» окажется суммой за всё время жизни процесса и ничего не
        # будет значить.
        self.channel_calls = {"llm": 0, "wiki": 0, "user": 0}
        self.channel_found = {"llm": 0, "wiki": 0, "user": 0}
        rep = self.learner.learn_from_text(text, source=source,
                                           allow_teacher=allow_teacher)
        rep.channel_calls = dict(self.channel_calls)
        rep.channel_found = dict(self.channel_found)
        # Опыт того же текста наблюдает предиктивный цикл. Отдельным шагом, а
        # не внутри Learner'а: у них РАЗНЫЕ права. Learner пишет знания во
        # Вселенную, цикл считает ожидания и не имеет права создавать ни узлов,
        # ни фактов — смешав это, нельзя было бы разобраться, откуда взялась
        # запись (тот же дефект, что чинили со счётчиками каналов).
        rep.experience = self.experience_from_text(text)
        # Учились — значит мир изменился: резолвер форм Вселенной обязан
        # увидеть новые имена, иначе следующее предложение спросит LLM про
        # уже выученное слово.
        self.lemmas_world.attach(sorted(self._load_world().get("entities", {})))
        if rep.learned_concepts:
            self.hormones.dopamine = 255
        if rep.unknown:
            self.hormones.cortisol = min(255, self.hormones.cortisol + 40)
        return rep

    # ── ПРЕДИКТИВНЫЙ ЦИКЛ: ОПЫТ И ЕГО СОХРАНЕНИЕ ───────────

    def experience_from_text(self, text: str, save: bool = True) -> Dict[str, Any]:
        """
        Пропустить текст через цикл «предсказание -> ошибка» и сохранить опыт.

        Возвращает метрики сеанса, а не молчит: точность цикла обязана быть
        видна рядом с точностью узнавания, иначе «система предсказывает»
        останется unverifiable утверждением. Отдельно от learn_from_text —
        чтобы вызывающий мог наблюдать и текст, который НЕ учит (вопрос
        пользователя тоже опыт).
        """
        if not self.learn_experience:
            return {"disabled": True}
        metrics = self.cortex.process_stream(text, verbose=False)
        if save and metrics.get("ticks"):
            self._save_transitions()
        # СОН ПОСЛЕ УРОКА (очередь п.23-доработка). Право «править вес онтологии
        # по опыту» есть только у brain/sleep.py, и вызывать его должен ВЛАДЕЛЕЦ
        # опыта, иначе сон так и остался бы функцией, которую никто не будит
        # (замер этой сессии: вызовов sleep из кода — 0).
        # Цикл при этом остаётся НАБЛЮДАТЕЛЕМ: сон — отдельный шаг после
        # записи опыта, а не действие внутри цикла (tests/
        # test_cortex_experience_contract.py закрывает циклу доступ к графу).
        self.ticks_since_sleep += int(metrics.get("ticks") or 0)
        if save:
            metrics["sleep"] = self.sleep_if_due()
        # СИЛА РАСХОДУЕТСЯ НЕ МОЛЧА (очередь п.25): расхождение онтологии и
        # опыта обязано быть видно сразу, а не обнаруживаться через неделю
        # по странным ответам. Мы НЕ правим вес здесь (у цикла нет прав
        # записи) — мы называем противоречие вслух.
        if metrics.get("ticks"):
            conflicts = self.experience_conflicts()
            if conflicts:
                metrics["conflicts"] = len(conflicts)
                worst = conflicts[0]
                self._say(f"⚠️ опыт ⟷ онтология: расхождений {len(conflicts)}; "
                          f"напр. «{worst['head']}→{worst['tail']}» "
                          f"(вес {worst['weight']}, опыт {worst['strength']}/15)")
            else:
                metrics["conflicts"] = 0
        return metrics

    def sleep_if_due(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """
        Спать, ПОРА (или force). Возвращает результат sleep_now либо None.

        Здесь нет «попробовать и промолчать»: сон либо отработал и назван в
        отчёте, либо не наступил, и причина (накоплено X тиков из Y) видна
        вызывающему. None возвращается в двух РАЗНЫХ случаях, и они
        различимы для читающего код: сон выключен и сон ещё не созрел.
        """
        if not self.sleep_enabled or not (force or self.sleep_auto):
            return None
        from brain.sleep import sleep_due
        if not force and not sleep_due(self.ticks_since_sleep):
            return None
        result = self.sleep_now()
        # Счётчик ОБНУЛЯЕТСЯ только на успешном сне. Иначе ошибка записи
        # («диск полон», битый файл) выглядела бы как «спали, ничего не
        # нашли», и мозг зациклился бы на попытке, никогда не дожидаясь
        # накопления нового опыта.
        if result.get("saved"):
            self.ticks_since_sleep = 0
        return result

    def _save_transitions(self) -> None:
        """
        Сохранить опыт секцией «transitions» Единой Вселенной.

        read-modify-write через merge_json, а не запись своего снимка: файл
        пишут и другие (Learner, Википедия, чистильщик), и прямой снимок стёр бы
        чужие переходы. Максимум при слиянии делает порядок записей
        несущественным. Ошибка сохранения НЕ глотается: потерю опыта надо
        видеть, а не обнаруживать её через неделю по нулевой точности.
        """
        try:
            # Транзакция ОБЯЗАТЕЛЬНА именно здесь: это единственный путь, где
            # два потока (цикл + Учитель) пишут ОДНУ секцию transitions.
            with world_transaction(self.world_filepath) as world:
                merged = merge_json(world.get("transitions"), self.transitions.to_json())
                world["transitions"] = merged
            # Живая таблица обязана стать ОБЪЕДИНЕНИЕМ, иначе процесс продолжит
            # работать без чужого опыта, который только что сохранил.
            self.transitions.from_json(merged)
        except Exception as exc:
            self._say(f"⚠️ опыт НЕ сохранён: {type(exc).__name__}: {exc}")

    # ── ОПЫТ УЧАСТВУЕТ В ОТВЕТЕ (очередь п.24) ────────────

    def predict_next(self, concept: str, k: int = 1) -> List[Dict[str, Any]]:
        """
        Что ОПЫТ цикла ждёт после этого концепта — БЕЗ обучения и БЕЗ записи.

        До этой правки Experience был измерительным прибором: цикл считал
        точность и молчал. Предсказание «кошка -> пить» не подсказывало
        brain/ir, какое знание показать первым (memory.txt, п.24). Здесь
        появляется ЕДИНСТВЕННЫЙ вход, которым ответы читают опыт: словарь
        {"tail", "strength", "norm"} для top-k хвостов.

        Три честных ограничения (каждое — не «на будущее», а прямо сейчас):
          1. Опыт НЕ создаёт знание. Если цикла нет или он выключен — пустой
             список, а не догадка. Выключенный канал обязан молчать.
          2. Имя концепта сводится к канону (лемма/резолвер), иначе «кошки»
             и «кошка» разошлись бы по двум разным рядкам таблицы.
          3. Возвращается СИЛА 0..15 отдельным полем `norm` (0.0..1.0):
             вызывающий сам решает, достаточен ли опыт. Порог не выдумывается
             здесь — его нет в замерах.
        """
        if not getattr(self, "learn_experience", False):
            return []
        canon = self.core.resolve_name(concept) or self.lemmas_world.resolve_name(concept)
        if not canon:
            canon = (concept or "").lower().strip()
        # Цикл хранит ЛЕММЫ (core/cortex.py::experience_sentences), а не формы
        # из запроса. Без этого шага «кошки» не находило рядок «кошка» и опыт
        # молча не участвовал в ответе на падежную форму — та же ловушка, что
        # закрывали у find_in_memory. Импорт ЛОКАЛЬНЫЙ: мозг обязан работать
        # без pymorphy3 (эвристика), а не падать на импорте.
        if not self.transitions.known_tails(canon):
            try:
                from core.morph import Morph
                lemma = Morph.lemma(canon)
                if lemma and self.transitions.known_tails(lemma):
                    canon = lemma
            except Exception:
                # Морфологии нет — остаёмся на сыром имени. Молчать нельзя,
                # но и падать нечем: пустой результат ниже сам скажет «опыта
                # нет», а не притворится «опыт есть».
                pass
        out = []
        for tail, strength in self.transitions.top(canon, k):
            if strength <= 0:
                continue
            out.append({"tail": tail, "strength": strength,
                        "norm": strength / float(self.transitions.max_strength)})
        return out

    def experience_conflicts(self) -> List[Dict[str, Any]]:
        """
        Противоречия ОНТОЛОГИЯ ↔ ОПЫТ — только ЧТЕНИЕ, без записи.

        Очередь п.25: сила (переход) накоплена, но не расходовалась — цикл не
        умел ни подсказывать, ни ТОРМОЗИТЬ запись факта, которому опыт
        противоречит. Подсказывать теперь умеет predict_next (п.24); здесь —
        вторая половина: противоречие обязано быть ВИДНО, а не лежать молча.

        Что считается противоречием (берём из experiments/PLASTICITY_CONTRACT.md,
        а не из головы):
          * онтология УТВЕРЖДАЕТ (weight >= 150), а опыт почти не видел связи
            (strength <= 3) -> факт в базе есть, но наблюдения его не
            подтверждают;
          * опыт СИЛЬНО ждёт (strength >= 8), а онтологической связи нет
            совсем -> опыт накопил, а знание не оформлено (это же лечит
            consolidate_during_sleep, п.23 — но там нужна запись, а здесь
            только сообщение).

        ВАЖНО: возвращается список диагнозов, НИЧЕГО не пишется в Вселенную.
        У цикла нет прав записи, и этот инспектор их не выдаёт: он называет
        расхождение, а решение (спать, понизить вес, добавить связь) — не его.
        """
        conflicts: List[Dict[str, Any]] = []
        # 1. онтология сильна, опыт молчит
        for v in (self._load_world().get("observations") or {}).values():
            if not isinstance(v, dict) or v.get("negated"):
                continue
            if v.get("relation") != "CAN_DO":
                continue
            head = str(v.get("head", "")).lower().strip()
            tail = str(v.get("tail", "")).lower().strip()
            if not head or not tail:
                continue
            # ВЕС берётся через observation_weight, а НЕ obs.get("weight", 0).
            # ЗАМЕР этой сессии: ключа weight в observations НЕ было вовсе, и
            # условие `weight >= 150` не выполнялось НИКОГДА — диагноз
            # ontology_without_experience был недостижим, хотя тесты на него
            # имелись. Читатель, который вечно читает нули, выглядит ровно как
            # работающий.
            weight = observation_weight(v)
            strength = self.transitions.get(head, tail)
            if weight >= 150 and strength <= 3:
                conflicts.append({"kind": "ontology_without_experience",
                                  "head": head, "tail": tail,
                                  "weight": weight, "strength": strength})
        # 2. опыт силён, онтологии нет
        for head in self.transitions.heads():
            for tail in self.transitions.known_tails(head):
                strength = self.transitions.get(head, tail)
                if strength < 8:
                    continue
                if not self._has_can_do(head, tail):
                    conflicts.append({"kind": "experience_without_ontology",
                                      "head": head, "tail": tail,
                                      "weight": 0, "strength": strength})
        return conflicts

    def _has_can_do(self, head: str, tail: str) -> bool:
        """Есть ли CAN_DO(head, tail) в наблюдениях Вселенной. Только чтение."""
        for v in (self._load_world().get("observations") or {}).values():
            if not isinstance(v, dict) or v.get("negated"):
                continue
            if v.get("relation") != "CAN_DO":
                continue
            if str(v.get("head", "")).lower().strip() == head \
                    and str(v.get("tail", "")).lower().strip() == tail:
                return True
        return False

    # ── SLEEP CYCLE (очередь memory.txt, п.23-доработка, P1) ───────────

    def sleep_now(self, dry_run: bool = False,
                  decay_rate: float = None) -> Dict[str, Any]:
        """
        ОТСПАТЬ один цикл: перевести накопленный опыт в вес онтологии.

        ЕДИНСТВЕННОЕ место в мозге, где опыт получает право править weight
        (memory.txt). Сами правила, числа и причины — в brain/sleep.py; здесь
        только обвязка: права, транзакция и отчёт.

        Почему мир берётся ВНУТРИ транзакции, а не из self._load_world().
        _load_world возвращает ЗАКЭШИРОВАННЫЙ словарь, который мозг раздаёт
        всем читателям (find_in_memory, ответы, инспекторы). Консолидация
        мутирует мир (создаёт наблюдения, правит веса). Мутировать кэшированный
        словарь — значит без спроса поправить всем читателям данные посреди их
        работы; это та же гонка, ради которой существовал WorldTransaction.
        Поэтому сон работает со СВЕЖИМ снимком из-под блокировки и пишет туда же.

        dry_run=True ничего не пишет: «покажи, что сделает сон» обязано быть
        возможным без побочных эффектов — иначе проверка контракта сама была бы
        правкой памяти.
        """
        from brain.sleep import DECAY_RATE
        from brain.sleep import consolidate as _consolidate
        from brain.sleep import decay_transitions as _decay
        from brain.sleep import record_in_world, reconcile_saved, table_of_section

        _table_of = table_of_section
        rate = DECAY_RATE if decay_rate is None else decay_rate
        try:
            # ОПЫТ БЕРЁТСЯ ОБЪЕДИНЕНИЕМ «живой таблицы процесса» и того, что
            # лежит в файле. Причина — ЗАМЕР, а не осторожность: цикл накапливает
            # силу в памяти процесса и сохраняет её только на _save_transitions,
            # а сон, прочитавший ТОЛЬКО файл, лечил бы переходы, которых у
            # процесса уже нет в таком виде (замер: файл показал силу 1 там, где
            # живая таблица помнила 12). merge_json по максимуму — единственная
            # операция, которая делает это объединение не зависящим от порядка.
            live = self.transitions.to_json()
            if dry_run:
                # Отдельные снимки: пробный сон обязан НЕ трогать живую таблицу.
                world = read_world(self.world_filepath)
                table = _table_of(merge_json(live, world.get("transitions")))
                rep = _consolidate(world, table, dry_run=True)
                d = _decay(table, rate, dry_run=True)
                rep.decayed, rep.compacted = d.decayed, d.compacted
                return {"report": rep, "text": rep.as_text(), "saved": False}
            with world_transaction(self.world_filepath) as world:
                table = _table_of(merge_json(live, world.get("transitions")))
                baseline = table.to_json()
                rep = _consolidate(world, table)
                d = _decay(table, rate)
                rep.decayed, rep.compacted = d.decayed, d.compacted
                # Секцию transitions правит ТА ЖЕ транзакция: выход из блока
                # сохранит и observations, и transitions одним атомарным
                # снимком. Иначе между двумя записями наблюдался бы мир, где
                # знание уже выросло, а опыт ещё не затух.
                section, lost = reconcile_saved(
                    baseline, table.to_json(), world.get("transitions") or {})
                world["transitions"] = section
                rep.file_conflict = lost
                # ИТОГ СНА — В ДАННЫЕ. Вопрос «спит ли система и что ей это
                # дало» обязан решаться чтением файла памяти, а не догадкой по
                # строке журнала, которая живёт до перезапуска окна. Сухой
                # сон сюда не попадает: он ничего не сделал, а статистика
                # логичного «не было» неотличима от статистики «не пытались».
                record_in_world(world, rep)
            # Живая таблица процесса обязана сравняться с файлом: иначе мозг
            # продолжил бы предсказывать по силам, которых нет уже нигде.
            self.transitions.from_json(section)
            # Наблюдения поменялись — граф пересобирается из них, иначе новое
            # знание останется в JSON и не станет связью до перезапуска.
            if rep.created or rep.raised or rep.lowered:
                self.learner.replay_observations(self._load_world())
                self.core.sync_lemmas()
                self._say("🌙 " + rep.as_text())
            return {"report": rep, "text": rep.as_text(), "saved": True}
        except Exception as exc:
            self._say(f"⚠️ сон НЕ состоялся: {type(exc).__name__}: {exc}")
            return {"report": None,
                    "text": f"сон не состоялся: {exc}", "saved": False,
                    "error": f"{type(exc).__name__}: {exc}"}

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
        elif intent_type == "ACTION":
            return self._handle_actions(subject)
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

    def _handle_actions(self, concept: str) -> str:
        """
        «Что делает / что умеет X» — ответ из наблюдений, а не из головы.

        ЕДИНСТВЕННЫЙ носитель этого знания — observations вида
        «вода|CAN_DO|гасить». Никакой отдельной обработки глагола не нужно:
        action-узел это тот же хвост связи, а профиль — его подпись.
        """
        data = self.find_in_memory(concept)
        name = data.get("matched_concept", concept) if data else concept
        obs = self._load_world().get("observations", {})
        acts = [v["tail"] for k, v in obs.items()
                if v.get("head") == name and v.get("relation") == "CAN_DO"
                and not v.get("negated")]
        if not acts:
            known = " (я учил «" + name + "», но умений не записывал)" \
                if data else ""
            return (f"[Lipid-2]: Я не знаю, что делает '{name}'"
                    f"{known}. Скажи сам — я запомню.")
        # ОПЫТ ПОДСКАЗЫВАЕТ ПОРЯДОК (п.24). Не заменяет знание и не выдумывает
        # новое: он лишь ставит ожидаемое продолжение первым и говорит вслух,
        # ПОЧЕМУ оно первое. Цикл не имеет прав записи — здесь только чтение.
        predicted = self.predict_next(name, k=1)
        ordered = sorted(set(acts))
        hint = ""
        if predicted and predicted[0]["tail"] in ordered:
            exp = predicted[0]["tail"]
            ordered.remove(exp)
            ordered.insert(0, exp)
            hint = (f" Опыт ждёт первым «{exp}» "
                    f"(сила {predicted[0]['strength']}/15).")
        return (f"[Lipid-2]: '{name}' умеет: "
                f"{', '.join(ordered)}.{hint}")

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
        """Обработка взаимодействия двух и более сущностей.

        ОПЫТ УЧАСТВУЕТ (п.24): для первого субъекта цикл подсказывает, чего он
        ждёт следующим. Если среди перечисленных субъектов есть ожидаемый —
        говорим это прямо, а не рисуем «векторный просчёт симулятора» (та
        строка была заглушкой: она не читала ни графа, ни опыта).
        """
        head = subjects[0] if subjects else ""
        other = subjects[1] if len(subjects) > 1 else ""
        facts = self._facts_between(subjects)
        parts = []
        if facts:
            parts.append("известные связи: " + ", ".join(facts))
        predicted = self.predict_next(head, k=1) if head else []
        if predicted and other and predicted[0]["tail"] == other.lower().strip():
            exp = predicted[0]
            parts.append(f"опыт ждёт после '{head}' именно '{exp['tail']}' "
                         f"(сила {exp['strength']}/15)")
        if parts:
            return f"[Lipid-2]: Про {' и '.join(subjects)} — " + "; ".join(parts) + "."
        return (f"[Lipid-2]: Прямых наблюдений между {' и '.join(subjects)} "
                f"нет, и опыт не предсказывает их порядок.")

    def _facts_between(self, subjects: list) -> List[str]:
        """
        Связи ВСЕЛЕННОЙ между перечисленными субъектами — текстовыми фактами.

        Только чтение. Пустой список = «связей нет», а НЕ «не искали»:
        вызывающий (_handle_interaction) сам решает, что сказать в пустоте.
        """
        names = {s.lower().strip() for s in subjects if s}
        out: List[str] = []
        for v in (self._load_world().get("observations") or {}).values():
            if not isinstance(v, dict):
                continue
            h = str(v.get("head", "")).lower().strip()
            t = str(v.get("tail", "")).lower().strip()
            if h in names and t in names and not v.get("negated"):
                out.append(f"{h} —{v.get('relation', '?')}-> {t}")
        return out

    # ── ДОСТУП ДЛЯ GUI / ХОРМОНЫ ──────────────────────────

    @property
    def noradrenaline(self):
        return self.hormones.norepinephrine

    @property
    def dopamine(self):
        return self.hormones.dopamine