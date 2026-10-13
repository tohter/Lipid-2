import os
import json
import urllib.request
import urllib.error
import re
import sys
from typing import Dict, Any

# Правило слияния записей — ОДНО на все каналы (core/records.py).
# Путь добавляем явно: учитель запускается отдельным процессом и не обязан
# запускаться из корня проекта.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
try:
    from core.records import merge_entity
except Exception as _exc:
    # Молча дублировать правило нельзя: каждая копия теряет данные по-своему.
    # Поэтому вариант «только новая запись» громко анонсирован и явно хуже.
    print(f"[teacher] core.records недоступен ({_exc}) — слияние ВЫКЛЮЧЕНО, "
          "новая запись будет перезаписывать старую", file=sys.stderr, flush=True)

    def merge_entity(prev, new):
        return new if isinstance(new, dict) else (prev or {})

try:
    # Атомарная запись: json.dump в открытый на "w" файл при обрыве процесса
    # оставляет ОБОРВАННЫЙ JSON — следующая загрузка считает мир повреждённым.
    # world_lock закрывает гонку «прочитал -> изменил -> записал» с
    # фоновым обучением (оба пишут один lipid_world.json).
    from core.worldfile import write_world, default_memory_dir, world_lock
except Exception as _exc:                              # pragma: no cover
    print(f"[teacher] core.worldfile недоступен ({_exc}) — запись НЕатомарная", file=sys.stderr, flush=True)
    write_world = None

    from contextlib import nullcontext as world_lock

    def default_memory_dir(memory_dir=None, filename="lipid_world.json"):
        """Каталог памяти без core.worldfile — тот же порядок источников."""
        if not memory_dir:
            memory_dir = os.environ.get("LIPID2_MEMORY_DIR")
        if not memory_dir:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            memory_dir = os.path.join(root, "data", "memory")
        os.makedirs(memory_dir, exist_ok=True)
        return memory_dir, os.path.join(memory_dir, filename)

try:
    import shutil
except Exception:                                      # pragma: no cover
    shutil = None

try:
    # Канал статуса — чистый stdlib, поэтому учителю он разрешён (в отличие от
    # torch/numpy): учитель может жить в отдельном процессе и в другой venv.
    from core.logchannel import status as _status
except Exception as _exc:                              # pragma: no cover
    print(f"[teacher] core.logchannel недоступен ({_exc}) — статус только в stdout",
          file=sys.stderr, flush=True)

    def _status(message, **_kw):
        print(f"[l2] {message}", flush=True)

class AgentTeacherV2:
    """
    Продвинутый Учитель Lipid-2.
    Работает с единой базой знаний Вселенной `lipid_world.json`.
    """
    def __init__(self, host: str = "http://localhost:8080", memory_dir: str = None):
        self.host = host.rstrip('/')
        self.endpoint = f"{self.host}/v1/chat/completions"
        self.memory_dir, self.world_filepath = default_memory_dir(memory_dir)
        # Лимиты ЗАМЕРЕНЫ, а не взяты «с запасом на всякий случай».
        # reasoning-модель на полном промпте тратит ~1400-1600 токенов на
        # «мысль» перед вызовом, поэтому прежние 2000 — гонка с обрезкой
        # (замер: finish=length, tool_calls НЕТ, канал объявлялся пустым).
        # С выключенными рассуждениями вызов стоит ~440 токенов.
        self.max_tokens = int(os.environ.get("LIPID2_TEACHER_MAX_TOKENS", 4000))
        self.timeout = float(os.environ.get("LIPID2_TEACHER_TIMEOUT", 180))
        # Механический отказ последнего сеанса (транспорт/лимит), либо None.
        # Нужен затем, чтобы вызывающий ОТЛИЧАЛ «модель не ответила» от
        # «модель ответила: не знаю». Без этого упавший llama.cpp ставил
        # словам статус «спрашивали — пусто», и они запрещались к изучению.
        self.last_failure = None
        # Концепты текущего запроса (см. _requested в run_agent). Вне сеанса
        # агента — None, то есть разрешена запись любого узла.
        self._requested = None

    @staticmethod
    def _echo_message(message: dict) -> dict:
        """
        Ответ модели для ОБРАТНОЙ отправки в историю диалога.

        Сервер кладёт в message reasoning_content (замер: 3549 символов на
        концепте «вода»). Переотправлять его нельзя: контекст раздувается, и
        на следующем шаге модель замолкает до упора в лимит токенов.
        ЗАМЕР (одинаковое состояние диалога, шаг 2 после [Tool Error]):
            с reasoning_content .... 74 с, finish=length, tool_calls НЕТ
            без reasoning_content .. 17 с, finish=stop
        Оставляем ровно то, чего ждёт OpenAI-совместимый протокол.
        """
        echo = {"role": message.get("role", "assistant"),
                "content": message.get("content") or ""}
        if message.get("tool_calls"):
            echo["tool_calls"] = message["tool_calls"]
        return echo
            


    def _say(self, message: str) -> None:
        """
        Шаг работы учителя в общий канал. В stdout шло и раньше, но GUI stdout
        не читает: пока агент думает, окно стояло мёртвым. Теперь тот же текст
        видит пользователь, а не только терминал.
        """
        _status(message)

    def _target_key(self, ent: dict, role: str) -> str:
        """
        Под каким ключом Вселенной лежит запись, которую прислала модель.

        Модель пишет имя то в `concept`, то в `name`/`id` (замер: на «вода»
        она вернула base_entity {id: water_substance, name: "Вода
        (Субстанция)"} — без `concept`). Прежний код читал только `concept`,
        не находил его и МОЛЧА проглатывал весь ответ: канал отрабатывал минуту
        и возвращал «не нашёл», хотя знание пришло.

        Спасать будем ТОЛЬКО известным нам целевым словом: `name` модель
        сочиняет свободно («Вода (Субстанция)», «action_extinguish»), и
        писать по нему — плодить фантомные узлы. Если похожего нет среди
        запрошенного концепта и уже существующих узлов — запись НЕОДНОЗНАЧНА,
        о ней кричим, а не молчим.
        """
        if not isinstance(ent, dict):
            return ""
        for key in ("concept", "name", "id", "label"):
            val = str(ent.get(key) or "").strip().lower()
            if val:
                break
        if not val:
            self._say("⚠️ учитель прислал запись без имени — НЕ записываю")
            return ""
        if self._requested is None:
            # Сеанс агента НЕ активен: вызов прямой (тест, скрипт, legacy).
            # Запись доверяется как есть — фильтровать имена смысла нет, а
            # запрет ломает создание первой записи о слове.
            return val
        if val in self._requested:
            return val
        # ищем запрошенный концепт ВНУТРИ имени («вода (субстанция)» -> «вода»)
        for want in sorted(self._requested):
            if want and len(want) >= 3 and want in val:
                return want
        # имени среди запрошенного нет: записывать можно только ТОГДА, когда
        # узел уже живёт во Вселенной (то есть мы его дополняем, а не создаём)
        world_entities = self._load_world().get("entities") or {}
        if val in world_entities:
            return val
        self._say(f"⚠️ учитель назвал «{val}», а просили "
                  f"{sorted(self._requested)} — НЕ записываю")
        return ""

    @staticmethod
    def _parse_tool_args(raw) -> dict:
        """
        Разбор аргументов tool-call от локальной модели.

        Прежний код делал `json.loads(...) except: pass` и ОСТАВЛЯЛ строку, после
        чего `tool_map[fn](**fn_args)` падал с «argument after ** must be a
        mapping, not str», а внешний `except` разрЫвал весь цикл агента. То есть
        одно кривое слово модели превращалось в «Учитель ничего не сделал».

        Локальные модели РЕГУЛЯРНО выдают аргументы с обрывком по max_tokens, в
        ```json-обёртке или с хвостовой запятой, поэтому тут разбор с
        восстановлением, а не строгий json.loads.
        """
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return {}
        text = raw.strip()
        for _ in range(3):        # снимаем markdown-обёртку, она бывает вложенной
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()
        if not text:
            return {}
        for candidate in (text,
                          AgentTeacherV2._clip_json(text),
                          re.sub(r",\s*([}\]])", r"\1", text)):
            if not candidate:
                continue
            try:
                val = json.loads(candidate)
            except Exception:
                continue
            if isinstance(val, dict):
                return val
            if isinstance(val, list) and val and isinstance(val[0], dict):
                return val[0]
        return {}

    @staticmethod
    def _clip_json(text: str) -> str:
        """
        Дочитать обрезанный по max_tokens JSON: добавляем недостающие } и ].
        Лучше половина знания, полученная и исправленная на следующем шаге, чем
        молча уроненный агент.
        """
        stack, in_str, esc = [], False, False
        for ch in text:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack and ((ch == "}" and stack[-1] == "{") or
                              (ch == "]" and stack[-1] == "[")):
                    stack.pop()
        if in_str:
            text += '"'
        return text + "".join("}" if c == "{" else "]" for c in reversed(stack))

    def _load_world(self) -> dict:
        """Загружает весь мир из единого файла."""
        if os.path.exists(self.world_filepath):
            try:
                with open(self.world_filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                # Повреждённый файл Вселенной — не «пустой мир»: молча вернув
                # пустоту, мы на следующем сохранении перезаписали бы файлы
                # знания пустышкой.
                print(f"[teacher]: lipid_world.json НЕ читаем ({e}) — "
                      "мир считается ПОВРЕЖДЁННЫМ, запись блокируется",
                      file=sys.stderr, flush=True)
                if shutil is not None:
                    # копия битых байтов: «разобраться потом» возможно, только
                    # пока их не перезаписали
                    try:
                        target = self.world_filepath + ".broken"
                        n = 1
                        while os.path.exists(target):
                            target = f"{self.world_filepath}.broken{n}"
                            n += 1
                        shutil.copyfile(self.world_filepath, target)
                        print(f"[teacher]: битая копия сохранена: {target}",
                              file=sys.stderr, flush=True)
                    except OSError:
                        pass
                raise
        return {"entities": {}, "relations": []}

    def write_json_file(self, filename: str = None, content: str = None) -> str:
        """Интегрирует новый онтологический разбор в единый файл lipid_world.json.

        filename НЕОБЯЗАТЕЛЕН и НИ НА ЧТО НЕ ВЛИЯЕТ. ЗАМЕРЕНО: Gemma зовёт
        write_json_file и с filename, и без него; когда схема ТРЕБОВАЛА оба
        аргумента, модель их не передавала, ловила TypeError, получала
        [Tool Error] — и канал обрывался на этом. Имя файла модели неизвестно
        и не нужно: адрес памяти задаёт ОДНА функция default_memory_dir, а
        чужое имя из устного ответа — прямой путь к записи мимо Вселенной.
        """
        if content is None:
            return "[Tool Error]: пустой content — записывать нечего"
        asked = str(filename or "").strip()
        if asked and os.path.basename(asked) != os.path.basename(self.world_filepath):
            self._say(f"учитель просил файл {asked!r} — игнорирую, пишу в "
                      f"{os.path.basename(self.world_filepath)} (каталог задаёт "
                      f"default_memory_dir)")
        try:
            if isinstance(content, str):
                clean_str = re.sub(r'```json|```', '', content).strip()
                data = json.loads(clean_str)
            else:
                data = content

            with world_lock():
                world = self._load_world()
                written = []

                def put(ent, role):
                    """Записать одну сущность ответа модели, разобрав её имя.

                    Имя берёт _target_key (он же отбивает фантомы), а ключ
                    запоминается в written, чтобы не перечитывать мир повторно.
                    """
                    key = self._target_key(ent, role)
                    if not key or not isinstance(ent, dict) or not ent:
                        return
                    prev = world["entities"].get(key)
                    if isinstance(prev, dict) and prev:
                        # САМОЕ ЧАСТОЕ: базовый и производный концепт — одно слово
                        # (запросили «флюгер», производного нет). Прежний код писал
                        # base, а СРАЗУ ПОСЛЕ него перезаписывал ту же ячейку
                        # вырожденным derived — и иерархия, свойства и действия
                        # базовой сущности МОЛЧА СТИРАЛИСЬ. Отсюда пустые «свойства»
                        # и «флюгер — это флюгер».
                        #
                        # Слияние делает merge_entity — ОДНА реализация на все
                        # каналы: новое имеет право ДОБАВИТЬ (hierarchy, properties,
                        # actions, delta_properties), но не СТЕРЕТЬ уже известное.
                        # Раньше здесь лежала своя копия правил, и она расходилась
                        # с общей при каждой правке.
                        world["entities"][key] = merge_entity(prev, ent)
                    else:
                        world["entities"][key] = dict(ent)
                        world["entities"][key].setdefault("concept", key)
                    if key not in written:
                        written.append(key)

                # САМОЕ ЧАСТЕ: порядок важен. Сначала base, потом derived, чтобы
                # вырожденный derived не перетянул на себя пустоту.
                for role in ("base_entity", "derived_entity"):
                    ent = data.get(role)
                    if ent is not None:
                        put(ent, role)

                # Известка об ИТОГЕ, а не о намерениях модели: written наполняется
                # только реально записанными ключами, поэтому фраза «записал во
                # Вселенную» больше не звучит после отклонённых фантомов.
                for concept in list(written):
                    ent = world["entities"].get(concept)
                    # Самоссылка parent_concept == concept — мусор, он превращает
                    # «X — это X» в вечный цикл обхода иерархии.
                    if isinstance(ent, dict) and \
                            str(ent.get("parent_concept", "")).lower().strip() == concept:
                        del ent["parent_concept"]

                if write_world is not None:
                    write_world(self.world_filepath, world)
                else:
                    with open(self.world_filepath, "w", encoding="utf-8") as f:
                        json.dump(world, f, ensure_ascii=False, indent=2)

            if written:
                self._say("Учитель записал во Вселенную: " + ", ".join(written))
            else:
                self._say("⚠️ Учитель: из его ответа НЕ записано ничего — см. "
                          "предыдущие строки журнала")

            if not written:
                return ("[Tool Error]: ни одна присланная сущность не подошла "
                        "по имени (записываем только запрошенный концепт)")
            return f"[Tool Success]: Новые знания успешно записаны в единую Вселенную '{self.world_filepath}'."
        except Exception as e:
            return f"[Tool Error]: Ошибка обновления Вселенной: {e}"

    def read_json_file(self, filename: str) -> str:
        """Читает единый файл Вселенной для проверки инвариантов."""
        try:
            if not os.path.exists(self.world_filepath):
                return "[Tool Error]: Файл Вселенной 'lipid_world.json' ещё не создан."
            with open(self.world_filepath, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"[Tool Error]: Ошибка чтения Вселенной: {e}"

    def get_tools_schema(self) -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": "write_json_file",
                    "description": "Записывает полученный граф знаний Lipid-2 в единый файл Вселенной.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            # filename НЕ в required. ЗАМЕРЕНО: Gemma зовёт
                            # write_json_file только с content, а требуемое имя
                            # файла рождало TypeError -> [Tool Error] -> шаг 2,
                            # на котором канал обрывался. Имя файла модели
                            # неизвестно и не нужно: пишим в свой каталог.
                            "filename": {"type": "string", "description": "Имя файла (по умолчанию lipid_world.json)"},
                            "content": {"type": "string", "description": "JSON-строка с сущностями base_entity и derived_entity"}
                        },
                        "required": ["content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_json_file",
                    "description": "Читает текущее состояние Вселенной для проверки инвариантов.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Имя файла"}
                        },
                        "required": ["filename"]
                    }
                }
            }
        ]

    def run_agent(self, base_concept: str = "кошка", derived_concept: str = "кот", filename: str = None):
        """
        Возвращает запись Вселенной, которую Учитель в ней оставил, либо None.

        Раньше функция ничего не возвращала, и вызывающий код читал мир с диска
        после каждого агента. Это работало, но: (а) лишнее чтение файла ради
        одного concept, (б) при конкурентном писателе прочитывался чужой снимок,
        (в) «агент отработал» было неотличимо от «агент записал то, что просили».
        filename оставлен в сигнатуре для совместимости старых вызовов и
        БОЛЬШЕ НЕ ИСПОЛЬЗУЕТСЯ: адрес памяти задаёт default_memory_dir.
        """
        # Разрешённые цели записи (см. _target_key). Без этого списка модель
        # свободна в именах и плодит узлы вроде «гасить (пожар)».
        self._requested = {c for c in ((base_concept or "").lower().strip(),
                                       (derived_concept or "").lower().strip()) if c}
        self.last_failure = None
        self._say(f"Учитель: онтологический разбор «{base_concept}» → «{derived_concept}»")
        self._say("цель записи: lipid_world.json")

        system_prompt = (
            "Ты — главный аналитик онтологии нейроморфной системы Lipid-2.\n"
            "Твоя задача — извлечь из памяти строгий, научно и физически корректный граф знаний для базового концепта и его производного узла.\n\n"
            "ЖЕСТКИЕ ЛОГИЧЕСКИЕ ИНВАРИАНТЫ LIPID-2 (НАРУШЕНИЕ ЗАПРЕЩЕНО):\n"
            "1. СТРОГОСТЬ ИЕРАРХИИ (`hierarchy`):\n"
            "   - `hierarchy[0]` — это ВСЕГДА БЛИЖАЙШЕЕ семейство/класс (например, 'кошачьи' или 'тягачи').\n"
            "   - `hierarchy[-1]` — это ВСЕГДА ВЕРХНИЙ АБСТРАКТНЫЙ КОРЕНЬ (например, 'животные' или 'объекты').\n"
            "   - Пример порядка: ['кошачьи', 'хищники', 'млекопитающие', 'позвоночные', 'животные'].\n\n"
            "2. ОБЯЗАТЕЛЬНАЯ ДЕЛЬТА ПОЛА (`delta_properties`):\n"
            "   - Если производный узел — это гендерный вариант (кот, бык, кобель), `delta_properties` ОБЯЗАН содержать 'самец'!\n"
            "   - Вторичные признаки (мощный, крупный) идут строго ПОСЛЕ пола.\n\n"
            "3. ФИЗИЧЕСКИЙ СМЫСЛ ВЕКТОРОВ [K, M, S] (0..255):\n"
            "   - K (Кинетика): Механическое движение (бег=200, сон=0).\n"
            "   - M (Метаболизм): ВНУТРЕННЯЯ ЖИЗНЕДЕЯТЕЛЬНОСТЬ И БИОЛОГИЯ! Сон, дыхание, переваривание — это ВЫСОКИЙ метаболизм (M=180..255).\n"
            "   - S (Семиотика): Внешние сигналы, звуки, запахи (мяуканье=255).\n\n"
            "ПРИМЕР СТРУКТУРЫ ВЫХОДА:\n"
            "{\n"
            '  "base_entity": {\n'
            '    "concept": "кошка",\n'
            '    "hierarchy": ["кошачьи", "хищники", "млекопитающие", "позвоночные", "животные"],\n'
            '    "properties": ["пушистая", "гибкая", "хищная"],\n'
            '    "actions": [{"name": "спать", "action_profile": [0, 190, 5]}]\n'
            '  },\n'
            '  "derived_entity": {\n'
            '    "concept": "кот",\n'
            '    "parent_rel": "IS_A",\n'
            '    "parent_concept": "кошка",\n'
            '    "delta_properties": ["самец", "крупный"]\n'
            '  }\n'
            "}\n\n"
            "АЛГОРИТМ РАБОТЫ:\n"
            "1. Сформируй JSON и вызови `write_json_file`.\n"
            "2. Прочитай Вселенную через `read_json_file`.\n"
            "3. ПРОВЕРЬ ИНВАРИАНТЫ: Идет ли иерархия от 'кошачьи' к 'животные'? Есть ли 'самец' у кота? M у сна > 180?\n"
            "4. Если есть нарушения — ПЕРЕЗАПИШИ с исправленной версией.\n"
            "5. Выдай финальный краткий отчет."
        )

        user_prompt = (
            f"Проведи онтологический разбор.\n"
            f"Базовый концепт: '{base_concept}'\n"
            f"Производный концепт: '{derived_concept}'\n"
            f"Интегрируй результаты в Вселенную."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        tool_map = {
            "write_json_file": self.write_json_file,
            "read_json_file": self.read_json_file
        }

        max_steps = 6
        wrote_something = False
        for step in range(1, max_steps + 1):
            self._say(f"Учитель: шаг {step}/{max_steps} — запрос к локальной модели "
                      f"({self.host})")

            payload = {
                "messages": messages,
                "tools": self.get_tools_schema(),
                "temperature": 0.1,
                "max_tokens": self.max_tokens,
                # Gemma здесь reasoning-модель: она думает в reasoning_content.
                # ЗАМЕР на http://localhost:8080 (gemma-4-12B-it-Q8_0), концепт
                # «гасить», одинаковый промпт:
                #   как есть, max_tokens=2000 ....... 54 с, finish=tool_calls
                #   reasoning_effort=none ........... 15 с, finish=tool_calls
                #   reasoning_effort=low ............ 56 с, finish=tool_calls
                # Результат тот же, в 3.5 раза быстрее и без гонки с лимитом
                # токенов. Рассуждать нам нечем: разбор вызова инструмента.
                "reasoning_effort": "none",
            }

            headers = {"Content-Type": "application/json"}
            req = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode('utf-8'),
                headers=headers
            )

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    raw_bytes = response.read()
                    result = json.loads(raw_bytes.decode('utf-8'))

                    choice = result['choices'][0]
                    message = choice['message']
                    finish = choice.get('finish_reason')
                    # В историю диалога кладём СЖАТУЮ версию ответа.
                    # Эхо reasoning_content — главная причина смерти канала 1:
                    # сервер возвращает ~3500 символов «мыслей», мы переотправляем
                    # их обратно, и на шаге 2 контекст раздувается, модель снова
                    # молчит до упора в лимит. ЗАМЕР (одинаковое состояние диалога,
                    # шаг 2 после [Tool Error]):
                    #   с reasoning_content ......... 74 с, finish=length, tools НЕТ
                    #   без reasoning_content ....... 17 с, finish=stop
                    messages.append(self._echo_message(message))

                    tool_calls = message.get("tool_calls", [])

                    if not tool_calls:
                        content = (message.get("content") or "").strip()
                        if finish == "length":
                            # Это НЕ «закончил рассуждение». Модель упёрлась в
                            # лимит токенов и tool-call не был порождён. Врать про
                            # «закончил» нельзя: после такой строки канал 1
                            # объявлялся пустым, а настоящая причина (лимит)
                            # оставалась невидимой.
                            self.last_failure = (f"ответ обрезан по лимиту токенов "
                                                 f"(max_tokens={self.max_tokens})")
                            self._say(f"⚠️ Учитель: ответ ОБРЕЗАН по лимиту токенов "
                                      f"(max_tokens={self.max_tokens}), вызова "
                                      f"инструмента нет — знание не получено")
                        else:
                            self._say("Учитель закончил рассуждение "
                                      "(ответ без вызова инструмента)")
                        if content:
                            self._say(f"Вывод учителя: {content[:300]}")
                        break

                    for tool_call in tool_calls:
                        fn_name = tool_call["function"]["name"]
                        fn_args = self._parse_tool_args(
                            tool_call["function"].get("arguments"))

                        self._say(f"инструмент: {fn_name}({', '.join(str(k) for k in list(fn_args)[:3])})")

                        if fn_name not in tool_map:
                            # Неизвестный инструмент — НЕ выходим из цикла: модель
                            # должна получить об этом сообщение и исправиться.
                            res = f"[Tool Error]: неизвестный инструмент '{fn_name}'"
                            self._say(f"⚠️ {res}")
                        else:
                            try:
                                res = tool_map[fn_name](**fn_args)
                            except TypeError as exc:
                                # Кривые аргументы — это НОРМА для локальной модели,
                                # а не катастрофа. Раньше здесь был `pass`, из-за чего
                                # строка уходила в **kwargs и роняла весь агент.
                                res = (f"[Tool Error]: аргументы не подходят "
                                       f"({exc}). Передай "
                                       f"{fn_name} строго по схеме.")
                            except Exception as exc:
                                res = f"[Tool Error]: {type(exc).__name__}: {exc}"
                            else:
                                if fn_name == "write_json_file" and \
                                        not str(res).startswith("[Tool Error]"):
                                    wrote_something = True
                        self._say(f"   ↳ {str(res)[:200]}")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", f"call_{step}"),
                            "content": str(res)
                        })

            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", "replace")[:200]
                except Exception:
                    pass
                self.last_failure = f"HTTP {e.code} от {self.host}"
                self._say(f"⛔ Учитель: локальная модель ОТВЕТИЛА ОШИБКОЙ "
                          f"(HTTP {e.code}) — сервер жив, но запрос не обработан. "
                          f"{body}")
                break
            except urllib.error.URLError as e:
                self.last_failure = f"модель недоступна: {e.reason}"
                self._say(f"⛔ Учитель: локальная модель недоступна ({e.reason}) — "
                          "проверяй llama.cpp на 8080")
                break
            except Exception as e:
                self.last_failure = f"{type(e).__name__}: {e}"
                self._say(f"⛔ Учитель: {type(e).__name__}: {e}")
                break

        if not wrote_something:
            # Ни одного успешного write_json_file за весь сеанс. Возвращать
            # запись из мира в этом случае нельзя: её там мог оставить любой
            # другой канал (Вики, пользователь, прошлый урок), и тогда канал 1
            # отчитается «нашёл» на чужой работе — а следующий код, наоборот,
            # пропустит Вики, решив что знание уже получено.
            return None
        return self._read_back(base_concept, derived_concept)

    def _read_back(self, base_concept: str, derived_concept: str):
        """Снимок только что записанных сущностей в формате Вселенной."""
        world = self._load_world()
        entities = world.get("entities") or {}
        key = (base_concept or "").lower().strip()
        base = entities.get(key)
        if not isinstance(base, dict):
            return None
        derived = {}
        dkey = (derived_concept or "").lower().strip()
        if dkey and dkey != key and isinstance(entities.get(dkey), dict):
            derived = entities[dkey]
        return {"base_entity": base, "derived_entity": derived,
                "matched_concept": key}
