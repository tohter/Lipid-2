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
    from core.worldfile import write_world, default_memory_dir
except Exception as _exc:                              # pragma: no cover
    print(f"[teacher] core.worldfile недоступен ({_exc}) — запись НЕатомарная", file=sys.stderr, flush=True)
    write_world = None

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
            


    def _say(self, message: str) -> None:
        """
        Шаг работы учителя в общий канал. В stdout шло и раньше, но GUI stdout
        не читает: пока агент думает, окно стояло мёртвым. Теперь тот же текст
        видит пользователь, а не только терминал.
        """
        _status(message)

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

    def write_json_file(self, filename: str, content: str) -> str:
        """Интегрирует новый онтологический разбор в единый файл lipid_world.json."""
        try:
            if isinstance(content, str):
                clean_str = re.sub(r'```json|```', '', content).strip()
                data = json.loads(clean_str)
            else:
                data = content

            world = self._load_world()

            # Добавляем или обновляем базовую сущность
            if "base_entity" in data:
                base = data["base_entity"]
                concept = base.get("concept", "").lower().strip()
                if concept:
                    # Присваивание целиком стирало всё, что у сущности уже
                    # было: наблюдения из рассказов, provenance, выученные
                    # свойства. Повторный вопрос про известное слово выглядел
                    # как успех, а внутри был откат базы.
                    world["entities"][concept] = merge_entity(
                        world["entities"].get(concept), base)

            # Добавляем или обновляем производную сущность
            if "derived_entity" in data:
                derived = data["derived_entity"]
                concept = derived.get("concept", "").lower().strip()
                if concept:
                    # САМОЕ ЧАСТОЕ: базовый и производный концепт — одно слово
                    # (запросили «флюгер», производного нет). Прежний код писал
                    # base, а СРАЗУ ПОСЛЕ него перезаписывал ту же ячейку
                    # вырожденным derived — и иерархия, свойства и действия
                    # базовой сущности МОЛЧА СТИРАЛИСЬ. Отсюда пустые «свойства»
                    # и «флюгер — это флюгер».
                    #
                    # Правила слияния ЖЁСТКИЕ: derived имеет право ДОБАВИТЬ
                    # delta_properties и parent_concept, но не имеет права
                    # СТЕРЕТЬ то, что уже есть у base. LLM пишет эти два блока
                    # в разном составе от запуска к запуску, и полагаться на
                    # «сейчас она положила hierarchy в derived» нельзя.
                    prev = world["entities"].get(concept)
                    if isinstance(prev, dict):
                        merged = dict(prev)
                        for k, v in derived.items():
                            if k in ("hierarchy", "properties", "actions"):
                                have = merged.get(k) or []
                                add = v or []
                                merged[k] = have + [x for x in add if x not in have]
                            elif k == "delta_properties":
                                merged[k] = list(merged.get(k) or []) + \
                                    [x for x in (v or [])
                                     if x not in (merged.get(k) or [])]
                            elif not merged.get(k):
                                merged[k] = v
                        world["entities"][concept] = merged
                    else:
                        world["entities"][concept] = derived

            # Самоссылка parent_concept == concept — мусор, он превращает
            # «X — это X» в вечный цикл обхода иерархии.
            ent = world["entities"].get(concept) if concept else None
            if isinstance(ent, dict) and ent.get("parent_concept", "").lower().strip() == concept:
                del ent["parent_concept"]

            if write_world is not None:
                write_world(self.world_filepath, world)
            else:
                with open(self.world_filepath, "w", encoding="utf-8") as f:
                    json.dump(world, f, ensure_ascii=False, indent=2)

            written = [c for c in (base.get("concept", "") if "base_entity" in data else "",
                                   (data.get("derived_entity") or {}).get("concept", "")) if c]
            if written:
                self._say("Учитель записал во Вселенную: " + ", ".join(written))

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
                            "filename": {"type": "string", "description": "Имя файла (сохраняется в lipid_world.json)"},
                            "content": {"type": "string", "description": "JSON-строка с сущностями base_entity и derived_entity"}
                        },
                        "required": ["filename", "content"]
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

    def run_agent(self, base_concept: str = "кошка", derived_concept: str = "кот", filename: str = "lipid_world.json"):
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
        for step in range(1, max_steps + 1):
            self._say(f"Учитель: шаг {step}/{max_steps} — запрос к локальной модели "
                      f"({self.host})")

            payload = {
                "messages": messages,
                "tools": self.get_tools_schema(),
                "temperature": 0.1,
                "max_tokens": 2000
            }

            headers = {"Content-Type": "application/json"}
            req = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode('utf-8'),
                headers=headers
            )

            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    raw_bytes = response.read()
                    result = json.loads(raw_bytes.decode('utf-8'))

                    message = result['choices'][0]['message']
                    messages.append(message)

                    tool_calls = message.get("tool_calls", [])

                    if not tool_calls:
                        content = message.get("content", "").strip()
                        self._say("Учитель закончил рассуждение (ответ без вызова инструмента)")
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
                        self._say(f"   ↳ {str(res)[:200]}")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", f"call_{step}"),
                            "content": str(res)
                        })

            except urllib.error.URLError as e:
                self._say(f"⛔ Учитель: локальная модель недоступна ({e.reason}) — "
                          "проверяй llama.cpp на 8080")
                break
            except Exception as e:
                self._say(f"⛔ Учитель: {type(e).__name__}: {e}")
                break