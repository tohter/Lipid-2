import os
import json
import urllib.request
import urllib.error
import re
from typing import Dict, Any

class AgentTeacherV2:
    """
    Продвинутый Учитель Lipid-2.
    Работает с единой базой знаний Вселенной `lipid_world.json`.
    """
    def __init__(self, host: str = "http://localhost:8080", memory_dir: str = None):
        self.host = host.rstrip('/')
        self.endpoint = f"{self.host}/v1/chat/completions"
        if memory_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.memory_dir = os.path.join(project_root, "data", "memory")
        else:
            self.memory_dir = memory_dir
            
        os.makedirs(self.memory_dir, exist_ok=True)
        self.world_filepath = os.path.join(self.memory_dir, "lipid_world.json")

    def _load_world(self) -> Dict[str, Any]:
        """Загружает весь мир из единого файла."""
        if os.path.exists(self.world_filepath):
            try:
                with open(self.world_filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
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
                    world["entities"][concept] = base

            # Добавляем или обновляем производную сущность
            if "derived_entity" in data:
                derived = data["derived_entity"]
                concept = derived.get("concept", "").lower().strip()
                if concept:
                    world["entities"][concept] = derived

            with open(self.world_filepath, "w", encoding="utf-8") as f:
                json.dump(world, f, ensure_ascii=False, indent=2)

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
        print(f"=== ЗАПУСК УЧИТЕЛЯ V2 (Анализ: '{base_concept}' -> '{derived_concept}') ===")
        print(f"Целевая база знаний: lipid_world.json\n")

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
            print(f"--- [Шаг {step}] Запрос к локалке ---")

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
                        print(f"\n[Ответ Учителя]:\n{content}")
                        break

                    for tool_call in tool_calls:
                        fn_name = tool_call["function"]["name"]
                        fn_args = tool_call["function"]["arguments"]

                        if isinstance(fn_args, str):
                            try:
                                fn_args = json.loads(fn_args)
                            except Exception:
                                pass

                        print(f" -> Вызов инструмента: {fn_name}(...)")

                        if fn_name in tool_map:
                            res = tool_map[fn_name](**fn_args)
                            print(f"    {res}")

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.get("id", f"call_{step}"),
                                "content": str(res)
                            })

            except urllib.error.URLError as e:
                print(f"[DEBUG ERROR]: Ошибка подключения/таймаут: {e.reason}")
                break
            except Exception as e:
                print(f"[DEBUG ERROR]: {e}")
                break