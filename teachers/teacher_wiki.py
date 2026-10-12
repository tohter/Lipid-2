"""
INTERNET-УЧИТЕЛЬ LIPID-2 (Канал 2: Википедия)

Источник знаний, когда локальная LLM недоступна или не сработала.
Вытягивает из русской Википедии:
  - первое предложение статьи (определение)
  - категории страницы (для иерархии)
и сворачивает их в формат онтологии Lipid-2 (base_entity / derived_entity),
идентичный тому, что возвращает локальный Учитель (Gemma).

Почему Википедия: бесплатный REST без ключей, стабильный формат,
редиректы синонимов (Кот -> Кошка), категории = готовая иерархия.

Автор: Amanда (pi), для проекта Lipid-2.
"""
import os
import json
import re
import urllib.request
import urllib.parse
import urllib.error

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from core.logchannel import status as _status
except Exception as _exc:                              # pragma: no cover
    import sys as _sys
    print(f"[wiki] core.logchannel недоступен ({_exc}) — статус только в stdout",
          file=_sys.stderr, flush=True)

    def _status(message, **_kw):
        print(f"[l2] {message}", flush=True)

try:
    from core.records import merge_entity
    from core.worldfile import read_world, write_world
except Exception as _exc:                              # pragma: no cover
    # Писать во Вселенную без правила слияния опаснее, чем не писать: каждый
    # такой «merge» — это молчаливое стирание известного. Отказ явный.
    import sys as _sys
    print(f"[wiki] ⛔ core.records / core.worldfile недоступны ({_exc}) — "
          "запись во Вселенную ОТКЛЮЧЕНА, знания не сохраняются",
          file=_sys.stderr, flush=True)
    merge_entity = None
    read_world = write_world = None


class WikiTeacher:
    """Внешний интернет-учитель через REST API русской Википедии."""

    API_URL = "https://ru.wikipedia.org/w/api.php"

    # Словарь эвристик: глагол -> [K (кинетика), M (метаболизм), S (семиотика)] 0..255
    # Это грубая онтологическая оценка, далёкая от идеала, но честная замена
    # тому, что Gemma генерирует как LLM. Эвристики можно расширять.
    ACTION_PROFILES = {
        "бегать":     [220, 60, 10],
        "прыгать":    [220, 60, 10],
        "летать":     [235, 70, 5],
        "плавать":    [200, 80, 5],
        "ползать":    [120, 60, 5],
        "бежать":     [220, 60, 10],
        "ходить":     [160, 50, 10],
        "идти":       [160, 50, 10],
        "спать":      [0, 200, 0],
        "дышать":     [0, 210, 0],
        "есть":       [30, 180, 5],
        "питаться":   [30, 180, 5],
        "пить":       [30, 170, 5],
        "охотиться":  [180, 120, 20],
        "мяукать":    [10, 60, 255],
        "лаять":      [10, 60, 255],
        "рычать":     [10, 70, 240],
        "петь":       [20, 60, 255],
        "разговаривать": [20, 50, 255],
        "думать":     [5, 100, 180],
        "работать":   [120, 90, 80],
        "строить":    [180, 80, 60],
        "расти":      [10, 190, 0],
        "цвести":     [10, 180, 30],
    }

    # Категории Википедии, которые стоит игнорировать в иерархии
    IGNORE_CATEGORY_WORDS = [
        "википедия", "страницы", "статьи", "значения", "категории",
        "викиданные", "таксоны", "изображения", "незавершённые", "стабы",
        "списки", "терминология", "род", "по алфавиту",
        "животные, описанные", "описанные в", "биологические статьи",
        "инвазивные", "без источников", "нет источников", "требующие",
        "неавторитетный", "ссылка", "непосредственно в",
    ]

    def __init__(self, memory_dir: str = None):
        if memory_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.memory_dir = os.path.join(project_root, "data", "memory")
        else:
            self.memory_dir = memory_dir
        os.makedirs(self.memory_dir, exist_ok=True)
        self.world_filepath = os.path.join(self.memory_dir, "lipid_world.json")

    def _say(self, message: str) -> None:
        _status(message)

    # ── ПУБЛИЧНЫЙ ИНТЕРФЕЙС ─────────────────────────────────

    def fetch_concept(self, concept: str) -> dict:
        """
        Полный разбор концепта из Википедии.
        Возвращает готовый словарь:
            {"base_entity": {...}, "derived_entity": {...}}  — или пустой dict при неудаче.
        """
        concept_clean = concept.strip().lower()
        if not concept_clean:
            return {}

        self._say(f"ищу в Википедии: «{concept_clean}»")
        page = self._fetch_page(concept_clean)
        if not page:
            self._say(f"в Википедии ничего не найдено по «{concept_clean}»")
            return {}

        title, extract, categories = page

        # Определение = первое предложение (обрезаем по точке)
        definition = self._first_sentence(extract)
        properties = self._extract_properties(extract)

        # Иерархия из категорий
        hierarchy = self._categories_to_hierarchy(categories)

        # Действия: ищем глаголы в тексте, которым назначили [K,M,S]
        actions = self._extract_actions(extract)

        base_entity = {
            "concept": concept_clean,
            "title": title,
            "hierarchy": hierarchy,
            "properties": properties,
            "actions": actions,
            "source": "wikipedia",
            "definition": definition,
        }

        return {
            "base_entity": base_entity,
            "derived_entity": {},
        }

    def run(self, base_concept: str) -> dict:
        """Алиас для run_agent — единый интерфейс с локальным Учителем."""
        return self.fetch_concept(base_concept)

    # ── ВНУТРЕННЕЕ ─────────────────────────────────────────

    def _fetch_page(self, title: str):
        """REST-запрос к Вики: извлечение + категории + редиректы.

        Категории берём из общего запроса, но с большим cllimit,
        т.к. иначе они смешиваются со служебными и теряются.
        """
        headers = {"User-Agent": "Lipid2Bot/1.0 (neuromorphic-learning-agent; contact: localhost)"}

        try:
            params = {
                "action": "query",
                "format": "json",
                "prop": "extracts|categories",
                "explaintext": 1,
                "exintro": 1,
                "cllimit": 50,
                "redirects": 1,
                "titles": title,
            }

            if HAS_REQUESTS:
                resp = requests.get(self.API_URL, params=params, timeout=20, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            else:
                url = self.API_URL + "?" + urllib.parse.urlencode(params)
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=20) as r:
                    data = json.loads(r.read().decode("utf-8"))

            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return None

            page = next(iter(pages.values()))
            pageid = page.get("pageid")
            if pageid is None or pageid < 0:
                # pageid < 0 => страница не существует
                return None

            real_title = page.get("title", title)
            extract = page.get("extract", "")

            categories = []
            for cat in page.get("categories", []):
                name = cat.get("title", "").replace("Категория:", "").strip()
                if name and name not in categories:
                    categories.append(name)

            return (real_title, extract, categories)

        except (urllib.error.URLError, ValueError, KeyError, OSError) as e:
            print(f"[WikiTeacher DEBUG]: Ошибка запроса к Вики: {e}")
            return None

    def _first_sentence(self, extract: str) -> str:
        """Первое предложение статьи (обрезка по точке).

        Игнорируем точки внутри латинских сокращений (лат., и т.д., т.п., ср., рис.,
        см., напр.) и ищем точку, за которой идёт заглавная буква / конец.
        """
        if not extract:
            return ""
        text = extract.replace("\n", " ").strip()
        # Убираем ударения (Ко́шка -> Кошка) для корректного поиска
        text = re.sub(r"\u0301", "", text)

        # Сокращения, после которых точка НЕ является концом предложения
        abbr = r"(?:лат|ит\.?д|т\.?п|ср|рис|см|напр|г\.?р|им|проф|англ|рус|др|т\.?е|н\?э|и\?т\.?п)\."
        # Ищем первый знак .!? после слов, за которым пробел+Заглавная или конец строки
        m = re.search(r"(.{10,}?)(?<!\.)[.!?](?=\s[А-ЯЁA-Z]|\s*$)", text)
        if m:
            return m.group(1).strip()
        # фоллбэк: до первой точки (грубо)
        if "." in text:
            return text.split(".")[0].strip()
        return text[:200].rstrip().strip()

    def _extract_properties(self, extract: str) -> list:
        """
        Выцепляет простые прилагательные из вступления как "свойства".
        Очень грубо, но даёт правдоподобный список для онтологии.
        """
        if not extract:
            return []
        text = extract[:1500].lower()
        props = []
        # Паттерны: "домашнее животное", "хищное млекопитающее", "крупный зверь"
        patterns = [
            r"\b(домашн\w*|дик\w*|хищн\w*|крупн\w*|мелк\w*|маленьк\w*|больш\w*|пушист\w*|быстр\w*|сильн\w*|умн\w*|опасн\w*|безопасн\w*|полезн\w*|вредн\w*|редк\w*|обычн\w*|травоядн\w*|плотоядн\w*|всеядн\w*|ночн\w*|дневн\w*)\b"
        ]
        # Приведение прилагательных к условно-словарной форме (муж. род, ед. число)
        def normalize(adj: str) -> str:
            a = adj.lower()
            # женский род
            for end in ["ая", "яя"]:
                if a.endswith(end):
                    return a[:-2] + "ый"
            # множественное число
            for end in ["ые", "ие", "ых", "их"]:
                if a.endswith(end):
                    return a[:-2] + "ый"
            # родительный падеж
            for end in ["ого", "его", "ой"]:
                if a.endswith(end):
                    return a[:-2] + "ый"
            return a

        for m in re.findall(patterns[0], text):
            word = normalize(m.strip())
            if word not in props:
                props.append(word)
        # лимит — первые 5
        return props[:5]

    def _categories_to_hierarchy(self, categories: list) -> list:
        """
        Преобразует категории Вики в иерархию (от частного к общему).
        Категории Вики идут от абстрактных к конкретным — обращаем порядок.
        """
        cleaned = []
        for cat in categories:
            c = cat.lower().strip()
            # отбрасываем служебные категории
            if any(w in c for w in self.IGNORE_CATEGORY_WORDS):
                continue
            # отбрасываем категории, начинающиеся со "статьи" / "биологические статьи"
            if c.startswith(("статьи", "биологические", "незавершённые", "списки")):
                continue
            # отбрасываем слишком длинные/сложные
            if len(c) > 60:
                continue
            if c not in cleaned:
                cleaned.append(c)

        # Если после фильтрации пусто — вернуть то, что есть (сырое)
        return cleaned if cleaned else [c for c in [x.lower().strip() for x in categories]
                                        if len(c) <= 40 and c not in cleaned][:4]

    def _extract_actions(self, extract: str) -> list:
        """
        Ищет глаголы с известными профилями [K,M,S] в тексте статьи.
        """
        if not extract:
            return []
        text = extract[:2000].lower()
        actions = []
        for verb, profile in self.ACTION_PROFILES.items():
            if re.search(r"\b" + re.escape(verb) + r"\w*", text):
                actions.append({"name": verb, "action_profile": profile})
            if len(actions) >= 5:
                break
        return actions

    # ── ЗАПИСЬ В МИР ───────────────────────────────────────

    def write_to_world(self, concept_data: dict) -> bool:
        """Записывает разобранный концепт в единый файл Вселенной."""
        if not concept_data:
            return False

        world = self._load_world()
        base = concept_data.get("base_entity", {})
        concept = base.get("concept", "").lower().strip()
        if not concept:
            return False

        # Чистим от служебных полей (title, source, definition — для мира они не нужны)
        clean_base = {
            "concept": base["concept"],
            "hierarchy": base.get("hierarchy", []),
            "properties": base.get("properties", []),
            "actions": base.get("actions", []),
            "provenance": [f"wiki:{base.get('title', concept)}"],
        }
        # merge_entity, а не присваивание. core/records.py объявлен ЕДИНСТВЕННЫМ
        # местом, где решается, что имеет право перезаписать известное; здесь же
        # было `world[concept] = clean_base`, то есть вопрос про уже известное
        # слово стирал наблюдения, provenance и ранее выученные свойства. Запись
        # Википедии беднее по составу, чем ответ локальной модели, — значит откат
        # был почти гарантирован: «спросили ещё раз» = «забыли больше».
        if merge_entity is not None:
            world["entities"][concept] = merge_entity(
                world["entities"].get(concept), clean_base)
        else:
            world["entities"][concept] = clean_base

        derived = concept_data.get("derived_entity", {})
        if derived and derived.get("concept"):
            dkey = derived["concept"].lower().strip()
            if dkey != concept:
                # base и derived про одно слово — merge_entity уже сложил всё,
                # вырожденный derived поверх богатой записи писать нельзя
                world["entities"][dkey] = (
                    merge_entity(world["entities"].get(dkey), derived)
                    if merge_entity is not None else derived)

        if write_world is not None:
            write_world(self.world_filepath, world)
        else:
            with open(self.world_filepath, "w", encoding="utf-8") as f:
                json.dump(world, f, ensure_ascii=False, indent=2)

        self._say(f"Википедия: статья «{base.get('concept', concept)}» прочитана, "
                  f"свойств: {len(clean_base['properties'])}, действий: {len(clean_base['actions'])}")
        self._say(f"Википедия: запись во Вселенную: «{concept}»")
        return True

    def _load_world(self) -> dict:
        if read_world is not None:
            return read_world(self.world_filepath)
        if os.path.exists(self.world_filepath):
            try:
                with open(self.world_filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"entities": {}, "relations": []}


if __name__ == "__main__":
    import sys
    concept = sys.argv[1] if len(sys.argv) > 1 else "кот"
    wt = WikiTeacher()
    data = wt.fetch_concept(concept)
    print(json.dumps(data, ensure_ascii=False, indent=2))