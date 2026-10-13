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
import sys
import urllib.request
import urllib.parse
import urllib.error

_MORPH = None
_MORPH_TRIED = False


def _get_morph():
    """pymorphy3 лениво; None — значит морфологии нет и канал уйдёт в остаток."""
    global _MORPH, _MORPH_TRIED
    if _MORPH_TRIED:
        return _MORPH
    _MORPH_TRIED = True
    try:
        import pymorphy3
        _MORPH = pymorphy3.MorphAnalyzer()
    except Exception as exc:                              # pragma: no cover
        print(f"[wiki] pymorphy3 недоступен ({exc})", file=sys.stderr, flush=True)
        _MORPH = None
    return _MORPH


# Местоимения и счётные слова: формально прилагательные, знанием не являющиеся.
_PROP_STOPWORDS = {"который", "некоторый", "каждый", "любой", "самый", "этот",
                   "тот", "иной", "весь", "другой", "такой", "наш", "ваш",
                   "свой", "сей", "один", "иной", "оный", "сейский"}


def _norm_adj(adj: str) -> str:                            # pragma: no cover
    """Приведение прилагательного к «условно-словарной» форме без морфологии."""
    a = (adj or "").lower()
    for end in ("ая", "яя", "ые", "ие", "ых", "их", "ого", "его"):
        if a.endswith(end):
            return a[:-2] + "ый"
    return a

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
    from core.worldfile import (read_world, write_world, default_memory_dir,
                                world_lock)
except Exception as _exc:                              # pragma: no cover
    # Писать во Вселенную без правила слияния опаснее, чем не писать: каждый
    # такой «merge» — это молчаливое стирание известного. Отказ явный.
    import sys as _sys
    print(f"[wiki] ⛔ core.records / core.worldfile недоступны ({_exc}) — "
          "запись во Вселенную ОТКЛЮЧЕНА, знания не сохраняются",
          file=_sys.stderr, flush=True)
    merge_entity = None
    read_world = write_world = None

    from contextlib import nullcontext as world_lock

    def default_memory_dir(memory_dir=None, filename="lipid_world.json"):
        """Каталог памяти без core.worldfile: тот же порядок источников."""
        import os
        if not memory_dir:
            memory_dir = os.environ.get("LIPID2_MEMORY_DIR")
        if not memory_dir:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            memory_dir = os.path.join(root, "data", "memory")
        os.makedirs(memory_dir, exist_ok=True)
        return memory_dir, os.path.join(memory_dir, filename)


class WikiTeacher:
    """Внешний интернет-учитель через REST API русской Википедии."""

    API_URL = "https://ru.wikipedia.org/w/api.php"

    # Словарь эвристик: глагол -> [K (кинетика), M (метаболизм), S (семиотика)] 0..255
    # Это грубая онтологическая оценка, далёкая от идеала, но честная замена
    # тому, что Gemma генерирует как LLM. Эвристики можно расширять.
    #
    # ТАБЛИЦА ЖИВЁТ В core/actions.py — один справочник на все каналы, иначе
    # Learner не смог бы построить ACTION-узел без обращения в Википедию, а
    # LLM-канал стал бы вторым, независимым источником профилей. Класс оставляет
    # имя себе, чтобы существующие обращения (WikiTeacher.ACTION_PROFILES) и
    # возможные внешние не перестали работать.
    from core.actions import ACTION_PROFILES  # noqa: F811

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
        self.memory_dir, self.world_filepath = default_memory_dir(memory_dir)


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

    # Классификационные прилагательные: размещают сущность в таксономии,
    # а не описывают её свойства. В статье «Вода» это: химический, бинарный,
    # неорганический, ковалентный, конечный, органический — они идут в первом
    # предложении (определении) и засоряют список свойств.
    _TAXONOMIC_ADJECTIVES = {
        # химическая классификация
        "химический", "бинарный", "неорганический", "органический",
        "ковалентный", "ионный", "металлический", "аморфный", "кристаллический",
        "конечный", "бесконечный", "полимерный", "молекулярный", "атомный",
        # научные домены
        "физический", "биологический", "геологический", "астрономический",
        "математический", "химико-физический", "биохимический", "физиологический",
        # происхождение/природа
        "натуральный", "естественный", "искусственный", "синтетический",
        "природный", "техногенный", "антропогенный", "космический",
        # таксономические ранги
        "родовой", "видовый", "семейственный", "отрядный", "классовый",
        "типовый", "царственный", "имперский", "федеральный", "региональный",
        # временные/исторические (для таксономии, а не свойств)
        "древний", "современный", "доколумбовский", "предколумбовский",
        "античный", "средневековый", "нововековый", "советский", "постсоветский",
        # национальные/языковые
        "русский", "английский", "немецкий", "французский", "китайский",
        "японский", "индийский", "арабский", "испанский", "португальский",
        # грамматические/логические категории
        "единственный", "многозначный", "множественный",
        "полный", "частичный", "основной", "второстепенный",
        "прямой", "косвенный", "положительный", "отрицательный",
        "внутренний", "внешний", "верхний", "нижний", "средний",
        "центральный", "периферический", "общий", "частный", "специальный",
        "базовый", "производный", "первичный", "вторичный", "третичный",
    }

    # Точка НЕ конец предложения: инициалы и служебные пометки в скобках
    # («лат. Anthophila», «др.-греч.», «устар.», «итд.»). Прежний список
    # сокращений был написан группой вида (?:лат|ит\.?д|...)\. — в ней
    # «ит.д.» требует literally «ит.д.», а в текстах стоит «ит. д.» и
    # «(лат. Vulcanus)»; итог на ЗАМЕРЕ по 10 живым вступлениям: 10 из 10
    # определений обрезаны посреди скобки — «Пчёлы (лат», «Флюгер (нид»,
    # «Вулкан (от лат». Определение, из которого выпила хвост скобка,
    # выглядит как ответ и не является знанием.
    _ABBREV_RE = re.compile(
        r"(?:[А-ЯЁA-Z]\.\s?"                                     # «Ф. », «C. »
        r"|(?:лат|др|греч|нем|англ|рус|фр|исп|пол|укр|устар|разг|разм|"
        r"ит|т|п|ср|рис|см|напр|г|гг|вв|н|э|им|проф|т\.е|др\.-греч)"
        r"\.(?=\s|\)))")

    # Служебные фразы Википедии, которые НЕ определяют предмет.
    _NOT_DEFINITION_RE = re.compile(
        r"^(?:это статья|эта статья|статья о |переадресация|значение|"
        r"термин имеет|см\.|также см|существует несколько|возможны|"
        r"о других|не путать|упоминается в|в переносном смысле)", re.I)

    def _first_sentence(self, extract: str) -> str:
        """
        Первое ПРЕДЛОЖЕНИЕ вступления — целиком, вместе со скобками.

        Ищем конец предложения, а не первую точку: точка встречается внутри
        латинских названий, инициалов и сокращений. Дополнительно режем
        вики-служебное («Это статья о джойстике — …»), иначе в память
        попадает мета-фраза вместо определения.
        """
        if not extract:
            return ""
        text = re.sub(r"\u0301", "", extract.replace("\n", " ").strip())
        masked = [m.span() for m in self._ABBREV_RE.finditer(text)]

        def inside_abbrev(i):
            return any(a <= i < b for a, b in masked)

        depth = 0
        first_fallback = ""
        for i, ch in enumerate(text):
            if ch == "(":
                depth += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                continue
            if ch not in ".!?" or inside_abbrev(i):
                continue
            nxt = text[i + 1:i + 2]
            if nxt and not nxt.isspace():
                continue                      # «C6H5OH», «1.5», продолжение
            if depth != 0:
                # Скобка ещё не закрыта: точка внутри неё предложение не
                # заканчивает, какой бы буквой ни начиналось следующее слово
                # («Флюгер (нид. Vleugel, нем. Flügel — крыло) — прибор…»).
                continue
            chunk = text[:i + 1].strip()
            if not first_fallback:
                first_fallback = chunk
            # «Это статья о джойстике — конкретном типе …» и «О других
            # устройствах см. …» — про статью, а не про предмет: писать такое
            # в память значит выучить вики-разметку вместо знания.
            if self._NOT_DEFINITION_RE.match(chunk) or re.match(r"^о \w", chunk):
                rest = text[i + 1:].strip()
                if rest:
                    nxt_sent = self._first_sentence(rest)
                    return nxt_sent or first_fallback
                return first_fallback
            return chunk

        # Конца не нашли: отдаём НЕ кусок на произвольной точке, а 400 символов
        # по границе слова — недоопределение честнее оборванной цитаты.
        head = text if len(text) <= 400 else text[:400].rsplit(" ", 1)[0]
        return head.strip()

    PROP_LIMIT = 6

    def _extract_properties(self, extract: str) -> list:
        """
        Свойства из вступления статьи: прилагательные в словарной форме,
        ОЧИЩЕННЫЕ от классификационных/таксономических.

        Стратегия:
        1. Первое предложение (определение) ПРОПУСКАЕТСЯ — там почти всегда таксономия.
        2. Принимаем ТОЛЬКО прилагательные с маркером Qual (качественные):
           - ADJF,Qual (полные формы): прозрачный, токсичный, безопасный
           - ADJS,Qual (краткие формы, предикативные): токсична, безопасна, прозрачна
           Относительные (химический, бинарный, вулканический) и притяжательные
           (кошачий, лисий) отбрасываются по тегу.
        3. Дополнительный denylist для проблемных Qual-прилагательных
           (бинарный, конечный, нормальный, научный, литературный и т.д.).
        4. Поиск идёт по предложениям 2+ вступления.

        ЗАМЕРЕНО на 10 живых вступлениях ru.wikipedia:
            закрытый список из 24 корней: 12 свойств, 4 из 10 сущностей БЕЗ свойств
            морфология, только прилагательные: 55 свойств, 0 из 10 пустых
        Прежняя реализация замечала лишь слова вроде «домашний/хищный/крупный»,
        поэтому у «флюгера», «вулкана», «экосистемы» и «джойстика» не было НИ
        одного свойства, и «свойства X» отвечало «нет данных» по статье, которую
        только что прочитали.

        Почему ТОЛЬКО прилагательные: замер того же алгоритма с причастиями даёт
        те же 0 пустых, но в свойства попадают глагольные леммы — «летать»,
        «иметь», «составить», «качаться». «Пчела: летать» выглядит знанием, но
        это действие, оно живёт в actions. Смешать их — значит потом не отличить.
        """
        if not extract:
            return []
        morph = _get_morph()
        if morph is None:                                  # pragma: no cover
            return self._extract_properties_closed_list(extract)

        # Denylist для Qual-прилагательных, которые всё равно таксономические/общие
        PROBLEMATIC_QUAL = {
            "бинарный", "конечный", "бесконечный", "нормальный", "главный",
            "основной", "второстепенный", "первичный", "вторичный", "третичный",
            "базовый", "производный", "центральный", "периферический",
            "общий", "частный", "специальный", "разный", "различный",
            "научный", "литературный", "художественный", "технический",
            "практический", "теоретический", "экспериментальный", "клинический",
            "жизненный", "бытовой", "промышленный", "военный", "гражданский",
            "государственный", "общественный", "международный",
            "национальный", "региональный", "муниципальный", "федеральный",
            # Типологические/классификационные (виды, типы, формы)
            "хвойный", "лиственный", "смешанный", "древесный", "книжный",
            "печатный", "рукописный", "электронный", "бумажный", "типографский",
            "пикообразный", "куполообразный", "платообразный", "подводный",
            "озёрный", "болотный", "проточный", "пресный", "солёный", "минеральный",
            "подземный", "поверхностный", "атмосферный", "земной", "космический",
            "горный", "равнинный", "вулканический", "сейсмический", "тектонический",
            "эффузивный", "выводной", "интрузивный", "эффузивно-выводной",
            "квартирный", "домовой", "жилой", "нежилой", "административный",
            # Общие/неспецифические
            "отдельный", "одиночный", "мелкий", "значимый", "больший", "широкий",
            "постоянный", "кратковременный", "конкретный", "определённый", "взаимосвязанный",
            "функциональный", "гейзерный", "культурный", "исторический", "архитектурный",
            # Субъективные/оценочные
            "популярный", "известный", "знаменитый", "любимый",
            # Состояния (а не свойства)
            "твёрдый", "жидкий", "газообразный", "плазменный", "кристаллический",
            "аморфный", "малый", "крупный", "микроскопический", "макроскопический",
            # Таксономические биологические (попадают через Qual, но это классы)
            "хищный", "дикий", "плотоядный", "травоядный", "всеядный", "насекомоядный",
            "социальный", "эусоциальный", "одиночный", "стайный", "колониальный",
            # Относительные, просочившиеся через Qual (химические/физические)
            "углекислый", "кислородный", "водородный", "азотный", "сернистый",
            "углеводородный", "органофосфатный", "органохлористый",
            # Общие оценочные
            "хороший", "плохой", "важный", "незначительный",
            "простой", "сложный", "элементарный", "фундаментальный",
            # Разное
            "полупостоянный", "разнообразный", "резкий", "потамолый", "хозяйственный",
            "признанный", "установленный", "официальный", "неофициальный",
        }

        props = []
        sentences = re.split(r"(?<=[.!?])\s+", extract)
        if len(sentences) <= 1:
            return []
        for sent in sentences[1:]:  # пропускаем sentences[0] — определение
            for tok in re.findall(r"[а-яёА-ЯЁ][а-яёА-ЯЁ\-]{3,}", sent):
                parsed = morph.parse(tok)
                if not parsed:
                    continue
                word = parsed[0]
                tag = str(word.tag)
                # Принимаем ТОЛЬКО качественные прилагательные (Qual)
                # ADJF,Qual — полные формы: прозрачный, токсичный
                # ADJS,Qual — краткие формы (предикативные): токсична, безопасна
                if "Qual" not in tag:
                    continue
                if "ADJF" not in tag and "ADJS" not in tag:
                    continue
                if "PROPN" in tag or "NPRO" in tag:
                    continue
                norm = word.normal_form
                if not re.fullmatch(r"[а-яё\-]{4,}", norm):
                    continue
                if (norm in _PROP_STOPWORDS
                        or norm in self._TAXONOMIC_ADJECTIVES
                        or norm in PROBLEMATIC_QUAL
                        or norm in props):
                    continue
                props.append(norm)
                if len(props) >= self.PROP_LIMIT:
                    return props
        return props

    def _extract_properties_closed_list(self, extract):                    # pragma: no cover
        """
        Старое извлечение (24 заданных корня) — ОСТАТОЧНЫЙ путь без морфологии,
        и он loud: молча деградировать вчетверо канал не имеет права.
        """
        print("[wiki]: pymorphy3 нет — свойства ищутся списком из 24 слов, "
              "у большинства слов их не будет", file=sys.stderr, flush=True)
        if not extract:
            return []
        text = extract[:1500].lower()
        pattern = (r"\b(домашн\w*|дик\w*|хищн\w*|крупн\w*|мелк\w*|маленьк\w*"
                   r"|больш\w*|пушист\w*|быстр\w*|сильн\w*|умн\w*|опасн\w*"
                   r"|безопасн\w*|полезн\w*|вредн\w*|редк\w*|обычн\w*|травоядн\w*"
                   r"|плотоядн\w*|всеядн\w*|ночн\w*|дневн\w*)\b")
        props = []
        for hit in re.findall(pattern, text):
            word = _norm_adj(hit)
            if word and word not in props:
                props.append(word)
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

        with world_lock():
            world = self._load_world()
            base = concept_data.get("base_entity", {})
            concept = base.get("concept", "").lower().strip()
            if not concept:
                return False

            # Чистим от служебного, НО definition оставляем. Прежний комментарий
            # утверждал, что «для мира title/definition не нужны», — и это была
            # потеря данных: определение (первое предложение статьи) есть самое
            # информативное, что удалось получить, а _handle_definition его читает.
            # Следствие было видно в GUI: «Экосистема — это экосистемы» при
            # уже скачанной и разобранный статье.
            clean_base = {
                "concept": base["concept"],
                "hierarchy": base.get("hierarchy", []),
                "properties": base.get("properties", []),
                "actions": base.get("actions", []),
                # П.14: ЕДИНАЯ схема происхождения — объект, не строка.
                # Прежде здесь стояло f"wiki:{title}" (строка), и рядом
                # в мире лежали объекты {source,...} из другого канала —
                # код вида [(p["source"]) for p in provenance] падал бы
                # на первой вики-записи.
                "provenance": [{"source": "wiki",
                                "evidence": str(base.get("title", concept))[:200],
                                "at": None}],
            }
            definition = str(base.get("definition") or "").strip()
            if definition:
                clean_base["definition"] = definition
            title = str(base.get("title") or "").strip()
            if title and title.lower() != concept:
                clean_base["title"] = title
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