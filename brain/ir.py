import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    import pymorphy3
    HAS_PYMORPHY = True
except ImportError:
    HAS_PYMORPHY = False


@dataclass
class IRIntent:
    """Промежуточное представление (IR) входящего запроса."""
    intent_type: str            # DEFINITION, PROPERTIES, INTERACTION, ACTION, UNKNOWN
    subjects: List[str] = field(default_factory=list)  # Выцепленные сущности (лемматизированные)
    raw_query: str = ""         # Исходный текст


class IRParser:
    """
    Универсальный IR-парсер русского языка.
    Выделяет интенты и сущности без завязки на конкретные слова.

    ИЗМЕНЕНИЯ v2 (п.9 очереди):
    - Лемматизация через pymorphy3: «гвозди» → «гвоздь», «флюгера» → «флюгер»
    - Захват фраз: «ржавый гвоздь» → целая фраза, а не только «ржавый»
    - Улучшенные паттерны для притяжательных и прилагательных конструкций

    ДВЕ ВЕЩИ, КОТОРЫЕ ЗДЕСЬ ЛОМАЛИСЬ И ИЗ-ЗА КОТОРЫХ МОЗГ ВЫГЛЯДЕЛ ГЛУПЫМ:

    1. ПОРЯДОК ШАБЛОНОВ ЗНАЧИТ. Движок возвращает ПЕРВОЕ совпадение, поэтому
       общий шаблон «какой/какие X» раньше стоял раньше и съедал «какие свойства
       у собаки», теряя объект. Специфичное обязано быть выше общего.

    2. ПАДЕЖИ. «Свойства флюгера» — родительный падеж; прежний шаблон требовал
       конструкция «какие свойства у X» и не матчил ни «свойства X», ни «у X есть
       свойства». Перебирать русские падежные окончания regex'ом бессмысленно,
       поэтому ловим слово ЛЮБОЙ формы, а приведение формы к концепту делает
       SDR-узнавание: «собаки» ложится на узел «собака» пересечением 17-19 бит
       из 20 (замер в tests/test_sdr_contract.py). Разделение ответственности:
       парсер ищет ФРАГУ, ядро — СМЫСЛ.
    """

    # Регэксп для слова (поддержка дефисов в составе слов)
    WORD = r"([а-яёa-z0-9][а-яёa-z0-9_-]*)"
    # Фраза: одно или несколько слов (жадный захват до стоп-слова или конца)
    # Используем жадный захват: (.+)
    PHRASE = r"(.+)"

    # Служебные слова не должны становиться «сущностями» в UNKNOWN-ветке.
    STOPWORDS = {"такое", "такой", "такая", "такие", "какие", "какой", "какая",
                 "какое", "что", "кто", "где", "когда", "почему", "зачем", "как",
                 "много", "сколько", "немного", "есть", "быть", "это", "этот", "эта", "эти", "сам",
                 "сама", "самое", "свойства", "признаки", "характеристики",
                 "умения", "действия", "знаешь", "умеешь", "расскажи", "назови",
                 "который", "которая", "которое", "которые", "и", "а", "но", "или"}

    # Предлоги и союзы, которые НЕ должны быть частью леммы
    PREPOSITIONS = {"в", "на", "под", "над", "за", "перед", "после", "для", "от",
                    "до", "из", "к", "у", "о", "об", "про", "с", "со", "без",
                    "через", "между", "при", "по", "около", "возле"}

    def __init__(self):
        W = self.WORD
        PH = self.PHRASE
        PROPS = r"(?:свойства|признаки|характеристики|умения|действия)"

        # Инициализируем морфологический анализатор
        self._morph = pymorphy3.MorphAnalyzer() if HAS_PYMORPHY else None

        self.patterns = [
            # 1. ОПРЕДЕЛЕНИЕ: "что такое X", "кто такой X"
            ("DEFINITION", rf"^(?:что|кто)\s+(?:такое|такой|такая|такие)\s+{PH}"),

            # 2. СВОЙСТВА — все русские построения, какие встречаются в живом
            #    вводе. Разные падежи, разный порядок слов.
            #    "свойства ржавый гвоздь", "какие свойства у ржавого гвоздя"
            ("PROPERTIES", rf"^(?:какие\s+)?(?:свойства|признаки|характеристики)\s+(?:у\s+)?{PH}"),
            #    "у ржавого гвоздя есть свойства", "у кошки признаки"
            ("PROPERTIES", rf"^у\s+{PH}\s+(?:какие\s+)?(?:есть\s+)?{PROPS}"),
            #    "какие свойства имеет ржавый гвоздь", "какие умеет действия X"
            ("PROPERTIES", rf"^(?:какие|какой|какая|какое)\s+(?:ещё\s+)?(?:свои\s+)?{PROPS}\s*(?:у|имеет|есть\s+у)?\s*{PH}"),
            #    "какой ржавый гвоздь", "какая красивая кошка" — самый общий, поэтому ПОСЛЕдних
            ("PROPERTIES", rf"^(?:какой|какая|какое)\s+{PH}"),

            # 3. ВЗАИМОДЕЙСТВИЕ: "что происходит, когда X встречает Y"
            ("INTERACTION", rf"(?:когда|если)\s+{PH}\s+(?:встречает|видит|взаимодействует\s+с|нападает\s+на|трогает|ест|кусает)\s+{PH}"),

            # 4. ДЕЙСТВИЕ: "что делает X", "что умеет X"
            ("ACTION", rf"^что\s+(?:может\s+)?(?:делать|делает|умеет)\s+{PH}"),
        ]

    def _lemmatize(self, text: str) -> str:
        """Лемматизирует текст: приводит каждое слово к начальной форме."""
        if not self._morph:
            return text.lower().strip()
        
        words = text.lower().strip().split()
        lemmas = []
        for word in words:
            # Убираем пунктуацию на концах
            clean = word.strip(".,?!:;\"'()[]{}")
            if not clean:
                continue
            # Парсим и берём нормальную форму
            parsed = self._morph.parse(clean)
            if parsed:
                # Выбираем разбор с наибольшей вероятностью
                lemma = parsed[0].normal_form
                lemmas.append(lemma)
            else:
                lemmas.append(clean)
        return " ".join(lemmas)

    def _clean_phrase(self, phrase: str) -> str:
        """
        Очищает захваченную фразу: убирает предлоги на краях, стоп-слова,
        лишние пробелы.
        """
        phrase = phrase.lower().strip().rstrip(".,?!:;\"'()[]{}")
        words = phrase.split()
        # Убираем предлоги и союзы на краях
        while words and words[0] in self.PREPOSITIONS:
            words.pop(0)
        while words and words[-1] in self.PREPOSITIONS:
            words.pop()
        # Убираем стоп-слова с концов
        while words and words[0] in self.STOPWORDS:
            words.pop(0)
        while words and words[-1] in self.STOPWORDS:
            words.pop()
        return " ".join(words)

    def _extract_subjects(self, match: re.Match) -> List[str]:
        """Извлекает и лемматизирует субъекты из групп захвата."""
        subjects = []
        for g in match.groups():
            if g:
                clean = self._clean_phrase(g)
                if clean:
                    lemma = self._lemmatize(clean)
                    if lemma and lemma not in self.STOPWORDS:
                        subjects.append(lemma)
        return subjects

    def parse(self, text: str) -> IRIntent:
        """Главный метод парсинга. Принимает строку, возвращает IRIntent."""
        clean_text = text.lower().strip().rstrip("?!.:;").strip()

        for intent_type, pattern in self.patterns:
            match = re.search(pattern, clean_text)
            if not match:
                continue
            subjects = self._extract_subjects(match)
            if subjects:                      # пустой объект = ложное срабатывание
                return IRIntent(intent_type=intent_type,
                                subjects=subjects, raw_query=text)

        # Если ни один шаблон не подошел, отдаем UNKNOWN
        # (В будущем сюда подкрутится Natasha/SpaCy для свободного текста)
        words = [w for w in clean_text.split()
                 if len(w) > 3 and w not in self.STOPWORDS]
        # Лемматизируем и в UNKNOWN
        lemmatized = [self._lemmatize(w) for w in words]
        return IRIntent(intent_type="UNKNOWN", subjects=lemmatized, raw_query=text)

    def parse_query(self, text: str) -> IRIntent:
        """Алиас для совместимости."""
        return self.parse(text)


if __name__ == "__main__":
    p = IRParser()
    cases = [
        # Базовые случаи
        ("что такое кошка", "DEFINITION", "кошка"),
        ("кто такой флюгер", "DEFINITION", "флюгер"),
        ("свойства флюгера", "PROPERTIES", "флюгер"),
        ("какие свойства у собаки", "PROPERTIES", "собака"),
        ("у кошки есть свойства", "PROPERTIES", "кошка"),
        ("какие признаки у стола", "PROPERTIES", "стол"),
        ("что делает собака", "ACTION", "собака"),
        ("что умеет кот", "ACTION", "кот"),
        ("что происходит когда кошка встречает собаку", "INTERACTION", "кошка"),
        ("сколько весит слон", "UNKNOWN", "весить"),
        
        # Новые тесты: лемматизация
        ("что такое гвозди", "DEFINITION", "гвоздь"),
        ("свойства ржавый гвоздь", "PROPERTIES", "ржавый гвоздь"),
        ("какие свойства у ржавого гвоздя", "PROPERTIES", "ржавый гвоздь"),
        ("у красивой кошки признаки", "PROPERTIES", "красивая кошка"),
        ("какие признаки у бегущего человека", "PROPERTIES", "бегущий человек"),
        
        # Тесты с предлогами
        ("свойства для кошки", "PROPERTIES", "кошка"),
        ("какие свойства у стола", "PROPERTIES", "стол"),
    ]
    bad = 0
    for text, want_intent, want_subj in cases:
        got = p.parse(text)
        ok = got.intent_type == want_intent and any(want_subj in s for s in got.subjects)
        bad += 0 if ok else 1
        status = "OK " if ok else "FAIL"
        print(f"  {status} «{text}» -> {got.intent_type} {got.subjects}")
    print("\nВсе шаблоны отработали." if not bad else f"\nОШИБОК: {bad}")