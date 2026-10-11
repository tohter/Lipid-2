import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class IRIntent:
    """Промежуточное представление (IR) входящего запроса."""
    intent_type: str            # DEFINITION, PROPERTIES, INTERACTION, ACTION, UNKNOWN
    subjects: List[str] = field(default_factory=list)  # Выцепленные сущности
    raw_query: str = ""         # Исходный текст


class IRParser:
    """
    Универсальный IR-парсер русского языка.
    Выделяет интенты и сущности без завязки на конкретные слова.

    ДВЕ ВЕЩИ, КОТОРЫЕ ЗДЕСЬ ЛОМАЛИСЬ И ИЗ-ЗА КОТОРЫХ МОЗГ ВЫГЛЯДЕЛ ГЛУПЫМ:

    1. ПОРЯДОК ШАБЛОНОВ ЗНАЧИТ. Движок возвращает ПЕРВОЕ совпадение, поэтому
       общий шаблон «какой/какие X» раньше стоял раньше и съедал «какие свойства
       у собаки», теряя объект. Специфичное обязано быть выше общего.

    2. ПАДЕЖИ. «Свойства флюгера» — родительный падеж; прежний шаблон требовал
       конструкция «какие свойства у X» и не матчил ни «свойства X», ни «у X есть
       свойства». Перебирать русские падежные окончанияregex'ом бессмысленно,
       поэтому ловим слово ЛЮБОЙ формы, а приведение формы к концепту делает
       SDR-узнавание: «собаки» ложится на узел «собака» пересечением 17-19 бит
       из 20 (замер в tests/test_sdr_contract.py). Разделение ответственности:
       парсер ищет ФРАГУ, ядро — СМЫСЛ.
    """

    WORD = r"([а-яёa-z0-9][а-яёa-z0-9_-]*)"

    # Служебные слова не должны становиться «сущностями» в UNKNOWN-ветке.
    STOPWORDS = {"такое", "такой", "такая", "такие", "какие", "какой", "какая",
                 "какое", "что", "кто", "где", "когда", "почему", "зачем", "как",
                 "много", "сколько", "немного", "есть", "быть", "это", "этот", "эта", "эти", "сам",
                 "сама", "самое", "свойства", "признаки", "характеристики",
                 "умения", "действия", "знаешь", "умеешь", "расскажи", "назови"}

    def __init__(self):
        W = self.WORD
        PROPS = r"(?:свойства|признаки|характеристики|умения|действия)"
        self.patterns = [
            # 1. ОПРЕДЕЛЕНИЕ: "что такое X", "кто такой X"
            ("DEFINITION", r"^(?:что|кто)\s+(?:такое|такой|такая|такие)\s+" + W),

            # 2. СВОЙСТВА — все русские построения, какие встречаются в живом
            #    вводе. Разные падежи, разный порядок слов.
            #    "свойства собаки", "какие свойства у собаки", "свойства есть у X"
            ("PROPERTIES", r"^(?:какие\s+)?(?:свойства|признаки|характеристики)\s+"
                           r"(?:у\s+)?" + W),
            #    "у собаки есть свойства", "у кошки признаки"
            ("PROPERTIES", r"^у\s+" + W + r"\s+(?:какие\s+)?(?:есть\s+)?" + PROPS),
            #    "какие свойства имеет собака", "какие умеет действия X"
            ("PROPERTIES", r"^(?:какие|какой|какая|какое)\s+(?:ещё\s+)?(?:свои\s+)?"
                           + PROPS + r"\s*(?:у|имеет|есть\s+у)?\s*" + W),
            #    "какая X", "какой X" — самый общий, поэтому ПОСЛЕдных
            ("PROPERTIES", r"^(?:какой|какая|какое)\s+" + W),

            # 3. ВЗАИМОДЕЙСТВИЕ: "что происходит, когда X встречает Y"
            ("INTERACTION", r"(?:когда|если)\s+" + W +
                            r"\s+(?:встречает|видит|взаимодействует\s+с|нападает\s+на|"
                            r"трогает|ест|кусает)\s+" + W),

            # 4. ДЕЙСТВИЕ: "что делает X", "что умеет X"
            ("ACTION", r"^что\s+(?:может\s+)?(?:делать|делает|умеет)\s+" + W),
        ]

    def parse(self, text: str) -> IRIntent:
        """Главный метод парсинга. Принимает строку, возвращает IRIntent."""
        clean_text = text.lower().strip().rstrip("?!.:;").strip()

        for intent_type, pattern in self.patterns:
            match = re.search(pattern, clean_text)
            if not match:
                continue
            subjects = [g.strip() for g in match.groups() if g]
            subjects = [s for s in subjects if s and s not in self.STOPWORDS]
            if subjects:                      # пустой объект = ложное срабатывание
                return IRIntent(intent_type=intent_type,
                                subjects=subjects, raw_query=text)

        # Если ни один шаблон не подошел, отдаем UNKNOWN
        # (В будущем сюда подкрутится Natasha/SpaCy для свободного текста)
        words = [w for w in clean_text.split()
                 if len(w) > 3 and w not in self.STOPWORDS]
        return IRIntent(intent_type="UNKNOWN", subjects=words, raw_query=text)

    def parse_query(self, text: str) -> IRIntent:
        """Алиас для совместимости."""
        return self.parse(text)


if __name__ == "__main__":
    p = IRParser()
    cases = [
        ("что такое кошка", "DEFINITION", "кошка"),
        ("кто такой флюгер", "DEFINITION", "флюгер"),
        ("свойства флюгера", "PROPERTIES", "флюгера"),
        ("какие свойства у собаки", "PROPERTIES", "собаки"),
        ("у кошки есть свойства", "PROPERTIES", "кошки"),
        ("какие признаки у стола", "PROPERTIES", "стола"),
        ("что делает собака", "ACTION", "собака"),
        ("что умеет кот", "ACTION", "кот"),
        ("что происходит когда кошка встречает собаку", "INTERACTION", "кошка"),
        ("сколько весит слон", "UNKNOWN", "весит"),
    ]
    bad = 0
    for text, want_intent, want_subj in cases:
        got = p.parse(text)
        ok = got.intent_type == want_intent and want_subj in " ".join(got.subjects)
        bad += 0 if ok else 1
        print(f"  {'OK ' if ok else 'FAIL'} «{text}» -> {got.intent_type} {got.subjects}")
    print("\nВсе шаблоны отработали." if not bad else f"\nОШИБОК: {bad}")
