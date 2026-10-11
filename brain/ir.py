import re
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class IRIntent:
    """Промежуточное представление (IR) входящего запроса."""
    intent_type: str            # DEFINITION, PROPERTIES, INTERACTION, ACTION
    subjects: List[str] = field(default_factory=list) # Выцепленные сущности
    raw_query: str = ""         # Исходный текст


class IRParser:
    """
    Универсальный IR-парсер русского языка.
    Выделяет интенты и сущности без завязки на конкретные слова.
    """
    def __init__(self):
        # Регулярные шаблоны для вычленения сущностей (абстрактные группы)
        self.patterns = [
            # 1. ОПРЕДЕЛЕНИЕ (DEFINITION): "что такое X", "кто такой X"
            (
                "DEFINITION",
                r"^(?:что|кто)\s+(?:такое|такой|такая|такие)\s+([а-яа-яa-z0-9_-]+)"
            ),
            # 2. СВОЙСТВА (PROPERTIES): "какие свойства у X", "какой X", "какая X"
            (
                "PROPERTIES",
                r"^(?:какие\s+свойства\s+у|какой|какая|какие)\s+([а-яа-яa-z0-9_-]+)"
            ),
            # 3. ВЗАИМОДЕЙСТВИЕ (INTERACTION): "что происходит когда X встречает Y", "как X связан с Y"
            (
                "INTERACTION",
                r"(?:когда|если)\s+([а-яа-яa-z0-9_-]+)\s+(?:встречает|видит|взаимодействует с|нападает на)\s+([а-яа-яa-z0-9_-]+)"
            ),
            # 4. ДЕЙСТВИЕ (ACTION): "что делает X"
            (
                "ACTION",
                r"^что\s+делает\s+([а-яа-яa-z0-9_-]+)"
            )
        ]

    def parse(self, text: str) -> IRIntent:
        """
        Главный метод парсинга. Принимает строку, возвращает IRIntent.
        """
        clean_text = text.lower().strip().rstrip("?!.")

        for intent_type, pattern in self.patterns:
            match = re.search(pattern, clean_text)
            if match:
                # Извлекаем все захваченные группы (сущности) и чистим от None
                subjects = [group.strip() for group in match.groups() if group is not None]
                return IRIntent(
                    intent_type=intent_type,
                    subjects=subjects,
                    raw_query=text
                )

        # Если ни один шаблон не подошел, отдаем UNKNOWN
        # (В будущем сюда подкрутится Natasha/SpaCy для свободного текста)
        words = [w for w in clean_text.split() if len(w) > 3]
        return IRIntent(
            intent_type="UNKNOWN",
            subjects=words,
            raw_query=text
        )

    def parse_query(self, text: str) -> IRIntent:
        """Алиас для совместимости."""
        return self.parse(text)