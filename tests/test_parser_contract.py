# -*- coding: utf-8 -*-
"""
Понималка (кирпич №4): текст -> ролевая структура (каркас говорилки).

Обратный полюс к speaker.py. Говорилка строит предложение из каркаса;
понималка строит каркас из предложения. Вместе они замыкают круг
«структура -> предложение -> структура», который советчики назвали пробой
центрального объекта: если движок речи верно собирает «фонарь освещал аптеку
на ночной улице», а понималка верно разбирает обратно в {predicate, subject,
object, place, time}, — значит ролевая структура реальна, а не сочинена
под конкретные фразы.

ГЛАВНЫЙ ЗАКОН ПРОЕКТА: ложная связь хуже отсутствующей. Поэтому понималка
обязана ОТКАЗАТЬ (вернуть None), когда текст не ложится ни в одну известную
раму, а не выдумывать каркас. Отказ и пустота — разные исходы (not_asked /
asked_empty): не поняла форму — это отказ, а не «ничего не знаю».

Цель контракта — ровно обратные каркасы к примерам 1-4 постановки хозяина
(те же, что говорунья уже умеет собирать). Список ниже задаёт пары
«текст -> ожидаемый каркас»; они же суть проба цикла: speaker_sentence над
каркасом обязана вернуть текст, лемматически равный исходному.

ИЗОЛЯЦИЯ ПАМЯТИ. Понималка НЕ пишет в мир: она читает форму предложения и
отдаёт каркас. Правила записи — это отдельный слой (core/records.py). Тест
проверяет ПРОЦЕСС, а не состояние после перезапуска.
"""
from __future__ import annotations

import unittest

from lang.ru.parser import parse_sentence
from lang.ru.speaker import speaker_sentence


# Обратные каркасы: текст -> каркас, который говорунья собирает обратно.
CIRCLE = [
    # 1) простое переходное прошедшего времени с местом
    ("фонарь освещал аптеку на ночной улице",
     {"predicate": "освещать", "subject": "фонарь", "object": "аптека",
      "place": "улица", "time": "ночь"}),
    # 2) безличный инфинитив с причиной и «при виде»
    ("дрожать от ужаса при виде чудовища",
     {"predicate": "дрожать", "cause": {"lemma": "ужас"},
      "view": {"lemma": "чудовище"}}),
    # 3) экзистенциальное настоящее с причиной и «по».
    # subject: pymorphy3 считает «мурашки» pluralia-tantum (лемма «мурашки»,
    # а не ед. «мурашка»), поэтому лемма честно «мурашки», а число — pl.
    ("от приятной музыки по коже бегут мурашки",
     {"predicate": "бежать", "predicate_tense": "present",
      "subject": "мурашки", "subject_number": "pl",
      "cause": {"lemma": "музыка", "adjective": "приятный"},
      "along": {"lemma": "кожа"}}),
    # 4) переходное настоящее с посессором объекта
    ("ветер шатает трубу твоего дома",
     {"predicate": "шатать", "predicate_tense": "present",
      "subject": "ветер",
      "object": {"lemma": "труба",
                  "possessor": {"lemma": "дом", "possessive": "твой"}}}),
]


class ParserKnownFrames(unittest.TestCase):
    """Каждый известный текст понималка разбирает в ровно ожидаемый каркас."""

    def test_each_text_parses_to_expected_frame(self):
        for text, frame in CIRCLE:
            got = parse_sentence(text)
            self.assertEqual(got, frame, f"разбор «{text}»")


class ParserCircle(unittest.TestCase):
    """Проба цикла: структура -> предложение -> структура держит смысл."""

    def test_speaker_round_trips_parse(self):
        for text, _ in CIRCLE:
            frame = parse_sentence(text)
            self.assertIsNotNone(frame, f"не разобрано: «{text}»")
            rendered = speaker_sentence(frame)
            # Говорилка и понималка должны сойтись по леммам в том же порядке:
            # повторный разбор выхода обязан дать ТОТ ЖЕ каркас.
            re_frame = parse_sentence(rendered)
            self.assertEqual(re_frame, frame,
                             f"круг разошёлся: «{text}» -> «{rendered}»")


class ParserHonestRefusal(unittest.TestCase):
    """Неизвестная рама -> честный отказ, а не выдуманный каркас."""

    def test_no_verb_refuses(self):
        # нет глагола — нечего собрать в каркас
        self.assertIsNone(parse_sentence("фонарь на ночной улице"))

    def test_word_soup_refuses(self):
        # слова есть, а ролевой рамы нет
        self.assertIsNone(parse_sentence("чудовище ужас при виде"))

    def test_unknown_role_connective_refuses(self):
        # «трубу к дому» — связь не из известных рам (к + дат. как роль не
        # существует у говорилки) — отказаться, а не выдумать «ветер шатает»
        self.assertIsNone(parse_sentence("ветер шатает трубу к дому"))


if __name__ == "__main__":
    unittest.main()