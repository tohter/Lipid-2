# -*- coding: utf-8 -*-
"""
lang/ru/speaker.py — ГОВОРИЛКА-КОНСТРУКТОР (кирпич №1 по плану советчиков).

Задача: ручная ролевая структура -> ОДНО грамматически корректное русское
предложение. Это самый изолированный кирпич: вход детерминирован, думалка не
участвует, выход сверяется строкой.

Философия (по консультациям Жозефины и босса): движок НЕ генерирует текст
«как человек», он раскладывает семантическую структуру по грамматическому
скелету. Русский — только выходной интерфейс; смысловая структура языково-
независима.

ЧТО ДЕЛАЕТ МОДУЛЬ
-----------------
1. ИНФЛЕКТ. По лемме и граммемам возвращает словоформу (pymorphy3).
2. УПРАВЛЕНИЕ ГЛАГОЛА. Таблица падежей, которые глагол назначает объекту
   («освещать» -> винительный: аптека -> аптеку).
3. СОГЛАСОВАНИЕ ГЛАГОЛА ПРОШЕДШЕГО ВРЕМЕНИ с субъектом: фонарь ОСВЕЩАЛ,
   лампа ОСВЕЩАЛА (род), фонари ОСВЕЩАЛИ (число). Род/число субъекта берётся
   из морфологии.
4. ОБСТОЯТЕЛЬСТВО МЕСТА. «улица» -> loc с предлогом «на». Время «ночь»
   превращается в согласованное прилагательное «ночной».
5. ДЕРИВАЦИЯ (время->прилагательное) — по словарю, а не по правилу: босс
   назвал это опасностью №4 («ночной» не выводится композиционно).
6. СБОРКА: канонический порядок [субъект] [действие] [объект] [место].

НЕ СОДЕРЖИТ думалки и поиска недостающих понятий — это следующий кирпич.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from core.morph import Morph

# ── инфлект ────────────────────────────────────────────────────────
# Замечено: нетривиальный инфлект решает pymorphy3, а не правила окончаний.
# ЛОВУШКА, замеренная сегодня: pymorphy3.inflect требует ПОЛНЫЕ граммемы
# (accs/loct/sing/plur/masc/femn), а короткие имена (acc/loc/sg/pl/m/f) —
# из того же разговорника, что core.morph.CASE, — молча не распознаёт и
# возвращает лемму. Поэтому здесь обратный словарь: короткое -> полное.
_GRAM = {
    "nom": "nomn", "acc": "accs", "dat": "datv", "gen": "gent",
    "ins": "ablt", "loc": "loct",
    "sg": "sing", "pl": "plur",
    "m": "masc", "f": "femn", "n": "neut",
    "past": "past",
}


def _infl(lemma: str, grams, failback: Optional[str] = None) -> str:
    """Лемма + короткие граммемы -> словоформа. При неудаче — лемма/запас."""
    if not lemma:
        return ""
    full = {_GRAM.get(g, g) for g in grams}
    if Morph.available and Morph._m is not None:
        try:
            p = Morph._m.parse(lemma)[0]
            form = p.inflect(full)
            if form is not None:
                return form.word
        except Exception:
            pass
    return failback if failback is not None else lemma


def _gender_of(lemma: str) -> str:
    """Род существительного-леммы по словарю: m/f/n ('' если не решено)."""
    if Morph.available and Morph._m is not None:
        try:
            for v in Morph._m.parse(lemma):
                if v.tag.POS == "NOUN" and v.tag.gender is not None:
                    g = str(v.tag.gender)
                    return {"masc": "m", "femn": "f", "neut": "n"}.get(g, "")
        except Exception:
            pass
    return ""


def _number_of(subject_number: Optional[str], lemma: str) -> str:
    """Число субъекта: явный флаг или по морфологии леммы."""
    if subject_number:
        return subject_number if subject_number in ("sg", "pl") else "pl"
    if Morph.available and Morph._m is not None:
        try:
            for v in Morph._m.parse(lemma):
                if v.tag.POS == "NOUN" and v.tag.number is not None:
                    n = str(v.tag.number)
                    if n == "plur":
                        return "pl"
            return "sg"
        except Exception:
            pass
    return "sg"


def _case_adjective(adj_lemma: str, noun_form_case: str, noun_number: str,
                    noun_gender: str) -> str:
    """Согласовать прилагательное с существительным: те же падеж/число/род."""
    grams = {noun_form_case}
    if noun_number:
        grams.add(noun_number)
    if noun_gender and noun_number == "sg":
        grams.add(noun_gender)
    return _infl(adj_lemma, grams, adj_lemma)


# ── управление глаголов (объект: падеж) ───────────────────────────
# Первый кирпич — малая таблица. Расширяется сознательно, а не тихо.
VERB_GOVERNMENT: Dict[str, str] = {
    "освещать": "acc",
    "осветить": "acc",
    "видеть": "acc",
    "видать": "acc",
    "держать": "acc",
    "шатать": "acc",
    "строить": "acc",
    "любить": "acc",
    "покупать": "acc",
    "петь": "acc",
    "слушать": "acc",
    "читать": "acc",
    "писать": "acc",
}
# Глаголы по умолчанию: прямой переходный берёт винительный.
_DEFAULT_OBJ_CASE = "acc"

# ── деривация: время -> прилагательное (словарь, а не правило) ────
TIME_ADJECTIVE: Dict[str, str] = {
    "ночь": "ночной",
    "зима": "зимний",
    "лето": "летний",
    "весна": "весенний",
    "осень": "осенний",
    "утро": "утренний",
    "вечер": "вечерний",
    "день": "дневной",
}

# Предлоги обстоятельств: место — «на», причина — «от», средство — «по».
_PLACE_PREP = "на"
_TIME_PREP = ""


def _verb_past(lemma: str, number: str, gender: str) -> str:
    """Глагол в прошедшем времени, согласованный с субъектом в числе/роде."""
    grams = {"past"}
    if number:
        grams.add(number)
    if gender and number == "sg":
        grams.add(gender)
    return _infl(lemma, grams, lemma)


def _build_place(place: str, time: Optional[str]) -> str:
    """«на ночной улице»: существительное места в loc + прилаг. времени."""
    case = "loc"
    pn = _number_of(None, place)  # число места (обычно sg)
    pg = _gender_of(place)
    form = _infl(place, {case}, place)
    if time and time in TIME_ADJECTIVE:
        adj = _case_adjective(TIME_ADJECTIVE[time], case, pn, pg)
        return f"{_PLACE_PREP} {adj} {form}"
    return f"{_PLACE_PREP} {form}"


def speaker_sentence(frame: Dict) -> str:
    """
    Ролевая структура -> предложение.

    frame: {predicate, subject, object?, place?, time?, subject_number?}
    Вход рукописный (кирпич №1). Думалка сюда НЕ подключается.
    """
    pred = frame.get("predicate", "")
    subj = frame.get("subject", "")
    obj = frame.get("object")
    place = frame.get("place")
    time = frame.get("time")
    subj_number = frame.get("subject_number")

    # субъект в именительном
    snum = _number_of(subj_number, subj)
    sgen = _gender_of(subj)
    subj_word = _infl(subj, {"nom"}, subj)

    # глагол прошедшего времени, согласован с субъектом
    verb_word = _verb_past(pred, snum, sgen)

    # объект: падеж по управлению глагола
    obj_case = VERB_GOVERNMENT.get(pred, _DEFAULT_OBJ_CASE)
    obj_word = _infl(obj, {obj_case}, obj) if obj else None

    # обстоятельство места (согласовано с временем-прилагательным)
    place_word = _build_place(place, time) if place else None

    parts: List[str] = [subj_word, verb_word]
    if obj_word:
        parts.append(obj_word)
    if place_word:
        parts.append(place_word)
    return " ".join(parts)


if __name__ == "__main__":
    samples = [
        {"predicate": "освещать", "subject": "фонарь", "object": "аптека",
         "place": "улица", "time": "ночь"},
        {"predicate": "освещать", "subject": "лампа", "object": "аптека",
         "place": "улица", "time": "ночь"},
        {"predicate": "освещать", "subject": "фонарь", "subject_number": "pl",
         "object": "аптека", "place": "улица", "time": "ночь"},
        {"predicate": "освещать", "subject": "фонарь", "object": "аптека"},
    ]
    for f in samples:
        print(speaker_sentence(f))