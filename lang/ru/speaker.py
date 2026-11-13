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
    "3per": "3per", "pres": "pres", "inf": "INFN",
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


def _build_cause(cause: dict) -> str:
    """«от ужаса» / «от приятной музыки»: от + род., прилаг. согласуется.
    Вход: {lemma, adjective?}. Прилагательное причины берётся согласованным
    с существительным (род., ж.р., ед.ч. -> «приятной»)."""
    lemma = cause.get("lemma", "")
    adj = cause.get("adjective")
    case = "gen"
    pn = _number_of(None, lemma)
    pg = _gender_of(lemma)
    form = _infl(lemma, {case}, lemma)
    if adj:
        aword = _case_adjective(adj, case, pn, pg)
        return f"от {aword} {form}"
    return f"от {form}"


def _build_view(view: dict) -> str:
    """«при виде чудовища»: идиома «при виде» + дополнение в род. падеже.
    Вход: {lemma}."""
    lemma = view.get("lemma", "")
    gen = _infl(lemma, {"gen"}, lemma)
    return f"при виде {gen}"


def _build_along(along: dict) -> str:
    """«по коже»: по + дат. падеж. Вход: {lemma}."""
    lemma = along.get("lemma", "")
    dat = _infl(lemma, {"dat"}, lemma)
    return f"по {dat}"


def _object_word(obj, obj_case: str) -> Optional[str]:
    """Объект в падеже управления глагола.
    Вход: лемма-строка ИЛИ {lemma, possessor:{lemma, possessive?}}.
    possessive-оборот: «трубу ТВОЕГО ДОМА» — посессор в род., притяж.
    прилагательное согласуется с ним (род. ед.ч.)."""
    if isinstance(obj, dict):
        lemma = obj.get("lemma", "")
        ow = _infl(lemma, {obj_case}, lemma)
        poss = obj.get("possessor")
        if poss:
            plemma = poss.get("lemma", "")
            ppos = poss.get("possessive")
            pn = _number_of(None, plemma)
            pg = _gender_of(plemma)
            pgen = _infl(plemma, {"gen"}, plemma)
            if ppos:
                pword = _case_adjective(ppos, "gen", pn, pg)
                return f"{ow} {pword} {pgen}"
            return f"{ow} {pgen}"
        return ow
    return _infl(obj, {obj_case}, obj) if obj else None


def speaker_sentence(frame: Dict) -> str:
    """
    Ролевая структура -> предложение.

    frame: {predicate, subject?, object?, place?, time?, subject_number?,
            predicate_tense?, cause?, view?, along?}
    Вход рукописный (кирпич №1). Думалка сюда НЕ подключается.

    Расширение (примеры 2/3/4 постановки хозяина):
      * predicate_tense: "past" (по умолчанию) | "present";
      * cause: {lemma, adjective?} -> «от ужаса» / «от приятной музыки»;
      * view: {lemma} -> «при виде чудовища»;
      * along: {lemma} -> «по коже»;
      * object: лемма ИЛИ {lemma, possessor:{lemma, possessive?}} ->
        «трубу твоего дома».
    Без субъекта — безличный инфинитив: глагол ведёт, затем обстоятельства.
    """
    pred = frame.get("predicate", "")
    subj = frame.get("subject")
    obj = frame.get("object")
    place = frame.get("place")
    time = frame.get("time")
    subj_number = frame.get("subject_number")
    tense = frame.get("predicate_tense", "past")
    cause = frame.get("cause")
    view = frame.get("view")
    along = frame.get("along")

    obj_case = VERB_GOVERNMENT.get(pred, _DEFAULT_OBJ_CASE)

    # число субъекта — общее для подлежащего и согласования глагола
    snum = _number_of(subj_number, subj) if subj is not None else None

    # ── глагол: инфинитив / настоящее / прошедшее ──
    if subj is None:
        verb_word = _infl(pred, {"inf"}, pred)
    elif tense == "present":
        verb_word = _infl(pred, {"3per", snum}, pred)
    else:
        sgen = _gender_of(subj)
        verb_word = _verb_past(pred, snum, sgen)

    cause_word = _build_cause(cause) if cause else None
    view_word = _build_view(view) if view else None
    along_word = _build_along(along) if along else None
    obj_word = _object_word(obj, obj_case) if obj else None
    place_word = _build_place(place, time) if place else None

    parts: List[str] = []
    if subj is None:
        # безличный инфинитив: глагол ведёт, затем обстоятельства
        parts.append(verb_word)
        if cause_word:
            parts.append(cause_word)
        if view_word:
            parts.append(view_word)
    else:
        # с субъектом: обстоятельства причины и «по» открывают. При «по»-
        # обстоятельстве — экзистенциальная инверсия: глагол перед подлежащим
        # («по коже бегут мурашки»). Без него — подлежащее ведёт.
        if cause_word:
            parts.append(cause_word)
        if along_word:
            parts.append(along_word)
            parts.append(verb_word)
            parts.append(_infl(subj, {"nom", snum}, subj))
        else:
            parts.append(_infl(subj, {"nom", snum}, subj))
            parts.append(verb_word)
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
        {"predicate": "дрожать", "cause": {"lemma": "ужас"},
         "view": {"lemma": "чудовище"}},
        {"predicate": "бежать", "predicate_tense": "present",
         "subject": "мурашка", "subject_number": "pl",
         "cause": {"lemma": "музыка", "adjective": "приятный"},
         "along": {"lemma": "кожа"}},
        {"predicate": "шатать", "predicate_tense": "present",
         "subject": "ветер",
         "object": {"lemma": "труба",
                     "possessor": {"lemma": "дом", "possessive": "твой"}}},
    ]
    for f in samples:
        print(speaker_sentence(f))