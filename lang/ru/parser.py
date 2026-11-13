# -*- coding: utf-8 -*-
"""
lang/ru/parser.py — понималка: текст -> ролевая структура (каркас говорилки).

Обратный полюс к speaker.py (кирпич №4, по плану советчиков 11.03).
Говорилка собирает предложение из каркаса {predicate, subject?, object?,
place?, time?, cause?, view?, along?}; понималка разбирает предложение обратно
в тот же каркас. Вместе они замыкают круг «структура -> предложение ->
структура» — пробу того, что ролевая структура реальна, а не сочинена под
конкретные фразы (см. tests/test_parser_contract.py::ParserCircle).

РОЛИ РЕШАЮТСЯ ПРЕДЛОГОМ И ПАДЕЖОМ, НЕ ПОРЯДКОМ СЛОВ — тот же закон, что у
экстрактора:
    «от» + род.          -> cause   (причина)
    «по» + местн.        -> along   (распространение: «по коже»)
    «на» + местн.        -> place   (место)
    «при виде» + род.    -> view    (идиома «при виде N»)
    имен. падеж          -> subject
    винит. падеж         -> object
    род. после объекта   -> possessor объекта

ПРИЛАГАТЕЛЬНОЕ ПРИНАДЛЕЖИТ СВОЕМУ СУЩЕСТВИТЕЛЬНОМУ, А НЕ СВОЕМУ ПАДЕЖУ.
pymorphy3 на «приятной» даёт твор., а слово прилеплено к «музыки» (род.).
Роль решает СУЩЕСТВИТЕЛЬНОЕ; прилагательное идёт туда, куда ушло его
существительное. Прилагательное места, к тому же, суть время («ночной» ->
time="ночь" через обратный словарю говорилки).

ГЛАВНЫЙ ЗАКОН ПРОЕКТА: ложная связь хуже отсутствующей. Если текст не
ложится ни в одну известную раму — понималка возвращает None, а не
выдумывает каркас. Отказ и пустота — разные исходы.

ИЗОЛЯЦИЯ. Понималка НЕ пишет в мир: читает форму предложения и отдаёт каркас.
Правила записи — отдельный слой (core/records.py).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from core.morph import Morph


# Ролевые предлоги понималки (зеркально говорилке speaker.py).
_CAUSE_PREP = "от"      # «от ужаса» -> cause
_ALONG_PREP = "по"      # «по коже» -> along
_PLACE_PREP = "на"      # «на ночной улице» -> place
_VIEW_PREP = "при"      # «при виде чудовища» -> view
_VIEW_NOUN = "вид"      # маркер идиомы: «при виде N»
_POSSESS_LEMMA = "твой" # притяжательное местоимение посессора

# Обратный словарю говорилки: прилагательное времени -> время-лемма.
_ADJ_TIME = {
    "ночной": "ночь",
    "зимний": "зима",
    "летний": "лето",
    "весенний": "весна",
}


def _tag(word: str) -> dict:
    """Наиболее вероятный разбор: {word, lemma, pos, case, number, grammemes}."""
    t = Morph.parse(word)
    t["word"] = word
    return t


def _lem(tok: dict) -> str:
    return tok.get("lemma") or tok["word"]


def parse_sentence(text: str) -> Optional[dict]:
    """
    Текст -> каркас говорилки, или None при честном отказе.
    """
    words = text.lower().strip().split()
    if not words:
        return None

    tokens: List[dict] = []
    for w in words:
        w = w.strip(".,!?;:«»\"'()")
        if not w:
            continue
        tokens.append(_tag(w))
    n = len(tokens)
    if n == 0:
        return None

    # ── глагол (VERB/INFN): из него predicate и время ──
    verb_idx: Optional[int] = None
    for i, t in enumerate(tokens):
        if t.get("pos") in ("VERB", "INFN", "PRTF", "GRND"):
            verb_idx = i
            break
    if verb_idx is None:
        return None                      # нет глагола — нечего собрать
    vtok = tokens[verb_idx]
    vgram = set(vtok.get("grammemes") or ())
    is_inf = "INFN" in (vtok.get("pos") or "")
    predicate = _lem(vtok)
    # Настоящее: verb может быть омонимичен по виду («бегут» = и «бежать»
    # impf+pres, и «побежать» perf+futr, скан поровну). Если лемма та же,
    # а среди вариантов есть present — берём present (обманывать нельзя:
    # будущее/прошлое у той же леммы не то же слово, что настоящее).
    vpres = _verb_is_present(vtok["word"], predicate)

    # привязка прилагательных к их существительному (head = индекс сущ.)
    for i, t in enumerate(tokens):
        if t.get("pos") == "NOUN":
            t["kind"] = "noun"
    for i, t in enumerate(tokens):
        if t.get("pos") in ("ADJF", "ADJS", "PRTF", "COMP"):
            for j in range(i + 1, n):
                if tokens[j].get("kind") == "noun":
                    t["head"] = j
                    break

    # ── проход с флагом предлога: роли по предлогу и падежу ──
    frame: Dict = {"predicate": predicate}
    subject = None
    subject_number = None
    cause = None
    cause_adj = None
    along = None
    view = None
    place = None
    time = None
    obj = None
    obj_adj = None
    poss = None
    poss_adj = None

    in_cause = False
    in_along = False
    in_place = False
    in_view = False      # после «при виде»: следующий род. = view
    seen_view_marker = False   # «вид» уже прошёл, род. за ним = view

    for i, t in enumerate(tokens):
        pos = t.get("pos")
        case = t.get("case") or ""
        lem = _lem(t)

        # предлог
        if pos == "PREP":
            if lem == _CAUSE_PREP:
                in_cause = True
            elif lem == _ALONG_PREP:
                in_along = True
            elif lem == _PLACE_PREP:
                in_place = True
            elif lem == _VIEW_PREP:
                # «при» должно открывать идиому «при виде»
                # маркер «вид» ищем среди следующих существительных
                j = i + 1
                marker = False
                while j < n:
                    nt = tokens[j]
                    if nt.get("kind") == "noun":
                        if _lem(nt) == _VIEW_NOUN and nt.get("case") == "loc":
                            marker = True
                            in_view = True
                        break
                    j += 1
                if not marker:
                    return None          # «при» без «вида» — чужая рама
            else:
                return None              # иной предлог — роль не из набора
            continue

        # имен. существительное -> субъект
        if pos == "NOUN" and case == "nom":
            if subject is None:
                subject = lem
                subject_number = "pl" if "plur" in vgram else "sg"
            continue

        # винит. существительное -> объект
        if pos == "NOUN" and case == "acc":
            if in_place or in_along:
                return None
            if obj is None:
                obj = lem
                obj_adj = _adj_at(tokens, i)
            continue

        # местн. существительное: роль по предлогу
        if pos == "NOUN" and case == "loc":
            if lem == _VIEW_NOUN:
                continue                 # маркер «вид» сам не роль
            if in_place:
                place = lem
                # прилагательное места -> время (если известно)
                a = _adj_at(tokens, i)
                if a in _ADJ_TIME:
                    time = _ADJ_TIME[a]
                in_place = False
            elif in_along:
                along = lem
                in_along = False
            elif in_view:
                # «при виде» уже дало род. (см. ниже); loc вне места/распр. — отказ
                return None
            else:
                return None
            continue

        # род. существительное
        if pos == "NOUN" and case == "gen":
            if in_view:
                view = lem
                in_view = False
            elif in_cause:
                if cause is None:
                    cause = lem
                    cause_adj = _adj_at(tokens, i)
                in_cause = False
            elif in_place or in_along:
                return None              # предлог места/распр. ждал местн.
            else:
                # род. после объекта -> посессор
                if obj is not None and poss is None:
                    poss = lem
                    poss_adj = _adj_at(tokens, i)
                else:
                    return None
            continue

        # прилагательное: роль ушла его существительному через head
        if pos in ("ADJF", "ADJS", "PRTF", "COMP"):
            continue

        # притяжательное местоимение («твой») — уходит как possessive
        if lem == _POSSESS_LEMMA:
            continue

        # прочее (частицы, служебные) — игнор
        continue

    # ── согласованность и сборка ──
    if is_inf:
        # безличный инфинитив: субъекта быть не должно
        if subject is not None:
            return None
        if cause is not None:
            c: Dict = {"lemma": cause}
            if cause_adj:
                c["adjective"] = cause_adj
            frame["cause"] = c
        if view is not None:
            frame["view"] = {"lemma": view}
        if vpres:
            frame["predicate_tense"] = "present"
        return frame

    # с субъектом
    if subject is None:
        return None
    frame["subject"] = subject
    # число субъекта — только когда не единственное (по умолчанию ед.ч.):
    # зеркально говорилке, где subject_number опускается для единственного.
    if subject_number == "pl":
        frame["subject_number"] = "pl"
    if vpres:
        frame["predicate_tense"] = "present"

    if cause is not None:
        c = {"lemma": cause}
        if cause_adj:
            c["adjective"] = cause_adj
        frame["cause"] = c
    if along is not None:
        frame["along"] = {"lemma": along}
    if view is not None:
        frame["view"] = {"lemma": view}
    if place is not None:
        frame["place"] = place
        if time is not None:
            frame["time"] = time
    if obj is not None:
        if poss is not None:
            p = {"lemma": poss}
            if poss_adj:
                p["possessive"] = poss_adj
            frame["object"] = {"lemma": obj, "possessor": p}
        else:
            frame["object"] = obj

    return frame


def _verb_is_present(word: str, top_lemma: str) -> bool:
    """Глагол в настоящем времени, даже если top-вариант омонимичен по виду.

    pymorphy3 на «бегут» даёт два равновероятных разбора: «бежать» impf+pres
    и «бежать» perf+futr. Оба — то же слово «бежать»; настоящее — верный
    выбор для не-прошедшей фразы. Проверяем present среди ВСЕХ вариантов с той
    же леммой, а не только в top."""
    try:
        for v in Morph.parse_all(word):
            if (v.get("lemma") or "") != top_lemma:
                continue
            if "pres" in set(v.get("grammemes") or ()):
                return True
    except Exception:
        return False
    return False


def _adj_at(tokens: List[dict], noun_idx: int) -> Optional[str]:
    """Прилагательное, прилепленное к существительному noun_idx."""
    for t in tokens:
        if t.get("head") == noun_idx:
            return _lem(t)
    return None


if __name__ == "__main__":
    for s in [
        "фонарь освещал аптеку на ночной улице",
        "дрожать от ужаса при виде чудовища",
        "от приятной музыки по коже бегут мурашки",
        "ветер шатает трубу твоего дома",
        "фонарь на ночной улице",
        "чудовище ужас при виде",
        "ветер шатает трубу к дому",
    ]:
        print(f"«{s}» -> {parse_sentence(s)}")