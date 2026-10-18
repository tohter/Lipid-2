"""
brain/question.py — РАЗБОР ВОПРОСА ШАБЛОНАМИ (шаг C из п.29).

ЗАЧЕМ
  Симуляторы умеют отвечать, но вызывались вручную: why_stopped("пожар"),
  why_acted("кошка", "съесть"). Между свободным вопросом человека
  ("Почему кошка съела канарейку?") и вызовом симулятора не было звена.
  Этот модуль — звено.

ЧТО ДЕЛАЕТ И ЧЕГО НЕ ДЕЛАЕТ
  Разбирает вопрос на роли (тип / субъект / действие / объект) тем же
  core.morph, что и рассказ: одна морфология на проект, не две.
  НЕ отвечает на вопрос: ответ собирает симулятор из пути по данным.
  Учителя не спрашивает никогда (правило п.29: учитель заполняет базу и
  размечает события, но на вопрос не отвечает).

ПОЧЕМУ ШАБЛОН, А НЕ LLM
  Роли вопроса решаются падежом, как и факты рассказа. Это проверяемо по
  одному предложению, ошибка видна сразу. LLM — только fallback, и его
  отказ обязан быть громким (что именно не разобрано), не подменой ответа.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

if __package__ in (None, ""):
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from core.morph import analyze, split_sentences
# Типы вопросов.
WHY = "why"        # «почему ...»
WHAT = "what"      # «что такое ...» / «кто такой ...»
UNKNOWN = "unknown"

# Маркеры типа по словам вопроса.
WHY_MARKERS = ("почему", "отчего", "зачем", "из-за", "по какой причине")
WHAT_MARKERS = ("что такое", "кто такой", "кто такая", "что значит")




@dataclass
class Question:
    """Разобранный вопрос. ok=False — что именно не понято (missing)."""
    text: str
    qtype: str = UNKNOWN
    subject: str = ""     # о ком/о чём: ИМ.п.
    action: str = ""      # что сделал: глагол-сказуемое
    obj: str = ""         # над кем/чем: ВИН.п. без предлога
    missing: List[str] = field(default_factory=list)
    ok: bool = False

    def as_text(self) -> str:
        if self.ok:
            parts = [f"тип={self.qtype}", f"субъект={self.subject}"]
            if self.action:
                parts.append(f"действие={self.action}")
            if self.obj:
                parts.append(f"объект={self.obj}")
            return "вопрос: " + ", ".join(parts)
        return "не понял: " + "; ".join(self.missing)


def _detect_type(text: str) -> str:
    low = " " + (text or "").lower().strip() + " "
    for m in WHAT_MARKERS:
        if m in low:
            return WHAT
    for m in WHY_MARKERS:
        if m in low:
            return WHY
    return UNKNOWN


def _role_of(tok: dict) -> str:
    """Роль токена по его ЛУЧШЕМУ варианту разбора."""
    v = tok["variants"][0]
    pos, case = v["pos"], v["case"]
    if pos in ("VERB", "INFN"):
        return "action"
    if pos == "NOUN":
        if case == "nom":
            return "subject"
        if case == "acc":
            return "object"
    return "other"


def parse_question(text: str) -> Question:
    """Разобрать вопрос на тип и роли. Гадать запрещено."""
    q = Question(text=(text or "").strip())
    if not q.text:
        q.missing.append("пустой вопрос")
        return q
    q.qtype = _detect_type(q.text)
    if q.qtype == UNKNOWN:
        q.missing.append("тип вопроса не распознан (нет «почему/что такое»)")
        return q

    sents = split_sentences(q.text)
    toks = analyze(sents[0] if sents else q.text)
    content = [t for t in toks if t["content"]]

    # Действие — первый глагол/инфинитив.
    for t in content:
        if _role_of(t) == "action":
            q.action = t["variants"][0]["lemma"]
            break

    # Субъект — первое существительное в ИМ.п.; объект — в ВИН.п. без предлога.
    for t in content:
        r = _role_of(t)
        if r == "subject" and not q.subject:
            q.subject = t["variants"][0]["lemma"]
        elif r == "object" and not q.obj:
            q.obj = t["variants"][0]["lemma"]

    # «Что такое доверие?»: предмет вопроса — ПЕРВОЕ существительное, даже если
    # разбор дал ВИН.п. («доверие» омонимично им./вин.). Спрашивают не о
    # действии, а о слове — поэтому роль берётся не из падежа, а из позиции.
    if q.qtype == WHAT and not q.subject:
        for t in content:
            if t["variants"][0]["pos"] == "NOUN":
                q.subject = t["variants"][0]["lemma"]
                break

    # Для WHY нужен субъект; если есть глагол-прекращение — субъекта достаточно
    # (fire-симулятор сам ищет событие «subject прекратился»).
    if not q.subject:
        q.missing.append("не найден субъект (ИМ.п.)")
    if q.qtype == WHY and not q.action:
        q.missing.append("не найдено действие (глагол) ")
    if q.qtype == WHAT and not q.subject:
        q.missing.append("не найден предмет вопроса (что/кто)")

    q.ok = not q.missing
    return q


def candidate_schemes(sims, roles):
    """
    Все схемы всех симуляторов, подходящие под роли вопроса.

    Отсев делает САМА СХЕМА (Scheme.matches по requires/forbids/stop_actions),
    поэтому в этом модуле нет ни одного имени мира и ни одного списка глаголов:
    какие вопросы объясняет данная схема — решение ДАННЫХ этого мира.

    Порядок — по priority, затем по ключу мира и имени схемы, чтобы перебор не
    зависел от порядка, в котором вызывающий положил симуляторы в словарь.
    """
    out = []
    for key, sim in (sims or {}).items():
        for name, scheme in (getattr(sim, "schemes", None) or {}).items():
            if scheme.matches(roles):
                out.append((scheme.priority, str(key), name, sim, scheme))
    out.sort(key=lambda c: (c[0], c[1], c[2]))
    return out


def answer(text, sims):
    """
    Ответить на вопрос, выбрав СХЕМУ ИЗ ДАННЫХ, а не по имени симулятора.

    sims — словарь доступных симуляторов {ключ: <Simulator>}. Ключ нужен только
    вызывающему (GUI), маршрутизация его не читает: класть можно сколько угодно
    миров, хоть один.

    Порядок: разобрали вопрос -> candidate_schemes отобрала подходящие схемы ->
    каждая ПРОБУЕТСЯ, первая успешная и есть ответ. Схема, прошедшая отсев но не
    прошедшая путь, НЕ замолкается: причина провала каждой из них попадает в
    missing, иначе «не ответил» неотличимо от «не смог».
    """
    from brain.simulator import SimulationResult

    q = parse_question(text)
    res = SimulationResult(q.text or text)
    if not q.ok:
        res.missing.extend(q.missing)
        return res

    roles = {"subject": q.subject, "action": q.action, "obj": q.obj,
             "qtype": q.qtype}
    cands = candidate_schemes(sims, roles)
    if not cands:
        have = sorted(k for k, v in roles.items() if v)
        known = []
        for key, sim in (sims or {}).items():
            for name, scheme in (getattr(sim, "schemes", None) or {}).items():
                need = list(scheme.requires) + list(scheme.stop_actions)
                known.append(f"«{name}» (мир «{key}») ждёт {need}")
        res.missing.append(
            f"ни одна схема не подошла; роли вопроса: {have}")
        res.missing.append("доступные схемы: " + ("; ".join(known)
                                                 if known else "НИ ОДНОЙ"))
        return res

    for _prio, key, name, sim, scheme in cands:
        if not scheme.handler:
            res.missing.append(f"схема «{name}» (мир «{key}») не объявила "
                               "handler — непонятно, каким методом её звать")
            continue
        handler = getattr(sim, scheme.handler, None)
        if handler is None:
            res.missing.append(f"у симулятора мира «{key}» нет метода "
                               f"«{scheme.handler}», который требует схема "
                               f"«{name}»")
            continue
        if not scheme.args:
            res.missing.append(f"схема «{name}» не объявила args — непонятно, "
                               "с чем её вызывать")
            continue
        vals, missing_role = [], None
        for role in scheme.args:
            val = roles.get(role)
            if not val:
                missing_role = role
                break
            vals.append(val)
        if missing_role:
            res.missing.append(f"схеме «{name}» не хватает роли «{missing_role}»")
            continue
        r = handler(*vals)
        if getattr(r, "ok", False):
            return r
        reasons = getattr(r, "missing", None) or ["не ответила без причины"]
        res.missing.extend(f"схема «{name}» (мир «{key}»): {m}" for m in reasons)

    if not res.missing:
        res.missing.append("подошедшие схемы смолчали без причины — это баг")
    return res
