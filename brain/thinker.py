"""
brain/thinker.py — ДОМАШНЯЯ ДУМАЛКА: поиск связи, а не угадывание ответа.

ЧТО ОНА УМЕЕТ И ЧЕГО НЕ УМЕЕТ
-----------------------------
Отвечает на вопрос о СПОСОБНОСТИ («может ли X сделать Y»), которого у неё нет
в памяти: правильного ответа в этом файле нет ни в одной строке. Есть ПЕРЕБОР:

    носители  чем данный subject может оказаться (вверх по IS_A)
    действия  что эти понятия умеют (CAN_DO)
    объекты   над чем эти действия происходят (OBJECT_OF)
    цели      сам предмет вопроса и то, что он в себе содержит (HAS_PART)

Пересечение и есть ответ. Ход коня не учит 64 ответам, он учит одному правилу.

ЯЗЫКА ЗДЕСЬ НЕТ (закон границы, experiments/IR_BOUNDARY.md)
-----------------------------------------------------------
Русские фразы вопроса и ответа живут в lang/ru/ability.py: мозгу положено
получать РОЛИ (субъект, действие, объект) и отдавать СТРУКТУРУ. Первая версия
этого файла импортиала морфологию и сама собирала русское предложение —
измеритель границы поймал это как рост швов 6 -> 7. Правка была не «поднять
потолок», а убрать язык отсюда.

ГЛАВНОЕ ПРАВИЛО, КОТОРОГО НЕ ХВАТАЛО (замер 07.11)
---------------------------------------------------
Глагол ВОПРОСА обязан лежать на пути. Прежняя версия сравнивала только ЦЕЛЬ,
и «может ли дождь съесть огонь» получало ответ «да» по ребру «вода гасит
огонь»: вопрос про еду превращался в ответ про воду. Перенос способности
разрешён ровно двумя способами:
  * найденное действие = запрошенное или его синоксима (SYNONYM_OF);
  * найденное действие — ВИД запрошенного (найденное -IS_A-> запрошенное):
    умеет частное => умеет общее. Обратное запрещено: «вода гасит вообще» не
    доказывает, что она потушит именно лесной пожар.
Никакого переноса через HAS_PART/PART_OF (ведро содержит воду, но не тушит) и
через HAS_PROPERTY (мокрая дорожка не научилась мыть).

ОТКАЗ ЧЕСТНЫЙ (закон проекта: ложное ребро хуже отсутствующего)
---------------------------------------------------------------
  UNKNOWN_WORDS    в вопросе есть слово, которого в базе нет: это ПРОБЕЛ, по
                   нему обязаны учиться, а не запрещать.
  MISSING_ACTION   всё знакомо, действие над целью есть, но оно НЕ совпало с
                   действием вопроса; в gaps лежит пара «что связать».
  MISSING_CARRIER  действие подошло, но носить его некому: subject не является
                   и не содержит носителя этого действия (ведро содержит воду,
                   но ведро не вода). Тоже заявка на учёбу, но ДРУГОГО факта.
  NO_LINK          связывающего ребра нет вовсе.
  TOO_DEEP         связка есть глубже разрешённого — «не знаю» честнее
                   выдуманной причины.
Молчания среди кодов нет: третьего не дано по закону проекта.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

try:                                    # запуск как скрипта, не как пакета
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except Exception:                       # pragma: no cover
    pass

# Связи, по которым способность ПЕРЕНОСИТСЯ. Право переноса ИЗМЕРЕНО, а не
# выбрано по вкусу: related() наследования не делает ни по какой связи, поэтому
# перенос считается здесь — и обязан различать природу связи.
UPWARD_RELATIONS: Tuple[str, ...] = ("IS_A",)
PART_RELATIONS: Tuple[str, ...] = ("HAS_PART", "PART_OF")
# Чем одно действие может оказаться для другого. Только синоксия и вид.
ACTION_IDENTITY: Tuple[str, ...] = ("SYNONYM_OF",)

DEFAULT_MAX_HOPS = 4
DEFAULT_MAX_PATHS = 64
#: Сколько чужих действий над целью просмотреть на предмет совпадения. ЧИСЛО,
#: а не список разрешённых слов: список слов был бы тем самым зашитым ответом.
DEFAULT_MAX_ACTIONS = 32


@dataclass
class Link:
    """Одно найденное объяснение: чем subject воздействует на цель."""
    source: str          # чем оказался носитель
    action: str          # найденное действие
    target: str          # на чём оно происходит
    goal: str            # к чему привели части цели
    asked: str = ""      # каким словом спросили (для отчёта о разрыве)
    via: str = ""        # чем найденное действие пришлось запрошенному
    hops: Tuple[Tuple[str, str, str], ...] = ()


@dataclass
class Verdict:
    """Итог думания. Структура, не фраза: фразу собирает lang/ru."""
    question: str = ""
    subject: str = ""
    action: str = ""
    object: str = ""
    ok: bool = False
    code: str = "NO_LINK"
    links: List[Link] = field(default_factory=list)
    gaps: List[Tuple[str, str]] = field(default_factory=list)
    unknown_words: List[str] = field(default_factory=list)
    examined: int = 0
    depth_used: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def explains(self) -> bool:
        """Есть ли ХОТЬ ОДНО объяснение. Отказ объяснением не является."""
        return self.ok and bool(self.links)

    @property
    def multiple(self) -> bool:
        return self.explains and len(self.links) > 1


class Thinker:
    """
        think(question_text)            — сама попросит lang/ru разобрать
        think_link(subject, act, obj)   — готовые роли (контракты, поиск дыр)

    Ни один из входов не знает правильного ответа.
    """

    def __init__(self, knowledge, max_hops: int = DEFAULT_MAX_HOPS,
                 max_paths: int = DEFAULT_MAX_PATHS,
                 max_actions: int = DEFAULT_MAX_ACTIONS) -> None:
        self.k = knowledge
        self.max_hops = max_hops
        self.max_paths = max_paths
        self.max_actions = max_actions

    # ── вход ────────────────────────────────────────────────
    def think(self, question: str) -> Verdict:
        """Русский вопрос -> вердикт. Разбор отдаёт lang/ru, решение здесь."""
        from lang.ru.ability import parse_ability

        p = parse_ability(question)
        return self.think_link(str(p.get("subject", "")),
                               str(p.get("action", "")),
                               str(p.get("object", "")),
                               question=question,
                               negative=bool(p.get("negative")),
                               content=[str(c) for c in p.get("content", [])])

    def think_link(self, subject: str, action: str, obj: str,
                   question: str = "", negative: bool = False,
                   content: Sequence[str] = ()) -> Verdict:
        v = Verdict(question=question or f"{subject} {action} {obj}",
                    subject=subject, action=action, object=obj)

        # 1. ПРОБЕЛ, а не «нельзя».
        v.unknown_words = [w for w in dict.fromkeys(content)
                           if w and not self.k.knows(w)]
        unknown_core = [w for w in (subject, obj) if w and not self.k.knows(w)]
        if unknown_core:
            v.code = "UNKNOWN_WORDS"
            v.unknown_words = list(dict.fromkeys(unknown_core + v.unknown_words))
            return v

        # 2. ЦЕЛЬ: предмет вопроса или то, что он в себе содержит.
        goals: List[str] = [obj] if obj else ([subject] if subject else [])
        for rel in PART_RELATIONS:
            for part in self.k.related(obj, rel):
                if part not in goals:
                    goals.append(part)

        # 3. НОСИТЕЛИ: чем может оказаться subject.
        carriers = self._carriers(subject, self.max_hops)

        # 4. ПЕРЕБОР целей и действий над ними. Действие ОБЯЗАНО совпасть с
        #    запрошенным (синоксия или вид) — иначе это ответ на другой вопрос.
        seen = set()
        matched: List[Tuple[str, str, str]] = []   # (цель, действие, чем пришлось)
        for goal in goals[: self.max_paths]:
            v.examined += 1
            for verb, via in self._matching_verbs(goal, action):
                matched.append((goal, verb, via))
                for carrier, depth in carriers:
                    if verb not in self.k.related(carrier, "CAN_DO"):
                        continue
                    key = (carrier, verb, goal)
                    if key in seen:
                        continue
                    seen.add(key)
                    v.links.append(Link(
                        source=carrier, action=verb, target=goal, goal=goal,
                        asked=action, via=via,
                        hops=((carrier, "CAN_DO", verb),
                              (verb, "OBJECT_OF", goal))))
                    v.depth_used = max(v.depth_used, depth)

        if v.links:
            v.ok = not negative
            v.code = "LINK_FOUND" if not negative else "LINK_FOUND_NEGATED"
            if negative:
                # КОД, а не фраза: вывод разворачивает «не может ли» в «может»,
                # и сказать это по-русски обязан lang/ru, а не мозг.
                v.notes.append("NEGATED_QUESTION_BUT_LINK_FOUND")
            return v

        # 5. ОТКАЗ С ИМЕНЕМ НЕХВАТКИ. Три разных дыры, и путать их нельзя:
        #    действие над целью ЕСТЬ, но ДРУГОЕ        -> MISSING_ACTION
        #    действие подошло, а носить его некому    -> MISSING_CARRIER
        #    над целью нет ничего                     -> NO_LINK / TOO_DEEP
        # Различие стоит того: у первой дыры не хватает лексического факта о
        # языке, у второй — ребра в мире. Искать их нужно у разных источников.
        if matched:
            v.gaps = list(dict.fromkeys(
                (subject, verb) for _goal, verb, _via in matched))
            v.code = "MISSING_CARRIER"
            return v
        near = [verb for goal in goals[: self.max_paths]
                for verb in self._verbs_over(goal)]
        if near and action:
            v.gaps = list(dict.fromkeys(
                (action, verb) for verb in dict.fromkeys(near)))
            v.code = "MISSING_ACTION"
            return v

        v.code = ("TOO_DEEP" if self._link_exists_beyond(obj, subject, action)
                  else "NO_LINK")
        return v

    # ── вспомогательное ─────────────────────────────────────
    def _matches(self, found: str, asked: str) -> str:
        """Чем найденное действие приходится запрошенному ('' = не приходится).

        Разрешены ДВА переноса:
          ''            совпало дословно;
          'SYNONYM_OF'  видовая пара/синоксима (данные, а не правило);
          'IS_A'        найденное — ВИД запрошенного: умеет частное => умеет
                        общее. Обратное НЕЛЬЗЯ: вода гасит вообще не значит,
                        что она потушит именно лесной пожар.
        """
        if not asked:
            return "ANY"                       # вопрос без глагола: не гадаем
        if found == asked:
            return "SAME"
        for rel in ACTION_IDENTITY:
            if asked in self.k.related(found, rel) or \
               found in self.k.related(asked, rel):
                return rel
        if asked in self.k.related(found, "IS_A"):
            return "IS_A"
        return ""

    def _matching_verbs(self, goal: str, asked: str):
        """Действия над целью, прошедшие проверку на совпадение."""
        out = []
        for verb in self._verbs_over(goal)[: self.max_actions]:
            via = self._matches(verb, asked)
            if via and via != "ANY":
                out.append((verb, via))
        return out

    def _verbs_over(self, goal: str) -> List[str]:
        """
        Глаголы, происходящие НАД целью: головы наблюдений вида
        «гасить OBJECT_OF огонь». Обратный ход обязателен: мир пишет
        наблюдение ОДНОЙ строкой, у существительного CAN_DO нет вовсе, поэтому
        «что делают с огнём» находится только движением от хвоста к голове.
        """
        out: List[str] = []
        for head, rel in self.k.referenced_by(goal):
            if rel != "OBJECT_OF":
                continue
            if head and head not in out:
                out.append(head)
        return out

    def _carriers(self, subject: str, hops: int) -> List[Tuple[str, int]]:
        """Чем может оказаться носитель: он сам + всё выше по IS_A.

        Глубина нужна ОТВЕТУ: «дождь может, потому что он вода» — путь в один
        шаг обобщения, и выдавать его за прямое свойство нельзя.
        """
        carriers: List[Tuple[str, int]] = [(subject, 0)]
        known = {subject}
        frontier = [subject]
        for depth in range(1, max(0, hops) + 1):
            nxt: List[str] = []
            for node in frontier:
                for rel in UPWARD_RELATIONS:
                    for up in self.k.related(node, rel):
                        if up and up not in known:
                            known.add(up)
                            carriers.append((up, depth))
                            nxt.append(up)
            frontier = nxt
            if not nxt:
                break
        return carriers

    def _link_exists_beyond(self, obj: str, subject: str,
                            asked: str = "") -> bool:
        """
        Есть ли связка глубже разрешённого — чтобы отличить «связей нет» от
        «я дальше не ищу». Оба отказа честные, но разные; третий (молчание)
        запрещён законом проекта.

        Честность меры: та же думалка, та же логика пересечения, только
        глубина вдвое больше. Глубина удваивается ОДИН раз, не в цикле: обход
        обязан заканчиваться.
        """
        deep = self._carriers(subject, self.max_hops * 2)
        if len(deep) <= len(self._carriers(subject, self.max_hops)):
            return False        # глубже смотреть некуда: мир узкий, не мы
        goals = [obj] + [p for rel in PART_RELATIONS
                         for p in self.k.related(obj, rel)]
        for goal in goals[: self.max_paths]:
            for verb, _via in self._matching_verbs(goal, asked):
                for carrier, _d in deep:
                    if verb in self.k.related(carrier, "CAN_DO"):
                        return True
        return False


def think(world, question: str, **kw) -> Verdict:
    """Однострочный вход для мозга и контрактов: мир + вопрос -> вердикт."""
    from brain.simulator import StaticKnowledge

    k = world if hasattr(world, "related") else StaticKnowledge(world)
    return Thinker(k, **kw).think(question)
