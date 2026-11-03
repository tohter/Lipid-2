"""
brain/simulator.py — ЧЕРНОВИК симулятора Lipid-2.

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ
----------------------
Обучение и симуляция — РАЗНЫЕ типы знания (согласовано с хозяином):

  * СТАТИКА (база знаний, обучение) отвечает на «что есть» и «что может»:
    дождь IS_A вода, вода CAN_DO гасить, пожар HAS_PART огонь. Вне времени.
  * ДИНАМИКА (симулятор) отвечает на «что и когда произошло»:
    начался пожар -> полил дождь -> пожар остановился. Только здесь есть
    время, порядок и (позже) состояния одушевлённых (эмоции 0..255).

Статика НЕ умеет ответить на «почему» — в ней нет порядка событий. Поэтому
вопрос «почему прекратился пожар» — вопрос К СИМУЛЯТОРУ, опирающемуся на
статику. Симулятор здесь пока не читает текст: события задаются руками (как
договорено), чтобы проверить саму механику вывода.

ЧЕСТНОСТЬ ВЫВОДА
----------------
Симулятор не подставляет готовый ответ. Он обязан ПРОЙТИ путь
     дождь --(IS_A)--> вода --(CAN_DO)--> гасить --(OBJECT_OF)--> огонь
и убедиться, что это тот огонь, что внутри пожара (пожар --HAS_PART--> огонь).
Ответ собирается ИЗ ПРОЙДЕННОГО ПУТИ, и путь предъявляется. Если звена нет —
симулятор говорит, КАКОГО звена нет, а не молчит (тихая деградация = баг).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Корень проекта в sys.path — чтобы запускать как `python brain/simulator.py`
# (тот же приём, что в core/cortex.py и brain/social_sim.py).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.schemes import SchemeInterpreter, load_schemes


# ─────────────────────────────────────────────────────────────
# СОБЫТИЕ — единица динамики. Статике такое неведомо.
# ─────────────────────────────────────────────────────────────

@dataclass
class Event:
    """Одно событие рассказа. order — позиция во времени (0,1,2...)."""
    subject: str
    action: str
    obj: str = ""
    order: int = 0
    text: str = ""

    def describe(self) -> str:
        tail = f" -> {self.obj}" if self.obj else ""
        return f"{self.subject} {self.action}{tail}"


# ─────────────────────────────────────────────────────────────
# ЧТЕНИЕ СТАТИКИ: симулятор только ЧИТАЕТ базу знаний, не пишет.
# ─────────────────────────────────────────────────────────────

class StaticKnowledge:
    """
    Обёртка над миром (entities/observations), дающая ТИПИРОВАННЫЕ вопросы:

        is_a(x)          -> [родители...]      (IS_A: x есть <родитель>)
        parts(x)         -> [части...]        (HAS_PART: у x есть <часть>)
        can_do(x)        -> [умения...]       (CAN_DO: x умеет <умение>)
        object_of(x)     -> [объекты...]      (OBJECT_OF: x направлено на <объект>)
        causes(x)        -> [следствия...]    (CAUSES)
        referenced_by(x, rel) -> [(кто, rel)]  ОБРАТНЫЙ вопрос: кто ссылается

    Только чтение: у симулятора нет прав менять базу (та же дисциплина, что у
    предиктивного цикла).

    ЗАЧЕМ ОБРАТНЫЙ ВОПРОС (П2, замер experiments/P2_BEFORE.txt). Кирпичи
    загадки «невестка Зинаиды = жена Егора» в мире есть:
        невестка IS_ROLE жена   жена PART_OF сын
        егор     IS_ROLE сын    сын  PART_OF зинаида
    а пути нет: все вопросы выше спрашивают «КУДА ссылается x». Шаг
    «КТО называет «сын» своей ролью» (он же егор IS_ROLE сын) был НЕЧЕМ,
    и цепочка обрывалась на середине — при полном знании. Замер на ОДНОМ
    и том же мире: вперёд — «нет пути», в обе стороны — путь есть.
    Значит дыра была в приборе, а не в знании.

    ОБРАТНЫЙ ИНДЕКС СТРОИТСЯ КАК ТРАНСПОЗИЦИЯ _by_head, а не вторым проходом
    по миру. Это не лень, а требование: два независимых прохода по одним и
    тем же наблюдениям расходятся при любом будущем изменении источника
    ( поля hierarchy уже исключение, и оно обязано быть видно в обе
    стороны). Транспозиция делает расхождение невозможным по построению.
    """

    def __init__(self, world: Dict):
        self.world = world or {}
        self._by_head: Dict[str, List[Tuple[str, str]]] = {}
        self._readable_cache = None
        # Имена, которыми ОТВЕЧАЮТ на вопрос related(). Реестр — потому что
        # отношение могут спросить ДО того, как кто-то его записал; рёбра
        # мира — потому что спрошенное из мира не имеет права называться
        # «нельзя прочитать».
        try:
            from core.storage import RelationType
            self._registry = {e.name for e in RelationType}
        except Exception:      # ядро без реестра живёт; тогда мерило — мир
            self._registry = set()
        # 1) Типизированные наблюдения (главный источник связей).
        for obs in (self.world.get("observations") or {}).values():
            if not isinstance(obs, dict) or obs.get("negated"):
                continue
            head = str(obs.get("head", "")).lower().strip()
            rel = str(obs.get("relation", "")).upper().strip()
            tail = str(obs.get("tail", "")).lower().strip()
            if head and tail:
                self._by_head.setdefault(head, []).append((rel, tail))
        # 2) Поля сущностей (hierarchy / actions). Они тоже ЗНАНИЕ, и мозг их
        #    использует (п.12: рекурсивные предки; action_profile для глаголов).
        #    Читать только observations значило бы, что «пламя IS_A огонь» из
        #    поля hierarchy невидимо симулятору — и вывод «гасит пламя» падал
        #    бы при живом знании в базе. Это ЗАМЕРЕНО на тесте «пламя».
        for name, ent in (self.world.get("entities") or {}).items():
            if not isinstance(ent, dict):
                continue
            low = str(name).lower().strip()
            for parent in (ent.get("hierarchy") or []):
                parent = str(parent).lower().strip()
                if parent:
                    self._by_head.setdefault(low, []).append(("IS_A", parent))
            for act in (ent.get("actions") or []):
                if isinstance(act, dict):
                    act = act.get("name", "")
                act = str(act).lower().strip()
                if act:
                    self._by_head.setdefault(low, []).append(("CAN_DO", act))
        # Обратный индекс — ПОСЛЕ того как _by_head собран целиком, включая
        # поля hierarchy/actions: иначе обратный ход видел бы наблюдения, но
        # не знал бы про предков, и два хода разошлись бы тихо.
        self._build_reverse()

    def _build_reverse(self) -> None:
        """_by_tail: хвост -> [(голова, отношение)]. Строится ИЗ _by_head."""
        self._by_tail: Dict[str, List[Tuple[str, str]]] = {}
        for head, pairs in self._by_head.items():
            for rel, tail in pairs:
                self._by_tail.setdefault(tail, []).append((head, rel))
        # Узлы, о которых миру что-то известно. Нужны НЕ для красоты: пустой
        # ответ про знакомый узел означает «ссылок нет», а про незнакомый —
        # «такого слова в мире нет». Свалить это в один ноль значит однажды
        # прочитать «у невестки нет хозяина» там, где невестки не существует.
        self._known = set(self._by_head) | set(self._by_tail)
        self._known.update(str(n).lower().strip()
                           for n in (self.world.get("entities") or {}))

    def knows(self, x: str) -> bool:
        return (x or "").lower().strip() in self._known

    def referenced_by(self, x: str, rel: str = "") -> List[Tuple[str, str]]:
        """
        ОБРАТНЫЙ ЗАПРОС: кто и каким отношением ссылается на x.

            referenced_by("сын", "IS_ROLE")  -> [("егор", "IS_ROLE")]
            referenced_by("сын", "PART_OF")  -> [("жена", "PART_OF")]

       rel пустой -> все входящие ссылки сразу. Ничего нового метод не
        выводит: он ОТДАЁТ РОВНО рёбра мира (замок
        test_reverse_invents_nothing). Неизвестный узел — KeyError, а не
        пустой список: отказ обязан быть ОТЛИЧИМ от «ссылок нет».
        """
        key = (x or "").lower().strip()
        if key not in self._known:
            raise KeyError(f"в мире нет узла «{x}» — это НЕ то же самое, "
                           f"что «на него никто не ссылается»")
        out = []
        for head, r in self._by_tail.get(key, []):
            if rel and r != (rel or "").upper().strip():
                continue
            if (head, r) not in out:
                out.append((head, r))
        return out

    def _read(self, x: str, rel: str) -> List[str]:
        """ЕДИНСТВЕННОЕ место, где чтение связей вообще происходит.
        Приватное и НЕ проверяющее имя: им пользуются обёртки, которым нужна
        ПОЛНОТА (схемы подают литералы и обязаны деградировать в «не хватило»,
        а не падать). Проверять имя — задача related()."""
        out = []
        for r, t in self._by_head.get((x or "").lower().strip(), []):
            if r == rel and t not in out:
                out.append(t)
        return out

    #: Все имена отношений, которые ЭТОТ мир реально пишет, плюс реестр.
    def _readable(self) -> set:
        if self._readable_cache is None:
            names = {r for pairs in self._by_head.values() for r, _t in pairs}
            names.update(self._registry)
            self._readable_cache = names
        return self._readable_cache

    def related(self, x: str, rel: str) -> List[str]:
        """
        ПУБЛИЧНЫЙ ЧИТАТЕЛЬ ПО ИМЕНИ ОТНОШЕНИЯ — то, чего в дыре F не было.

            related("невестка", "IS_ROLE")     -> ["жена"]
            related("жена",      "PART_OF")    -> ["сын"]
            related("гостить",   "LOCATED_IN") -> ["семья"]

        ЗАЧЕМ ОН ОДИН, а не шесть новых. В реестре 17 имён (core.storage.
        RelationType), а специализированных методов было шесть. Значит
        ЛЮБОЕ следующее отношение — INSTRUMENT, RECIPIENT, ABOUT, VALUE —
        родилось бы нечитаемым, и дыра повторилась бы молча. Один вход по
        имени закрывает класс, а не список.

        ДВА ОТВЕТА РАЗДЕЛЕНЫ ОСОЗНАННО (и это НЕ случайная асимметрия с
        referenced_by):
          * неизвестное ОТНОШЕНИЕ  -> KeyError. Имя отношения — параметр
            программы; опечатка (IS_ROL) не имеет права становиться «знаний
            нет», иначе вывод замолчит вместо поломки.
          * неизвестный УЗЕЛ       -> []. Обход путей тыкает в соседей, и
            отсутствие узла — обычный вопрос к миру, на который честен
            пустой список. Для узла есть ОТДЕЛЬНЫЙ вопрос: knows(x).
        referenced_by строго к узлу, потому что отвечает на вопрос «КТО
        ссылается», а спрашивать его у несуществующего слова — ошибка
        вопроса, а не пустой факт (заперто контрактом П2).
        """
        name = (rel or "").upper().strip()
        if name not in self._readable():
            raise KeyError(f"отношение «{rel}» нельзя прочитать: его нет ни в "
                           f"реестре, ни в этом мире — это НЕ то же самое, "
                           f"что «таких связей нет»")
        return self._read(x, name)

    def _tails(self, x: str, rel: str) -> List[str]:
        return self._read(x, (rel or "").upper().strip())

    def is_a(self, x: str) -> List[str]:
        return self._tails(x, "IS_A")

    def parts(self, x: str) -> List[str]:
        return self._tails(x, "HAS_PART")

    def can_do(self, x: str) -> List[str]:
        return self._tails(x, "CAN_DO")

    def object_of(self, x: str) -> List[str]:
        return self._tails(x, "OBJECT_OF")

    def causes(self, x: str) -> List[str]:
        return self._tails(x, "CAUSES")

    def action_purpose(self, x: str) -> List[str]:
        return self._tails(x, "PURPOSE")

    # ------------------------------------------------------------------
    # ПРАВИЛО ШАГА (П4). Обход обязан уметь ОТКАЗАТЬ.
    # ------------------------------------------------------------------

    def is_event(self, x: str) -> bool:
        """Событие ли это по ЗАПИСИ МИРА.

        ТИП ЧИТАЕТСЯ ИЗ МИРА, А НЕ ИЗ ЯДРА — и это не придирка, а замер
        (experiments/P4_STEP_RULE.txt): узел ядра на «гостить» отвечает
        NodeType.ENTITY (ACTION не у ОДНОГО из двадцати узлов рассказа), и
        node_type как РЕШЕНИЕ в проекте не читает никто. Спроси правило типа
        у ядра — оно не запретило бы РОВНО НИЧЕГО и осталось бы зелёным над
        ложью. Полю мира kind == "action" доверяют четыре читателя мозга
        (brain/lipid2_brain.py:1355, :1454, :1456, brain/learner.py:688).

        ОТСУТСТВИЕ ЗАПИСИ — НЕ СОБЫТИЕ. У лиц kind нет ВООБЩЕ (зинаида, егор,
        жена, сын не лежат в entities). Считать неизвестное событие было бы
        тем же грехом, что свести «нет ребра» и «нельзя спросить» к одному
        нулю: запрет стал бы всеобщим и честный путь исчез бы вместе с ложью.
        """
        ent = (self.world.get("entities") or {}).get(
            (x or "").lower().strip())
        return isinstance(ent, dict) and str(ent.get("kind") or "") == "action"

    def _neighbors(self, node: str):
        """Соседи в обе стороны, ЛЮБЫМ отношением. Порядок не важен: обход
        ширинный, а ответ — путь, а не первый случай."""
        for rel, tail in self._by_head.get(node, []):
            yield tail, f"{rel}>"
        for head, rel in self._by_tail.get(node, []):
            yield head, f"<{rel}"

    def _walk(self, start: str, goal: str, forbidden: frozenset,
              max_hops: int):
        """Обход в обе стороны, НЕ ЗАХОДЯ в forbidden. Путь или None."""
        from collections import deque
        if start == goal:
            return []
        seen = {start} | set(forbidden)
        q = deque([(start, [])])
        while q:
            node, trail = q.popleft()
            if len(trail) >= max_hops:
                continue
            for nxt, mark in self._neighbors(node):
                if nxt in seen:
                    continue
                step = (node, mark.strip("<>"), nxt)
                if nxt == goal:
                    return trail + [step]
                seen.add(nxt)
                q.append((nxt, trail + [step]))
        return None

    def identity_path(self, start: str, goal: str,
                      max_hops: int = 4) -> "PathResult":
        """Путь ТОЖДЕСТВА ЛИЦА: можно ли от одного лица осмысленно заключить
        про другое. ОТВЕТ — СТРУКТУРА с причиной, а не список рёбер.

            identity_path("невестка", "егор").path     -> честная цепочка
            identity_path("невестка", "егор").code     -> "OK"

        ЗАЧЕМ ОН ВООБЩЕ НУЖЕН, если есть related(). Замер рассказа (см.
        experiments/STORY_WITHOUT_LESSON.txt): обход, которому разрешено
        ходить ЛЮБЫМ отношением, отвечает на «невестка и Егор» так:

            невестка <-OBJECT_OF- недолюбливать -CAN_DO-> зинаида
                     -CAN_DO-> гостить -OBJECT_OF-> егор

        Ни одно ребро не выдумано, вывод ЛОЖЕН: цепочка прошла сквозь
        событие и объявила связь лиц. Отказ должен быть возможен ДО ответа,
        потому что ложная связь для этого проекта хуже отсутствующей.

        ЗАПРЕТОМ ПРОХОДА, А НЕ ТРЕБОВАНИЕМ ЛИЦА. У лиц типа нет (см.
        is_event), поэтому «разрешены только известные лица» запретило бы
        зинаиду, егора и жену — то есть убило бы и честный путь
        «невестка -IS_ROLE-> жена -PART_OF-> сын <-IS_ROLE- егор».
        Запрещено только ОДНО: оказаться внутри события по дороге.

        КОДЫ (IR, их читает вывод; русскую фразу в ответ выводам не кормят —
        тот же закон границы, что с unreadable у дыры F):
          OK              путь есть
          UNKNOWN_NODE    такого слова в мире нет (вопрос к словарю, не к)
          NO_PATH         слова знакомы, связи кончились (вопрос к базе)
          BLOCKED_BY_EVENT путь СУЩЕСТВУЕТ, но весь проходит сквозь события
                      (вопрос к ВЫВОДУ; врать отказались)
        """
        a = (start or "").lower().strip()
        b = (goal or "").lower().strip()
        if not self.knows(a) or not self.knows(b):
            return PathResult([], [], "UNKNOWN_NODE")
        events = frozenset(n for n in self._known if self.is_event(n))
        path = self._walk(a, b, events, max_hops)
        if path is not None:
            return PathResult(path, [], "OK")
        # Путь запретили МЫ, а не мир. Проверить это можно только одним
        # способом: снять запрет и посмотреть, был ли путь. Делать это надо
        # ПОСЛЕ отказа, а не вместо: иначе «запрет» нельзя отличить от
        # «связей и правда нет», а это разные починки (вывод против базы).
        loose = self._walk(a, b, frozenset(), max_hops)
        if loose is None:
            return PathResult([], [], "NO_PATH")
        crossed = sorted({n for h, _r, t in loose for n in (h, t)
                          if n in events})
        return PathResult([], crossed, "BLOCKED_BY_EVENT")


class PathResult:
    """ОТВЕТ на запрос пути тождества: путь + ПРИЧИНА отказа кодом.

    Поле code — IR. Русскую формулировку (если она понадобится человеку)
    имеет право ставить мозг, не этот класс: см. unreadable_to_text и закон
    границы A0.
    """

    __slots__ = ("path", "blocked", "code")

    def __init__(self, path, blocked, code):
        self.path = list(path or [])
        self.blocked = list(blocked or [])
        self.code = code

    def ok(self) -> bool:
        return self.code == "OK"

    def __repr__(self):
        return (f"PathResult(code={self.code!r}, path={self.path}, "
                f"blocked={self.blocked})")


def unreadable_to_text(pairs: List[Tuple[str, str]]) -> List[str]:
    """
    Как мозГ называет человеку нечитаемое отношение.

    ЖИВЁТ ЗДЕСЬ, А НЕ В core/schemes.py, и это не перенос строк ради
    зелёных тестов. Ядро обязано уметь говорить о мире без русского (закон
    границы A0): оно возвращает ПАРУ (роль, отношение) — факт о том, что
    прибор не смог спросить. Фраза — свойство ЧЕЛОВЕКА, а не знания, поэтому
    она в мозге, где граница языка разрешает русский.

    ЗАПРЕТ, ВАЖНЫЙ ДЛЯ ВЫВОДА: не путать с «не хватило». Пустой список
    здесь означает «прибор работал», непустой — «половина вывода не была
    проверена вообще»; по первому чинят базу знаний, по второму — код.
    """
    out = []
    for role, rel in pairs:
        out.append(f"роль «{role}»: отношение «{rel}» НЕ ЧИТАЕТСЯ прибором — "
                   f"это не «связи нет», а сломанный вопрос; починка — "
                   f"brain/simulator.py::related, а не база знаний")
    return out


def absorb_walk(res, walk) -> None:
    """
    Единственная точка, где мозг перекладывает ход схемы в свой результат:
    путь + чего не хватило + чего НЕ ПРОЧИТАЛИ.

    ОДНА точка, а не две строки в двух модулях: brain/simulator.py и
    brain/social_sim.py оба сводят схему в ответ, и если второй забудет про
    unreadable, вывод social-мира молча потеряет различение «прибор сломан»
    / «связи нет» — ровно та дыра F, из-за которой всё это затевалось.
    """
    res.chain.extend(walk.chain)
    res.missing.extend(walk.missing)
    res.missing.extend(unreadable_to_text(walk.unreadable))


# ─────────────────────────────────────────────────────────────
# СИМУЛЯТОР
# ─────────────────────────────────────────────────────────────

# Действия, которыми помечается «процесс прекратился». Список — не догадка:
# в рассказе сказано «пожар остановился», и любое из этих слов означает конец.
STOP_ACTIONS = ("остановиться", "остановить", "прекратиться", "прекратить",
                "закончиться", "потухнуть", "погаснуть")


class SimulationResult:
    """Итог вывода: ОТВЕТ + ПУТЬ, которым он получен, + чего не хватило."""

    def __init__(self, question: str):
        self.question = question
        self.answer = ""
        self.chain: List[str] = []
        self.missing: List[str] = []
        self.ok = False

    def as_text(self) -> str:
        lines = [f"❓ {self.question}"]
        if self.ok:
            lines.append(f"✅ {self.answer}")
        else:
            lines.append("⛔ вывести не удалось")
        for step in self.chain:
            lines.append(f"   · {step}")
        for miss in self.missing:
            lines.append(f"   ✗ не хватает: {miss}")
        return "\n".join(lines)


class Simulator:
    """
    Симулятор рассказа: держит СОБЫТИЯ во времени и выводит «почему».

    Пока не читает текст: события добавляются руками через add_event().
    Знания берёт ТОЛЬКО из статической базы (StaticKnowledge) и туда не пишет.

    ПУТЬ ВЫВОДА — В СХЕМЕ (core/schemes.py + секция "schemes" мира), не здесь.
    Этот класс умеет только: хранить события, находить событие-прекращение и
    выбирать событие перед ним. Дальше ход делает интерпретатор схемы.
    """

    def __init__(self, world: Dict):
        self.knowledge = StaticKnowledge(world)
        self.schemes = load_schemes(world)
        self.events: List[Event] = []
        # Событий в статической базе НЕТ — они живут в отдельном файле рассказа
        # (load_story/load_events). Конструктор статику не трогает.

    # ── наполнение рассказа из данных ────────────────────
    def load_events(self, records) -> int:
        """
        Загрузить СОБЫТИЯ рассказа.

        Принимает ЛИБО список записей, ЛИБО мир (dict) с секцией "events" —
        по тому же формату, что пишет Learner. Формат записи:
        {subject, action, obj?, tense?, order?, evidence?}. События сортируются
        по order: рассказ читается по предложениям, и порядок в файле может не
        совпасть с порядком в тексте.
        """
        if isinstance(records, dict):
            records = records.get("events") or []
        for rec in records or []:
            if not isinstance(rec, dict):
                continue
            subject = str(rec.get("subject", "")).lower().strip()
            action = str(rec.get("action", "")).lower().strip()
            if not subject or not action:
                continue
            self.events.append(Event(
                subject=subject, action=action,
                obj=str(rec.get("obj", "")).lower().strip(),
                order=int(rec.get("order", len(self.events))),
                text=str(rec.get("evidence", ""))))
        self.events.sort(key=lambda e: e.order)
        return len(self.events)

    def load_story(self, path: str) -> int:
        """Прочитать файл рассказа ({"events": [...]}) и загрузить события."""
        with open(path, encoding="utf-8") as fh:
            story = json.load(fh)
        return self.load_events(story.get("events") or [])

    # ── наполнение рассказа (пока вручную) ─────────────────
    def add_event(self, subject: str, action: str, obj: str = "",
                  text: str = "") -> Event:
        ev = Event(subject=subject.lower().strip(),
                   action=action.lower().strip(),
                   obj=obj.lower().strip(),
                   order=len(self.events), text=text)
        self.events.append(ev)
        return ev

    # ── поиск события-прекращения ──────────────────────────
    def _stop_event(self, subject: str) -> Optional[Event]:
        s = subject.lower().strip()
        for ev in reversed(self.events):
            if ev.subject == s and ev.action in STOP_ACTIONS:
                return ev
        return None

    def _event_before(self, ev: Event) -> Optional[Event]:
        idx = self.events.index(ev)
        return self.events[idx - 1] if idx > 0 else None

    # ── сведение сущности к «действующему веществу» ────────
    def _agents_of(self, subject: str) -> List[str]:
        """
        Во что «входит» subject снизу вверх по IS_A, включая сам subject.
        «дождь» -> [дождь, вода]. Именно так дождь становится водой БЕЗ
        отдельной строки «дождь гасит огонь»: вывод честный.
        """
        chain = [subject]
        frontier = [subject]
        seen = {subject}
        while frontier:
            cur = frontier.pop(0)
            for parent in self.knowledge.is_a(cur):
                if parent not in seen:
                    seen.add(parent)
                    chain.append(parent)
                    frontier.append(parent)
        return chain

    # ── ГЛАВНЫЙ ВЫВОД: СХЕМА ИЗ ДАННЫХ, НЕ КОД ────────────
    def why_stopped(self, subject: str,
                    scheme_name: str = "process_stopped") -> SimulationResult:
        """
        Вывести «почему прекратился subject», пройдя СХЕМУ ИЗ ДАННЫХ
        (секция "schemes" мира), а не зашитый в Python путь.

        Симулятор здесь лишь: находит событие-прекращение, берёт событие
        перед ним — и отдаёт роли интерпретатору. Какой именно путь пройти,
        решает схема; симулятор её не знает.
        """
        q = f"Почему прекратился {subject}?"
        res = SimulationResult(q)

        scheme = self.schemes.get(scheme_name)
        if scheme is None:
            res.missing.append(f"в мире нет схемы «{scheme_name}» — "
                               f"выводить нечем")
            return res

        stop = self._stop_event(subject)
        if stop is None:
            res.missing.append(f"в рассказе нет события «{subject} прекратился»")
            return res

        before = self._event_before(stop)
        if before is None:
            res.missing.append("перед прекращением нет ни одного события — "
                               "причину брать неоткуда")
            return res

        res.chain.append(f"перед прекращением: «{before.describe()}»")

        interp = SchemeInterpreter(self.knowledge)
        walk = interp.run(scheme, {"subject": subject.lower().strip(),
                                   "cause_subject": before.subject})
        # ОДИН вызов: absorb_walk кладёт и путь, и отказ. Раньше стояло
        # res.chain.extend(...) плюс absorb_walk — путь печатался ДВАЖДЫ.
        absorb_walk(res, walk)
        if not walk.ok:
            return res

        res.answer = interp.render(scheme.answer, walk.bindings)
        res.ok = True
        return res


# ─────────────────────────────────────────────────────────────
# ДЕМОНСТРАЦИЯ: рассказ про дождь и пожар (события — вручную)
# ─────────────────────────────────────────────────────────────

def build_fire_rain_simulator() -> Simulator:
    """
    Собирает демонстрационный симулятор: СТАТИКА из fire_rain_world.json,
    СОБЫТИЯ — из отдельного файла рассказа fire_rain_story.json.

    События больше НЕ добавляются здесь руками (разрыв A): рассказ живёт в
    данных, а симулятор — его интерпретатор.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp = os.path.join(root, "experiments")
    with open(os.path.join(exp, "fire_rain_world.json"), encoding="utf-8") as fh:
        world = json.load(fh)
    sim = Simulator(world)
    sim.load_story(os.path.join(exp, "fire_rain_story.json"))
    return sim


if __name__ == "__main__":
    s = build_fire_rain_simulator()
    print("Рассказ:")
    for ev in s.events:
        print(f"  {ev.order}. {ev.text}")
    print()
    print(s.why_stopped("пожар").as_text())
