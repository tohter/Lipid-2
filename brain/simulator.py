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

    Только чтение: у симулятора нет прав менять базу (та же дисциплина, что у
    предиктивного цикла).
    """

    def __init__(self, world: Dict):
        self.world = world or {}
        self._by_head: Dict[str, List[Tuple[str, str]]] = {}
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

    def _tails(self, x: str, rel: str) -> List[str]:
        out = []
        for r, t in self._by_head.get((x or "").lower().strip(), []):
            if r == rel and t not in out:
                out.append(t)
        return out

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
        res.chain.extend(walk.chain)
        if not walk.ok:
            res.missing.extend(walk.missing)
            return res

        res.answer = interp.render(scheme.answer, walk.bindings)
        res.ok = True
        return res


# ─────────────────────────────────────────────────────────────
# ДЕМОНСТРАЦИЯ: рассказ про дождь и пожар (события — вручную)
# ─────────────────────────────────────────────────────────────

def build_fire_rain_simulator() -> Simulator:
    """Собирает демонстрационный симулятор из учебной базы fire_rain_world.json."""
    import json
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db = os.path.join(root, "experiments", "fire_rain_world.json")
    with open(db, encoding="utf-8") as fh:
        world = json.load(fh)
    sim = Simulator(world)
    # Три предложения рассказа — пока прописаны вручную (симулятор не читает).
    sim.add_event("пожар", "начаться", text="На опушке леса начался пожар.")
    sim.add_event("дождь", "лить", text="Но тут полил сильный дождь.")
    sim.add_event("пожар", "остановиться", text="Пожар остановился.")
    return sim


if __name__ == "__main__":
    s = build_fire_rain_simulator()
    print("Рассказ:")
    for ev in s.events:
        print(f"  {ev.order}. {ev.text}")
    print()
    print(s.why_stopped("пожар").as_text())
