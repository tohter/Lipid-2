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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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
    """

    def __init__(self, world: Dict):
        self.knowledge = StaticKnowledge(world)
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

    # ── совпадение цели закона с тем, что прекращается ────
    def _matches(self, rule_target: str, targets: List[str]) -> bool:
        """
        Совпадает ли цель закона с тем, что прекратилось... с учётом IS_A.
        «пламя» IS_A «огонь», поэтому закон «гасить огонь» применим и к пламени.
        Только чтение; без этого законы пришлось бы дублировать под каждый синоним.
        """
        for t in targets:
            if rule_target == t:
                return True
            if rule_target in self._agents_of(t):   # «пламя» -> [пламя, огонь]
                return True
        return False

    # ── ГЛАВНЫЙ ВЫВОД ──────────────────────────────────────
    def why_stopped(self, subject: str) -> SimulationResult:
        q = f"Почему прекратился {subject}?"
        res = SimulationResult(q)
        stop = self._stop_event(subject)
        if stop is None:
            res.missing.append(f"в рассказе нет события «{subject} прекратился»")
            return res

        before = self._event_before(stop)
        if before is None:
            res.missing.append("перед прекращением нет ни одного события — "
                               "причину брать неоткуда")
            return res

        # 1) Кандидат-причина: событие непосредственно перед прекращением.
        cause_subject = before.subject
        res.chain.append(f"перед прекращением: «{before.describe()}»")

        # 2) Чего КОСНУЛОСЬ прекращение: часть/суть прекращаемого.
        #    «пожар» --HAS_PART--> «огонь». Если частей нет — берём сам subject.
        targets = self.knowledge.parts(subject) or [subject]
        res.chain.append(f"у «{subject}» прекращается: {', '.join(targets)}")

        # 3) Кем/чем могло быть вызвано: сводим дождь к воде по IS_A.
        agents = self._agents_of(cause_subject)
        if len(agents) > 1:
            res.chain.append(f"«{cause_subject}» есть " +
                             " -> ".join(agents[1:]) + " (по IS_A)")

        # 4) Ищем закон: агент умеет действие, направленное на цель.
        for agent in agents:
            for action in self.knowledge.can_do(agent):
                for target in self.knowledge.object_of(action):
                    if self._matches(target, targets):
                        res.answer = (f"потому что {agent} {self._verb(action)} "
                                      f"{target}")
                        res.chain.append(
                            f"закон: {agent} CAN_DO {action}, "
                            f"{action} OBJECT_OF {target}")
                        res.ok = True
                        return res

        # 5) Закона не нашлось — сказать ЧЕГО именно нет (не молчать).
        res.missing.append(
            f"нет закона: ни один из {agents} не умеет действие, направленное на "
            f"{targets}")
        return res

    @staticmethod
    def _verb(action: str) -> str:
        """
        Привести инфинитив к форме 3-го лица ед.ч.: «гасить» -> «гасит».
        Это НЕ морфология ради красоты: ответ «вода гасить огонь» звучит как
        сломанный, и пользователь решит, что сломан движок. Правило простое и
        покрывает глаголы нашего рассказа; неизвестный глагол вернётся как есть
        (лучше честный инфинитив, чем выдуманная форма).
        """
        a = (action or "").strip()
        if a.endswith("ить"):
            return a[:-3] + "ит"
        if a.endswith("ать"):
            return a[:-3] + "ает"
        if a.endswith("еть"):
            return a[:-3] + "еет"
        if a.endswith("ыть"):
            return a[:-3] + "оет"
        if a.endswith("ть"):
            return a[:-2] + "ет"
        return a


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
