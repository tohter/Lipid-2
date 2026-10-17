"""
brain/social_sim.py — СОЦИАЛЬНЫЙ симулятор: ожидания, их нарушения, эмоции.

ЗАЧЕМ ВТОРОЙ СИМУЛЯТОР
----------------------
Первый пример («дождь и пожар», brain/simulator.py) выводил ответ из СТАТИКИ:
закон физики + событие перед прекращением. Ответ не зависел от времени.

Здесь ответ РОЖДАЕТСЯ ВО ВРЕМЕНИ и его нельзя получить из статики:
    ожидание -> нарушение -> обида (нарастает по тикам) -> гнев -> агрессия
    -> действие.
Поэтому это отдельный модуль, а не ветка в fire-симуляторе: разная природа
вопроса. Общее — только ЧТЕНИЕ статики (StaticKnowledge) и МОДЕЛЬ ЭМОЦИЙ
(EmotionSystem); оба переиспользуются, не копируются.

ГЛАВНОЕ ПРАВИЛО ЧЕСТНОСТИ
-------------------------
Ответ «кошка съела канарейку, потому что разозлилась» нельзя ПОДСТАВИТЬ.
Тесты-замки требуют, чтобы симулятор:
  1. нашёл ОЖИДАНИЕ, порождённое просьбой;
  2. нашёл НАРУШЕНИЕ этого ожидания;
  3. НАКОПИЛ «плохую» эмоцию по тикам и показал, что она ПЕРЕШЛА ПОРОГ до
     действия (иначе действия не было бы);
  4. прошёл по статике нарушение -> предательство -> обида -> гнев -> агрессия;
  5. если любого звена нет — сказал, какого, а не выдумал ответ.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Корень проекта в sys.path — чтобы можно было запускать как `python brain/social_sim.py`
# (тот же приём, что в core/cortex.py, brain/learner.py).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.schemes import SchemeInterpreter
from brain.emotions import EmotionSystem
from brain.simulator import Event, SimulationResult, Simulator, StaticKnowledge

# Порог «плохой» шкалы, при котором агент ПЕРЕХОДИТ К ДЕЙСТВИЮ.
# DRAFT (догадка): согласуется с уровнем «агрессия» в emotions.py.
AGGRESSION_THRESHOLD = 230


# ─────────────────────────────────────────────────────────────
# ОЖИДАНИЕ — то, чего в статике нет. Оно рождается из события-просьбы.

@dataclass
class Expectation:
    """
    «holder ждёт, что target (не) сделает action».

    Пример из рассказа: кошка попросила канарейку не рассказывать тайну ->
        Expectation(holder="кошка", target="канарейка",
                    action="рассказывать", negated=True)
    Ожидание живёт во времени: у него есть тик рождения (born_tick).
    """
    holder: str
    target: str
    action: str
    negated: bool = True
    born_tick: int = 0
    source: str = ""

    def violated_by(self, ev: Event) -> bool:
        """Нарушает ли событие ev это ожидание?"""
        if ev.subject != self.target or ev.action != self.action:
            return False
        # Ждали, что НЕ сделает (negated=True), а он сделал -> нарушение.
        # Ждали, что сделает, а он не сделал -> это НЕ событие, а его отсутствие,
        # здесь не проверяется (нужен отдельный механизм «не сделал»).
        return self.negated


@dataclass
class Violation:
    """Зафиксированное нарушение ожидания + эмоциональный заряд."""
    expectation: Expectation
    event: Event
    charge: float          # величина заряда (до множителя близости)
    closeness: float       # степень близости holder->target
    applied: float         # итоговый заряд = charge * closeness


class SocialSimulator(Simulator):
    """
    Расширяет Simulator ожиданиями и эмоциями агентов.

    Наследуемся, потому что события, поиск события-прекращения и чтение статики
    устроены одинаково. Добавляем: агентов с эмоциями, ожидания, нарушения и
    вывод «почему агент так поступил».
    """

    def __init__(self, world: Dict, closeness: float = None,
                 script: List[Dict] = None):
        super().__init__(world)
        self.agents: Dict[str, EmotionSystem] = {}
        self.expectations: List[Expectation] = []
        self.violations: List[Violation] = []
        self.tick_index: int = 0
        # Снимок эмоций ВСЕХ агентов на момент каждого события. Нужен, чтобы
        # судить о состоянии НА МОМЕНТ действия, а не на момент вопроса: к
        # моменту вопроса эмоция уже могла затухнуть, и это было бы враньём.
        self.snapshots: List[Dict[str, Dict[str, int]]] = []
        # Сценарий рассказа — ИЗ ДАННЫХ (файл рассказа, секция "script").
        # Порядок директив и есть порядок во времени; в Python остаётся только
        # ИНТЕРПРЕТАТОР. Ручная сборка рассказа в build_*_simulator была
        # разрывом A (см. experiments/EVENTS_TASK.md).
        self.load_script(script or [], closeness=closeness)

    # ── ИНТЕРПРЕТАТОР СЦЕНАРИЯ ─────────────────────────────
    def load_script(self, raw: List[Dict],
                    closeness: float = None) -> int:
        """
        Выполнить директивы сценария из секции "script" мира.

        Директивы ("do"):
          agent  — {name, attack?, decay?} завести агента с эмоциями;
          please — {name, amount, closeness?} обрадовать («хорошая» шкала);
          ticks  — {n} пустые тики времени;
          event  — {subject, action, obj?, closeness?, text?} событие рассказа;
          expect — {holder, target, action, negated?, source?} ожидание.
        Возвращает число выполненных директив; неизвестная директива —
        громкий пропуск (не молчание).
        """
        if closeness is None:
            closeness = CLOSENESS_LIVING_TOGETHER
        done = 0
        for d in raw or []:
            if not isinstance(d, dict):
                continue
            kind = str(d.get("do", "")).strip()
            if kind == "agent":
                self.agent(str(d.get("name", "")),
                           attack=int(d.get("attack", 40)),
                           decay=int(d.get("decay", 8)))
            elif kind == "please":
                self.agent(str(d.get("name", ""))).please(
                    int(d.get("amount", 0)),
                    closeness=float(d.get("closeness", closeness)))
            elif kind == "ticks":
                self.ticks(int(d.get("n", 0)))
            elif kind == "event":
                self.act(str(d.get("subject", "")), str(d.get("action", "")),
                         obj=str(d.get("obj", "")),
                         text=str(d.get("text", "")),
                         closeness=float(d.get("closeness", closeness)))
            elif kind == "expect":
                self.expect(str(d.get("holder", "")), str(d.get("target", "")),
                            str(d.get("action", "")),
                            negated=bool(d.get("negated", True)),
                            source=str(d.get("source", "")))
            else:
                print(f"[social_sim] ⚠️ неизвестная директива сценария: «{kind}»",
                      file=sys.stderr, flush=True)
                continue
            done += 1
        return done

    # ── агенты и время ─────────────────────────────────────
    def agent(self, name: str, attack: int = 40, decay: int = 8) -> EmotionSystem:
        """Завести агента с эмоциями. Одушевлённые несут ОБЕ шкалы сразу."""
        name = name.lower().strip()
        if name not in self.agents:
            self.agents[name] = EmotionSystem(attack=attack, decay=decay)
        return self.agents[name]

    def tick(self) -> None:
        """Один тик времени: эмоции ВСЕХ агентов живут и затухают."""
        self.tick_index += 1
        for emo in self.agents.values():
            emo.tick()

    def ticks(self, n: int) -> None:
        for _ in range(max(0, n)):
            self.tick()

    # ── ожидания ───────────────────────────────────────────
    def expect(self, holder: str, target: str, action: str,
               negated: bool = True, source: str = "") -> Expectation:
        exp = Expectation(holder=holder.lower().strip(),
                          target=target.lower().strip(),
                          action=action.lower().strip(),
                          negated=negated, born_tick=self.tick_index,
                          source=source)
        self.expectations.append(exp)
        return exp

    def violations_of(self, holder: str) -> List[Violation]:
        h = holder.lower().strip()
        return [v for v in self.violations if v.expectation.holder == h]

    # ── события, которые МОГУТ нарушить ожидание ───────────
    def act(self, subject: str, action: str, obj: str = "", text: str = "",
            closeness: float = 1.0,
            base_charge: float = 200.0) -> Tuple[Event, List[Violation]]:
        """
        Записать событие. Если оно нарушает чьё-то ожидание — в «плохую» шкалу
        ожидающего идёт заряд (base_charge x closeness).

        base_charge и closeness — DRAFT-числа (см. шапку модуля): замером станет
        прогон рассказа. closeness здесь — та самая «степень близости» хозяина:
        гадость от близкого ранит сильнее, от чужого — слабее.
        """
        ev = self.add_event(subject, action, obj, text)
        hit: List[Violation] = []
        for exp in self.expectations:
            if exp.violated_by(ev) and exp.born_tick <= ev.order:
                emo = self.agent(exp.holder)
                self.agent(exp.target)          # у обеих сторон есть эмоции
                emo.hurt(base_charge, closeness=closeness)
                viol = Violation(expectation=exp, event=ev, charge=base_charge,
                                 closeness=closeness,
                                 applied=base_charge * closeness)
                self.violations.append(viol)
                hit.append(viol)
        # Снимок ПОСЛЕ применения заряда: тогда состояние на момент события
        # включает и его собственный эффект — так честнее.
        self.snapshots.append({name: emo.state()
                               for name, emo in self.agents.items()})
        return ev, hit

    # ── связь момента события со снимком эмоций ────────────
    def _snapshot_at(self, ev: Event) -> Dict[str, Dict[str, int]]:
        idx = self.events.index(ev)
        if 0 <= idx < len(self.snapshots):
            return self.snapshots[idx]
        return {}

    # ── ГЛАВНЫЙ ВЫВОД: СХЕМА ИЗ ДАННЫХ ────────────────────
    def why_acted(self, agent: str, action: str,
                  scheme_name: str = "social_betrayal") -> SimulationResult:
        """
        Вывести, ПОЧЕМУ агент совершил действие.

        Динамику (какое ожидание нарушено и каким событием) доставляет
        симулятор — это живёт во времени и в статике не лежит. СТАТИЧЕСКУЮ
        ЦЕПОЧКУ («нарушение доверия -> предательство -> обида -> гнев ->
        агрессия») и ПОРОГ проходит интерпретатор схемы из данных.
        """
        agent = agent.lower().strip()
        action = action.lower().strip()
        q = f"Почему {agent} {action}?"
        res = SimulationResult(q)

        scheme = self.schemes.get(scheme_name)
        if scheme is None:
            res.missing.append(f"в мире нет схемы «{scheme_name}» — "
                               f"выводить нечем")
            return res

        # 1) Само действие в рассказе.
        act_ev = None
        for ev in self.events:
            if ev.subject == agent and ev.action == action:
                act_ev = ev
                break
        if act_ev is None:
            res.missing.append(f"в рассказе нет события «{agent} {action}»")
            return res

        # 2) Эмоции агента вообще есть?
        if agent not in self.agents:
            res.missing.append(f"у «{agent}» нет эмоционального состояния "
                               f"(не одушевлён или не заведён)")
            return res

        # 3) Нарушение ожидания, которое зарядило ЭТОГО агента ДО действия.
        viol = None
        for v in self.violations:
            if v.expectation.holder == agent and v.event.order < act_ev.order:
                viol = v
        if viol is None:
            res.missing.append(f"нет нарушённого ожидания у «{agent}» до действия — "
                               f"объяснить нечем")
            return res

        viol_ev = viol.event
        res.chain.append(
            f"ожидание: {viol.expectation.holder} ждал(а), что "
            f"«{viol.expectation.target}» не станет «{viol.expectation.action}»")
        res.chain.append(f"нарушение: «{viol_ev.describe()}» — это предательство")
        res.chain.append(
            f"заряд обиды: {viol.charge:.0f} x близость {viol.closeness:.1f} "
            f"= {viol.applied:.0f}")

        # 4) Статика + порог — ИНТЕРПРЕТАТОРУ. Числа берём СО СНИМКА НА
        #    МОМЕНТ действия, а не на момент вопроса.
        scales = {agent: dict(self._snapshot_at(act_ev).get(agent)
                              or self.agents[agent].state())}
        interp = SchemeInterpreter(self.knowledge, scales=scales)
        walk = interp.run(scheme, {"agent": agent,
                                   "target": viol.expectation.target,
                                   "action": action})
        res.chain.extend(walk.chain)
        if not walk.ok:
            res.missing.extend(walk.missing)
            return res

        res.answer = interp.render(scheme.answer, walk.bindings)
        res.ok = True
        return res



# ─────────────────────────────────────────────────────────────
# ДЕМОНСТРАЦИЯ: рассказ про кошку и канарейку
# ─────────────────────────────────────────────────────────────

CLOSENESS_LIVING_TOGETHER = 2.0    # «жили вместе», «кошка любила» — близкие. DRAFT.

# Сколько тиков проходит между ключевыми событиями рассказа. Выбрано НЕ
# наобум: при прогоне рассказа замерено, что «плохая» шкала кошки к моменту
# действия равна 128 (4 тика, рано), 247 (8 тиков, пик), 231 (12), 199 (16,
# уже прошло). То есть есть ОКНО: гнев зреет и затухает. 8 — середина окна.
# Это и есть «время или порог активации» — не украшение, а замер.
RIPEN_TICKS = 8


def build_cat_canary_simulator(closeness: float = CLOSENESS_LIVING_TOGETHER,
                               between_ticks: int = RIPEN_TICKS) -> SocialSimulator:
    """
    Собирает симулятор из учебной базы cat_canary_world.json.

    Рассказ (агенты, просьба, ожидание, тики, события) лежит в данных
    (секция "script") и исполняется интерпретатором. Здесь — только
    параметры, которые тесты хотят варьировать: between_ticks подставляется
    ВМЕСТО плейсхолдера "$between_ticks", a closeness уходит в конструктор.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp = os.path.join(root, "experiments")
    with open(os.path.join(exp, "cat_canary_world.json"), encoding="utf-8") as fh:
        world = json.load(fh)
    with open(os.path.join(exp, "cat_canary_story.json"), encoding="utf-8") as fh:
        script = json.load(fh).get("script") or []
    # Подстановка параметров в сценарий (данные остаются источником порядка).
    for d in script:
        if d.get("n") == "$between_ticks":
            d["n"] = int(between_ticks)
    sim = SocialSimulator(world, closeness=closeness, script=script)
    return sim


if __name__ == "__main__":
    s = build_cat_canary_simulator()
    print("Рассказ:")
    for ev in s.events:
        print(f"  {ev.order}. {ev.text}")
    print()
    for name, emo in s.agents.items():
        print(f"Эмоции «{name}»:")
        print(emo.timeline())
        print(f"  итог: {emo.state()}  уровни: {emo.level()}")
        print()
    print(s.why_acted("кошка", "съесть").as_text())
