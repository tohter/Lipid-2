"""
core/schemes.py — СХЕМЫ ВЫВОДА КАК ДАННЫЕ (шаг 1 из архитектуры п.29).

ЗАЧЕМ ЭТОТ МОДУЛЬ
-----------------
До него путь вывода был ЗАШИТ В PYTHON в двух местах:
  * brain/simulator.py::why_stopped — HAS_PART -> IS_A -> CAN_DO/OBJECT_OF;
  * brain/social_sim.py::why_acted — Violation -> IS_A(предательство) ->
    CAUSES* -> порог 230.
Пока путь в коде, учить модель нечему: схеме негде лежать, и в C переносить
нечего (переносится ИНТЕРПРЕТАТОР, а не частный обход).

Теперь схема — это СПИСОК ШАГОВ с РОЛЯМИ, лежащий в данных (секция "schemes"
файла мира). Симулятор лишь ИНТЕРПРЕТИРУЕТ схему:

    связал роли по рёбрам статики -> проверил числовые пороги ->
    собрал ответ из привязанных значений.

ДВА ТИПА ШАГОВ (согласовано с хозяином):
  * kind="edge"      — «пройти по ребру relation из текущего узла»;
  * kind="threshold" — «проверить числовую шкалу агента против порога».

ПОРОГИ ЖИВУТ В СХЕМЕ, А НЕ В КОДЕ. Числа 230 и 8 раньше были константами
модулей; теперь это поля шага threshold / поля схемы. В коде остаётся только
интерпретатор: ни одного порога вывода он больше не знает.

ЧЕСТНОСТЬ
---------
Если шаг не сработал (нет ребра, число ниже порога) — в результат идёт запись
о ТОМ, ЧЕГО НЕ ХВАТИЛО, с именем роли и шага. Молчаливого пустого ответа нет.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# СХЕМА: читается из данных, не из кода
# ─────────────────────────────────────────────────────────────

@dataclass
class Step:
    """Один шаг схемы. Поля зависят от kind (edge | threshold)."""
    kind: str
    role: str = ""              # имя переменной, которую связывает шаг
    frm: str = ""               # источник значения: "$role" или литерал
    relation: str = ""          # для kind=edge
    mode: str = "single"        # single | transitive
    match: str = ""             # "$role": результат должен совпасть с ролью
    optional: bool = False      # шаг может не сработать — это не провал схемы
    fallback: str = ""          # "self": при пустом результате взять вход
    scale: str = ""             # для kind=threshold: "bad"/"good"
    op: str = ">="              # >=, <=, >, <, ==
    value: int = 0              # порог


@dataclass
class Scheme:
    """
    Схема вывода: имя, человеческий вопрос, шаги и шаблон ответа.

    Шаблон ответа — ТЕКСТ с подстановками вида {role} и модификаторами
    {role:fin} (глагол в 3-м лице ед.ч.). Это тоже данные, а не код: смена
    формулировки ответа не трогает интерпретатор.
    """
    name: str
    question: str = ""
    steps: List[Step] = field(default_factory=list)
    answer: str = ""
    # КОМУ АДРЕСОВАНА СХЕМА. Раньше выбор делал вызывающий по ИМЕНИ мира
    # (sims.get("fire")), то есть имена рассказов жили в Python. Теперь схема
    # сама говорит, какие роли она требует на входе (requires) и при какой
    # роли её точно НЕ надо пробовать (forbids). Отсев по ДАННЫМ мира.
    requires: List[str] = field(default_factory=list)
    forbids: List[str] = field(default_factory=list)
    # КАК СХЕМУ ВЫЗВАТЬ. Имя метода симулятора и ПОРЯДОК его аргументов —
    # тоже данные. Благодаря этому brain/question.py не знает ни имён миров
    # ("fire", "soc"), ни того, что у одной схемы один аргумент, а у другой два.
    handler: str = ""
    args: List[str] = field(default_factory=list)
    # Глаголы «процесс прекратился» (ЛЕММЫ). Живут В МИРЕ, а не в Python:
    # один список питает и маршрутизацию вопроса, и поиск события- конца
    # в рассказе (Simulator._stop_event). Пусто = схема не про прекращение.
    stop_actions: List[str] = field(default_factory=list)
    # Порядок перебора схем обязан быть определён однозначно: dict сохраняет
    # порядок вставки, но порядок симуляторов задаёт вызывающий. Меньше —
    # раньше.
    priority: int = 100

    def matches(self, roles: Dict[str, object]) -> bool:
        """
        Подходит ли схема под НАБОР РОЛЕЙ разобранного вопроса.

        Только дешёвый отсев по наличию ролей — он НЕ решает, получится ли
        вывод. Пройдёт ли путь по статике, решает интерпретатор: у схемы,
        которая прошла отсев но не прошла путь, причина провала ОБЯЗАНА
        попасть в ответ (см. brain.question.answer).
        """
        have = {k for k, v in (roles or {}).items() if v}
        if not all(r in have for r in self.requires):
            return False
        if any(r in have for r in self.forbids):
            return False
        # Схема про прекращение процесса обязана видеть прекращение в действии
        # вопроса, иначе она подхватит любой «почему X ...». Обратная проверка
        # НЕ нужна: схема «поступок агента» не содержит этих глаголов, а
        # отсев по stop_actions у неё пустой, поэтому вопрос о прекращении
        # отсеивается на стороне fire-схемы, а не здесь.
        if self.stop_actions:
            act = str((roles or {}).get("action") or "").lower().strip()
            if act not in self.stop_actions:
                return False
        return True


def load_schemes(world: Dict) -> Dict[str, Scheme]:
    """Прочитать секцию "schemes" файла мира. Нет секции -> пустой словарь."""
    raw = (world or {}).get("schemes") or {}
    out: Dict[str, Scheme] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        steps = []
        for s in (spec.get("steps") or []):
            if not isinstance(s, dict):
                continue
            steps.append(Step(
                kind=str(s.get("kind", "edge")).strip().lower(),
                role=str(s.get("role", "")).strip(),
                frm=str(s.get("from", "")).strip(),
                relation=str(s.get("relation", "")).upper().strip(),
                mode=str(s.get("mode", "single")).strip().lower(),
                match=str(s.get("match", "")).strip(),
                optional=bool(s.get("optional", False)),
                fallback=str(s.get("fallback", "")).strip().lower(),
                scale=str(s.get("scale", "")).strip(),
                op=str(s.get("op", ">=")).strip(),
                value=int(s.get("value", 0)),
            ))
        out[name] = Scheme(
            name=name,
            question=str(spec.get("question", "")).strip(),
            steps=steps,
            answer=str(spec.get("answer", "")).strip(),
            requires=[str(r).strip() for r in (spec.get("requires") or [])
                      if str(r).strip()],
            forbids=[str(r).strip() for r in (spec.get("forbids") or [])
                     if str(r).strip()],
            handler=str(spec.get("handler", "")).strip(),
            stop_actions=[str(a).strip().lower() for a in (spec.get("stop_actions") or [])
                          if str(a).strip()],
            args=[str(a).strip() for a in (spec.get("args") or [])
                  if str(a).strip()],
            priority=int(spec.get("priority", 100)),
        )
    return out


# ─────────────────────────────────────────────────────────────
# РЕЗУЛЬТАТ ПРОХОДА ПО СХЕМЕ
# ─────────────────────────────────────────────────────────────

class SchemeResult:
    """Итог интерпретации схемы: связанные роли, путь, чего не хватило."""

    def __init__(self, scheme_name: str, question: str):
        self.scheme = scheme_name
        self.question = question
        self.bindings: Dict[str, object] = {}
        self.chain: List[str] = []
        self.missing: List[str] = []
        self.ok = False


# ─────────────────────────────────────────────────────────────
# ИНТЕРПРЕТАТОР
# ─────────────────────────────────────────────────────────────

class SchemeInterpreter:
    """
    Проходит схему ПО ДАННЫМ СТАТИКИ. Ничего не знает про «пожар» и «кошку»:
    ему дают схему, статику и входные роли — он связывает остальные роли.

    Вход:  bindings вроде {"subject": "пожар", "cause_subject": "дождь"}.
    Статика: любой объект с методами is_a/can_do/object_of/parts/causes
             (то есть brain.simulator.StaticKnowledge — не импортируем его,
             чтобы core не зависел от brain).
    Числа:  scales вроде {"кошка": {"bad": 247, "good": 12}} — снимок
             эмоций агента на момент события. Интерпретатор их только читает.
    """

    def __init__(self, static, scales: Optional[Dict[str, Dict[str, int]]] = None):
        self.static = static
        self.scales = scales or {}

    # ── доступ к рёбрам по имени отношения (единая точка) ──
    def _neighbors(self, node: str, relation: str, mode: str) -> List[str]:
        """
        Все соседи узла по ребру relation. mode=transitive — вглубь
        (для IS_A: дождь -> вода -> вещество). Только чтение.
        """
        rel = (relation or "").upper()
        if mode == "transitive":
            out: List[str] = []
            frontier = [node]
            seen = {node}
            while frontier:
                cur = frontier.pop(0)
                for nxt in self._direct(cur, rel):
                    if nxt not in seen:
                        seen.add(nxt)
                        out.append(nxt)
                        frontier.append(nxt)
            return out
        return self._direct(node, rel)

    def _direct(self, node: str, rel: str) -> List[str]:
        table = {
            "IS_A": self.static.is_a,
            "CAN_DO": self.static.can_do,
            "OBJECT_OF": self.static.object_of,
            "HAS_PART": self.static.parts,
            "CAUSES": self.static.causes,
            "PURPOSE": self.static.action_purpose,
        }
        fn = table.get(rel)
        return list(fn(node)) if fn else []

    # ── значение ссылки "$role" / литерал ──────────────────
    def _resolve(self, spec: str, bindings: Dict[str, object]) -> List[str]:
        s = (spec or "").strip()
        if s.startswith("$"):
            val = bindings.get(s[1:])
            if val is None:
                return []
            if isinstance(val, (list, tuple, set)):
                return [str(x) for x in val]
            return [str(val)]
        return [s] if s else []

    # ── ГЛАВНЫЙ ПРОХОД ─────────────────────────────────────
    def run(self, scheme: Scheme, bindings: Optional[Dict[str, object]] = None
            ) -> SchemeResult:
        res = SchemeResult(scheme.name, scheme.question)
        res.bindings.update(bindings or {})
        for idx, step in enumerate(scheme.steps):
            before = len(res.missing)
            if step.kind == "edge":
                self._run_edge(step, res)
            elif step.kind == "threshold":
                self._run_threshold(step, res)
            else:
                res.missing.append(f"шаг {idx}: неизвестный kind «{step.kind}»")
            if len(res.missing) > before and not step.optional:
                return res
        res.ok = True
        return res

    def _run_edge(self, step: Step, res: SchemeResult) -> None:
        sources = self._resolve(step.frm, res.bindings)
        found: List[str] = []
        for src in sources:
            for nxt in self._neighbors(src, step.relation, step.mode):
                if nxt not in found:
                    found.append(nxt)
        if step.match:
            allowed = self._resolve(step.match, res.bindings)
            found = [f for f in found if self._matches(f, allowed)]
        fell_back = False
        if not found and step.fallback == "self":
            found = list(sources)
            fell_back = True
        if not found:
            res.missing.append(
                f"роль «{step.role}»: нет ребра {step.relation} из "
                f"{sources or '—'}"
                + (f" с совпадением {step.match}" if step.match else ""))
            return
        res.bindings[step.role] = found
        if fell_back:
            res.chain.append(f"{step.role}: {', '.join(found)} "
                             f"(сам субъект: {step.relation} нет)")
        else:
            res.chain.append(f"{step.role}: {', '.join(found)} "
                             f"(по {step.relation})")

    def _matches(self, candidate: str, allowed: List[str]) -> bool:
        """
        Совпадение с учётом IS_A В ОБЕ СТОРОНЫ: кандидат и цель совпадают,
        либо один из них является видом другого («пламя» вида «огонь» — закон
        «гасить огонь» применим и к пламени; обратное тоже честно).
        """
        for a in allowed:
            if candidate == a:
                return True
            if a in self._neighbors(candidate, "IS_A", "transitive"):
                return True
            if candidate in self._neighbors(a, "IS_A", "transitive"):
                return True
        return False

    def _run_threshold(self, step: Step, res: SchemeResult) -> None:
        subject = self._resolve(step.frm, res.bindings)
        if not subject:
            res.missing.append(f"порог «{step.role}»: нет субъекта {step.frm}")
            return
        who = subject[0]
        snap = self.scales.get(who) or {}
        scale = step.scale
        got = snap.get(scale)
        if got is None:
            res.missing.append(
                f"порог «{step.role}»: у «{who}» нет шкалы «{scale}» "
                f"(эмоции не заведены или не сняты на момент действия)")
            return
        ok = self._compare(int(got), step.op, step.value)
        res.bindings[step.role] = int(got)
        res.chain.append(f"порог «{scale}» у «{who}» = {got} "
                         f"({step.op} {step.value}: {'да' if ok else 'нет'})")
        if not ok:
            res.missing.append(
                f"порог «{step.role}»: «{scale}» = {got}, а нужно {step.op} "
                f"{step.value} — чисел НЕ хватает, вывод был бы натяжкой")

    @staticmethod
    def _compare(got: int, op: str, want: int) -> bool:
        return {
            ">=": got >= want,
            "<=": got <= want,
            ">": got > want,
            "<": got < want,
            "==": got == want,
        }.get(op, False)

    # ── СБОРКА ОТВЕТА ИЗ ШАБЛОНА ───────────────────────────
    @staticmethod
    def render(template: str, bindings: Dict[str, object]) -> str:
        """
        Подставить роли в шаблон ответа. Поддерживается {role} и {role:fin}
        (последний — глагол в 3-м лице: «гасить» -> «гасит»).

        Берём ПЕРВОЕ значение роли: ответ — одна фраза, а не перечисление.
        Перечисление остаётся в chain (путь), оно не теряется.
        """
        out = template
        for m in re.finditer(r"\{(\w+)(?::(\w+))?\}", template):
            role, mod = m.group(1), m.group(2)
            val = bindings.get(role)
            if isinstance(val, (list, tuple)):
                val = val[0] if val else ""
            text = str(val or "")
            if mod == "fin":
                text = SchemeInterpreter.finitive(text)
            out = out.replace(m.group(0), text)
        return out

    @staticmethod
    def finitive(verb: str) -> str:
        """
        Инфинитив -> 3-е лицо ед.ч.: «гасить» -> «гасит».
        Правило простое и покрывает глаголы рассказов; неизвестный глагол
        возвращается как есть (честный инфинитив лучше выдуманной формы).
        Было brain/simulator.py::_verb — теперь это свойство ЯЗЫКА ответа,
        а не симулятора.
        """
        a = (verb or "").strip()
        for suf, rep in (("ить", "ит"), ("ать", "ает"), ("еть", "еет"),
                         ("ыть", "оет")):
            if a.endswith(suf):
                return a[:-3] + rep
        if a.endswith("ть"):
            return a[:-2] + "ет"
        return a