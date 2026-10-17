"""
brain/emotions.py — СЛОЙ ЭМОЦИЙ симулятора (черновик, «лёгкая симуляция»).

СОГЛАСОВАНО С ХОЗЯИНОМ (эта сессия):
  * Эмоции — НЕ узлы базы знаний, а ЧИСЛА. Они временные, поэтому живут
    ТОЛЬКО в симуляторе и никогда не попадают в entities/observations.
  * ДВЕ НЕЗАВИСИМЫЕ ШКАЛЫ 0..255: «хорошее» и «плохое». Не одна ось:
    любовь и ненависть могут жить одновременно, и одна переменная это
    не выражает.
  * Эмоция — ПРОЦЕСС ВО ВРЕМЕНИ, а не мгновенное значение. Есть:
      скорость нарастания (attack), скорость затухания (decay), пороги.
  * Всё считается ПО ТИКАМ.

ПОЧЕМУ НАРАСТАНИЕ ЧЕРЕЗ «ОЖИДАЮЩИЙ ЗАПАС»
-----------------------------------------
Если стимул (обида) сразу прибавлялся бы к шкале, «скорость нарастания» была
бы фикцией: шкала прыгнула бы мгновенно, как при отсутствии параметра. Поэтому
стимул попадает в ОТДЕЛЬНЫЙ запас канала (pending), и за каждый тик в шкалу
перетекает не больше attack единиц. Стимул может быть сильнее attack — тогда
нарастание занимает несколько тиков. Это и есть кривая: подъём -> пик ->
затухание, а не ступенька.

ПАРАМЕТРЫ — ПОКА ДОГАДКА, НЕ ЗАМЕР
-----------------------------------
attack/decay/thresholds заданы «на глаз». В проекте правило: порог без замера —
ложь. Здесь замер появится ПЕРВЫМ ПРОГОНОМ рассказа; до тех пор числа честно
помечены как DRAFT и их нельзя считать истиной (см. experiments/).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


MAX_EMOTION = 255


# Пороги «плохой» шкалы. Слова — из примера хозяина: безразличие, лёгкое
# неудовольствие, раздражение, злость, гнев, агрессия. Числа — DRAFT (догадка),
# их задача — дать ПОВЕДЕНИЮ различимые ступени, а не быть точными.
BAD_LEVELS: List[Tuple[int, str]] = [
    (0, "безразличие"),
    (20, "лёгкое неудовольствие"),
    (60, "раздражение"),
    (120, "злость"),
    (180, "гнев"),
    (230, "агрессия"),
]

# Пороги «хорошей» шкалы — DRAFT, симметричны по смыслу.
GOOD_LEVELS: List[Tuple[int, str]] = [
    (0, "безразличие"),
    (20, "легкая приязнь"),
    (60, "радость"),
    (120, "воодушевление"),
    (180, "восторг"),
    (230, "эйфория"),
]


@dataclass
class EmotionScale:
    """
    Одна шкала 0..255 с нарастанием и затуханием.

    Близость (степень близости к другому агенту) ЗАДЕЛОМ учтена как множитель
    к нарастанию: stimulus() умножает входящий заряд на closeness. Сейчас
    closeness=1.0 (хозяин просил тест без неё), но место готово: чужая гадость
    -> 0.2, обида близкого -> 2.0 — это ДАННЫЕ, а не отдельный код.
    """
    value: int = 0
    attack: int = 40          # сколько единиц из запаса перетекает в шкалу за тик
    decay: int = 8            # сколько шкала теряет сама за тик
    pending: float = 0.0      # накопленный, но ещё не «прочувствованный» заряд
    ceiling: int = MAX_EMOTION

    def stimulus(self, amount: float, closeness: float = 1.0) -> None:
        """Внешний заряд. Идёт в ЗАПАС, а в шкалу попадёт по тикам (attack)."""
        if amount <= 0:
            return
        self.pending += float(amount) * max(0.0, float(closeness))

    def tick(self) -> int:
        """Один шаг времени: утечка запаса в шкалу, затем собственное затухание."""
        if self.pending > 0:
            step = min(self.pending, float(self.attack))
            self.pending -= step
            self.value = min(self.ceiling, self.value + int(round(step)))
        if self.value > 0:
            self.value = max(0, self.value - self.decay)
        return self.value


class EmotionSystem:
    """
    Обе шкалы одного агента + история по тикам.

    tick() двигает ВРЕМЯ (обе шкалы затухают), stimulus() вбрасывает событие.
    Это ровно то разделение, о котором просил хозяин: пустые тики нужны, чтобы
    эмоция успела и нарасти, и упасть между событиями.
    """

    def __init__(self, attack: int = 40, decay: int = 8,
                 good_attack: Optional[int] = None,
                 good_decay: Optional[int] = None):
        # НАРАСТАНИЕ и ЗАТУХАНИЕ плохого и хорошего могут различаться (обида
        # живёт дольше радости) — поэтому параметры задаются раздельно, а не
        # одним числом на всё. По умолчанию совпадают.
        self.bad = EmotionScale(attack=attack, decay=decay)
        self.good = EmotionScale(attack=good_attack if good_attack is not None
                                 else attack,
                                 decay=good_decay if good_decay is not None
                                 else decay)
        self.t = 0
        self.history: List[Dict[str, int]] = []

    # ── время ──────────────────────────────────────────────
    def tick(self) -> Dict[str, int]:
        """Прокрутить один тик времени. Обе шкалы живут независимо."""
        self.t += 1
        bad = self.bad.tick()
        good = self.good.tick()
        snap = {"t": self.t, "good": good, "bad": bad}
        self.history.append(snap)
        return snap

    def tick_n(self, n: int) -> Dict[str, int]:
        """Прогнать n пустых тиков (время между событиями)."""
        snap = {"t": self.t, "good": self.good.value, "bad": self.bad.value}
        for _ in range(max(0, n)):
            snap = self.tick()
        return snap

    # ── стимулы ────────────────────────────────────────────
    def hurt(self, amount: float, closeness: float = 1.0) -> None:
        """Плохое событие (обида, обман): в «плохую» шкалу."""
        self.bad.stimulus(amount, closeness)

    def please(self, amount: float, closeness: float = 1.0) -> None:
        """Хорошее событие: в «хорошую» шкалу."""
        self.good.stimulus(amount, closeness)

    # ── чтение состояния ───────────────────────────────────
    def state(self) -> Dict[str, int]:
        return {"t": self.t, "good": self.good.value, "bad": self.bad.value}

    def level(self) -> Tuple[str, str]:
        """Имена уровней обеих шкал (для ответа человеку)."""
        return self._name(self.good.value, GOOD_LEVELS), \
               self._name(self.bad.value, BAD_LEVELS)

    @staticmethod
    def _name(value: int, levels: List[Tuple[int, str]]) -> str:
        name = levels[0][1]
        for threshold, label in levels:
            if value >= threshold:
                name = label
        return name

    def bad_at_least(self, value: int) -> bool:
        """Достигла ли «плохая» шкала порога — момент, когда агент ДЕЙСТВУЕТ."""
        return self.bad.value >= value

    def timeline(self) -> str:
        """Человекочитаемая история шкал — доказательство динамики."""
        lines = []
        for snap in self.history:
            g, b = self._name(snap["good"], GOOD_LEVELS), \
                   self._name(snap["bad"], BAD_LEVELS)
            lines.append(f"  t={snap['t']:>2}  хорошее={snap['good']:>3} ({g})"
                         f"   плохое={snap['bad']:>3} ({b})")
        return "\n".join(lines)


if __name__ == "__main__":
    emo = EmotionSystem()
    print("--- модель эмоций: обида от близкого (closeness=2.0), заряд 200 ---")
    emo.hurt(200, closeness=2.0)
    for _ in range(8):
        emo.tick()
    print(emo.timeline())
