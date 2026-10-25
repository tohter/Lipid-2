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

ПАРАМЕТРЫ — ЗАМЕРЕНЫ (experiments/pF_emotion_measure.py, очередь F)
----------------------------------------------------------------
attack/decay/thresholds долго были «на глаз». Замер показал, ЧТО именно они
значат — до калибровки это надо было знать (на шкале силы очередь E нашла,
что «5% за сон» — это минус один балл, а не процент):

  * реальная скорость нарастания = attack - decay, НЕ attack: tick() сначала
    вливает attack, в ЭТОМ ЖЕ тике отнимает decay. Замер: attack=40 decay=8
    -> шкала растёт на 32 за тик.
  * при decay >= attack эмоцию нельзя выразить НИ ПРИ КАКОМ заряде (пик 0).
    Это не должно молчать — см. __post_init__, там крик.
  * потолок шкалы при decay > 0 недостижим: максимум = MAX_EMOTION - decay
    (замер: decay=8 -> 247, decay=1 -> 254, decay=0 -> 255).
  * выше насыщения величина заряда превращается в СРОК, а не в силу: заряд
    400 живёт 41 тик, 3200 -> 111 тик, пик у обоих 247. Это принято как
    СМЫСЛ (сильная обида длится дольше) и заперто тестом.
"""
from __future__ import annotations

import sys

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


MAX_EMOTION = 255


# Пороги «плохой» шкалы. Слова — из примера хозяина: безразличие, лёгкое
# неудовольствие, раздражение, злость, гнев, агрессия.
#
# ЗАМЕР F/M3 (experiments/pF_emotion_measure.py): чем именно является каждое
# слово при дефолтных attack=40 decay=8 — какой ЗАРЯД его вызывает (пик шкалы
# попадает в полосу):
#   безразличие 0..27, неудовольствие 28..75, раздражение 76..151,
#   злость 152..227, гнев 228..293, агрессия от 294 и ДО БЕСКОНЕЧНОСТИ.
# т.е. полоса «агрессия» одна схлопнулась: все заряды от 294 называются ОДНИМ
# словом (шкала упирается в 247). Это не «ступени неточны», а свойство
# квантованной шкалы; заперто тестом.
BAD_LEVELS: List[Tuple[int, str]] = [
    (0, "безразличие"),
    (20, "лёгкое неудовольствие"),
    (60, "раздражение"),
    (120, "злость"),
    (180, "гнев"),
    (230, "агрессия"),
]

# Пороги «хорошей» шкалы — симметричны по смыслу. Границы полос важны для
# ПОКОЯ (floor): пол 19 ещё даёт слово «безразличие», пол 20 — уже «легкая
# приязнь» (замер F/M5). Число покоя выбирает хозяин, ЗНАЯ эти границы.
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

    floor — ПОКОЙ шкалы, ненулевой пол, ниже которого чувство не остывает.
    По умолчанию 0: нынешнее поведение сохраняется побитово, а значение пола
    — решение хозяина (для «хорошей» шкалы это «любовь не умирает от
    тишины»). Пол — ПОЛ, а не потолок: живое чувство он не режет.
    """
    value: int = 0
    attack: int = 40          # сколько единиц из запаса перетекает в шкалу за тик
    decay: int = 8            # сколько шкала теряет сама за тик
    pending: float = 0.0      # накопленный, но ещё не «прочувствованный» заряд
    ceiling: int = MAX_EMOTION
    floor: int = 0            # покой: ниже этого шкала не остывает

    def __post_init__(self):
        """
        МЁРТВАЯ ЗОНА decay >= attack — крик, а не молчание (замер F/M1).

        При такой паре шкала НЕ РАСТЁТ ни при каком заряде: подток и утечка
        случаются в одном тике. Замер: attack=40 decay=40, заряд 1000000 ->
        пик 0, слово «безразличие», без исключения и без единой строчки в
        отчёте. Тот же класс тихой деградации, что и у невызываемого
        параметра: калибровать молча невозможно, поэтому говорим сразу.
        """
        if self.attack <= self.decay:
            print(f"[emotions] ⚠ attack={self.attack} <= decay={self.decay}: "
                  f"за тик шкала растёт на attack-decay="
                  f"{self.attack - self.decay}, то есть НЕ растёт вовсе. "
                  f"Эмоцию нельзя выразить ни при каком заряде (замер F/M1). "
                  f"Нужно decay < attack.", file=sys.stderr, flush=True)
        if self.value < self.floor:
            # Покой — он и при отсутствии событий: агент рождается уже с ним,
            # иначе на t=0 было бы «безразличие», а на t=1 «приязнь» без
            # единого стимула — шкала врёт о начале чувства.
            self.value = self.floor

    def reachable_max(self) -> int:
        """
        Максимум, которого шкала ДОСТИГАЕТ при бесконечном заряде.

        ЗАМЕР F/M2: это НЕ ceiling. Подток и утечка в одном тике оставляют
        максимум = ceiling - decay (247 при decay=8). Разница важна не как
        косметика: порог выше reachable_max физически непроходим, и сказать про
        него «чисел не хватает» — значит соврать, будто помогает «сильнее
        обидеть».
        """
        return self.ceiling if self.decay <= 0 else self.ceiling - self.decay

    def saturated(self) -> bool:
        """Уперлась ли шкала в свой максимум: дальше сила НЕ измеряется."""
        return self.value >= self.reachable_max()

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
        # ПОЛ вместо нуля. ОДИН механизм, без запасающего `if value > floor`
        # вокруг: с таким `if` max(floor, ...) внутри был бы мёртвым кодом,
        # и мутация «забыл про пол» проходила бы незаметно (проверено).
        # При floor=0 это ровно прежний max(0, value - decay).
        self.value = max(self.floor, self.value - self.decay)
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
                 good_decay: Optional[int] = None,
                 good_floor: int = 0):
        # НАРАСТАНИЕ и ЗАТУХАНИЕ плохого и хорошего могут различаться (обида
        # живёт дольше радости) — поэтому параметры задаются раздельно, а не
        # одним числом на всё. По умолчанию совпадают.
        self.bad = EmotionScale(attack=attack, decay=decay)
        self.good = EmotionScale(attack=good_attack if good_attack is not None
                                 else attack,
                                 decay=good_decay if good_decay is not None
                                 else decay,
                                 floor=good_floor)
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
