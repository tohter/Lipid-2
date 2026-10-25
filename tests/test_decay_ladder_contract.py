"""
КОНТРАКТ, ОЧЕРЕДЬ E: ЧТО НА САМОМ ДЕЛЕ ДЕЛАЕТ ЗАТУХАНИЕ СНА.

Память ставила задачу так: «decay 5% убивает силу 1 за один сон
(int(1*0.95)=0) ... Это следствие пары (rate 0.05, trigger 80), а не поломка;
менять только с новым замером». Замер сделан. Он подтвердил рамку памяти и
переворачивает деталь: виноват не «5%», а то, как из 5% извлекают целое, и
что ручка, которой эту величину регулируют, не крутится.

ЗАМЕР №1, rate — мёртвая ручка (записан до любой правки).
new_s = int(s * (1 - rate)) на ВЕЙШЕЙ законной шкале силы 1..15:

    rate      итог на силах 1..15
    0.001     -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
    0.01      -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
    0.02      -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
    0.05      -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1   <-- текущее
    0.1       -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2
    0.3       -1 -1 -1 -2 -2 -2 -3 -3 -3 -3 -4 -4 -4 -5 -5

  rate = 0.001, 0.01, 0.02 и 0.05 дают ПОБИТОВО ОДИН результат: минус ровно
  один балл силы за сон, а не «пять процентов». 4-битный saturating-счётчик
  силы (core/transitions.py, 0..15) делает float-процент бессмысленным: пока
  rate <= 1/15 (~0.0667), усечение всегда срезает ровно единицу.

  Следствие для комментария brain/sleep.py у DECAY_RATE («Медленнее — редкий
  шум живёт вечно и продолжает побеждать в предсказании»): левая часть
  НЕВЫПОЛНИМА. Замедлить затухание ЭТИМ кодом нельзя ни при каком законном
  rate. Число 0.05 стоит не «на медленное забывание», а где попало внутри
  плато.

ЗАМЕР №2, цена редкости (добит до точной границы на живом PredictiveCortex,
  пара «пёс->мяч», заполнитель-шум; подтверждение даёт +1, delta_confirm=1):

    подтверждений за сон   тиков/подтв   за сколько подтверждений до порога 8
      2.00                     40        15
      1.33                     60        29
      1.14                     70        57
      1.07                     75       113
      1.03                     78       281
      1.01                     79       561
      1.00                     80        НЕ ДОХОДИТ (за 4000 подтверждений)
      0.89                     90        НЕ ДОХОДИТ

  Цена редкости растёт лавинообразно и упирается в СТЕНУ ровно на одном
  подтверждении на сон: 1.01 — это 561 повторение, 1.00 — никогда. Причина
  арифметическая: +1 за подтверждение против минус 1 за сон, сила 1 после сна
  даёт int(1*0.95)=0, опыт не помнит себя на нуле. Это СМЫСЛ (шум не
  выучивается), но смысл обязан быть назван и заперён, а не выводимую заново
  каждую сессию.
  (Первая версия вывода была неточна — «при 1.01 никогда». Точный замер даёт
  561. Неточность стоила отдельного промера и стоила бы неверного замка.)

ЗАМЕР №3, расхождение с доком. experiments/PLASTICITY_CONTRACT.md:
  «Частота: раз в N уроков (N=50)» и «delta_confirm = 2». Факт: delta_confirm
  = 1 (core/cortex.py), сон каждые 80 тиков при 10 тиков/урок = каждые 8
  уроков. Док — план до реализации, реализованы другие числа; расхождение
  6.25x по частоте.

ЗАМЕР №4, молчание. Переход, убитый затуханием, исчезает из таблицы, но в
  rep.notes НЕ идёт НИЧЕГО: остаётся только счётчик compacted. «Вот этот
  конкретный опыт умер» неотличимо от «такого опыта не было».

ЧЕГО ЗДЕСЬ НЕТ, И ПОЧЕМУ
  Ни одна константа (rate, trigger, порог 8) НЕ изменена: память требовала
  менять их только с замером, а замер показал, что смена rate вообще ничего
  не меняет, то есть калибровать этим нечего.

  Две кандидатные правки арифметики ОТВЕРГНУТЫ замером:
    round(s*(1-rate))  -> при 0.05 даёт 1..14 вместо 0..13: сила 1 перестаёт
                          умирать, а это запертый контракт
                          test_rare_noise_can_actually_die (шум обязан иметь
                          шанс исчезнуть). Правка ломает систему, а не чинит.
    s - max(1,int(s*rate)) -> при 0.05 побитово то же самое, но ручка оживает
                          только ВВЕРХ (0.2, 0.3), а вниз по-прежнему no-op:
                          плато не лечится, добавляется лишь новая шкала,
                          которую нечем калибровать.
  Настоящая двусторонняя ручка требует либо finer-шкалы силы, либо другой
  частоты сна — то есть решения владельца, а не молчаливой правки константы.
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.sleep import (CONSOLIDATION_THRESHOLD, DECAY_RATE,          # noqa: E402
                         SLEEP_TRIGGER_TICKS, TICKS_PER_STRENGTH_POINT,
                         decay_transitions)
from core.transitions import MAX_STRENGTH, TransitionTable              # noqa: E402

SCALE = tuple(range(1, MAX_STRENGTH + 1))


def table_of(pairs):
    t = TransitionTable()
    for head, tail, s in pairs:
        t.set_strength(head, tail, s)
    return t


def after_sleep(s, rate):
    """Сила одной пары после одного сна при данном rate."""
    t = table_of([("а", "б", s)])
    decay_transitions(t, rate)
    return t.get("а", "б") if t.known_tails("а") else 0


class TestDecayLadderIsMeasured(unittest.TestCase):
    """Затухание обязано описывать АРИФМЕТИКУ счётчика, а не желание."""

    def test_one_point_per_sleep_at_working_rate(self):
        """
        Главный смысл: DECAY_RATE=0.05 — это НЕ «5%», а «минус ОДИН балл за
        сон» на всей шкале. Формулировка для человека, читающего константу.
        """
        self.assertEqual([after_sleep(s, DECAY_RATE) for s in SCALE],
                         [s - 1 for s in SCALE],
                         "затухание на рабочем rate перестало быть «минус "
                         "один балл» — вся шкала пересчитана")

    def test_strength_one_dies_in_one_sleep(self):
        """int(1*0.95)=0 из очереди E: заперто как СМЫСЛ (шум обязан умирать)."""
        self.assertEqual(after_sleep(1, DECAY_RATE), 0)

    def test_plateau_makes_rate_a_dead_knob(self):
        """
        ЗАМЕР №1, ЗАПИСЬ ОГРАНИЧЕНИЯ, а не желаемого поведения.

        На всей нижней половине законных значений rate затухание ПОБИТОВО
        одинаково: «сделать забывание медленнее» этим кодом невозможно.
        Тест запирать факт, чтобы следующий читатель не выставил rate «на
        медленное» и не проверил потом, почему ничего не произошло.

        ЕСЛИ ЭТОТ ТЕСТ ПОКРАСНЕЛ — кто-то сделал ручку настоящим (или сменил
        шкалу силы). Тогда обновлять надо не только тест: калибровка
        (rate, SLEEP_TRIGGER_TICKS, порог 8) обязана быть перемерена, а
        комментарии у DECAY_RATE и experiments/PLASTICITY_CONTRACT.md —
        переписаны под новый замер.
        """
        for slow in (0.001, 0.01, 0.02):
            self.assertEqual([after_sleep(s, slow) for s in SCALE],
                             [after_sleep(s, DECAY_RATE) for s in SCALE],
                             f"rate={slow} совпадало с {DECAY_RATE} — ручка не "
                             f"крутилась; раз разъехалось, калибровка сдвинулась")

    def test_rate_above_plateau_starts_to_bite_upwards(self):
        """
        Второе полугодие того же замера: выше 1/15 усечение начинает срезать
        по два балла. Заперто, чтобы направление «быстрее» было рабочим,
        когда ручку починят, и чтобы текущий потолок плато был виден.
        """
        self.assertEqual(after_sleep(MAX_STRENGTH, 0.1), 13,
                         "0.1 обязан снимать 2 балла с 15: это единственное "
                         "законное место, где rate вообще что-то значит")
        self.assertLessEqual(DECAY_RATE, 1.0 / MAX_STRENGTH,
                             f"DECAY_RATE={DECAY_RATE} вне плато 1/{MAX_STRENGTH}"
                             " — затухание сильнее одного балла, шкала "
                             "пересчитана, см. замеры в шапке")

    def test_strong_tails_do_not_lose_two_steps_at_once(self):
        self.assertEqual(after_sleep(MAX_STRENGTH, DECAY_RATE), MAX_STRENGTH - 1)

    def test_header_table_is_measured_not_written(self):
        """
        Таблица замеров в шапке этого файла обязана сходиться с кодом.

        Причина не косметическая: пока я писал этот контракт, одна строка
        таблицы (rate=0.3) была приписана от руки и оказалась НЕВЕРНОЙ —
        фактическое затухание там -1 -1 -1 -2 -2 -2 -3..., а не «почти все
        по два». Цифра в комментарии, которую никто не сверяет, врёт так же
        уверенно, как правда.
        """
        import re
        here = os.path.abspath(__file__)
        doc = open(here, encoding="utf-8").read().split('"""')[1]
        rows = re.findall(r"^ {4}([\d.]+) {5,}(-?[\d -?]+)$", doc, re.M)
        self.assertTrue(rows, "таблица замеров пропала из шапки")
        for rate_text, claimed in rows:
            rate = float(rate_text)
            if rate >= 1.0:
                continue
            actual = [s - after_sleep(s, rate) for s in SCALE]
            self.assertEqual(claimed.split(),
                             [f"-{d}" for d in actual],
                             f"строка rate={rate} в шапке разошлась с кодом: "
                             f"заявлено {claimed.strip()}, факт {actual}")


class TestDeathIsNotSilent(unittest.TestCase):
    """ЗАМЕР №4: гибель опыта обязана быть названа, а не спрятана в счётчик."""

    def test_killed_transition_is_announced(self):
        t = table_of([("пёс", "шум", 1)])
        rep = decay_transitions(t, DECAY_RATE)
        self.assertEqual(t.known_tails("пёс"), [], "шум обязан исчезнуть")
        self.assertEqual(rep.compacted, 1)
        self.assertTrue(rep.notes,
                        f"опыт умер молча, заметок нет: {rep.notes}")
        text = " ".join(rep.notes)
        self.assertIn("шум", text, f"заметка не называет, ЧТО умерло: {text}")

    def test_healthy_sleep_stays_quiet(self):
        """
        Обратная сторона: громчать на каждом сне — обесценить громкость.
        Живой переход погибнуть не должен и заметку о смерти не плодит.
        """
        t = table_of([("а", "живо", 9)])
        rep = decay_transitions(t, DECAY_RATE)
        self.assertEqual((t.get("а", "живо"), rep.compacted), (8, 0))
        self.assertEqual([n for n in rep.notes if "умер" in n.lower()], [],
                         f"пустой сон зашумел: {rep.notes}")

    def test_only_the_dead_are_mourned(self):
        """Смешанный сон: из трёх пар умирает одна — заметка тоже одна."""
        t = table_of([("а", "шум", 1), ("а", "живо", 9), ("а", "крепко", 15)])
        rep = decay_transitions(t, DECAY_RATE)
        dead = [n for n in rep.notes if "умер" in n.lower()]
        self.assertEqual(len(dead), 1, f"заметок о смерти не одна: {rep.notes}")
        self.assertIn("шум", dead[0])

    def test_dry_run_does_not_claim_a_death_that_did_not_happen(self):
        """
        Пробный сон («Посмотреть сон» в GUI, dry_run=True) НЕ удаляет переход,
        поэтому обязан сказать «умер бы», а не «умер». Иначе витрина врёт про
        несделанное — ровно тот грех, который весь проект выжигает тестами.
        """
        t = table_of([("пёс", "шум", 1)])
        rep = decay_transitions(t, DECAY_RATE, dry_run=True)
        dead = [n for n in rep.notes if "умер" in n.lower()]
        self.assertEqual(len(dead), 1, f"пробный сон обязан предупредить: {rep.notes}")
        self.assertIn("умер бы", dead[0],
                      f"dry_run выдал совершившийся факт: {dead[0]}")
        self.assertEqual(t.get("пёс", "шум"), 1,
                         "dry_run всё-таки тронул силу")
        self.assertEqual(rep.compacted, 0)


class TestRarityPriceIsLocked(unittest.TestCase):
    """
    ЗАМЕР №2. Цена редкости — СМЫСЛ системы (шум не учится), и она обязана
    быть заперта числом: правка арифметики или частоты сна не имела права
    двигать её молча.
    """

    @staticmethod
    def _run(tpc, cycles):
        """
        Прогон реального PredictiveCortex: пару «пёс->мяч» подтверждают раз в
        `tpc` тиков, остальное время идёт шум, сон каждые
        SLEEP_TRIGGER_TICKS тиков. Возвращает (сила после каждого сна, номер
        подтверждения, на котором достигнут порог, или None).
        """
        from core.cortex import PredictiveCortex
        from core.graph import GraphMemory
        cx = PredictiveCortex(GraphMemory())
        ticks, seen, reached = 0, [], None
        for n in range(1, cycles + 1):
            cx.observe("пёс", "мяч")
            ticks += 1
            for _ in range(tpc - 1):
                cx.observe("дождь", "капля")
                ticks += 1
            while ticks >= SLEEP_TRIGGER_TICKS:
                decay_transitions(cx.transitions, DECAY_RATE)
                ticks -= SLEEP_TRIGGER_TICKS
            seen.append(cx.transitions.get("пёс", "мяч"))
            if reached is None and seen[-1] >= CONSOLIDATION_THRESHOLD:
                reached = n
        return seen, reached

    def test_two_per_sleep_measured_cost(self):
        """ЗАМЕР: 2 подтверждения на сон (раз в 40 тиков) = порог за 15 шт."""
        _, reached = self._run(40, 40)
        self.assertEqual(reached, 15,
                         "частота выучивания уехала — гонку силы и затухания "
                         "пересчитали, а док и память молчат")

    def test_cliff_edge_is_where_measured(self):
        """
        ЗАМЕР: на краю (1.14 подтверждения на сон, раз в 70 тиков) сила всё же
        доходит, но за 57 подтверждений вместо 15.
        """
        _, reached = self._run(70, 70)
        self.assertEqual(reached, 57, "край клиффа сдвинулся: цена редкости "
                                      "изменилась")

    def test_exactly_one_per_sleep_never_accumulates(self):
        """
        НАСТОЯЩИЙ ОБРЫВ измерен точно: ровно ОДНО подтверждение на сон
        (раз в 80 тиков) — неподвижная точка на НУЛЕ. Подтверждение даёт +1,
        ближайший сон снимает 1, опыт не помнит себя никогда, сколько бы раз
        ни повторился. Прежняя формулировка «при 1.01 не выучивается» была
        неточна: при 1.01 доходит, но за 561 подтверждение.
        """
        seen, reached = self._run(SLEEP_TRIGGER_TICKS, 200)
        self.assertEqual(set(seen), {0},
                         f"редкий переход всё-таки копится: {seen[:12]} — "
                         f"либо затухание, либо шаг подтверждения уехали")
        self.assertIsNone(reached)

    def test_confirm_step_is_one_not_two(self):
        """
        ЗАМЕР №3 внутри кода. PLASTICITY_CONTRACT.md утверждает delta_confirm
        = 2, brain/sleep.py в docstring decay_transitions тоже обещает «шаг
        подтверждения +2 (замер)». Факт — 1. Замер здесь, потому что вся
        гонка «сила против затухания» считается именно от этого шага: с +2
        обрыва по редкости не было бы вовсе.
        """
        from core.cortex import PredictiveCortex
        step = PredictiveCortex.PlasticityConfig().delta_confirm
        self.assertEqual(step, 1,
                         f"шаг подтверждения = {step}, а вся шкала редкости "
                         f"в этом файле замерена при 1")

    def test_trigger_arithmetic_is_stated(self):
        """
        «Подтверждений на сон» выводится из констант: 80 тиков / (10 тиков за
        урок) = сон каждые 8 уроков. PLASTICITY_CONTRACT.md утверждал «раз в
        N уроков (N=50)» — расхождение 6.25x, замерено тут, чтобы его
        нельзя было повторить молча.
        """
        self.assertEqual(SLEEP_TRIGGER_TICKS,
                         CONSOLIDATION_THRESHOLD * TICKS_PER_STRENGTH_POINT)
        self.assertEqual(SLEEP_TRIGGER_TICKS / TICKS_PER_STRENGTH_POINT, 8.0,
                         "частота сна уехала: док и память говорят про другое, "
                         "а гонка силы/затухания считается по ней")


class TestGuardStillRaises(unittest.TestCase):
    """Строгая граница rate обязана остаться: правка заметок не размяла её."""

    def test_out_of_range_rate_raises(self):
        for bad in (-0.1, 1.0, 2.0):
            with self.assertRaises(ValueError):
                decay_transitions(table_of([("а", "б", 5)]), bad)

    def test_zero_rate_is_still_exact_noop(self):
        t = table_of([("а", "б", 9)])
        rep = decay_transitions(t, 0.0)
        self.assertEqual((t.get("а", "б"), rep.decayed), (9, 0))


class TestBrainAnnouncesQuietSleep(unittest.TestCase):
    """
    Сон, который ТОЛЬКО убрал шум, обязан быть виден в журнале мозга.

    ЗАМЕР (записан до правки): мир с парой «пёс->мяч» силы 4 (до порога 8 ещё
    не дошла) и тремя шумовыми переходами силы 1 даёт сон
        created=0 raised=0 lowered=0 decayed=4 compacted=3
    а мозг печатал «🌙» лишь при created/raised/lowered — то есть весь этот
    сон в журнале НЕ СУЩЕСТВОВАЛ. Владелец видел бы «система спит» и не мог
    бы узнать, что сон регулярно выстригает опыт, ничего при этом не создав.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="sleep_announce_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_silent_killing_sleep_is_announced(self):
        from brain.lipid2_brain import Lipid2Brain
        b = Lipid2Brain(memory_dir=self.dir)
        b.transitions.confirm("пёс", "мяч", 4)      # до порога 8 не доросла
        b.transitions.confirm("пёс", "шум1", 1)
        b.transitions.confirm("пёс", "шум2", 1)
        seen = []
        b._say = lambda m, **kw: seen.append(m)
        rep = b.sleep_now()
        r = rep["report"]
        self.assertEqual((r.created, r.raised, r.lowered), (0, 0, 0),
                         "замер сломан: сон стал что-то создавать")
        self.assertGreater(r.compacted, 0, "замер сломан: шум не помер")
        text = " ".join(seen)
        self.assertIn("🌙", text,
                      f"сон убил опыт и промолчал в журнале: {seen}")
        self.assertIn("умер", text.lower(),
                      f"журнал не назвал, что именно умерло: {seen}")

    def test_truly_empty_sleep_stays_quiet(self):
        """
        Обратная сторона: сон, где умирать нечему, не обязан кричать — иначе
        журнал превращается в шум, и «🌙» перестанет значить «что-то вышло».
        """
        from brain.lipid2_brain import Lipid2Brain
        b = Lipid2Brain(memory_dir=self.dir)
        b.transitions.confirm("птенец", "летать", 15)
        seen = []
        b._say = lambda m, **kw: seen.append(m)
        rep = b.sleep_now()
        self.assertEqual(rep["report"].compacted, 0, "здесь умирать нечему")
        self.assertEqual([m for m in seen if "умер" in m.lower()], [],
                         f"пустой сон зашумел в журнале: {seen}")

    def test_sleep_panel_shows_the_dead_by_name(self):
        """
        Панель сна (ui/app_gui.py) перебирает rep.changes и rep.notes, но
        никогда не показывала ни одной строки о затухании: мозг собирал отчёт
        из consolidate() и переносил от сна лишь счётчики, теряя changes и
        notes. Проверка ровно того, что видит окно, а не внутреннего отчёта.
        """
        from brain.lipid2_brain import Lipid2Brain
        from ui.app_gui import Lipid2GUI
        b = Lipid2Brain(memory_dir=self.dir)
        b.transitions.confirm("птенец", "летать", 15)
        b.transitions.confirm("птенец", "шум", 1)
        res = b.sleep_now()
        app = type("App", (), {})()
        shown = Lipid2GUI._sleep_details(app, res)
        self.assertIn("шум", shown,
                      f"окно не показало погибший опыт: {shown}")
        self.assertIn("умер", shown.lower(),
                      f"окно назвало смерть не смертью: {shown}")

    def test_dry_run_panel_warns_without_doing(self):
        """
        Кнопка «Посмотреть сон (не записывая)» — тот же путь, но dry_run=True,
        и заметки туда приходят из ДРУГОЙ ветки sleep_now. Заперто отдельно,
        иначе пробный сон остался бы немым: он честно ничего не пишет, но и
        не говорит, что собирался убить, — а ровно для этого он и нужен.
        """
        from brain.lipid2_brain import Lipid2Brain
        from ui.app_gui import Lipid2GUI
        b = Lipid2Brain(memory_dir=self.dir)
        b.transitions.confirm("птенец", "летать", 15)
        b.transitions.confirm("птенец", "шум", 1)
        before = b.transitions.to_json()
        res = b.sleep_now(dry_run=True)
        app = type("App", (), {})()
        shown = Lipid2GUI._sleep_details(app, res)
        self.assertIn("шум", shown,
                      f"пробный сон скрыл, кого собрался убрать: {shown}")
        self.assertIn("умер бы", shown,
                      f"пробный сон выдал совершившийся факт: {shown}")
        self.assertEqual(b.transitions.to_json(), before,
                         "dry_run тронул живую таблицу")
        self.assertFalse(res["saved"])


if __name__ == "__main__":
    unittest.main()
