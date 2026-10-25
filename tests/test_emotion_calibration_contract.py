# -*- coding: utf-8 -*-
"""
Очередь F — ЗАМЕР эмоции заперт, а молчаливая невозможность — закричала.

Что измерено (experiments/pF_emotion_measure.py, перемеряется этим тестом) и
что из этого ДЕФЕКТ, а что СМЫСЛ:

  ДЕФЕКТ 1. МЁРТВАЯ ЗОНА. tick() сначала вливает attack, в ЭТОМ ЖЕ тике
      отнимает decay. Значит реальная скорость нарастания = attack - decay,
      а не attack, как обещает шапка модуля. При decay >= attack эмоцию
      нельзя выразить НИ ПРИ КАКОМ заряде: пик 0, слова «безразличие»,
      исключения нет, заметки нет. Молчание = тихая деградация (урок проекта).

  ДЕФЕКТ 2. ВЕРХНЯЯ ПОЛОСА НЕДОСТУПНА. Максимум шкалы при decay >0 равен
      MAX_EMOTION - decay (замер: decay=8 -> 247, decay=1 -> 254, decay=0 ->
      255). Порог «агрессии» 230 живёт при decay <= 25; при decay=26 спуск
      рассказа НЕВОЗМОЖЕН ни при каком заряде и ни за сколько тиков — а
      симулятор сейчас говорит только «порог не пройден», не называя причину.

  ДЕФЕКТ 3. НАСЫЩЕНИЕ ВЫДАЁТСЯ ЗА СИЛУ. Заряд 294 и заряд 100000 дают ОДНО
      слово («агрессия») и один пик 247; выше насыщения величина заряда
      превращается в СРОК (замер: 400->41 тик, 3200->111 тик). Смысл это или
      ошибка — решает хозяин, но молчать нельзя: отчет обязан сказать, что
      шкала упёрлась в потолок и дальше сила НЕ измеряется.

  НЕ ДЕФЕКТ (заперто как смысл): квантование полос, линейная жизнь заряда
      выше насыщения, окно спуска рассказа 8..12.

ЗАПРЕЩЕНО менять арифметику без нового замера: порядок «сперва влилось, потом
утекло» заперт test_decay_without_events (30, 20). Тест проверяет числа, а не
комментарии: п.16 памяти.
"""
import os
import re
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from brain.emotions import MAX_EMOTION, EmotionScale, EmotionSystem   # noqa: E402
from brain.social_sim import (AGGRESSION_THRESHOLD,                   # noqa: E402
                              CLOSENESS_LIVING_TOGETHER, RIPEN_TICKS,
                              build_cat_canary_simulator)

MEASURE = os.path.join(ROOT, "experiments", "pF_emotion_measure.py")

# Минимальный ЗАРЯД, при котором шкала вообще удостаивает слова «агрессия»
# при дефолтных attack=40 decay=8. Замер F/M3, пересобран тестом ниже.
AGGRESSION_MIN_CHARGE = 294


def run_measure():
    """Пересобрать замера живой командой. Число без команды — не число."""
    env = dict(os.environ, LIPID2_STATUS="0")
    return subprocess.run([sys.executable, MEASURE], capture_output=True,
                          text=True, env=env, cwd=ROOT, timeout=300).stdout


def summary(text):
    line = [l for l in text.splitlines() if l.strip().startswith("real_slope")]
    assert line, f"замер не выдал строку ИТОГ:\n{text[-500:]}"
    return dict(p.split("=", 1) for p in line[0].split() if "=" in p)


class TestMeasurementIsReproducible(unittest.TestCase):
    """
    Цифры этого файла живут не в этом файле: их пересобирает скрипт. Если
    арифметика уедет — упадёт здесь, а не враньём в комментарии.
    """

    @classmethod
    def setUpClass(cls):
        cls.out = run_measure()
        cls.s = summary(cls.out)

    def test_every_number_in_summary_is_recomputed_by_code(self):
        s = self.s
        self.assertEqual(int(s["real_slope_default"]), 40 - 8)
        self.assertEqual(int(s["deadzone_peak"]), 0)
        self.assertEqual(int(s["ceiling_default"]), MAX_EMOTION - 8)
        self.assertEqual(int(s["aggression_min_charge"]),
                         AGGRESSION_MIN_CHARGE)
        self.assertEqual(int(s["good_zero_ticks_120"]), 15)
        self.assertEqual(s["window"], "8..12")
        self.assertEqual(s["comment_numbers_match"], "True")
        self.assertEqual(float(s["closeness_cliff"]), 1.5)

    def test_ceiling_formula_holds_on_several_decays(self):
        for dec in (0, 1, 8, 25):
            sc = EmotionScale(attack=40, decay=dec)
            sc.stimulus(10 ** 6)
            top = max(sc.tick() for _ in range(600))
            want = MAX_EMOTION if dec == 0 else MAX_EMOTION - dec
            self.assertEqual(top, want,
                             f"decay={dec}: обещанный потолок {MAX_EMOTION}"
                             f" недостижим, факт {top}")

    def test_story_comment_numbers_are_the_measured_ones(self):
        """128/247/231/199 из комментария social_sim.py — замер, который повторяется."""
        for n, want in ((4, 128), (8, 247), (12, 231), (16, 199)):
            sim = build_cat_canary_simulator(between_ticks=n)
            got = sim.snapshots[-1]["кошка"]["bad"]
            self.assertEqual(got, want, f"between_ticks={n}: было {want}, стало {got}")


class TestDeadZoneScreams(unittest.TestCase):
    """ДЕФЕКТ 1: attack <= decay делает эмоцию невозможной — это должно быть СЛЫШНО."""

    def capture(self, **kw):
        import io
        import contextlib
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            e = EmotionSystem(**kw)
            e.hurt(10 ** 6)
            e.tick_n(50)
        return e, err.getvalue()

    def test_impossible_ratio_is_announced_not_silent(self):
        e, err = self.capture(attack=40, decay=40)
        self.assertEqual(e.bad.value, 0, "пик обязан остаться 0 (арифметику не трогаем)")
        self.assertIn("attack", err,
                      "decay >= attack: эмоцию нельзя выразить, молчать нельзя")

    def test_working_ratio_is_silent(self):
        e, err = self.capture(attack=40, decay=8)
        self.assertGreater(e.bad.value, 0)
        self.assertEqual(err.strip(), "",
                         f"при рабочей паре кричать нельзя, накричано: {err!r}")


class TestUnreachableThresholdIsNamed(unittest.TestCase):
    """ДЕФЕКТ 2: порог, который физически не пройти, надо НАЗЫВАТЬ."""

    def test_reachable_max_is_reported(self):
        sc = EmotionScale(attack=40, decay=8)
        self.assertEqual(sc.reachable_max(), MAX_EMOTION - 8)

    def test_refusal_names_unreachable_threshold(self):
        """
        max = 255 - decay. При decay=26 потолок 229, а порог рассказа 230:
        спуск НЕВОЗМОЖЕН ни при каком заряде и ни за сколько тиков. Сказать
        «порог не пройден» тут — враньё: оно намекает, что надо сильнее обидеть.
        """
        sim = build_cat_canary_simulator(between_ticks=RIPEN_TICKS, decay=26)
        res = sim.why_acted("кошка", "съесть")
        self.assertFalse(res.ok)
        text = " ".join(res.missing)
        low = text.lower()
        self.assertIn("недостижим", low,
                      "отказ обязан назвать порог физически недосягаемым,"
                      f" а не просто непройденным: {text!r}")
        self.assertIn("229", text,
                      "отказ обязан назвать И число потолка (229), иначе "
                      "человек не поймёт, насколько не хватает: "
                      f"{text!r}")
        self.assertIn("decay", low,
                      f"и назвать причину (decay), а не только симптом: {text!r}")

    def test_calibration_reaches_the_scale_from_outside(self):
        """
        Калибровка должна быть ВОЗМОЖНА. ЗАМЕР ДО правки: attack/decay были
        зашиты в сигнатуру agent(name, attack=40, decay=8) и EmotionSystem,
        наружу не отдавались НИКАК — «отдельная задача калибровки с замером»
        не имела ручки. Теперь параметры идут сквозь сборку рассказа.
        """
        sim = build_cat_canary_simulator(between_ticks=RIPEN_TICKS,
                                        attack=64, decay=4)
        self.assertEqual(sim.agents["кошка"].bad.attack, 64)
        self.assertEqual(sim.agents["кошка"].bad.decay, 4)
        self.assertEqual(sim.agents["кошка"].good.attack, 64)


class TestSaturationIsAdmitted(unittest.TestCase):
    """ДЕФЕКТ 3: насыщение шкалы нельзя выдавать за измерение силы."""

    def test_report_says_scale_is_saturated(self):
        sim = build_cat_canary_simulator(between_ticks=RIPEN_TICKS)
        res = sim.why_acted("кошка", "съесть")
        self.assertTrue(res.ok)
        txt = res.as_text()
        self.assertIn("насыщ", txt,
                      "шкала на пике (247 = максимум), сила дальше НЕ измеряется;"
                      " отчёт обязан это сказать")

    def test_two_different_charges_are_the_same_word_but_different_life(self):
        """Выше насыщения сила = срок. Это СМЫСЛ, он заперт, чтобы не уехал."""
        life = {}
        for amt in (400, 3200):
            e = EmotionSystem(attack=40, decay=8)
            e.hurt(amt)
            t = 0
            while t < 2000 and e.tick()["bad"] > 0:
                t += 1
            life[amt] = t
        self.assertGreater(life[3200], life[400],
                           "бОльший заряд обязан жить дольше (иначе срок — не смысл)")


class TestGoodBaselineMechanism(unittest.TestCase):
    """
    Ненулевой покой шкалы good: механизм есть, ЗНАЧЕНИЕ выбирает хозяин.

    Механизм обязан быть нейтральным по умолчанию: floor=0 сохраняет нынешнее
    поведение ПОБитово (иначе «починка любви» тайно сдвинула бы все рассказные
    числа). Число покоя — решение хозяина, кандидаты с замером в памяти.
    """

    # Золотой ряд снят СО СТАРОГО tick() (git HEAD до очереди F):
    #   git show HEAD:brain/emotions.py  ->  EmotionScale(attack, decay)
    #   .stimulus(заряд); [s.tick() for _ in range(N)]
    # Побитовая сверка с новым кодом: 290 тиков, 0 расхождений (замер).
    GOLDEN = {
        (40, 8, 200): [32, 64, 96, 128, 160, 152, 144, 136, 128, 120, 112,
                       104, 96, 88],
        (40, 1, 40): [39, 38, 37, 36, 35, 34, 33, 32, 31, 30],
    }

    def test_floor_zero_is_bitwise_current_behavior(self):
        """
        Пол 0 обязан сохранить нынешнее поведение ПОБитово.

        Иначе «починка любви» тайно сдвинула бы все рассказные числа
        (128/247/231/199) и все тесты шкал. Тест сверяет ряд значений с
        ЗОЛОТЫМ рядом старого кода, а не с тем же новым кодом — иначе это
        была бы тривиально истинная проверка.
        """
        for (atk, dec, charge), want in self.GOLDEN.items():
            sc = EmotionScale(attack=atk, decay=dec, floor=0)
            sc.stimulus(charge)
            got = [sc.tick() for _ in want]
            self.assertEqual(got, want,
                             f"attack={atk} decay={dec} заряд={charge}: пол 0 "
                             f"сдвинул поведение: {got} != {want}")

    def test_good_can_rest_above_zero_when_asked(self):
        e = EmotionSystem(good_floor=20)
        e.please(120)
        e.tick_n(400)
        self.assertEqual(e.state()["good"], 20,
                         "любовь не обязана умирать от тишины до безразличия")

    def test_floor_is_there_from_birth_not_from_first_tick(self):
        """
        Покой есть СРАЗУ, до всякого стимула и до первого тика.

        Иначе на t=0 агент был бы «безразличен», а на t=1 стал бы «приязнен»
        без единого события — шкала соврала бы о МОМЕНТЕ начала чувства.
        Мутация «пол только в tick()» проходила незаметно, пока это не
        запиралось: заперто здесь.
        """
        e = EmotionSystem(good_floor=20)
        self.assertEqual(e.state()["good"], 20)
        self.assertEqual(e.level()[0], "легкая приязнь",
                         "уровень на t=0 обязан соответствовать покою")
        self.assertEqual(e.t, 0, "проверка ДО первого тика")

    def test_floor_never_eats_a_live_emotion(self):
        e = EmotionSystem(good_floor=20)
        e.please(120)
        e.tick()
        self.assertGreater(e.state()["good"], 20,
                           "покой — ПОЛ, а не потолок: живое чувство не режется")

    def test_hurt_scale_keeps_resting_at_zero(self):
        e = EmotionSystem(good_floor=20)
        e.hurt(120)
        e.tick_n(400)
        self.assertEqual(e.state()["bad"], 0,
                         "обида, в отличие от приязни, гаснет в ноль (договорённость)")


class TestBadNumbersStayLocked(unittest.TestCase):
    """Смысловые, а не сломанные, свойства — заперты, чтобы не уехали молча."""

    def test_real_growth_speed_is_attack_minus_decay(self):
        sc = EmotionScale(attack=40, decay=8)
        sc.stimulus(10 ** 6)
        self.assertEqual(sc.tick(), 32,
                         "за тик шкала РАСТЁТ на attack-decay, а не на attack")

    def test_window_of_the_story(self):
        ok = [n for n in range(1, 25)
              if build_cat_canary_simulator(between_ticks=n).why_acted(
                  "кошка", "съесть").ok]
        self.assertEqual((ok[0], ok[-1]), (8, 12),
                         "окно спуска измерено 8..12; RIPEN_TICKS=8 — КРАЙ окна, не середина")

    def test_ripen_comment_carries_the_measured_window(self):
        """
        Комментарий у RIPEN_TICKS обязан нести ИЗМЕРЕННОЕ окно, а не фразу.

        Запрет слов — не цензура: прежний текст утверждал «8 — середина окна»,
        тогда как замер даёт окно 8..12 и середину 10. Тест требует, чтобы
        число окна стояло в комментарии, и запрещает именно ложное
        утверждение, оставляя право на честную формулировку.
        """
        src = open(os.path.join(ROOT, "brain", "social_sim.py"),
                   encoding="utf-8").read()
        blk = re.search(r"RIPEN_TICKS = \d+", src)
        self.assertTrue(blk)
        head = src[max(0, blk.start() - 900):blk.start()]
        self.assertNotIn("8 — середина окна", head,
                         "комментарий врёт: середина измеренного окна 10, а не 8")
        self.assertIn("8..12", head,
                      "в комментарии должно стоять измеренное окно спуска 8..12, "
                      "иначе следующий читатель снова будет угадывать")

    def test_closeness_cliff_is_measured(self):
        """
        Решает ПРОИЗВЕДЕНИЕ заряд x близость, а НЕ близость сама по себе.

        На этом утверждении я сама чуть не соврала в комментарии (написала
        «при близости 1.0 рассказа не будет», а контрольный замер дал: заряд
        294 при близости 1.0 проходит). Поэтому заперто и произведение, и
        Cliff близости отдельно.
        """
        def peak(charge, closeness):
            e = EmotionSystem(attack=40, decay=8)
            e.hurt(charge, closeness=closeness)
            return max(e.tick()["bad"] for _ in range(60))
        # одинаковое применённое число -> одинаковый пик
        self.assertEqual(peak(200, 1.5), peak(300, 1.0),
                         "шкала обязана знать ТОЛЬКО произведение заряда и близости")
        # 294 — это ЗАРЯД (F/M3), 230 — ПОРОГ ШКАЛЫ (агрессия). Не путать:
        # сравнение пика с 294 было бы сравниванием числа удара со шкалой.
        self.assertGreaterEqual(peak(294, 1.0), AGGRESSION_THRESHOLD,
                                "заряд 294 при близости 1.0 ДОСТАТОЧЕН — моя "
                                "фраза про «чужого» была ложью")
        self.assertLess(peak(293, 1.0), AGGRESSION_THRESHOLD,
                        "293 уже мало: cliff измерен, а не угадан")

        def fires(c):
            return build_cat_canary_simulator(
                closeness=c, between_ticks=RIPEN_TICKS).why_acted(
                    "кошка", "съесть").ok
        self.assertFalse(fires(1.25))
        self.assertTrue(fires(1.5))
        self.assertGreater(CLOSENESS_LIVING_TOGETHER, 1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
