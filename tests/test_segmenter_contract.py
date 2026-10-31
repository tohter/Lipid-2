"""
test_segmenter_contract.py — замок IR A2.4: разбор текста приносит язык.

Закон границы (experiments/IR_BOUNDARY.md):

    текст -> lang/xx -> IR -> core -> IR -> lang/xx -> текст

core/cortex.py до A2.4 знал русский сам: резал текст на предложения, звал
морфологию, решал, какое слово служебное. Теперь он принимает `segmenter`
(текст -> цепочки лемм), а русская часть живёт в lang/ru/text.py.

Что здесь заперто и почему именно это:
  1. ПОВЕДЕНИЕ НЕ ИЗМЕНИЛОСЬ. Перенос обязан быть дословным; золотые цепочки
     сняты ДО переезда и зашиты в файл, а не сочинены по текущему выводу.
  2. ЗАБЫТЫЙ ВНЕДРИТЕЛЬ ПАДАЕТ, а не деградирует молча. Конструктор PredictiveCortex
     принимает segmenter=None — значит можно создать цикл и никогда не передать
     разбор; таких мест 25+ только в тестах.
  3. ДЕМО НЕ ВИРЯЕТ. Самопроверка переехала в tools/cortex_demo.py; её вывод
     сверяется с эталоном experiments/CORTEX_DEMO_GOLDEN.txt побайтно.
  4. ПРОИЗВОДСТВО НЕ ЗАБЫТО: мозг обязан принести разбор, иначе GUI учится вхолостую.
"""
from __future__ import annotations

import ast
import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stderr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("LIPID2_STATUS", "0")

from core.cortex import PredictiveCortex
from core.graph import GraphMemory
from lang.ru.text import experience as ru_experience

GOLDEN_CHAINS = os.path.join(ROOT, "experiments", "SEGMENTER_GOLDEN.json")
GOLDEN_DEMO = os.path.join(ROOT, "experiments", "CORTEX_DEMO_GOLDEN.txt")

# Тексты, на которых сверяется перенос. СПИСОК ЗАКРЫТ ДО ПЕРЕЕЗДА: среди них
# есть специально «подлые» места — связка «есть», форма «вечеру» (проходит
# фильтр analyze по форме, отсекается по лемме), придаточное с «которая»,
# модальные глаголы, пустой вход. Подбирать тексты под удобный результат —
# тот же класс лжи, что эталон, подогнанный под вывод.
TEXTS = [
    "Кошка сидит на подоконнике. Кошка пьёт молоко. Собака бегает во дворе."
    " Кошка пьёт молоко.",
    "У красивой кошки признаки. Какие признаки у бегущего человека?",
    "Он думает о погоде из-за давления, но гуляет без зонта.",
    "Ветер гасит пламя, потому что огонь боится воды.",
    "Пчела опыляет цветы; делает это вечером.",
    "5000 рублей стоит ремонт, и это недорого.",
    "Кошка есть животное. Кошка спит.",
    "Небо темнеет; идёт дождь, и ветер воет.",
    "Маша, которая читает книгу, любит лето.",
    "Он начал строить дом. Она хочет прочитать книгу.",
    "Вечеру делает холоднее, а делают это поздно.",
    "",
    "   ",
    "Слово.",
]


class TestTransferIsExact(unittest.TestCase):
    """Перенос кода из ядра в lang/ не имело права менять поведение."""

    def test_golden_file_exists(self):
        self.assertTrue(os.path.exists(GOLDEN_CHAINS),
                        f"нет {GOLDEN_CHAINS}: без эталона, снятого ДО переезда,"
                        " проверка 'поведение не изменилось' — слово, а не факт")

    def test_segmenter_matches_golden(self):
        with open(GOLDEN_CHAINS, encoding="utf-8") as fh:
            golden = json.load(fh)
        for i, text in enumerate(TEXTS):
            with self.subTest(i=i, text=text[:40]):
                self.assertEqual(ru_experience(text), golden[str(i)],
                                 "разбор уехал из ядра и начал думать иначе")

    def test_underhand_cases_are_actually_present(self):
        """
        Мутация самого ЭТАЛОНА. Если подлые случаи исчезнут из списка TEXTS
        (кто-то «почистит» тест), сверка превратится в украшение. Здесь
        проверяется, что ловушки на месте и ЭТАЛОН их помнит.
        """
        with open(GOLDEN_CHAINS, encoding="utf-8") as fh:
            golden = json.load(fh)
        # связка «есть»: «кошка есть животное» обязана дать ЛОЖНУЮ связку
        # кошка->животное — это измеренный дефект разбора, перенесённый как есть
        copula = [c for c in golden[str(6)] if "кошка" in c and "животное" in c]
        self.assertTrue(copula,
                        "эталон перестал фиксировать ложную связку «есть»: либо"
                        " разбор починили (и это отдельная работа с замером),"
                        " либо эталон подделали под вывод")
        # «вечеру»: форма значима по analyze, но ЛЕММА в стоп-листе. Проверка
        # обязана быть сквозной (по ВСЕМ цепочкам), а не «хоть где-то нет»:
        # any() по одному предложению прошла бы всегда и ничего не ловил.
        words = [w for chains in golden.values() for c in chains for w in c]
        # ЗАМЕРЕНО: у 'вечеру' лемма 'вечер' — она в стоп-листе, слово выбрасывается;
        # у 'вечером' лемма 'вечером' — её в стоп-листе нет, слово остаётся.
        # Разница не в смысле слов, а в том, что морфология дала разные леммы.
        self.assertNotIn("вечер", words,
                         "'вечеру' перестал теряться: фильтр по лемме больше не"
                         " работает либо стоп-лист изменился")
        self.assertIn("вечером", words,
                      "'вечером' тоже исчез — значит фильтр режет шире, чем"
                      " стоп-лист (это уже не перенос, а поломка)")
        self.assertIn("холодный", words,
                      "в той же подлой фразе значимое слово пропало")

    def test_core_itself_no_longer_contains_segmentation(self):
        """
        Ядро не обязано ИМЕТЬ копию разбора «на всякий случай»: вторая копия
        языка — это второй сорт шва, только без import.
        """
        with open(os.path.join(ROOT, "core", "cortex.py"),
                    encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src)
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.add(n.module)
        for banned in ("core.morph", "core.lemma", "core.extract", "pymorphy3",
                       "lang", "lang.ru", "lang.ru.text"):
            self.assertNotIn(banned, mods,
                             f"core/cortex.py снова знает язык: {banned}")


class TestMissingSegmenterIsLoud(unittest.TestCase):
    """
    Забытый внедритель обязан пасть ГРОМКО.

    Без разбора цикл не видит ни одного слова. Молча вернуть [] означало бы
    «в тексте нечего было брать» там, где «разбор не принесли», — отчёт при
    этом выглядит как честный ноль. Тот же урок, что с парсерскими заметками
    (класс молчания, коммит B).
    """

    def _run(self):
        g = GraphMemory()
        cx = PredictiveCortex(g, teacher=None)   # segmenter НЕ передан
        err = io.StringIO()
        with redirect_stderr(err):
            out = cx.experience_sentences("Кошка пьёт молоко.")
        return out, err.getvalue()

    def test_returns_nothing_but_screams(self):
        out, msg = self._run()
        self.assertEqual(out, [], "без разбора цикл обязан не выдумывать опыт")
        self.assertIn("БЕЗ СЕГМЕНТАТЕРА", msg,
                      "деградация молчит: ноль в отчёте неотличим от украденного"
                      " разбора")

    def test_screams_once_not_every_call(self):
        g = GraphMemory()
        cx = PredictiveCortex(g, teacher=None)
        err = io.StringIO()
        with redirect_stderr(err):
            for _ in range(5):
                cx.experience_sentences("Кошка пьёт молоко.")
        self.assertEqual(err.getvalue().count("БЕЗ СЕГМЕНТАТЕРА"), 1,
                         "предупреждение на каждый вызов заглушает сигнал")

    def test_installed_segmenter_is_silent(self):
        g = GraphMemory()
        cx = PredictiveCortex(g, teacher=None, segmenter=ru_experience)
        err = io.StringIO()
        with redirect_stderr(err):
            out = cx.experience_sentences("Кошка пьёт молоко.")
        self.assertEqual(err.getvalue(), "",
                         "при установленной службе цикл обязан молчать")
        self.assertEqual(out, [["кошка", "пить", "молоко"]])

    def test_segmenter_exception_does_not_kill_the_loop(self):
        """
        Сломанный разбор — не повод ронять обучение: цикл обязан кричать
        и пропускать фразу, а не уносить с собой GUI.
        """
        def broken(text):
            raise RuntimeError("морфология умерла")
        g = GraphMemory()
        cx = PredictiveCortex(g, teacher=None, segmenter=broken)
        err = io.StringIO()
        with redirect_stderr(err):
            m = cx.process_stream("Кошка пьёт молоко.", verbose=False)
        self.assertEqual(m["ticks"], 0,
                         "на сломанном разборе не должно быть ни тика")
        self.assertIn("сегментатор", err.getvalue().lower(),
                      "исключение сегментатора проглочено молча")


class TestProductionWiresSegmenter(unittest.TestCase):
    """Мозг — единственный производственный вход цикла; он обязан принести язык."""

    def test_brain_constructs_cortex_with_segmenter(self):
        with open(os.path.join(ROOT, "brain", "lipid2_brain.py"),
                    encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src)
        found = False
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                fn = n.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if name == "PredictiveCortex":
                    found = any(k.arg == "segmenter" for k in n.keywords)
        self.assertTrue(found,
                        "мозг создаёт цикл без segmenter: GUI молча учится вхолостую")

    def test_brain_actually_learns_chains(self):
        """
        Не AST-заклинание, а поведение: на реальном тексте мозг обязан получить
        тики цикла. Если сегментатор приехал, но сломан, — упадёт здесь.
        """
        from brain.lipid2_brain import Lipid2Brain
        b = Lipid2Brain()
        m = b.cortex.process_stream("Кошка пьёт молоко. Кошка спит.",
                                    verbose=False)
        self.assertGreaterEqual(m["ticks"], 3,
                                "цикл в мозге не увидел опыта — разбор не дошёл")


class TestDemoOutputIsPinned(unittest.TestCase):
    """
    Самопроверка переехала из ядра в tools/cortex_demo.py. Её вывод — часть
    договора «перенесли дословно», поэтому сверяется побайтно.
    """

    def test_demo_output_matches_golden(self):
        self.assertTrue(os.path.exists(GOLDEN_DEMO),
                        f"нет эталона {GOLDEN_DEMO}")
        env = dict(os.environ, LIPID2_STATUS="0")
        r = subprocess.run([sys.executable, "tools/cortex_demo.py"],
                           cwd=ROOT, env=env, capture_output=True, text=True,
                           timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        with open(GOLDEN_DEMO, encoding="utf-8") as fh:
            golden = fh.read()
        self.assertEqual(r.stdout, golden,
                         "самопроверка после переезда печатает другое — значит"
                         " перенос был не дословным")


if __name__ == "__main__":
    unittest.main()
