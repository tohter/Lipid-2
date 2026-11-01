"""
Контракт цифр витрины PRESENTATION.md (очередь, п. ПРЕЗ, открыт 31.10).

ЗАЧЕМ ЭТОТ ТЕСТ. Витрина была списана 23.10 «по 516 зелёным тестам» и с тех
пор чисел не видела. Замер 31.10 дал: тестов 748, коммитов 87, а в витрине
сидели 516, «503 в 25 файлах», «12 586 строк ядра», «27 292 всего», «83
файла», «50 коммитов» и «103 с тесты». Витрина продавала состояние
десятидневной давности. Это тот же класс вранья, из-за которого в проекте
стоит трещотка границы языка: «число, которого нет в законе, всегда можно
сочинить» (memory.txt, урок 28).

ПОЧЕМУ ТРЕЩОТКА, А НЕ РУЧНАЯ ПОДГОНКА. Поправить цифры глазами — значит
починить витрину ровно до следующего шага: править их было бы нечем. Здесь
каждое число блока «Цифры» обязано совпадать с живым замером
tools/presnum.py, а число БЕЗ измерителя запрещено: нечем померить — нечем
и написать.

ЧТО ЗАПРЕЩЕНО ЭТИМ ТЕСТОМ
  1. Расхождение блока «Цифры» с замером.
  2. Число в блоке, для которого нет измерителя в tools/presnum.py.
  3. Размер, повторённый в прозе: «516 тестов» стояло и в шапке, и в блоке,
     и устарели ОБЕ копии. Измеренное число живёт в одном месте.
  4. Правку блока руками: он обязан содержаться в витрине побайтно, как его
     печатает `tools/presnum.py --print`.

ПОЧЕМУ countTestCases, А НЕ ПРОГОН. Сборка suite стоит доли секунды, прогон
— 130 с. Шлюз не обязан удваивать себя ради сверки строк в markdown; «OK»
заперт остальными тестами.

РАБОЧЕСТЬ МЕРЫ ПРОВЕРЯЕТСЯ МУТАЦИЯМИ (урок A2.1b: «нарушения нет» неотличимо
от «проверка нарушения сломана»). Класс TestMeasureCanSee портит ВРЕМЕННУЮ
копию витрины и требует, чтобы check() покраснел. Реальный файл не меняется
ни байтом.
"""
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import presnum  # noqa: E402


class TestNumbersMatchMeasure(unittest.TestCase):

    def test_doc_has_number_block_that_parses(self):
        """Если блок не разбирается, весь этот файл — декорация: он не
        сверяет ничего и всегда зелёный. Проверка обязательна первым делом."""
        self.assertTrue(os.path.exists(presnum.DOC),
                        "нет PRESENTATION.md — сверять не с чем")
        got = presnum.doc_numbers()
        self.assertGreaterEqual(len(got), len(presnum.MEASURES),
                                f"блок «Цифры» разобрался пусто ({len(got)} "
                                f"строк против {len(presnum.MEASURES)} мер) — "
                                f"проверка ослепла")

    def test_no_stale_numbers_in_doc(self):
        bad, _ = presnum.check()
        self.assertEqual(bad, [],
                         "витрина расходится с замером:\n  " + "\n  ".join(bad)
                         + "\n\nПереснять: .venv/bin/python tools/presnum.py"
                         "\nПерегенерировать блок: .venv/bin/python "
                         "tools/presnum.py --print")

    def test_every_number_in_block_has_a_measure(self):
        """Отдельной строкой: «просто поправить цифру» без замера — это и
        есть сочинительство, ради которого затеян весь механизм."""
        orphan = sorted(set(presnum.doc_numbers()) - set(presnum.MEASURES))
        self.assertEqual(orphan, [],
                         f"числа в витрине без измерителя: {orphan}. Либо "
                         f"добавь меру в tools/presnum.py, либо убери число.")

    # Ноль как ЦЕЛЬ, а не как сломанная мера. Швов языка внутри ядра ноль
    # — это итог A2.4, запертый tests/test_langborder_contract.py. Витрина
    # обязана иметь право на честный ноль, но он обязан БЫТЬ НАЗВАН здесь:
    # случайный нуль («мера перестала видеть») тест не пропустит.
    ZERO_IS_GOAL = {"шов языка внутри ядра"}

    def test_measures_return_sensible_numbers(self):
        """Мера, вечно возвращающая 0, делает блок красивым и пустым."""
        for name, fn in presnum.MEASURES.items():
            with self.subTest(name):
                v = fn()
                self.assertIsInstance(v, int, f"{name}: не целое")
                if name in self.ZERO_IS_GOAL:
                    self.assertEqual(v, 0,
                                     f"{name}: стал {v}, а это запертый ноль "
                                     f"(граница языка) — смотри langborder")
                else:
                    self.assertGreater(v, 0, f"{name}: вернул {v} — мера "
                                             f"ослепла или дерево не там")

    def test_block_is_generated_not_handwritten(self):
        self.assertIn(presnum.render_block(), presnum.doc_text(),
                      "блок «Цифры» в витрине побайтно отличается от вывода "
                      "tools/presnum.py --print. Перегенерируй блок целиком, "
                      "а не правь одну цифру на глаз — так он и протух.")

    def test_relation_list_matches_registry(self):
        """Список отношений в витрине обязан СОВПАДАТЬ с реестром, а не только
        количеством. В витрине лежали 13 имён при 17 в реестре: числа 17 само
        по себе не мешает показывать устаревший список, поэтому сверяется сам
        перечень."""
        self.assertEqual(presnum.doc_relations(), presnum.render_relations(),
                         "список отношений в витрине расходится с "
                         "core.storage.RelationType (--fix)")

    # Писано в этой же задаче трижды: «иероглиф» в докстринге, CJK в
    # формулировке теста, китайская вставка в объяснении. Глазом не видно,
    # редактор не ругается, python компилирует. Поэтому проверка символьная.
    # Граница сознательная: experiments/ARCHIVE_SESSIONS.md содержит CJK
    # СОЗНАТЕЛЬНО (там иероглифы обсуждаются как пример мусора), и гонять
    # правило на весь проект нельзя — оно начинало бы краснеть на правду.
    # СОСТАВ РАСШИРЕН 01.11. Прежний список ловил только витрину и её замок,
    # а CJK уже лежал в experiments/corpus_heldout.json (фраза про контаминацию
    # held-out) и никто его не видел: замок проверял три файла, а текст,
    # который читает человек, в проекте не три. Правило то же: любой файл,
    # который человек читает как прозу, обязан быть чистым.
    OWNED = ("tools/presnum.py", "tests/test_presentation_numbers_contract.py",
             "PRESENTATION.md", "README_ЭКСПЕРТУ.md",
             "experiments/corpus_dev.json", "experiments/corpus_heldout.json",
             "experiments/CORPUS_QUALITY.md", "tests/test_role_category_contract.py",
             "tests/test_relcov_contract.py", "tools/relcov.py",
             "memory.txt")
    CJK = __import__("re").compile(
        "[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")

    def test_no_stray_cjk_in_showcase_and_lock(self):
        for rel in self.OWNED:
            self.assertTrue(os.path.exists(os.path.join(ROOT, rel)),
                            f"замок потерял покрытие: файла {rel} нет — "
                            "список OWNED протух, а зелёный цвет этого не скажет")
            path = os.path.join(ROOT, rel)
            with open(path, encoding="utf-8") as fh:
                for n, line in enumerate(fh, 1):
                    self.assertIsNone(
                        self.CJK.search(line),
                        f"{rel}:{n}: посторонняя CJK-строка в тексте, который "
                        f"читает человек: «{line.strip()[:80]}»")

    def test_markers_present(self):
        """Генерируемые куски ищутся по маркерам. Пропал маркер — проверка
        должна Краснеть, а не «зеленеть от невнимания»."""
        text = presnum.doc_text()
        for mark in (presnum.MARK_NUM, presnum.MARK_REL):
            self.assertIn(mark, text,
                          f"в витрине нет маркера {mark}: соответствующий "
                          f"блок перестал проверяться")

    def test_no_size_duplicated_in_prose(self):
        hits = presnum.stale_copies()
        self.assertEqual(hits, [],
                         "размер повторён в прозе вне блока: " +
                         "; ".join(f"«{line}»" for line, _ in hits))


class TestMeasureCanSee(unittest.TestCase):
    """Мутации обязаны краснеть. Без этого класса предыдущие шесть тестов —
    украшение: зелёный цвет при слепом парсере выглядит идентично."""

    def setUp(self):
        """Портим ВРЕМЕННУЮ копию, у которой блок чисел СВЕЖИЙ.

        Зачем сначала перегенерировать блок: витрина сегодня протухшая, и
        мутация «ври число тестов» падала бы не потому, что замок работает,
        а потому, что строки в старой витрине просто нет (ValueError в первом
        варианте этого файла). Мутация обязана проверять МЕХАНИЗМ, поэтому
        отправная точка — корректный документ, а краснеет он от самой мутации.
        """
        # Копия витрины, ПРИВЕДЁННАЯ К ЗАМЕРУ тем же tools/presnum.py --fix.
        # Зачем: на протухшем документе check() красен и без мутации, и
        # мутация «соври про число тестов» прошла бы бесплатно — тест
        # проверял бы чужую вину, а не свой механизм. Отправная точка обязана
        # быть зелёной, иначе краснеть от мутации не с чего.
        self.tmp = tempfile.mkdtemp(prefix="presnum_mutation_")
        self.doc = os.path.join(self.tmp, "PRESENTATION.md")
        shutil.copy(presnum.DOC, self.doc)
        self._real = presnum.DOC
        presnum.DOC = self.doc
        presnum.fix()
        bad, _ = presnum.check()
        self.assertEqual(bad, [],
                         "приведённая копия всё ещё красная: " + ";".join(bad)
                         + " — мутации проверяли бы не свой механизм")

    def tearDown(self):
        presnum.DOC = self._real
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _set_number(self, key, value):
        """Заменить число меры key на value, оставив остальное нетронутым."""
        with open(self.doc, encoding="utf-8") as fh:
            text = fh.read()
        needle = key + ":"
        i = text.index(needle)
        j = i + len(needle)
        while j < len(text) and (text[j].isdigit() or text[j] in " \u00a0"):
            j += 1
        with open(self.doc, "w", encoding="utf-8") as fh:
            fh.write(text[:i] + f"{needle} {value} " + text[j:])

    def _assert_red(self, what):
        bad, _ = presnum.check()
        self.assertTrue(bad, f"мутация «{what}» НЕ поймана — замок слеп, "
                             f"он не защитит витрину от следующего шага")
        return bad

    def test_lie_about_test_count_is_caught(self):
        self._set_number("тестов всего", presnum.n_tests() + 7)
        self.assertIn("тестов всего", " ".join(self._assert_red("тестов")))

    def test_lie_about_lines_in_core_is_caught(self):
        self._set_number("строк в core/", presnum.py_tree("core")[0] + 4000)
        self.assertIn("строк в core/", " ".join(self._assert_red("строк core")))

    def test_honest_number_is_still_green(self):
        """Отрицательный контроль: честная витрина обязана ОСТАВАТЬСЯ зелёной
        после всех манипуляций setUp. Без этого теста класс мутаций мог бы
        краснеть на всё подряд и выглядеть работающим."""
        bad, _ = presnum.check()
        self.assertEqual(bad, [])

    def test_lie_about_lines_is_caught(self):
        self._set_number("строк Python всего", 999_999)
        bad = self._assert_red("строк Python")
        self.assertIn("строк Python всего", " ".join(bad))

    def test_inventing_a_number_is_caught(self):
        """Новая строка числа без измерителя: правдоподобно и ничем не
        проверяемо — ровно то, что запрещено."""
        with open(self.doc, encoding="utf-8") as fh:
            text = fh.read()
        with open(self.doc, "w", encoding="utf-8") as fh:
            fh.write(text.replace("файлов Python всего:", "узлов в мире:", 1))
        bad = self._assert_red("сочинённое число")
        self.assertTrue(any("без измерителя" in b for b in bad),
                        f"сочинённое число поймано, но не как сочинённое: {bad}")

    def test_removing_a_number_is_caught(self):
        """Молчание витрины — тоже враньё: блок обязан быть полным."""
        with open(self.doc, encoding="utf-8") as fh:
            text = fh.read()
        lines = [l for l in text.splitlines()
                 if not l.strip().startswith("отношений в реестре:")]
        with open(self.doc, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        self._assert_red("пропущенное число")

    def test_duplicate_size_in_prose_is_caught(self):
        with open(self.doc, encoding="utf-8") as fh:
            text = fh.read()
        # Вставляем абзац прямо ПЕРЕД маркером блока, а не «перед заголовком
        # раздела»: разделы перенумеровываются, и мутация, ищущая несуществую-
        # щий заголовок, тихо превращается в тест, который ничего не делал.
        i = text.index(presnum.MARK_NUM)
        head = text[:i].rstrip()
        rest = text[i:]
        with open(self.doc, "w", encoding="utf-8") as fh:
            fh.write(head + f"\n\nУ нас {presnum.n_tests()} тестов, все "
                            f"зелёные.\n\n" + rest)
        self.assertTrue(presnum.stale_copies(),
                        "дубль размера в прозе не найден — правило "
                        "«число живёт в одном месте» не работает")

    def test_parser_blinding_is_caught(self):
        """Самый подлый случай: убрать маркер, и парсер перестанет что-либо
        видеть, а тесты останутся зелёными. Проверка обязана краснеть и на
        это."""
        with open(self.doc, encoding="utf-8") as fh:
            text = fh.read()
        with open(self.doc, "w", encoding="utf-8") as fh:
            fh.write(text.replace(presnum.MARK_NUM, "<!-- спрятал -->", 1))
        bad = self._assert_red("пропажа маркера")
        self.assertTrue(any("маркера" in b for b in bad),
                        f"ослепление парсера не названо маркером: {bad}")

    def test_stale_relation_list_is_caught(self):
        """Возврат в витрину старого списка из 13 отношений (без SYNONYM_OF,
        CAUSES, IS_ROLE, HAS_PART) обязан краснеть, хотя «17 отношений»
        в тексте останется."""
        with open(self.doc, encoding="utf-8") as fh:
            text = fh.read()
        old = "`IS_A / HAS_PROPERTY / CAN_DO / OBJECT_OF / LOCATED_IN / " \
              "PART_OF / INSTRUMENT / RECIPIENT / PURPOSE / ABOUT / " \
              "CONDITION / CIRCUMSTANCE / VALUE`"
        with open(self.doc, "w", encoding="utf-8") as fh:
            fh.write(text.replace(presnum.render_relations(), old, 1))
        bad = self._assert_red("устаревший список отношений")
        self.assertTrue(any("отношени" in b for b in bad),
                        f"подмена списка не поймана: {bad}")

    def test_number_in_header_is_caught(self):
        """Шапка с числом тестов — тот самый дубль, из-за которого витрина
        протухла целиком: правка блока не лечила шапку."""
        with open(self.doc, encoding="utf-8") as fh:
            text = fh.read()
        with open(self.doc, "w", encoding="utf-8") as fh:
            fh.write(text.replace(
                "*составлено", f"*составлено {presnum.today_ru()} "
                f"по {presnum.fmt(presnum.n_tests())} зелёным", 1))
        bad = self._assert_red("число в шапке")
        self.assertTrue(any("шапка" in b for b in bad),
                        f"число в шапке не пойман: {bad}")


if __name__ == "__main__":
    unittest.main()
