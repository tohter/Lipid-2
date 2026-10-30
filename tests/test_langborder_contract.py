"""
Контракт границы языка (очередь, IR-блок, шаг A0).

ЗАКОН:

    текст -> lang/xx -> IR-НА-ВХОД -> core -> IR-С-ВЫХОДА -> lang/xx -> текст

core обязан знать ТОЛЬКО IR. Русский язык для него — чужие байты.

ПОЧЕМУ ТРЕЩОТКА, А НЕ «ПИЛАН ПОСЛЕ REFACTORINGA». Список нарушений в
документе умирает молча: правка добавляет импорт, и это становится видно
только когда ядро пытаются переставить на другой язык. Здесь рост любого
измеренного числа — падение теста, то есть нарушение границы замечается в
тот же день, когда сделано.

ПОТОЛКИ ВЗЯТЫ ИЗ ЗАМЕРА, А НЕ ИЗ ГОЛОВЫ: experiments/IR_BASELINE.json,
полученный этим же инструментом (tools/langborder.py). Каждый потолок —
«столько есть сегодня»; УМЕНЬШИТЬ — улучшение, ТЕСТ ПРИКАЗЫВАЕТ ПОНИЗИТЬ
ПОТОЛОК, чтобы откат был невозможен. РАСТИТЬ — отдельное решение хозяина,
поэтому тест говорит, куда смотреть, а не просто краснеет.

ЧЕГО ЗАПРЕЩЕНО НЕ ЗАМЕЧАТЬ (это и есть содержание A0, а не формулировка):
  * новый импорт языка в core/            -> падение
  * новый публичный метод, отдающий русскую строку из ядра -> падение
  * русские ДАННЫЕ в нейтральном core/    -> падение
  * print с кириллицей, демо-строка __main__, лишняя строка лога -> НЕ падение.
    Последнее важно не меньше первых трёх: трещотка, которая орёт на
    косметику, через месяц начинает обходиться, и тогда она не защищает
    ничего (урок п.21 про срез без числа).
"""
import ast
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.langborder import (LANGUAGE_MODULES, measure, py_files,
                              price_of_ir_output, is_language_file,
                              classify_ruslits)

# ── ПОТОЛКИ: сняты experiments/IR_BASELINE.json 26.10 ─────────────────
CEILINGS = {
    # C = 0 НАВСЕГДА, это не «сегодня ноль», а закон: через lang/ имеют право
    # ходить brain/ и ui/, ядро — только через IR. Пока язык лежал внутри
    # core/, импорт ловился категорией S; после переезда (A1/A2) он стал бы
    # невидим, и закон остался бы только на бумаге.
    "C_core_reaches_lang": 0,
    "D_debt_files": 3,          # core/extract.py, core/morph.py, core/lemma.py
    "S_seam_sites": 8,          # импортов языка не-языковым кодом. Было 12:
                                # A1 убрал шов brain/ir.py -> pymorphy3,
                                # A2.1 убрал core/graph.py -> core/lemma
                                # (резолвер форм теперь ПРИХОДИТ в ядро извне),
                                # A2.2 убрал core/records.py -> pymorphy3
                                # (лемматизатор приносит lang/ru/lemma.py),
                                # A2.3 убрал core/actions.py -> core.morph
                                # (часть речи приносит та же точка установки).
    "S_seams_inside_core": 2,   # из них ВНУТРИ ядра — это нарушения закона.
                                # Было 5: A2.1 закрыл graph.py, A2.2 закрыл
                                # records.py, A2.3 закрыл actions.py; остался
                                # один cortex (x2).
    "O_output_methods": 6,      # публичных методов, отдающих русскую строку
    "R_core_ruslits": 163,      # русских ДАННЫХ в нейтральном core/
                                # Было 169: до IR A2.4b предупреждения в
                                # stderr (`_complain`) считались ДАННЫМИ
                                # ядра. Понижение куплено мерой, а не дырой
                                # — см. test_sink_rule_... ниже.
}

BASELINE = os.path.join(ROOT, "experiments", "IR_BASELINE.json")

# Справочные величины: они НЕ запреты, но обязаны быть пришпилены к снимку.
# Число без потолка и без снимка — это число, которое может уехать и молчать.
REFERENCE_KEYS = (
    "M_core_messages",       # print/log/raise на русском
    "X_core_demo",           # строки __main__-самопроверок
    "S_demo_seam_sites",     # швы внутри демо
    "S_demo_seams_inside_core",
    "U_outer_seam_sites",    # швы в tools/ и teachers/
    "U_outer_seam_files",
    "G_lang_import_sites",   # не-языковой код, импортирующий lang/
    "G_lang_import_files",
    "Z_language_ruslits",    # кириллица в файлах-языках
    "P_ir_output_price",     # цена шага A4
    "P_ir_output_files",
)


class TestLawIsMeasured(unittest.TestCase):
    """
    Закон границы обязан быть ИЗМЕРИМ, иначе это декларация. Отдельно
    проверяется, что сам измеритель видит известные нарушения: зелёный тест
    на инструменте, который ничего не находит, — это ноль информации.
    """

    @classmethod
    def setUpClass(cls):
        cls.rep = measure()
        cls.tot = cls.rep["totals"]

    def test_measurer_finds_the_known_debt(self):
        """core/extract.py обязан считаться языковым долгом — иначе мера слепая."""
        self.assertIn("core/extract.py", self.rep["language_files"],
                      "измеритель не считает core/extract.py языком: он "
                      "объявляет границу там, где её нет")

    def test_measurer_finds_core_seams(self):
        """
        Ядровые швы, ЕЩЁ НЕ ЗАКРЫТЫЕ, обязаны быть вмерены — именно их закрывает
        A2. core/graph.py убран на A2.1, core/records.py на A2.2, core/actions.py
        на A2.3: швов там больше нет, и требовать их впредь значило бы запретить
        улучшение. Вместо требования «шить навсегда» — симметричная проверка
        ниже: закрытый шов обязан ИСЧЕЗНУТЬ из меры, а не прятаться в нуле.
        """
        inside = {f: v for f, v in self.rep["seams"].items()
                  if f.startswith("core" + os.sep)}
        for f in ("core/cortex.py",):
            self.assertIn(f, inside,
                          f"шва {f} в мере не видно — закон нечем проверять")

    def test_closed_seam_actually_disappears_from_measure(self):
        """A2.1: у core/graph.py больше нет языкового импорта. Если шов
        вернётся — упадёт эта проверка, а не «потолок»: цена молчания в том,
        что закрытый шов нельзя потерять из виду."""
        inside = {f: v for f, v in self.rep["seams"].items()
                  if f.startswith("core" + os.sep)}
        self.assertNotIn("core/graph.py", inside,
                         "у ядра снова языковый импорт в graph.py — A2.1 откатился")
        self.assertNotIn("core/records.py", inside,
                         "у ядра снова языковый импорт в records.py — A2.2 "
                         "откатился: лемматизатор обязан приходить из lang/")
        self.assertNotIn("core/actions.py", inside,
                         "у ядра снова языковый импорт в actions.py — A2.3 "
                         "откатился: часть речи обязана приходить из lang/")
        demo = [h for hits in inside.values() for h in hits if h.get("demo")]
        self.assertEqual(demo, [],
                         "демо-шов внутри core затерялся в общей мере — он обязан "
                         "быть виден отдельным числом S_demo_seam_sites")

    def test_measurer_finds_the_output_hole(self):
        """brain/lipid2_brain.py process_input() возвращает человеку строку."""
        hits = self.rep["output_methods"].get("brain/lipid2_brain.py", [])
        self.assertTrue(any(h["method"] == "process_input" for h in hits),
                        "выходная дыра не измерена — значит её можно "
                        "не лечить и считать границу закрытой")

    def test_ratchet_ceiling_equals_measured_today(self):
        """
        Потолок обязан РАВНЯТЬСЯ сегодняшнему замеру, а не «запасу на потом».
        Потолок больше замера = разрешение добавить нарушение, не нарушив
        тест, то есть трещотка с заранее прослабленным местом.
        """
        for key, cap in CEILINGS.items():
            self.assertEqual(self.tot[key], cap,
                             f"{key}: замерено {self.tot[key]}, а потолок {cap}. "
                             f"Если стало ЛУЧШЕ — понизь потолок в этом файле "
                             f"(и это единственное разрешённое изменение). Если "
                             f"стало ХУЖЕ — рост границы требует решения хозяина, "
                             f"а не правки числа.")

    def test_no_ceiling_key_is_invented(self):
        """Обратная сторона: нельзя тихо удалить категорию из замка."""
        for key in CEILINGS:
            self.assertIn(key, self.tot,
                          f"категории {key} нет в замере — её нельзя и ограничивать")


class TestCeilingsHold(unittest.TestCase):
    """Главная проверка: язык не имеет права проникать в ядро дальше."""

    @classmethod
    def setUpClass(cls):
        cls.tot = measure()["totals"]

    def test_russian_does_not_grow_in_core(self):
        bad = {k: (self.tot[k], cap) for k, cap in CEILINGS.items()
               if self.tot[k] > cap}
        self.assertEqual(bad, {},
                         "ГРАНИЦА НАРУШЕНА: значение (замер, потолок) — "
                         f"{bad}. См. experiments/IR_BOUNDARY.md §5; правка "
                         f"числа без решения хозяина запрещена.")

    def test_core_does_not_import_language(self):
        """
        Каждый ядровый шов — отдельная строчка отчёта, а не размытое число.
        A2 обязан довести это до нуля; сейчас это ИЗМЕРЕННЫЙ долг, и он заперт
        от роста (5 мест).
        """
        rep = measure()
        inside = {f: len(v) for f, v in rep["seams"].items()
                  if f.startswith("core" + os.sep)}
        self.assertLessEqual(sum(inside.values()),
                             CEILINGS["S_seams_inside_core"],
                             f"ядро тянет язык здесь: {inside}")

    def test_core_never_reaches_into_lang(self):
        """
        Отдельная проверка именно для C, потому что её нельзя «понизить»
        вместе с прогрессом: единственный допустимый результат — ноль.
        """
        rep = measure()
        self.assertEqual(rep["totals"]["C_core_reaches_lang"], 0,
                         f"ядро импортирует lang/ руками: "
                         f"{rep['core_reaches_lang']}. Это тот же импорт языка, "
                         f"что и S, просто ставший невидимым после переезда "
                         f"языка в lang/. Закон: ядро -> только через IR.")

    def test_language_dir_is_actually_measured(self):
        """
        Обратная сторона: lang/ обязан попадать в обход, иначе его файлы выпадают из меры и любой импорт из них не заметен.
        """
        rep = measure()
        live = {f.replace(os.sep, "/") for f in rep["lang_files"]}
        self.assertIn("lang/ru/questions.py", live,
                      "lang/ не сканируется — категория C мерит пустоту")

    def test_language_files_stay_the_only_language_in_core(self):
        """
        Новый языковой модуль в core/ возможен ТОЛЬКО осознанно: он обязан
        появиться в явном списке LANGUAGE_MODULES, иначе он просто не будет
        измеряться. Список — узкое горло, и оно проверяется здесь.
        """
        rep = measure()
        self.assertEqual(len(rep["language_files"]), CEILINGS["D_debt_files"],
                         "список языка внутри core изменился: "
                         f"{rep['language_files']}")


class TestPriceIsKnown(unittest.TestCase):
    """
    Цена шага A4 обязана быть известна ДО его начала. Это не ограничение, а
    защита от сочинённого обещания «осталось дописать lang/».
    """

    def test_price_of_ir_output_is_measured_not_recalled(self):
        checked, files, msg_only = price_of_ir_output()
        self.assertGreater(checked, 0, "цена не измерена")
        # Число записано в IR_BOUNDARY.md. Если оно поехало — документ врёт,
        # и знать об этом надо сейчас, а не на чтении документа.
        # 553 -> 554 на шаге A2.1: добавлен тест test_core_itself_recognizes_
        # forms, он СРАВНИВАЕТ РУССКУЮ СТРОКУ с выводом ядра, то есть честно
        # входит в цену A4. 554 -> 561 на шаге A2.2: добавлен
        # tests/test_lemma_service_contract.py, и его проверки тоже сравнивают
        # русские слова с ответом системы («кошки» -> «кошка») — он покупает
        # ровно тот же билет, что и остальные, а не идёт зайцем. 561 -> 564 на
        # шаге A2.3: в том же файле добавлены проверки части речи («чинить»
        # обязано остаться глаголом, «пчела» — нет), они тоже сравнивают
        # русское слово с ответом системы.
        # Числа не подгоняются: сначала замер, потом цифра.
        self.assertEqual((checked, files), (564, 38),
                         f"цена A4 = ({checked}, {files}), а в IR_BOUNDARY.md "
                         f"написано (564, 38) — обнови документ СЕЙЧАС")
        self.assertGreater(msg_only, 0,
                           "проверки только-с-сообщением перестали считаться: "
                           "цена будет завышена")

    def test_price_not_double_counted_with_message_strings(self):
        """Сравниваемое и сообщение — РАЗНЫЕ деньги; свалить их = завысить цену."""
        checked, _, msg_only = price_of_ir_output()
        self.assertNotEqual(checked + msg_only, checked,
                            "сообщения об ошибках попали в цену A4")


class TestRatchetDoesNotCryOnCosmetics(unittest.TestCase):
    """
    Трещотка, запрещающая ЛЮБУЮ кириллицу в core/, бесполезна: лог и тексты
    ошибок добавляют каждый день, и запрет начнут обходить. Проверяется, что
    категории M (сообщения) и X (демо) измерены ОТДЕЛЬНО и в потолок R не
    входят — иначе A2 придётся спорить с логами вместо границы.
    """

    @classmethod
    def setUpClass(cls):
        cls.rep = measure()

    def test_messages_are_counted_separately(self):
        tot = self.rep["totals"]
        self.assertIn("M_core_messages", tot)
        self.assertIn("X_core_demo", tot)
        self.assertNotIn("M_core_messages", CEILINGS,
                         "сообщения humans-ошибок закрыты потолком — трещотка "
                         "лечит косметику, а не границу")
        self.assertNotIn("X_core_demo", CEILINGS)

    def test_categories_cover_every_found_literal(self):
        """
        ДАННЫЕ + СООБЩЕНИЯ + ДЕМО обязаны равняться ВСЕМУ числу русских
        литералов в нейтральном core/. Иначе часть строк не попала ни в одну
        категорию — а неучтённая строка это ровно та дыра, из-за которой
        трещотка зелёная при настоящем нарушении (урок 24).
        """
        import ast as _ast
        from tools.langborder import parse, docstring_ids, RU
        rep = self.rep
        total_found = 0
        for rel in py_files("core"):
            if is_language_file(rel):
                continue
            tree = parse(rel)
            docs = docstring_ids(tree)
            total_found += sum(
                1 for n in _ast.walk(tree)
                if isinstance(n, _ast.Constant) and isinstance(n.value, str)
                and RU.search(n.value) and id(n) not in docs)
        bucketed = (sum(rep["core_ruslits"].values())
                    + sum(rep["core_messages"].values())
                    + sum(rep["core_demo"].values()))
        self.assertEqual(bucketed, total_found,
                         f"категории покрывают {bucketed} из {total_found} "
                         f"литералов: {total_found - bucketed} строк не учтены "
                         f"ни в одной категории — трещотка их не видит")
        self.assertNotIn("M_core_messages", CEILINGS)
        self.assertNotIn("X_core_demo", CEILINGS)


class TestMutationsAreCaught(unittest.TestCase):
    """
    Замок обязан ловить НАСТОЯЩЕЕ нарушение, а не только перестановку чисел.
    Проверяется на временной копии реального файла: подделать границу в
    памяти нельзя, а трогать рабочий код тест не имеет права.
    """

    def _patched_measure(self, rel, anchor, insert):
        """
        Замер с подменённым содержимым одного файла (в памяти, не на диске).

        СТРАЖ ОБЯЗАТЕЛЕН: первая версия мутаций подставляла код по якорю
        ROLE_TO_NODETYPE, которого в core/graph.py НЕТ. str.replace молча
        ничего не нашёл, исходник не изменился, и две мутации прошли
        зелёными — ровно урок 23 (эквивалентная мутация ничего не доказывает),
        повторённый мной же второй раз за день. Поэтому мутация обязана:
        (а) найти якорь, (б) сделать файл длиннее.
        """
        import tools.langborder as lb
        with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn(anchor, src,
                      f"якорь {anchor!r} не найден в {rel}: мутация стала "
                      f"ЭКВИВАЛЕНТНОЙ и ничего не проверяет")
        fake = src.replace(anchor, insert, 1)
        self.assertNotEqual(fake, src, "мутация не изменила исходник")
        self.assertGreater(len(fake), len(src),
                           "мутация укоротила файл — она не добавляет нарушение")

        real_parse = lb.parse

        def parse2(r):
            if r == rel:
                return ast.parse(fake, filename=r)
            with open(os.path.join(ROOT, r), encoding="utf-8") as fh:
                return ast.parse(fh.read(), filename=r)
        lb.parse = parse2
        try:
            rep = lb.measure()
        finally:
            lb.parse = real_parse
        # подделка обязана остаться валидным Python, иначе упёрлась бы в
        # SyntaxError, а не в нарушение границы
        ast.parse(fake, filename=rel)
        return rep

    # ядро модуля, куда безопасно вставлять код уровня модуля
    TAIL = 'if __name__ == "__main__":'

    # ЯКОРЬ НЕЛЬЗЯ брать из __main__: демо-самопроверка обязана кормиться
    # русским текстом, и импорт, вставленный в демо, нарушением библиотечного
    # пути не считается. Первый вариант мутации вставлял импорт в хвост файла
    # — то есть ровно в демо — и перестал ловить настоящее нарушение после
    # того, как A2.1 ввёл демо-разделение. Мутант обязан бить в рабочий путь.
    LIB = "from core.storage import StorageManager, NodeType, RelationType"

    def test_new_language_import_in_core_is_caught(self):
        rep = self._patched_measure(
            "core/graph.py", self.LIB,
            "from core.morph import Morph  # нарушение границы\n" + self.LIB)
        self.assertGreater(rep["totals"]["S_seams_inside_core"],
                           CEILINGS["S_seams_inside_core"],
                           "ядро получило новый импорт языка, а замер его не "
                           "увидел — закон нечем проверять")

    def test_measure_C_CAN_fire_not_only_return_zero(self):
        """
        Ноль нарушения — НЕ доказательство, что проверка работает.

        Этот замок появился из найденного дефекта, а не из осторожности. В
        core_reaches_lang попала ссылка на переменную spans, которой у функции
        нет: пока ядро честно НЕ импортирует lang/, функция не доходила до
        строки и C показывал правильный 0. Стоило нарушению появиться — NameError
        вместо поимки. Закон C = 0 на пустоте неотличим от закона C = 0,
        который сломался. Поэтому мутант ОБЯЗАН: (а) не уронить measure,
        (б) дать ровно C = 1, (в) не стать при этом швом категории S.
        """
        rep = self._patched_measure(
            "core/graph.py", self.LIB,
            "from lang.ru.questions import IRParser  # нарушение границы\n"
            + self.LIB)
        self.assertEqual(rep["totals"]["C_core_reaches_lang"], 1,
                         "ядро импортирует lang/, а мера C это не увидела "
                         "(или упала) — закон C = 0 нечем проверять")
        self.assertEqual(rep["totals"]["S_seams_inside_core"],
                         CEILINGS["S_seams_inside_core"],
                         "импорт lang/ посчитан швом S: категории наложились, "
                         "улучшение S можно спрятать за ростом C")

    def test_measure_G_SEES_lang_import_that_category_S_is_blind_to(self):
        """
        G обязана ловить то, чего S не видит ФИЗИЧЕСКИ.

        Ноль G неотличим от сломанной меры (тот же урок, что C: пока ядро
        честно не импортирует lang/, битая функция молчала и показывала
        правильный 0). Поэтому мутация, а не вера в код.

        Якорь выбран НЕ случайно: brain/question.py импортирует core.morph, то
        есть ЕЩЁ год назад его шов был виден категории S. Если теперь тот же
        файл импортирует lang.ru.lemma, S остаётся прежним — имя lang.ru.* не
        входит в LANGUAGE_MODULES. Без G работа просто исчезает из отчёта.
        """
        anchor = "from core.morph import analyze, split_sentences"
        base = measure()["totals"]
        rep = self._patched_measure(
            "brain/question.py", anchor,
            anchor + "\nfrom lang.ru.lemma import install  # переезд шва\n")
        self.assertEqual(rep["totals"]["S_seam_sites"],
                         base["S_seam_sites"],
                         "S перестала быть слепой к lang/ — значит список "
                         "LANGUAGE_MODULES тайно расширили и старые проверки "
                         "стали другими по смыслу")
        self.assertEqual(rep["totals"]["G_lang_import_sites"],
                         base["G_lang_import_sites"] + 1,
                         "шов уехал в lang/ и исчез из меры — ровно та дыра, "
                         "которую A1 оставил бы, не придумай он категорию C")
        self.assertEqual(rep["totals"]["G_lang_import_files"],
                         base["G_lang_import_files"] + 1,
                         "новый файл-носитель языка не проявился числом")

    def test_language_import_in_demo_is_counted_but_not_a_seam(self):
        """ОБРАТНАЯ мутация (демо), и она важна не меньше: импорт языка внутри
        `if __name__ == "__main__"` не попадает в библиотечный путь, поэтому
        швом НЕ считается — но обязан всплыть числом S_demo_seam_sites. Без
        этой пары демо-блок становится лазейкой: перенёс импорт в самопроверку
        и «очистил» ядро. С этой парой перенос виден как минимум в мере."""
        rep = self._patched_measure(
            "core/graph.py", self.TAIL,
            self.TAIL + "\n    from core.morph import Morph")
        self.assertEqual(rep["totals"]["S_seams_inside_core"],
                         CEILINGS["S_seams_inside_core"],
                         "демо-импорт посчитали швом — ядру нельзя писать "
                         "самопроверку вообще")
        self.assertEqual(rep["totals"]["S_demo_seams_inside_core"], 1,
                         "демо-шов внутри ядра пропал из меры — ноль, "
                         "дорисованный демо-блоком, это не чистота")

    def test_new_ru_returning_method_is_caught(self):
        rep = self._patched_measure(
            "core/graph.py", "class GraphMemory:",
            "class GraphMemory:\n"
            "    def answer(self) -> str:\n"
            '        return "кошка пушистая"\n')
        self.assertGreater(rep["totals"]["O_output_methods"],
                           CEILINGS["O_output_methods"],
                           "ядро начало возвращать русскую строку, а замер молчит")

    def test_new_russian_data_in_neutral_core_is_caught(self):
        rep = self._patched_measure(
            "core/graph.py", self.TAIL,
            'RU_SUFX = ("ся", "сь")  # нарушение: решение по окончанию в ядре\n'
            + self.TAIL)
        self.assertGreater(rep["totals"]["R_core_ruslits"],
                           CEILINGS["R_core_ruslits"],
                           "русские данные легли в нейтральное ядро незамеченными")

    def test_print_with_russian_is_NOT_a_violation(self):
        """
        Обратная мутация. Если бы лог считался нарушением, запретили бы
        логирование — и через месяц трещотку обходили бы по умолчанию.
        """
        rep = self._patched_measure(
            "core/graph.py", self.TAIL,
            'print("логи на русском — это не нарушение границы")\n'
            + self.TAIL)
        self.assertEqual(rep["totals"]["R_core_ruslits"],
                         CEILINGS["R_core_ruslits"],
                         "print с кириллицей засчитан ДАННЫМИ ядра — трещотка "
                         "орёт на косметику")
        self.assertGreater(rep["totals"]["M_core_messages"], 0,
                           "сообщение вообще не попало в категорию M")

    def test_stderr_warning_is_NOT_a_violation(self):
        """
        Кричалка в stderr — СООБЩЕНИЕ, а не данные ядра.

        Это не косметика меры: до неё предупреждения `_complain` из
        core/transitions.py и core/worldfile.py числились русскими ДАННЫМИ,
        то есть нарушением границы закона, которого в них нет — текст уходит в
        stderr и ни на одно решение не влияет. Замерено: 6 литералов в трёх
        файлах ядра (R 169 -> 163, M 105 -> 111).
        """
        rep = self._patched_measure(
            "core/graph.py", self.LIB,
            "def _cry(text):\n    import sys\n"
            "    sys.stderr.write(text)\n"
            '_cry("летяга летит")  # это сообщение, не данные\n' + self.LIB)
        self.assertEqual(rep["totals"]["R_core_ruslits"],
                         CEILINGS["R_core_ruslits"],
                         "предупреждение в stderr засчитано ДАННЫМИ ядра")
        self.assertGreater(rep["totals"]["M_core_messages"], 0)

    def test_sink_rule_cannot_hide_data_by_returning_it(self):
        """
        Потолок R понижен (169 -> 163), и обязан быть доказан обратный мутант:
        понижение куплено ЧЕСТНОЙ классификацией, а не новой лазейкой.

        Приёмником считается только функция, которая строку не забирает: без
        return и без присваивания. Если кто-то напишет «приёмник», который
        возвращает или сохраняет переданное, это уже не лог, а хранение русских
        данных в ядре — и обязано остаться нарушением.
        """
        rep = self._patched_measure(
            "core/graph.py", self.LIB,
            "def _keep(text):\n    import sys\n"
            "    sys.stderr.write(text)\n    return text\n"
            '_KEPT = _keep("летяга летит")  # нарушение: строку вернули\n'
            + self.LIB)
        self.assertGreater(rep["totals"]["R_core_ruslits"],
                           CEILINGS["R_core_ruslits"],
                           "функция, ВЕРНУВШАЯ русскую строку, прошла как лог: "
                           "правило приёмников стало дырой")

    def test_sink_rule_cannot_hide_data_by_storing_it(self):
        """То же для сохранённой строки: global-присваивание = данные ядра."""
        rep = self._patched_measure(
            "core/graph.py", self.LIB,
            "_LAST = None\n\n\ndef _stash(text):\n    import sys\n"
            "    global _LAST\n    sys.stderr.write(text)\n    _LAST = text\n"
            '_stash("летяга летит")  # нарушение: строку сохранили\n'
            + self.LIB)
        self.assertGreater(rep["totals"]["R_core_ruslits"],
                           CEILINGS["R_core_ruslits"],
                           "функция, СОХРАНИВШАЯ русскую строку, прошла как лог")

    def test_demo_literal_is_NOT_a_violation(self):
        """Строка внутри __main__-самопроверки — не данные ядра."""
        rep = self._patched_measure(
            "core/graph.py", self.TAIL,
            self.TAIL + '\n    DEMO_WORD = "кошка"  # демо-строка')
        self.assertEqual(rep["totals"]["R_core_ruslits"],
                         CEILINGS["R_core_ruslits"],
                         "демо-строка засчитана данными ядра: A2 пришлось бы "
                         "спорить с текстами самопроверок")
        self.assertGreater(rep["totals"]["X_core_demo"], 0)

    def test_language_module_itself_is_not_a_seam(self):
        """core/morph.py знать морфологию — не нарушение, а работа."""
        rep = measure()
        for lang_file in rep["language_files"]:
            self.assertNotIn(lang_file, rep["seams"],
                             f"{lang_file} записан себе в швы: ядро меряет "
                             f"само себя нарушителем")


class TestBaselineFileExists(unittest.TestCase):
    """Числа из документа обязаны иметь снимок, иначе их нечем подтвердить."""

    def test_snapshot_lies_next_to_the_law(self):
        self.assertTrue(os.path.exists(BASELINE),
                        "нет experiments/IR_BASELINE.json — цифры в "
                        "IR_BOUNDARY.md нечем проверить")

    def test_snapshot_agrees_with_live_measurement(self):
        import json
        with open(BASELINE, encoding="utf-8") as fh:
            snap = json.load(fh)
        live = measure()["totals"]
        for k in CEILINGS:
            self.assertEqual(snap["totals"][k], live[k],
                             f"{k}: снимок {snap['totals'][k]}, вживую "
                             f"{live[k]} — документ и код разошлись")

    def test_reference_numbers_are_pinned_too(self):
        """
        Числа БЕЗ потолка гниют МОЛЧА, и это хуже, чем быть незамеренными:
        незамеренное число нельзя сочинить, а протухшее — можно. Поэтому
        C demo/M/X/U сверяются со снимком ОБЯЗАНЫ, хоть потолка и не имеют.
        Рост такого числа не падает сам — его замечает эта проверка, и это
        единственная защита от «у нас же было 117».
        """
        import json
        with open(BASELINE, encoding="utf-8") as fh:
            snap = json.load(fh)
        live = measure()["totals"]
        for k in REFERENCE_KEYS:
            self.assertIn(k, live, f"категория {k} пропала из меры")
            self.assertEqual(snap["totals"].get(k), live[k],
                             f"{k}: снимок {snap['totals'].get(k)}, вживую "
                             f"{live[k]} — число без потолка уехало незаметно")
        # U обязана быть СПРАВОЧНОЙ: потолка у неё нет и быть не может, пока
        # инструмент имеет право знать язык (рост потолка — решение хозяина)
        for k in REFERENCE_KEYS:
            self.assertNotIn(k, CEILINGS,
                             f"{k} попала в трещотку: справочная величина "
                             f"стала запретом без решения хозяина")


class TestNothingIsWrittenToDisk(unittest.TestCase):
    """Замер границы читает код и ничего не пишет (правило проекта)."""

    def test_measure_has_no_side_effects(self):
        work = tempfile.mkdtemp()
        import shutil
        try:
            before = sorted(os.listdir(ROOT))
            measure()
            price_of_ir_output()
            self.assertEqual(sorted(os.listdir(ROOT)), before,
                             "измеритель оставил файлы в корне проекта")
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
