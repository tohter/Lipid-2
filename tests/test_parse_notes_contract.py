"""
Контракт «заметка разбора обязана дойти до отчёта» (очередь, пункт B).

ЧТО БЫЛО (замер 26.10, производственный путь Learner.learn_from_text по
золотому корпусу, 84 фразы):

    экстрактор оставляет sf.notes .................. 17 штук на 84 фразы
    до LearningReport доходит ........................ 0
    фраз, где заметка ПОТЕРЯНА ..................... 13  (15,5 %)

`sf.notes` в пакете brain/ не читается НИГДЕ (проверка: grep -rn "\\.notes"
brain/ core/ ui/ — в brain/ встречаются только rep.notes, это ДРУГОЕ поле).
Экстрактор честно пишет «придаточное „берега которой заросли ивняком" без
личного глагола — связи не записаны», «нет ни подлежащего (ИМ.п.), ни
сказуемого», «цена/мера есть, но субъект не найден» — и молчит для всех
потребителей: и для журнала, и для окна.

ПОЧЕМУ ЭТО ОПАСНО, а не косметика. Тот же класс дефекта, что E4 (сон убивал
переходы без единой строки в отчёте) и G-остаток. Разбор отказал — связь не
записана — а отчёт рапортует «фактов извлечено: 0» без причины. От «разбор
отказал» неотличимо от «в тексте нечего было брать». Молчание читается как
успех.

ЧТО ЗАПЕРАЕТСЯ (и чего сознательно НЕ запирается):

  1. заметка экстрактора доходит до отчёта НЕ ПОД ОДНИМ ИЗВЕСТНЫМ текстом;
  2. фраза, по которой НЕ записано НИЧЕГО, названа ВЯВО — это и есть «молча
     потеряно», и число её обязано быть в строке итога, а не только в списке
     (урок п.21: причины в notes, которые режутся срезом, не видно);
  3. заметки разбора НЕ вытесняют заметки про Учителя: у них РАЗНЫЕ причины и
     РАЗНОЕ лечение, свалить их в один список — значит вернуться к путанице
     «не нашли» vs «не спрашивали»;
  4. любой срез списка ОБЯЗАН объявить, сколько он скрыл. Прежний notes[:6]
     резал молча — именно из-за этого п.21 пришлось лечить отдельным
     счётчиком;
  5. повторный урок того же текста не плодит дублей (отчёт урока обязан быть
     идемпотентен по заметкам).

Цифры 17/13 в этот файл НЕ вписаны как ожидаемые: корпус правится, число
обязано плыть. Заперто поведение на конкретных фразах.
"""
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from brain.learner import Learner
from core.extract import SimpleExtractor
from core.worldfile import read_world, write_world

# Фразы, на которых экстрактор Guarantee оставляет заметку (замерено
# core/extract.py, см. таблицы ниже). Разных причин — три, намеренно: ловить
# проброс по одному тексту нельзя, замок обязан видеть канал, а не строку.
SENT_NO_SUBJECT = "Он думает о погоде."       # «нет ни подлежащего...»
SENT_MEASURE = "Слон весит пять тонн."         # «цена/мера записана как VALUE»
SENT_CLAUSE = ("Река, берега которой заросли ивняком, обмелела.")
# придаточное без личного глагола -> «связи не записаны»

ALL_THREE = [SENT_NO_SUBJECT, SENT_MEASURE, SENT_CLAUSE]


def make_learner(workdir):
    """Учитель ВЫКЛЮЧЕН: проверяется канал заметок, а не знание слов."""
    path = os.path.join(workdir, "w.json")
    return Learner(load_world=lambda: read_world(path),
                   save_world=lambda w: write_world(path, w),
                   core=None, teacher=None), path


class TestParseNotesReachReport(unittest.TestCase):
    """Заметка экстрактора обязана быть видна потребителю отчёта."""

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="lipid_notes_")
        cls.learner, cls.path = make_learner(cls.work)
        cls.ex = SimpleExtractor()
        cls.reports = {}
        for sent in ALL_THREE:
            cls.reports[sent] = cls.learner.learn_from_text(
                sent, source="text", allow_teacher=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def test_extractor_actually_leaves_a_note(self):
        """
        Условие задачи обязано быть проверено, а не предполагано: если
        экстрактор перестанет оставлять заметку, весь этот файл меряет
        пустоту и обязан упасть здесь, а не зелёным «проброс работает».
        """
        for sent in ALL_THREE:
            sf = self.ex.extract(sent)
            self.assertTrue(sf.notes,
                            f"«{sent}»: у экстрактора нет заметки — предпосылка "
                            f"задачи пропала, замок проверяет не то")

    def test_note_reaches_report_in_some_form(self):
        """
        Проверяется ФАКТ доставки, а не конкретное поле: где именно жить
        заметке (отдельный список или notes) — решение кода, а не контракт.
        Контракт: прочитав отчёт, человек видит ПРИЧИНУ, а не только «0 фактов».

        Сверка идёт по НАСТОЯЩЕМУ каналу — sf.notes данного предложения, — а
        не по «похожему слову из фразы»: stem-эвристика принимала отчёт, где
        причины нет, и отвергала верный (первая версия теста именно так и
        падала на «Он думает о погоде»: подлежащего в тексте заметки нет).
        """
        for sent in ALL_THREE:
            sf = self.ex.extract(sent)
            self.assertTrue(sf.notes, f"{sent}: предпосылка ушла")
            rep = self.reports[sent]
            blob = " ".join(list(getattr(rep, "parse_notes", []))
                            + list(rep.notes)) + rep.as_text()
            delivered = [n for n in sf.notes if n[:25] in blob]
            self.assertTrue(delivered,
                            f"«{sent}»: разбор отказал, а в отчёте НИ СЛОВА о "
                            f"причине — это и есть пункт B. ждалось {sf.notes}, "
                            f"в отчёте: {blob[:300]}")

    def test_reason_is_not_generic_zero(self):
        """
        Причина обязана быть РАЗНОЙ для трёх разных отказов. Если отчёт
        печатает одно и то же «разбор не удался» — это не причина, а глушение.
        """
        reasons = []
        for sent in ALL_THREE:
            rep = self.reports[sent]
            blob = " ".join(list(getattr(rep, "parse_notes", []))
                            + list(rep.notes)) + rep.as_text()
            reasons.append(blob)
        self.assertEqual(len({r[:400] for r in reasons}), 3,
                         "все три отказа описаны одинаково — причина не различается")

    def test_measure_sentence_reports_its_own_reason(self):
        """
        «Слон весит пять тонн»: самая безобидная фраза, где связь ВСЁ ЖЕ
        записана (VALUE), но экстрактор предупреждает о трактовке. Предупреж-
        дение теряет смысл, если до отчёта не доходит.
        """
        rep = self.reports[SENT_MEASURE]
        blob = " ".join(list(getattr(rep, "parse_notes", [])) + list(rep.notes))
        self.assertTrue("тонн" in blob or "цена" in blob or "мер" in blob,
                        f"предупреждение о мере пропало: {rep.notes}")

    def test_clause_note_names_the_clause(self):
        """
        «Река, берега которой заросли ивняком, обмелела» — заметка экстрактора
        называет придаточное. Обрывок придаточного обязан остаться в отчёте:
        без него «связь не записана» неотличимо от «связи и не было».
        """
        rep = self.reports[SENT_CLAUSE]
        blob = " ".join(list(getattr(rep, "parse_notes", [])) + list(rep.notes))
        self.assertIn("ивняк", blob,
                      f"придаточное не названо — нечего показывать человеку: {blob}")


class TestSilentSentencesAreNamed(unittest.TestCase):
    """
    Фраза, по которой не записано НИЧЕГО, обязана быть названа. Это ядро
    пункта B: «фактов 0» без перечня фраз = молчание, переодетое в отчёт.
    """

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="lipid_silent_")
        cls.learner, _ = make_learner(cls.work)
        cls.rep = cls.learner.learn_from_text(
            SENT_NO_SUBJECT + " " + SENT_CLAUSE, source="text",
            allow_teacher=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def test_silent_sentence_counted(self):
        """«Он думает о погоде» не даёт НИ ОДНОЙ связи — это обязано быть числом."""
        sf = SimpleExtractor().extract(SENT_NO_SUBJECT)
        self.assertEqual(sf.facts, [], "предпосылка ушла: фраза стала давать факты")
        self.assertEqual(sf.events, [])
        got = getattr(self.rep, "sentences_silent", None)
        self.assertIsNotNone(got,
                             "в отчёте нет счётчика фраз без единой связи: "
                             "молчаливые потери по-прежнему не измеримы")
        self.assertGreaterEqual(got, 1, f"молчаливая фраза не посчитана: {self.rep}")

    def test_silent_count_visible_in_summary_line(self):
        """
        Урок п.21: то, что лежит только в notes, не видно — notes режутся.
        Число молчаливых фраз обязано попасть в as_text (и, если потребитель
        показывает однострочный итог, в summarize_report тоже).
        """
        text = self.rep.as_text()
        rows = [ln for ln in text.splitlines()
                if ("молч" in ln.lower() or "без связи" in ln.lower()
                    or "без единой" in ln.lower())
                and any(ch.isdigit() for ch in ln)]
        self.assertTrue(rows,
                        f"в as_text нет СТРОКИ С ЧИСЛОМ о молчаливых фразах "
                        f"(одного упоминания мало — урок п.21):\n{text}")

    def test_speaker_who_produced_nothing_says_so(self):
        """Ни одной связи + ни одной заметки = полная тишина. Тишина запрещена."""
        rep = self.learner.learn_from_text(SENT_NO_SUBJECT, source="text",
                                           allow_teacher=False)
        produced = rep.observations_new + rep.events_new + rep.expectations_new
        said = " ".join(list(getattr(rep, "parse_notes", [])) + list(rep.notes))
        if produced == 0:
            self.assertTrue(said.strip(),
                            f"урок не дал НИЧЕГО и не объяснил почему: {rep.as_text()}")

    def test_status_bar_line_carries_the_silence(self):
        """
        Закон п.21 и урока 16: что не показано БЕЗ раскрытия журнала — того
        нет. Строка-итог статуса остаётся на экране, когда отчёт уехал в
        журнал и его никто не открыл; «0 фактов» без «молчало N фраз» читается
        как «в тексте нечего было брать».
        """
        from ui.presenter import summarize_report
        line = summarize_report(self.rep)
        self.assertIn("молчало", line,
                      f"в строке-итоге молчание разбора не видно: {line}")
        self.assertIn(str(self.rep.sentences_silent), line)


class TestParseNotesDoNotEvictTeacherNotes(unittest.TestCase):
    """
    Заметки разбора и заметки про Учителя — РАЗНЫЕ причины с РАЗНЫМ лечением.
    Свалить их в один список = вернуть путаницу «не нашли» vs «не спрашивали».
    """

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="lipid_mix_")
        cls.learner, _ = make_learner(cls.work)
        # Учитель выключен + неизвестное слово => учителевская заметка есть
        # ВСЕГДА, на неё и проверяем вытеснение.
        cls.rep = cls.learner.learn_from_text(
            SENT_NO_SUBJECT + " Флюгер крутится.", source="text",
            allow_teacher=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def test_teacher_reason_survives_parse_notes(self):
        teacherish = [n for n in self.rep.notes
                      if "Учител" in n or "не изучалось" in n or "НЕ СПРАШИВАЛСЯ" in n]
        self.assertTrue(teacherish,
                        f"причина по слову исчезла из отчёта: {self.rep.notes}")

    def test_parse_notes_live_separately(self):
        """
        Разделение полей — контракт, а не деталь реализации.

        ПЕРВАЯ ВЕРСИЯ БЫЛА СЛАБОЙ: она спрашивала hasattr(rep, "parse_notes"),
        а поле существует ПРИ ЛЮБОЙ правке — мутация «свалить разбор в notes»
        прошла ЗЕЛЁНОЙ. Настоящее свойство другое: заметка разбора НЕ обязана
        появляться в notes вообще. Иначе на длинном уроке она занимает место у
        причины по Учителю (notes режется [:6]) — и теряется ровно то, ради
        чего лечили п.21.
        """
        self.assertTrue(hasattr(self.rep, "parse_notes"))
        raw = [n for sf in SimpleExtractor().extract_text(
            SENT_NO_SUBJECT + " Флюгер крутится.") for n in sf.notes]
        self.assertTrue(raw, "предпосылка ушла: заметок разбора нет")
        leaked = [n for n in raw if n in self.rep.notes]
        self.assertEqual(leaked, [],
                         f"заметки разбора утекли в notes ({leaked}) — они "
                         f"вытесняют причины по Учителю, у которых срез короче")

    def test_teacher_reasons_survive_a_long_lesson(self):
        """
        Вытеснение обязано проверяться на ДЛИННОМ уроке, где срез [:6] реально
        режет. На двух фразах оно не проявляется — тест был бы зелёным при
        сломанном коде (замер: на корпусе в notes накапливается 173 строки).
        """
        work = tempfile.mkdtemp(prefix="lipid_long_")
        try:
            learner, _ = make_learner(work)
            rep = learner.learn_from_text(
                " ".join([SENT_NO_SUBJECT, SENT_MEASURE, SENT_CLAUSE,
                          "Флюгер крутится от ветра.", "Джойстик лежит.",
                          "Кактус колючий.", "Манго спелое."]),
                source="text", allow_teacher=False)
            shown = rep.as_text()
            self.assertTrue(rep.parse_notes,
                            "на длинном уроке заметки разбора пропали вовсе")
            # Причина по незнакомому слову обязана остаться ВИДИМОЙ или быть
            # объявленной числом (тот же закон, что в п.21).
            self.assertIn("НЕ спрашивали", shown,
                          f"число непонятых слов срезано без объявления:\n{shown}")
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def test_both_kinds_appear_in_as_text(self):
        text = self.rep.as_text()
        self.assertIn("Флюгер".lower(), text.lower())
        self.assertIn("погод", text.lower(),
                      f"разбор промолчал в итоговом тексте:\n{text}")


class TestTruncationIsNeverSilent(unittest.TestCase):
    """
    Любой срез списка в отчёте обязан СКАЗАТЬ, сколько он скрыл. Прежний
    notes[:6] резал молча — из-за этого п.21 лечили отдельным счётчиком.
    """

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="lipid_cut_")
        cls.learner, _ = make_learner(cls.work)
        cls.rep = cls.learner.learn_from_text(
            " ".join([
                "Он думает о погоде.", "Слон весит пять тонн.",
                "Река, берега которой заросли ивняком, обмелела.",
                "Мост построили за лето.", "Надо есть рыбу.",
                "Дом, в котором жил пёс, сгорел.",
                "Дорогу замело, так как метель усилилась.",
                "Кошка спрятала мышь, чтобы не есть её.",
            ]), source="text", allow_teacher=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    @staticmethod
    def _shown(rep):
        """Сколько заметок разбора ВИДНО в as_text — по их собственному тексту.

        Считать обязано по тексту самой заметки: слово «ещё» и «!» живут и в
        чужих строках отчёта (замерено), поэтому косвенные признаки не годятся.
        """
        text = rep.as_text()
        return [n for n in getattr(rep, "parse_notes", []) if n[:30] in text]

    @staticmethod
    def _band(rep):
        """
        Разобрать ПОЛОСУ заметок разбора в as_text: (объявлено_в_шапке,
        сколько_строк_напечатано, последняя_строка_полосы).

        ПЕРВЫЕ ВЕРСИИ ПРОВЕРЯЛИ КОСВЕННОЕ И ПОТОМУ ПРОПУСКАЛИ МУТАЦИИ:
          * слово «ещё» живёт и в чужих строках отчёта (замерено: «сущность
            ещё не создана», «ещё 2 значимых слова УТВЕРЖДАЮТСЯ»);
          * «!» начинают строку не только заметки;
          * цифра скрытых «1» находится в любом счётчике отчёта;
          * поиск строки по «ещё/скрыто/см.» подхватывал заметку «действие
            закреплено БЕЗ профиля ... не зная канонического» (в ней есть и
            «не», и цифра).
        Поэтому разбирается СТРУКТУРА: заголовок «разбор предупредил (N):» и
        идущие за ним строки полосы.
        """
        import re
        header = re.compile(r"разбор предупредил \((\d+)\):")
        lines = rep.as_text().splitlines()
        for i, ln in enumerate(lines):
            m = header.search(ln)
            if not m:
                continue
            declared = int(m.group(1))
            rows, tail = [], ""
            for nxt in lines[i + 1:]:
                t = nxt.strip()
                if not t.startswith("!"):
                    break
                # Строка-объявление «! ещё N — см. ...» НЕ есть заметка: её
                # нельзя класть в счёт напечатанных, иначе арифметика среза
                # расходится на единицу (первая версия теста так и падала —
                # правильно падала, но по моей ошибке, а не по дефекту кода).
                if re.match(r"^!\s*ещё\s+\d+", t):
                    tail = t
                    continue
                rows.append(t)
            return declared, len(rows), tail
        return None, 0, ""

    def test_shown_plus_hidden_equals_total(self):
        """
        Арифметика среза: ПОКАЗАНО + СКРЫТО = ВСЕГО — по структуре полосы, а
        не по косвенным признакам.

        ПЕРВАЯ ВЕРСИЯ брала восемь фраз: ПОСЛЕ дедуплики их меньше лимита,
        условие `total > shown` не срабатывало, мутация «скрытых = 0» прошла
        ЗЕЛЁНОЙ. Здесь — сам урок из setUpClass (в notes 173 строки, срез
        заведомо режет) и реальный корпус. skip вместо падения запрещён:
        «нечего проверять» обязано быть слышно (урок 19).
        """
        for rep in (self.rep, self._corpus_report()):
            total = len(rep.parse_notes)
            shown = len(self._shown(rep))
            declared, rows, tail = self._band(rep)
            self.assertIsNotNone(declared,
                                 f"{total} заметок разбора, а ПОЛОСЫ в отчёте нет")
            self.assertEqual(declared, total,
                             f"шапка полосы объявляет {declared}, а в поле "
                             f"{total} — отчёт врет о собственном объёме")
            self.assertEqual(rows, shown,
                             f"напечатано {rows} строк, а по тексту видно "
                             f"{shown} заметок")
            self.assertEqual(shown + rep.parse_notes_hidden, total,
                             f"ПОКАЗАНО {shown} + СКРЫТО "
                             f"{rep.parse_notes_hidden} ≠ ВСЕГО {total}: отчёт "
                             f"скрывает сам факт существования части причин")
            if rep.parse_notes_hidden:
                self.assertTrue(tail,
                                f"скрыто {rep.parse_notes_hidden} заметок, а "
                                f"строки-объявления в полосе нет")
                self.assertIn(str(rep.parse_notes_hidden), tail,
                              f"в строке объявления нет ЧИСЛА скрытых: {tail!r}")

    def test_truncation_band_exists_whenever_limit_is_hit(self):
        """
        Если заметок больше лимита — полоса ОБЯЗАНА быть и обязана называть
        число. Мутация «полосу не печатать» обязана падать здесь, а не
        размазываться по нескольким зелёным тестам.
        """
        from brain.learner import PARSE_NOTE_LIMIT
        rep = self._corpus_report()
        if len(rep.parse_notes) <= PARSE_NOTE_LIMIT:
            self.fail(f"случай перестал резать: {len(rep.parse_notes)} заметок "
                      f"при лимите {PARSE_NOTE_LIMIT} — мутационная проверка "
                      f"среза меряет пустоту")
        declared, rows, tail = self._band(rep)
        self.assertIsNotNone(declared, "срез есть, а полосы в отчёте нет")
        self.assertEqual(rows, PARSE_NOTE_LIMIT,
                         f"напечатано {rows} при лимите {PARSE_NOTE_LIMIT}")

    @classmethod
    def _corpus_report(cls):
        """Один урок на всём золотом корпусе: единственный замеренный случай,
        где срез заметок РЕАЛЬНО режет (9 уникальных причин при лимите 8)."""
        if "corpus" in cls.__dict__ and cls.corpus is not None:
            return cls.corpus
        from tools.golden_corpus import DEV_PATH, HELD_PATH, load_cases
        work = tempfile.mkdtemp(prefix="lipid_big_")
        try:
            learner, _ = make_learner(work)
            big = " ".join(c["text"] for path in (DEV_PATH, HELD_PATH)
                           for c in load_cases(path))
            cls.corpus = learner.learn_from_text(big, source="text",
                                                 allow_teacher=False)
            return cls.corpus
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def test_dedup_keeps_reason_once_but_counts_each_silent_sentence(self):
        """
        ДВЕ РАЗНЫЕ АРИФМЕТИКИ В ОДНОМ УРОКЕ (замерено, не выдумано): три
        фразы дают ОДНУ И ТУ ЖЕ причину «нет ни подлежащего», поэтому заметка
        обязана быть ОДНА, а счётчик молчаливых фраз — ТРИ.

        Мутация «снять дедупликацию» первой версией теста НЕ ловилась: та брала
        ОДНУ фразу, где дедупить нечего.
        """
        work = tempfile.mkdtemp(prefix="lipid_dedup_")
        try:
            learner, _ = make_learner(work)
            three = " ".join([SENT_NO_SUBJECT, "Надо есть рыбу.",
                              "Мост построили за лето."])
            rep = learner.learn_from_text(three, source="text",
                                          allow_teacher=False)
            self.assertEqual(len(rep.parse_notes), len(set(rep.parse_notes)),
                             f"дубли причин: {rep.parse_notes}")
            self.assertGreaterEqual(rep.sentences_silent, 3,
                                    f"три молчаливые фразы обязаны считаться "
                                    f"поштучно, получено {rep.sentences_silent}")
        finally:
            shutil.rmtree(work, ignore_errors=True)


class TestReportsAreIdempotent(unittest.TestCase):
    """Повторный урок того же текста не имеет права плодить дубли заметок."""

    def test_same_text_twice_does_not_duplicate_notes(self):
        work = tempfile.mkdtemp(prefix="lipid_idem_")
        try:
            learner, _ = make_learner(work)
            first = learner.learn_from_text(SENT_MEASURE, source="text",
                                            allow_teacher=False)
            second = learner.learn_from_text(SENT_MEASURE, source="text",
                                             allow_teacher=False)
            f = list(getattr(first, "parse_notes", []))
            s = list(getattr(second, "parse_notes", []))
            self.assertEqual(len(s), len(set(s)),
                             f"дубли внутри одного отчёта: {s}")
            self.assertLessEqual(len(s), len(f) + 1,
                                 f"второй урок раздул заметки: {len(f)} -> {len(s)}")
        finally:
            shutil.rmtree(work, ignore_errors=True)


class TestNothingIsWrittenToLiveMemory(unittest.TestCase):
    """Замер заметок — только в изолированной памяти (правило проекта)."""

    def test_live_world_untouched(self):
        from core.worldfile import default_memory_dir
        path = default_memory_dir()[1]

        def snap():
            if not os.path.exists(path):
                return None
            with open(path, "rb") as fh:
                return fh.read()

        before = snap()
        work = tempfile.mkdtemp(prefix="lipid_iso_b_")
        try:
            learner, _ = make_learner(work)
            for sent in ALL_THREE:
                learner.learn_from_text(sent, source="text", allow_teacher=False)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self.assertEqual(before, snap(),
                         "прогон по контракту тронул живую Вселенную")


class TestChannelIsNotStringMatching(unittest.TestCase):
    """
    Мутационная защита. Проброс нельзя «сделать» одной подстановкой строки:
    заметка обязана приходить ИЗ SentenceFacts, иначе замок обходится literals.
    """

    def test_note_text_is_taken_from_extractor_verbatim(self):
        """
        Формулировка экстрактора обязана дойти ДОСЛОВНО (или с префиксом
        предложения): перефразировать причину на стороне learner'а — значит
        через полгода утверждать в отчёте то, что правило давно перепечатали.
        """
        work = tempfile.mkdtemp(prefix="lipid_verb_")
        try:
            learner, _ = make_learner(work)
            for sent in ALL_THREE:
                sf = SimpleExtractor().extract(sent)
                if not sf.notes:
                    continue
                rep = learner.learn_from_text(sent, source="text",
                                              allow_teacher=False)
                blob = " ".join(list(getattr(rep, "parse_notes", []))
                                + list(rep.notes))
                self.assertTrue(
                    any(n[:25] in blob for n in sf.notes),
                    f"«{sent}»: ни одна формулировка экстрактора не дословна — "
                    f"{sf.notes} против {blob}")
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def test_no_note_invented_when_extractor_is_silent(self):
        """
        Обратная сторона: где экстрактор молчит, заметок быть НЕ должно.
        Иначе «причины есть на все случаи» — сочинённые причины.
        """
        work = tempfile.mkdtemp(prefix="lipid_quiet_")
        try:
            learner, _ = make_learner(work)
            sent = "Кошка пушистая."
            sf = SimpleExtractor().extract(sent)
            if sf.notes:
                self.skipTest("экстрактор теперь оставляет заметку и здесь — "
                              "условие теста устарело, пересмотреть")
            rep = learner.learn_from_text(sent, source="text",
                                          allow_teacher=False)
            self.assertEqual(list(getattr(rep, "parse_notes", [])), [],
                             f"придуманные заметки на фразе без отказов: "
                             f"{rep.parse_notes}")
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
