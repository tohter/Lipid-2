"""
Контракт логики окна (ui/presenter.py) — БЕЗ Tkinter и без мозга.

Зачем это отдельно от test_gui: проверять цвета и расположение кнопок дёшево
только в глаза, а толку мало. Здесь то, что реально ломалось и ломается:

  1. фоновый поток без try/except умирал молча, окно оставалось «работает»
     навсегда (реальный баг прежнего GUI);
  2. второй запуск обучения поверх первого — гонка двух writer'ов в один файл;
  3. вопрос мозгу без ответа вешал рабочий поток, а закрытие окна не помогало;
  4. журнал без ограничений превращает Tk-виджет в тормоз.

Все проверки — на реальном времени и реальных потоках, но с микроскопическими
таймаутами: суммарно секунды.
"""
import os
import sys
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ui.presenter import (AnswerBridge, JobController, Journal, format_report,
                          summarize_report)


def wait_until(pred, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


class TestJournal(unittest.TestCase):

    def test_add_and_tail(self):
        j = Journal(limit=10)
        j.add("изучаю слово «джойстик»")
        j.add("записано", kind="ok")
        self.assertEqual(len(j), 2)
        ts, kind, text = j.tail(1)[0]
        self.assertEqual((kind, text), ("ok", "записано"))

    def test_empty_lines_are_not_rows(self):
        j = Journal()
        j.add("")
        j.add("   ")
        self.assertEqual(len(j), 0)

    def test_limit_drops_oldest_and_COUNTS_it(self):
        j = Journal(limit=5)
        for i in range(9):
            j.add(f"строка {i}")
        self.assertEqual(len(j), 5)
        self.assertEqual(j.dropped, 4,
                         "сброшенные строки обязаны быть посчитаны, иначе "
                         "«журнал укоротился» неотличим от «журнал замолчал»")
        self.assertEqual(j.tail(5)[0][2], "строка 4")

    def test_last_status_skips_reports(self):
        j = Journal()
        j.add("изучаю слово «джойстик»")
        j.add("📚 Отчёт:\n  фактов: 8", kind="report")
        self.assertEqual(j.last_status(), "изучаю слово «джойстик»",
                         "индикатор «чем занята» не должен показывать простыню")

    def test_error_is_visible_in_last_status(self):
        j = Journal()
        j.add("работало")
        j.add("⛔ модель недоступна", kind="error")
        self.assertIn("недоступна", j.last_status())

    def test_threadsafe_add(self):
        j = Journal(limit=10_000)
        threads = [threading.Thread(target=lambda k=k: [j.add(f"{k}:{i}")
                                                         for i in range(200)])
                   for k in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(j), 1200)


class TestAnswerBridge(unittest.TestCase):

    def test_question_reaches_ui_and_answer_returns(self):
        seen = []
        bridge = AnswerBridge(on_question=lambda q: seen.append(q))

        def worker():
            bridge.result = bridge.handler("Что такое «флюгер»?")

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        self.assertTrue(wait_until(lambda: bool(seen)),
                        "вопрос не дошёл до интерфейса")
        self.assertEqual(seen, ["Что такое «флюгер»?"])
        self.assertTrue(bridge.submit("это прибор"))
        th.join(3)
        self.assertEqual(bridge.result, "это прибор")

    def test_empty_answer_is_none_like(self):
        bridge = AnswerBridge(on_question=lambda q: None)
        th = threading.Thread(target=lambda: setattr(
            bridge, "result", bridge.handler("вопрос")), daemon=True)
        th.start()
        self.assertTrue(wait_until(lambda: bridge.pending))
        bridge.submit("")
        th.join(3)
        self.assertEqual(bridge.result, "",
                         "пустой ответ — ответственность вызывающего: мозг "
                         "трактует пустую строку как «знания нет»")

    def test_answer_without_question_is_refused(self):
        bridge = AnswerBridge(on_question=lambda q: None)
        self.assertFalse(bridge.submit("ответ в пустоту"))

    def test_close_unblocks_waiting_thread(self):
        bridge = AnswerBridge(on_question=lambda q: None)
        th = threading.Thread(target=lambda: setattr(
            bridge, "result", bridge.handler("вопрос")), daemon=True)
        th.start()
        self.assertTrue(wait_until(lambda: bridge.pending))
        bridge.close()
        th.join(3)
        self.assertFalse(th.is_alive(),
                         "закрытие окна обязано отпустить рабочий поток, "
                         "иначе процесс живёт после закрытия окна")
        self.assertIsNone(bridge.result)
        self.assertFalse(bridge.submit("поздно"))


class TestJobController(unittest.TestCase):

    def setUp(self):
        self.journal = Journal()
        self.states = []
        self.ctrl = JobController(self.journal, on_state=self.states.append)

    def test_work_runs_in_background_and_reports(self):
        done = threading.Event()

        def work():
            self.journal.add("делаю")
            done.set()

        self.assertTrue(self.ctrl.start(work, label="обучение"))
        self.assertTrue(done.wait(3))
        self.assertTrue(wait_until(lambda: not self.ctrl.busy))
        self.assertEqual(self.states, [True, False])
        self.assertEqual(self.ctrl.finished, 1)
        text = " ".join(r[2] for r in self.journal.rows())
        self.assertIn("начал: обучение", text)
        self.assertIn("закончил: обучение", text)

    def test_exception_reaches_journal_not_silence(self):
        def boom():
            raise ValueError("нет связи с 8080")

        self.ctrl.start(boom, label="обучение")
        self.assertTrue(wait_until(lambda: not self.ctrl.busy, 3.0),
                        "поток умер, не сбросив busy — окно зависло бы навсегда")
        self.assertIn("ValueError", self.ctrl.last_error)
        text = " ".join(r[2] for r in self.journal.rows())
        self.assertIn("нет связи с 8080", text)
        kinds = [r[1] for r in self.journal.rows()]
        self.assertIn("error", kinds, "ошибка обязана быть помечена как ошибка")

    def test_second_job_is_refused_and_says_why(self):
        gate = threading.Event()
        release = threading.Event()

        def work():
            gate.set()
            release.wait(3)

        self.assertTrue(self.ctrl.start(work, label="обучение"))
        self.assertTrue(gate.wait(3))
        self.assertFalse(self.ctrl.start(lambda: None, label="второе"))
        text = " ".join(r[2] for r in self.journal.rows())
        self.assertIn("уже работаю", text)
        self.assertIn("lipid_world.json", text,
                      "причина отказа обязана объясняться: молчаливое «нет» "
                      "выглядит как сломанная кнопка")
        release.set()
        self.assertTrue(wait_until(lambda: not self.ctrl.busy))

    def test_busy_flag_released_even_if_work_lies(self):
        # работа, которая завершается нормально, но быстро
        self.ctrl.start(lambda: None, label="быстро")
        self.assertTrue(wait_until(lambda: not self.ctrl.busy))
        # и снова можно запускать
        self.assertTrue(self.ctrl.start(lambda: None, label="снова"))

    def test_on_done_failure_does_not_break_state(self):
        def bad():
            raise RuntimeError("GUI-обновление упало")

        ctrl = JobController(self.journal, on_state=lambda s: None, on_done=bad)
        ctrl.start(lambda: None, label="ok")
        self.assertTrue(wait_until(lambda: not ctrl.busy, 3.0))
        text = " ".join(r[2] for r in self.journal.rows())
        self.assertIn("on_done", text)
        self.assertIn("закончил", text,
                      "сбой пост-обработчика не должен отменять итог работы")


class TestReportFormatting(unittest.TestCase):

    def test_report_shows_degraded_channels(self):
        from brain.learner import LearningReport
        rep = LearningReport(content_words=10, known_words=4, facts=8,
                             observations_new=5, teacher_calls=2,
                             quality={"morph": False, "teacher": False})
        text = format_report(rep)
        self.assertIn("ЭВРИСТИКА", text)
        self.assertIn("ОТКЛЮЧЁН", text)
        self.assertIn("8 фактов", text.replace("фактов извлечено: 8", "8 фактов"))

    def test_summary_is_one_line(self):
        from brain.learner import LearningReport
        rep = LearningReport(content_words=10, known_words=4,
                             observations_new=5, teacher_calls=2)
        line = summarize_report(rep)
        self.assertNotIn("\n", line)
        self.assertIn("+5", line)

    def test_teacher_budget_visible_without_opening_journal(self):
        """
        П.21: исчерпанный лимит обязан быть виден ЧИСЛОМ в строке статуса и в
        отчёте — не только построчной причиной в notes.

        ЗАМЕРЕНО до правки: при лимите 3 и 12 незнакомых словах статус-бар
        говорил «учитель 3×», а список «не спрашивали: 8 — ...» читался как
        «знания нет». Causes по словам были, сводки не было ни в одной строке:
        notes режутся [:6], not_asked — [:8].
        """
        from brain.learner import LearningReport
        rep = LearningReport(content_words=12, known_words=0, teacher_calls=3,
                             teacher_budget=3, blocked_by_limit=8,
                             not_asked=["джойстик", "дыня", "кактус"])
        line = summarize_report(rep)
        self.assertNotIn("\n", line)
        self.assertIn("3/3", line, f"бюджет не виден в статус-баре: {line!r}")
        self.assertIn("8", line, f"число заблокированных слов не в статус-баре: {line!r}")
        text = format_report(rep)
        self.assertIn("3 из 3", text,
                      "в отчёте нет «N из M» — пользователь обязан видеть "
                      "границу бюджета, а не вычитать её из текста причины")
        self.assertIn("НЕ «не нашли»", text,
                      "заблокированные лимитом слова названы ненайденными")

    def test_no_fake_fraction_without_budget(self):
        """Бюджет не задан — значит «учитель N», а не врущее «N/0»."""
        from brain.learner import LearningReport
        line = summarize_report(LearningReport(teacher_calls=2))
        self.assertIn("учитель 2", line)
        self.assertNotIn("учитель 2/0", line,
                         "при незаданном бюджете статус обещает ноль обращений")
        self.assertNotIn("учитель 2/3", line)


class TestLogchannelIntegration(unittest.TestCase):
    """Презентер и канал статуса обязаны быть соединены, а не параллельны."""

    def test_status_reaches_subscriber(self):
        from core import logchannel
        logchannel.reset()
        got = []
        logchannel.subscribe(got.append)
        logchannel.status("изучаю слово «джойстик»")
        self.assertEqual(got, ["изучаю слово «джойстик»"])
        logchannel.reset()

    def test_broken_subscriber_dropped_not_fatal(self):
        from core import logchannel
        logchannel.reset()
        ok = []

        def broken(msg):
            raise RuntimeError("виджет мёртв")

        logchannel.subscribe(broken)
        logchannel.subscribe(ok.append)
        logchannel.status("долшло")                       # не должно бросить
        self.assertEqual(ok, ["долшло"])
        self.assertEqual(logchannel.subscribers(), 1,
                         "сломанный подписчик обязан быть отключён")
        self.assertTrue(logchannel.dropped())
        logchannel.reset()

    def test_request_without_handler_returns_none_loudly(self):
        from core import logchannel
        logchannel.reset()
        got = []
        logchannel.subscribe(got.append)
        self.assertIsNone(logchannel.request("Что такое X?"))
        self.assertTrue(any("обратный канал не подключён" in m for m in got),
                        "«спросить некого» обязано быть видно, иначе канал 3 "
                        "молча считается успешным")
        logchannel.reset()

    def test_request_goes_through_handler(self):
        from core import logchannel
        logchannel.reset()
        logchannel.ask_handler(lambda q: "это прибор")
        self.assertEqual(logchannel.request("Что такое X?"), "это прибор")
        logchannel.ask_handler(None)
        self.assertIsNone(logchannel.request("Что такое X?"))
        logchannel.reset()



class TestJournalKinds(unittest.TestCase):
    """
    ОТВЕТ и ОТЧЁТ обязаны различаться. ЗАМЕРЕНО на живом прогоне: журнал
    заканчивался блоком «📚 Обучение из текста … выучено: экосистема», а строка
    ответа шла тем же kind="report" и тем же цветом — последнее, что видел
    пользователь, выглядело как отчёт обучения, а не как ответ Аманды.
    """

    def test_answer_row_has_its_own_kind(self):
        from ui.presenter import answer_row, KIND_ANSWER, KIND_REPORT
        text, kind = answer_row("Флюгер — это прибор")
        self.assertEqual(kind, KIND_ANSWER)
        self.assertNotEqual(KIND_ANSWER, KIND_REPORT,
                            "ответ и отчёт сливаются в один вид журнала")
        self.assertIn("Флюгер", text)

    def test_journal_keeps_kinds_apart(self):
        from ui.presenter import Journal, answer_row, KIND_ANSWER, KIND_REPORT
        j = Journal()
        j.add("📚 Обучение из текста\n  выучено: флюгер", KIND_REPORT)
        j.add(*answer_row("Флюгер — это прибор"))
        kinds = [kind for _, kind, _ in j.rows()]
        self.assertEqual(kinds, [KIND_REPORT, KIND_ANSWER])
        self.assertNotEqual(kinds[-1], KIND_REPORT,
                            "журнал обязан заканчиваться ОТВЕТОМ, а не отчётом")

    def test_gui_paints_answer_differently_from_report(self):
        """Цвет ответа отличен от цвета отчёта — иначе различия не видно."""
        from ui.app_gui import KIND_COLORS
        self.assertIn("answer", KIND_COLORS,
                      "для kind=\"answer\" нет цвета: Tk покрасит его default")
        self.assertNotEqual(KIND_COLORS["answer"], KIND_COLORS["report"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
