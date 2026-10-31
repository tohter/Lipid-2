"""
КОНТРАКТ: решающие меры сквозного прогона п.6 (tools/e2e_verdict.py).

Откуда файл. Очередь, пункт F1: «один прогон из ~8 упал на [warn] уже
работаю, повтор в грязном каталоге даёт НЕ СВЯЗАНО». Замер 31.10 подтвердил
оба и добавил третье — молчаливое свойство самого стенда:

  1. ПОВТОР В ТОМ ЖЕ КАТАЛОГЕ. Первый прогон (чистый каталог) зелёный,
     второй в том же даёт rc=1 «НЕ СВЯЗАНО», хотя знание в файле ПОЛНОЕ
     (hierarchy/properties/definition на месте). Причина не в мозге: во
     второй раз спрашивать «что такое пчела» не о чем, мозг отвечает из
     памяти, вопрос пользователю НЕ задаётся, а стенд требовал asked_of — то
     есть объявлял провальным ровно то, что перестал проверять.
  2. ПАДЕНИЕ САМОГО СТЕНДА. После [warn] «уже работаю» ответа в журнале нет,
     и прежняя строка `max(i for ...)` падала ValueError, не напечатав ни
     ИТОГА, ни причины. Читатель лога не узнавал, что сломался прогон, а не
     цикл.
  3. НИ ОДИН ТЕСТ НЕ ЗАПУСКАЛ e2e_gui. Скрипт неимпортируем (окно поднимается
     на уровне модуля), а гонять целый GUI подтестом (~11 с, замерено) никто
     не будет. Значит «выучил -> вспомнил» держал только ручной запуск.
     Отдельный улов: tests/test_definition_curiosity_contract.py проверял
     маркеры отказа по СОБСТВЕННОМУ списку — две копии, способные разойтись.

Отсюда форма замка: логика вердикта вынесена в чистые функции, они подлинные
для скрипта и званены отсюда. Проверяется РАСКЛАДКА отказа (стенд/мозг/успех)
и признаки, а не текст лога.

Мутации, которые обязан поймать этот файл (проверены руками, см. отчёт):
  * is_refusal без какого-либо маркера -> отказ перестаёт быть отказом;
  * reports_after_answer без None -> падение вместо читаемого отказа;
  * start_problems, считающий непустой каталог годным -> ложное НЕ СВЯЗАНО;
  * verdict, забывший любой из шести фактов -> зелёный на дырявом прогоне;
  * rejected_in, не сверяемый с ui.presenter.REJECTED -> отклонение работы
    перестаёт быть отказом СТАНДА.
"""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.e2e_verdict import (                           # noqa: E402
    REFUSAL_MARKERS, is_refusal, knowledge_in_world, rejected_in,
    reports_after_answer, start_problems, verdict)

# Ответ-ЗНАНИЕ и ответ-ОТКАЗ — дословно те, что ловил прогон. ОТКАЗ взят из
# замера: он содержит спрашиваемое слово, поэтому «пчел» in answer НЕ признак
# знания (см. test_definition_curiosity_contract).
KNOWLEDGE = ("[Lipid-2]: Пчела — летаемое насекомое; опыляет цветы; "
             "живёт в улье")
REFUSAL = ("[Lipid-2]: О 'пчела' в памяти есть только само слово — знания "
           "нет. Спроси ещё раз с включённым Учителем или подскажи сам.")
ASKED = ["что такое пчела"]


def ok_inputs():
    """Входы заведомо ЗЕЛЁНОГО прогона. Мутации меряются против них."""
    return dict(asked_of=list(ASKED), answer_text=KNOWLEDGE,
                after_reports=[], remembered=True, contradictions=[])


class TestRefusalRecognition(unittest.TestCase):
    """Признак отказа обязан переживать правки текста, но не смену смысла.

    ПОЧЕМУ ЗДЕСЬ ВЫЗОВ МОЗГА, А НЕ ПЕРЕБОР СПИСКА. Первая версия проверки
    была: `for m in REFUSAL_MARKERS: assertTrue(is_refusal(m))`. Мутация
    «выбросить маркер из кортежа» проходила её ЗЕЛЁНОЙ (замерено) — список и
    проверяющий один и тот же объект, и мутация делала проверку меньше вместе
    с кодом. Проверять признак его же собственным списком — значит проверять
    ничего. Поэтому маркеры сверяются с НАСТОЯЩИМИ текстами отказа мозга.
    """

    def _refusals(self):
        """Три честных отказа мозга — тексты, а не мои пересказы."""
        os.environ["LIPID2_STATUS"] = "0"
        from brain.lipid2_brain import Lipid2Brain
        with tempfile.TemporaryDirectory() as d:
            b = Lipid2Brain(memory_dir=d)
        hollow = {"base_entity": {"concept": "нюхор", "actions": [],
                                  "hierarchy": [], "properties": [],
                                  "definition": ""}}
        known = {"base_entity": {"concept": "пчела",
                                 "actions": ["опылять"], "hierarchy": [],
                                 "properties": [], "definition": ""}}
        return [b._definition_hollow("нюхор", None),        # совершенная пустота
                b._definition_hollow("нюхор", hollow),      # только само слово
                b._definition_hollow("пчела", known)]       # есть действие

    def test_real_refusals_are_recognized(self):
        for text in self._refusals():
            self.assertTrue(is_refusal(text),
                            "настоящий отказ мозга не опознан как отказ — "
                            "прогон снова назовёт знание знанием, а отказ "
                            f"знанием: {text!r}")

    def test_markers_are_not_a_self_check(self):
        """Каждый маркер обязан встречаться в текстах мозга. Иначе он — балласт,
        создающий ложное покрытие: кортеж больше, чем реальность."""
        joined = " ".join(self._refusals()).lower()
        for m in REFUSAL_MARKERS:
            self.assertIn(m, joined,
                          f"маркер {m!r} не встречается НИ В ОДНОМ тексте "
                          f"отказа: он проверяет то, чего не бывает")

    def test_knowledge_line_carries_no_marker(self):
        """Знание не должно содержать ни одного признака отказа — иначе классы
        схлопнутся и ok станет недостижим."""
        for m in REFUSAL_MARKERS:
            self.assertNotIn(m, KNOWLEDGE.lower(),
                             f"признак {m!r} сработал на знание")

    def test_knowledge_is_not_refusal(self):
        self.assertFalse(is_refusal(KNOWLEDGE))

    def test_empty_text_is_not_refusal(self):
        """«НИЧЕГО» — отказ СТАНДА (ответ не дошёл), а не отказ мозга."""
        self.assertFalse(is_refusal(""))
        self.assertFalse(is_refusal(None))

    def test_refusal_contains_asked_word_but_is_still_refusal(self):
        """Почему подстрока не работает: в отказе спрашиваемое слово ЕСТЬ."""
        self.assertIn("пчел", REFUSAL.lower())
        self.assertTrue(is_refusal(REFUSAL))


class TestStandVsBrain(unittest.TestCase):
    """Три исхода вместо двух: поломку прогона нельзя вешать на мозг."""

    def test_green_run_is_green(self):
        ok, why = verdict(**ok_inputs())
        self.assertTrue(ok, f"честный прогон объявлен красным: {why}")
        self.assertEqual(why, [])

    def test_missing_answer_is_reported_not_raised(self):
        """Ответ может НЕ ДОЙТИ (окно не взяло работу). max() по пустому
        итератору ронял стенд без ИТОГА; здесь это ЗНАЧЕНИЕ."""
        self.assertIsNone(reports_after_answer(["status", "report"]))
        self.assertIsNone(reports_after_answer([]))

    def test_reports_after_last_answer_only(self):
        kinds = ["report", "answer", "report", "answer", "status"]
        self.assertEqual(reports_after_answer(kinds), [])
        self.assertEqual(reports_after_answer(["answer", "report", "status"]),
                         ["report"])

    def test_no_question_means_not_learned_by_brain(self):
        inp = ok_inputs() | {"asked_of": []}
        ok, why = verdict(**inp)
        self.assertFalse(ok)
        self.assertTrue(any("вопрос" in w for w in why), why)

    def test_refusal_answer_is_red(self):
        inp = ok_inputs() | {"answer_text": REFUSAL}
        ok, why = verdict(**inp)
        self.assertFalse(ok)
        self.assertTrue(any("ОТКАЗ" in w for w in why), why)

    def test_answer_about_other_word_is_red(self):
        inp = ok_inputs() | {"answer_text": "Слон — наземное млекопитающее"}
        ok, why = verdict(**inp)
        self.assertFalse(ok)
        self.assertTrue(any("пчел" in w for w in why), why)

    def test_reports_after_answer_are_red(self):
        inp = ok_inputs() | {"after_reports": ["report"]}
        ok, why = verdict(**inp)
        self.assertFalse(ok)
        self.assertTrue(any("отчёты обучения" in w for w in why), why)

    def test_absent_answer_is_red_and_names_stand(self):
        inp = ok_inputs() | {"after_reports": None}
        ok, why = verdict(**inp)
        self.assertFalse(ok)
        self.assertTrue(any("СТАНДА" in w for w in why),
                        f"пустой журнал обязан зваться отказом СТАНДА: {why}")

    def test_knowledge_only_in_text_not_file_is_red(self):
        inp = ok_inputs() | {"remembered": False}
        ok, why = verdict(**inp)
        self.assertFalse(ok)
        self.assertTrue(any("ФАЙЛЕ" in w for w in why), why)

    def test_contradiction_is_red(self):
        inp = ok_inputs() | {"contradictions": ["пчела"]}
        ok, why = verdict(**inp)
        self.assertFalse(ok)
        self.assertTrue(any("противореч" in w.lower() for w in why), why)


class TestCleanIsolationRequired(unittest.TestCase):
    """Каталог прогона обязан быть пуст ДО подъёма окна."""

    def test_empty_dir_is_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(start_problems(d), [])

    def test_dirty_dir_is_refused(self):
        """Ровно случай F1: повтор прогона в том же каталоге."""
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "lipid_world.json"), "w").write("{}")
            problems = start_problems(d)
            self.assertEqual(len(problems), 1)
            self.assertIn("НЕ пуст", problems[0])
            self.assertIn("lipid_world.json", problems[0])

    def test_missing_memory_dir_is_refused(self):
        problems = start_problems(None)
        self.assertTrue(problems)
        self.assertIn("LIPID2_MEMORY_DIR", problems[0])

    def test_absent_dir_is_allowed(self):
        """Каталога нет — не грязно, а чисто: он создастся по пути."""
        self.assertEqual(start_problems("/tmp/l2_такого_нет_9e31"), [])

    def test_hidden_files_do_not_deny(self):
        """Служебные точки (кэш по правилам проекта общий) не считаются
        состоянием прогона."""
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, ".cache_side"), "w").write("x")
            self.assertEqual(start_problems(d), [])


class TestWorkRejectionVisible(unittest.TestCase):
    """Отклонённую окном работу прогон обязан видеть как отказ СТАНДА."""

    def test_rejected_work_detected_by_shared_name(self):
        from ui.presenter import REJECTED
        rows = [(0.0, "warn", f"{REJECTED} — дождись окончания «обучение»")]
        self.assertEqual(len(rejected_in(rows)), 1)

    def test_ordinary_rows_are_not_rejection(self):
        rows = [(0.0, "answer", KNOWLEDGE), (1.0, "report", "готово за 1 с")]
        self.assertEqual(rejected_in(rows), [])

    def test_marker_comes_from_presenter_not_from_memory(self):
        """Признак берётся ИМЕНЕМ у окна. Копия текста в тесте — та же
        ошибка, что маркеры отказа в соседнем файле: её правка не краснит
        замок, а делает его молчаливым."""
        import inspect

        from ui.presenter import REJECTED
        src = inspect.getsource(rejected_in)
        self.assertIn("REJECTED", src)
        self.assertTrue(REJECTED, "пустой признак не отличит ничего ни от чего")


class TestKnowledgeInFile(unittest.TestCase):
    def test_any_knowledge_field_counts(self):
        for f in ("hierarchy", "properties", "definition"):
            self.assertTrue(knowledge_in_world({f: ["что-то"]}), f)

    def test_bare_word_is_not_knowledge(self):
        self.assertFalse(knowledge_in_world({"name": "пчела"}))
        self.assertFalse(knowledge_in_world({}))
        self.assertFalse(knowledge_in_world(None))

    def test_empty_lists_are_not_knowledge(self):
        self.assertFalse(knowledge_in_world({"hierarchy": [],
                                             "properties": []}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
