"""
Два разных «не знаю»: спрашивали — или не спрашивали.

Дефект, найденный на GUI-прогоне: урок со ВЫКЛЮЧЕННЫМ Учителем записывал
встреченные слова в unknown_words без всякого пометочного поля. Следующий урок
с включённым Учителем видел запись и молча выходил — слово оставалось
невыученным НАВСЕГДА, хотя его ни разу не спрашивали. «Опылять», «пчела» и
«цветок» погибли именно так: галочку сняли на одном уроке.

Отдельно чинится счётчик: unknown_words/unknown означали «слово было
незнакомым», из-за чего урок, который успешно выучил слово, рапортовал
«распознано 0%, неизвестных 1» — как будто учиться было бесполезно.
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.worldfile import read_world, write_world


class StubTeacher:
    def __init__(self, answers=None):
        self.answers = answers or {}
        self.calls = []

    def __call__(self, concept):
        self.calls.append(concept)
        return self.answers.get(concept)


class TestTwoKindsOfUnknown(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="l2status-")
        self.path = os.path.join(self.dir, "lipid_world.json")
        self._write({"entities": {}, "observations": {}, "unknown_words": {}})

    def _write(self, w):
        write_world(self.path, w)

    def _read(self):
        return read_world(self.path)

    def _learner(self, teacher=None):
        from brain.learner import Learner
        return Learner(load_world=self._read,
                       save_world=lambda w: write_world(self.path, w),
                       teacher=teacher)

    def test_no_teacher_means_not_asked(self):
        """Выключенный канал = «НЕ спрашивали», слово обязано остаться учеником."""
        from brain.learner import NOT_ASKED
        t = StubTeacher({"флакон": {"hierarchy": ["посуда"]}})
        lr = self._learner(t)
        rep = lr.learn_from_text("Флакон стоит на полке.", source="t",
                                 allow_teacher=False)
        self.assertEqual(t.calls, [], "Учитель выключен, но вызван")
        self.assertIn("флакон", rep.not_asked)
        # ВСЕ незнакомые слова урока помечаются как «не спрашивали»: иначе любое
        # из них потом станет «вечно неизвестным».
        unknowns = self._read()["unknown_words"]
        self.assertTrue(unknowns, "незнакомые слова вообще не записаны")
        for key, rec in unknowns.items():
            self.assertEqual(rec.get("status"), NOT_ASKED,
                             f"«{key}» помечено как испрошенное, хотя канала не было")

    def test_not_asked_is_asked_at_the_next_lesson(self):
        """Включили канал позже — слово СПРАШИВАЕТСЯ, а не считается вечным."""
        t = StubTeacher({"флакон": {"hierarchy": ["посуда"],
                                    "properties": ["стеклянный"]}})
        self._learner(t).learn_from_text("Флакон стоит.", source="t",
                                         allow_teacher=False)
        self.assertNotIn("флакон", self._read()["entities"])
        rep = self._learner(t).learn_from_text("Флакон стоит.", source="t",
                                               allow_teacher=True)
        self.assertIn("флакон", t.calls,
                      "слово ни разу не спросили, но оно уже «навсегда неизвестно»")
        self.assertIn("флакон", rep.learned_concepts)
        self.assertIn("флакон", self._read()["entities"])

    def test_asked_empty_is_not_asked_twice(self):
        """А вот сюда Учитель ходить повторно НЕ должен — это дороже всего."""
        t = StubTeacher({})   # ничего не знает
        lr = self._learner(t)
        lr.learn_from_text("Зюзя сидит.", source="t", allow_teacher=True)
        first = len(t.calls)
        self._learner(t).learn_from_text("Зюзя сидит.", source="t",
                                         allow_teacher=True)
        self.assertEqual(len(t.calls), first,
                         f"повторный вопрос о уже испрошенном слове: {t.calls}")
        self.assertEqual(self._read()["unknown_words"]["зюзя"]["status"],
                         "asked_empty")

    def test_report_does_not_call_learned_word_unknown(self):
        """
        Урок выучил слово — отчёт не имеет права рапортовать его неизвестным.
        ЗАМЕРЕНО до правки: «распознано: 0 (0%)  незнакомых: 1 …
        осталось неизвестным: флюгер» про урок, который создал сущность.
        """
        # «лежать» тоже слово: учитель на него отвечает, чтобы в отчёте остался
        # ровно один кандидат — флакон. Иначе проверка считает чужие слова.
        t = StubTeacher({"флакон": {"hierarchy": ["посуда"],
                                    "properties": ["стеклянный"]},
                         "лежать": {"hierarchy": ["состояние"]}})
        rep = self._learner(t).learn_from_text("Флакон лежит.", source="t")
        self.assertIn("флакон", rep.learned_concepts)
        self.assertNotIn("флакон", rep.unknown,
                         "выученное слово записано в «осталось неизвестным»")
        self.assertEqual(rep.unknown_words, 0,
                         "урок выучил все слова — «неизвестных» обязано быть 0")
        self.assertEqual(rep.seen_unknown_words, 2,
                         "сколько слов ПРИШЛО незнакомыми — отдельная цифра, "
                         "её терять нельзя (иначе непонятно, чему учились)")
        text = rep.as_text()
        self.assertNotIn("осталось неизвестным: флакон", text)
        self.assertIn("НЕ спрашивали: 0", text)

    def test_limit_is_reported_as_not_asked(self):
        """Лимит обращений — это «не спросили», а не «не нашли».
        Молча закрыть слово статусом asked_empty значило бы потерять его."""
        asked = {"абрикос": {"hierarchy": ["фрукт"], "properties": ["сладкий"]},
                 "мять": {"hierarchy": ["действие"]},
                 "баклажан": {"hierarchy": ["овощ"]}}
        t = StubTeacher(asked)
        lr = self._learner(t)
        lr.max_teacher_calls = 1
        rep = lr.learn_from_text("Абрикос мнёт баклажан.", source="t",
                                 allow_teacher=True)
        self.assertEqual(len(t.calls), 1,
                         f"лимит {lr.max_teacher_calls} нарушен: {t.calls}")
        self.assertEqual(len(rep.not_asked), 2,
                         "исчерпание лимита выдало неспрошеные слова за «не найдено»")
        self.assertTrue(any("лимит" in n.lower() for n in rep.notes),
                        "лимит исчерпан молча")
        unknowns = self._read()["unknown_words"]
        # Спрошенное и найденное слово уходит из unknown_words совсем;
        # неспрошеные остаются со статусом not_asked.
        self.assertNotIn("абрикос", unknowns,
                         "выученное слово осталось в списке незнания")
        rest = sorted(v.get("status") for v in unknowns.values())
        self.assertEqual(rest, ["not_asked", "not_asked"],
                         f"статусы расставлены неверно: {unknowns}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
