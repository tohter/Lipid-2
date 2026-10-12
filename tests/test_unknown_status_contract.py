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
        Молча закрыть слово статусом asked_empty значило бы потерять его.

        Глагол «мнёт» вunknown_words НЕ попадает: он изучается действием, а не
        сущностью, поэтому в списке остаётся ровно одно существительное.
        """
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
        self.assertIn("баклажан", rep.not_asked,
                      "исчерпание лимита выдало неспрошеное слово за «не найдено»")
        self.assertTrue(any("лимит" in n.lower() for n in rep.notes),
                        "лимит исчерпан молча")
        unknowns = self._read()["unknown_words"]
        # Спрошенное и найденное слово уходит из unknown_words совсем;
        # неспрошеные остаются со статусом not_asked.
        self.assertNotIn("абрикос", unknowns,
                         "выученное слово осталось в списке незнания")
        self.assertEqual(unknowns["баклажан"]["status"], "not_asked",
                         f"статус расставлен неверно: {unknowns}")

    def test_disabled_channels_never_become_asked_empty(self):
        """
        Все каналы выключены — ни одно слово НЕ имеет права получить
        asked_empty: мы НИЧЕГО не спрашивали, а файл не имеет права писать
        «мы искали и не нашли». ЗАМЕРЕНО: при трёх снятых галочках каждый
        встреченный слово получало пожизненный asked_empty.
        """
        from brain.learner import ASKED_EMPTY
        t = StubTeacher({})
        rep = self._learner(t).learn_from_text("Зюзя сидит на табурете.",
                                               source="t", allow_teacher=False)
        unknowns = self._read()["unknown_words"]
        self.assertTrue(unknowns, "нечего проверять — слова не записаны")
        for key, rec in unknowns.items():
            self.assertNotEqual(rec.get("status"), ASKED_EMPTY,
                                f"«{key}» помечено испрошенным без вопроса")
        self.assertEqual(rep.teacher_calls, 0,
                         "обращение к каналу засчитано, хотя канала не было")


class TestChannelCounters(unittest.TestCase):
    """
    Мозг обязан считать ОБРАЩЕНИЯ К КАЖДОМУ КАНАЛУ и публиковать последний
    отчёт. Причина не в косметике: ЗАМЕРЕНО, сквозной прогон утверждал
    «выучил и выучил и вспомнил» при выключенном LLM, потому что проверял
    только текст журнала. Число обращений — единственное доказательство, что
    канал работали; без него «знание дала Gemma» необратимо в утверждение.
    """

    def _brain(self, tmp):
        from brain.lipid2_brain import Lipid2Brain
        os.makedirs(tmp, exist_ok=True)
        with open(os.path.join(tmp, "lipid_world.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"entities": {}, "unknown_words": {},
                       "observations": {}}, f, ensure_ascii=False)
        brain = Lipid2Brain(memory_dir=tmp)

        class Silent:
            last_failure = None

            def run_agent(self, **kw):
                return None

        brain.teacher_llm = Silent()
        brain.use_wiki_teacher = False
        brain.ask_user_enabled = False
        return brain

    def test_counters_count_calls_per_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = self._brain(tmp)
            rep = brain.learn_from_text("Вода гасит пламя.", allow_teacher=True)
            self.assertGreater(rep.channel_calls.get("llm", 0), 0,
                               "канал 1 вызывался — счётчик обязан это показать")
            self.assertEqual(rep.channel_found.get("llm", 0), 0,
                             "молчаливый учитель не обязан становиться нашедшим")
            self.assertEqual(rep.channel_calls.get("user", 0), 0,
                             "канал пользователя выключен — обращений нет")

    def test_counters_reset_between_lessons(self):
        """Счётчики сеанса, а не процесса: сумма за всё время ничего не значит."""
        with tempfile.TemporaryDirectory() as tmp:
            brain = self._brain(tmp)
            first = brain.learn_from_text("Вода гасит пламя.", allow_teacher=True)
            again = brain.learn_from_text("Пчёлы опыляют цветы.",
                                          allow_teacher=True)
            self.assertLessEqual(again.channel_calls.get("llm", 0),
                                 first.channel_calls.get("llm", 0) + 3,
                                 "счётчик не накопился за два урока")

    def test_disabled_channel_counts_zero_calls(self):
        """Контроль ВВЕРХ: снятый канал обязан дать ровно ноль обращений.
        Если счётчик рисует вызовы, вся проверка «знание дала Gemma» — ложь."""
        with tempfile.TemporaryDirectory() as tmp:
            brain = self._brain(tmp)
            rep = brain.learn_from_text("Вода гасит пламя.", allow_teacher=False)
            self.assertEqual(rep.channel_calls.get("llm", 0), 0)
            self.assertEqual(rep.channel_calls.get("wiki", 0), 0)
            self.assertEqual(rep.channel_calls.get("user", 0), 0)
            # unknown = «осталось невыученным», not_asked = «мы НЕ искали».
            # Без учителей первое равно второму; ложью было бы status=
            # asked_empty в мире («искали — там пусто»).
            self.assertEqual(set(rep.unknown), set(rep.not_asked),
                             "невыученное без учителей обязано числиться неисканным")
            world = read_world(brain.world_filepath)
            statuses = {k: v.get("status")
                        for k, v in world["unknown_words"].items()}
            self.assertNotIn("asked_empty", statuses.values(),
                             f"выключенные каналы породили пожизненный "
                             f"«искали и не нашли»: {statuses}")


class TestVerbsAreNotEntities(unittest.TestCase):
    """
    Глагол — отношение, а не «сущность, у которой не нашли иерархию».

    Дефект ЗАМЕРЕН дважды. На «Вода гасит пламя»learner спрашивал «гасить»
    как сущность; канал 1 падал на Internal Server Error, слово получало
    asked_empty и больше никогда не спрашивалось. На «Гладкий шар катится по
    столу» «катиться» уезжало в Википедию за онтологией, которой для глагола
    не существует.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="l2verb-")
        self.path = os.path.join(self.dir, "lipid_world.json")
        write_world(self.path, {"entities": {}, "observations": {},
                                "unknown_words": {}})

    def _read(self):
        return read_world(self.path)

    def _learner(self, teacher=None):
        from brain.learner import Learner
        return Learner(load_world=self._read,
                       save_world=lambda w: write_world(self.path, w),
                       teacher=teacher)

    def test_verb_not_asked_as_entity(self):
        """Глагол НЕ гоняется через «онтологическую сущность»."""
        t = StubTeacher({})
        self._learner(t).learn_from_text("Вода гасит пламя.", source="t")
        self.assertNotIn("гасить", t.calls,
                         "глагол спросили как сущность — категориальная ошибка")

    def test_verb_never_gets_asked_empty(self):
        """Провал поиска «сущности» не имеет права вешаться на глагол."""
        self._learner(StubTeacher({})).learn_from_text("Вода гасит пламя.",
                                                        source="t")
        unknowns = self._read()["unknown_words"]
        self.assertNotIn("гасить", unknowns,
                         "глагол записан как невыученная сущность")

    def test_verb_becomes_action_node(self):
        """Глагол закрепляется узлом ACTION с участниками предложения."""
        rep = self._learner(StubTeacher({})).learn_from_text(
            "Вода гасит пламя.", source="t")
        ent = self._read()["entities"].get("гасить")
        self.assertIsInstance(ent, dict, "ACTION-узел глагола не создан")
        self.assertEqual(ent.get("kind"), "action")
        self.assertEqual(ent.get("hierarchy"), ["действие"])
        self.assertEqual(ent["participants"]["agent"], ["вода"])
        self.assertEqual(ent["participants"]["object"], ["пламя"])
        self.assertIn("гасить", rep.learned_actions,
                      "действие не попало в отдельную строку отчёта")
        self.assertNotIn("гасить", rep.learned_concepts,
                         "действие записано в выученные СУЩНОСТИ")

    def test_unknown_profile_is_announced_not_invented(self):
        """
        Профиля K/M/S в справочнике нет — его НЕ ВЫДУМЫВАЮТ, а громко говорят.
        Выдуманное число молча породило бы ложные предсказания предиктивного
        цикла, и проверить это было бы нечем.

        Глагол Примера — «шить»: его в справочнике нет и не было. (Раньше тут
        стояло «гасить» как пример пустоты; профиль пришёл с числом от
        пользователя, и тест стал бы лгать о справочнике.)
        """
        from core.actions import profile_of
        self.assertIsNone(profile_of("шить"),
                          "справочник внезапно обогатился неподтверждённым числом")
        rep = self._learner(StubTeacher({})).learn_from_text(
            "Портниха шьёт платье.", source="t")
        ent = self._read()["entities"].get("шить")
        self.assertIsInstance(ent, dict, "ACTION-узел обязан появиться и без числа")
        self.assertNotIn("action_profile", ent)
        self.assertTrue(any("профиля" in n for n in rep.notes),
                        f"отсутствие профиля промолчано: {rep.notes}")

    def test_known_verb_keeps_reference_profile(self):
        """
        Из справочника профиль берётся ДОСЛОВНО — перенос чисел не исказить.

        ЗАМЕРЕНО: форма «летит» даёт лемму «лететь», которой в справочнике нет
        (есть «летать»). Склеивать эти леммы молча я не вправе: справочник
        заполнялся по инфинитивам, и каждая новая форма обязана приходить с
        замером, а не «похоже, это то же действие». Пока это значит, что
        «летит» закрепится ACTION-узлом БЕЗ профиля — и скажет об этом в notes.
        """
        from core.actions import profile_of
        self._learner(StubTeacher({})).learn_from_text(
            "Птица летает над лугом.", source="t")
        prof = profile_of("летать")
        self.assertEqual(prof, [235, 70, 5],
                         "числа справочника изменены — это договор, а не догадка")
        # «гасить» подтверждён пользователем: [кинетика, стоимость сигнала,
        # метаболизм] = [20, 50, 10]. Проверяется ПОЗИЦИЯМИ, а не «на глаз».
        self.assertEqual(profile_of("гасить"), [20, 50, 10])
        ent = self._read()["entities"].get("летать")
        self.assertEqual((ent or {}).get("action_profile"), prof)

    def test_verb_form_outside_reference_stays_loud(self):
        """«летит» = лемма «лететь»: профиля нет, и это СЛЫШНО, а не тихо."""
        rep = self._learner(StubTeacher({})).learn_from_text(
            "Птица летит над лугом.", source="t")
        ent = self._read()["entities"].get("лететь")
        self.assertIsInstance(ent, dict, "ACTION-узел всё равно обязан появиться")
        self.assertNotIn("action_profile", ent)
        self.assertTrue(any("профиля" in n for n in rep.notes),
                        f"пустой профиль промолчан: {rep.notes}")

    def test_stale_asked_empty_on_verb_is_lifted(self):
        """
        Память, ИСПОРЧЕННАЯ прежним поведением, лечится: у глагола стоит
        asked_empty, выставленный неверным вопросом. Он снимается, а не
        уважается как «результат поиска».
        """
        write_world(self.path, {"entities": {}, "observations": {},
                                "unknown_words": {
                                    "гасить": {"status": "asked_empty",
                                               "count": 3,
                                               "evidence": "Вода гасит пламя"}}})
        self._learner(StubTeacher({})).learn_from_text("Вода гасит пламя.",
                                                       source="t")
        w = self._read()
        self.assertNotIn("гасить", w["unknown_words"],
                         "неверный диагноз пережил урок")
        self.assertIn("гасить", w["entities"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
