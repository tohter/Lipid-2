"""
tests/test_cause_extraction_contract.py — замок очереди G: причина из текста.

ЗАМЕР (experiments/pG_causes_measure.py, 25.10) показал, что дефект острее
постановки «экстрактор НЕ порождает CAUSES»:

  1. Причина не теряется — она записывается ЛОЖЬЮ: «Пожар гаснет из-за дождя»
     давало «гаснуть --LOCATED_IN--> дождь» (predicate=fact, confidence 0.85,
     сквозным прогоном через мозг связь легла в observations мира).
  2. «вследствие», «ввиду», «благодаря» вообще НЕ были предлогами для
     экстрактора (нет в PREP_CASES, ALL_PREPS = set(PREP_CASES)): отношение
     выходило наугад — OBJECT_OF, RECIPIENT.
  3. Ориентация. В данных мира CAUSES направлен «причина -> следствие»
     (cat_canary_world.json: предательство -> обида -> гнев). Переименование
     прежнего хвоста без переворота дало бы «погаснуть --CAUSES--> дождь» =
     «гаснуть ВЫЗЫВАЕТ дождь»: причину шиворот-навыворот. Поэтому замок на
     НАПРАВЛЕНИЕ здесь важнее замка на само отношение.

ДВУСМЫСЛЕННОЕ («из-за», «от») намеренно НЕ тронуто: корпус замера — место
7/7 и причина 6/6 дают ОДНО И ТО ЖЕ отношение, морфология неразличима
(«угла»/«дождя» оба неодушевлённые). Сменить дефолт молча = соврать одной из
групп; это решение хозяина (записано в memory.txt, ЖДЁТ РЕШЕНИЯ). Замок ниже
держит СТАТУС-КВО, чтобы смена произошла только вместе с решением.

Числа тест НЕ хранит в голове: ИТОГ пересобирается скриптом-замером командой
(урок 16: цифра без команды замера — догадка).
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.extract import PREP_CASES, SimpleExtractor        # noqa: E402
from brain.simulator import StaticKnowledge                 # noqa: E402

MEASURE = os.path.join(ROOT, "experiments", "pG_causes_measure.py")

ex = SimpleExtractor()

# Однозначные причинные предлоги: место они выразить НЕ МОГУТ никогда,
# поэтому перекосить на них что-либо безопаснее, чем оставить OBJECT_OF наугад.
UNAMBIG = ("вследствие", "ввиду", "благодаря")


def triples(sentence):
    return [(f.head, f.relation, f.tail) for f in ex.extract(sentence, order=1).facts]


class TestCausePrepositionsAreData(unittest.TestCase):
    """Причинные предлоги — строка в таблице, а не ветка кода."""

    def test_unambiguous_cause_prepositions_are_registered(self):
        for p in UNAMBIG:
            self.assertIn(p, PREP_CASES,
                          f"«{p}» не в PREP_CASES — экстрактор не считает его "
                          f"предлогом и отношениe берёт наугад (замер G2)")
            cases, rel = PREP_CASES[p]
            self.assertEqual(rel, "CAUSES", f"«{p}» обязан давать CAUSES")
            # Падежи РАЗНЫЕ: «вследствие/ввиду» + род., «благодаря» + дат.
            # (pymorphy: «дождя» gen, «тренеру» dat). Свести их к одному
            # родовому «gen» нельзя — предлог перестал бы узнавать свою форму.
            self.assertEqual(cases, {"вследствие": ("gen",),
                                     "ввиду": ("gen",),
                                     "благодаря": ("dat",)}[p],
                             f"неожиданный падеж у «{p}»: {cases}")

    def test_cause_preposition_is_not_a_place(self):
        """prep_is_place обязан отвечать False: иначе придаточная ветка
        «вследствие» сочла бы местом и повела бы себя иначе, чем плоская."""
        from core.extract import prep_is_place
        for p in UNAMBIG:
            self.assertFalse(prep_is_place(p), f"«{p}» не место")


class TestCauseDirection(unittest.TestCase):
    """
    Направление CAUSES: голова — ПРИЧИНА, хвост — СЛЕДСТВИЕ (действие).

    Сверено с ориентацией ручных данных мира (предательство --CAUSES--> обида).
    Наивная правка без переворота давала «погаснуть --CAUSES--> дождь».
    """

    def test_vseledstvie_yields_cause_first(self):
        got = triples("Пожар погас вследствие дождя.")
        self.assertIn(("дождь", "CAUSES", "погаснуть"), got,
                      f"причина обязана быть головой: {got}")
        self.assertNotIn(("погаснуть", "CAUSES", "дождь"), got,
                         "перевёрнутая причина: «гаснуть вызывает дождь»")
        self.assertNotIn(("погаснуть", "OBJECT_OF", "дождь"), got,
                         "старая ложь вернулась: следствие как объект")

    def test_vvidu_keeps_lemma_not_flexed_form(self):
        """Замер: старый OBJECT_OF нёс хвост «жары» — форму, а не лемму."""
        got = triples("Урожай сгорел ввиду жары.")
        self.assertIn(("жара", "CAUSES", "сгореть"), got,
                      f"хвост обязан быть леммой и причиной: {got}")
        for h, r, t in got:
            self.assertNotEqual(t, "жары", f"флексия в графе: {(h, r, t)}")
            self.assertNotEqual(h, "жары")

    def test_blagodarya_is_not_recipient(self):
        got = triples("Успех пришёл благодаря тренеру.")
        self.assertIn(("тренер", "CAUSES", "прийти"), got,
                      f"{got}")
        self.assertNotIn(("прийти", "RECIPIENT", "тренер"), got,
                         "старая ложь вернулась: тренер — получатель прихода")

    def test_cause_confidence_clears_learner_floor(self):
        """
        Факт с confidence ниже 0.30 обучающий отсекает МОЛЧА (min_confidence,
        brain/learner.py). Замок не даёт правке породить CAUSES, который никуда
        не дойдёт: отношение появилось бы в экстракторе и исчезло в мозге.
        """
        for s in ("Пожар погас вследствие дождя.",
                  "Урожай сгорел ввиду жары.",
                  "Успех пришёл благодаря тренеру."):
            for f in ex.extract(s, order=1).facts:
                if f.relation == "CAUSES":
                    self.assertGreaterEqual(
                        f.confidence, 0.30,
                        f"CAUSES из «{s}» отсекается фильтром и мир его не увидит")


class TestAmbiguousStaysOwnerDecision(unittest.TestCase):
    """
    «из-за» и «от» НЕ переведены в CAUSES: корпус замера — МЕСТО 7/7,
    ПРИЧИНА 6/6, отношение ОДНО И ТО ЖЕ, морфология молчит. Дефолт выбирает
    хозяин; замок держит статус-кво, чтобы смена была только с решением.
    """

    def test_place_reading_untouched(self):
        got = triples("Мальчик выглянул из-за угла.")
        self.assertIn(("выглянуть", "LOCATED_IN", "угол"), got,
                      f"место сломано правкой: {got}")
        self.assertFalse([t for t in got if t[1] == "CAUSES"],
                         f"место стало причиной без решения хозяина: {got}")

    def test_ambiguous_cause_still_located_in(self):
        got = triples("Пожар гаснет из-за дождя.")
        self.assertIn(("гаснуть", "LOCATED_IN", "дождь"), got,
                      f"статус-кво сменён молча — это решение хозяина: {got}")
        self.assertFalse([t for t in got if t[1] == "CAUSES"])

    def test_ambiguous_prepositions_are_marked_as_open(self):
        """Статус-кво обязан быть ВИДЕН как класс данных, а не забыт:
        множество открыто и перечисляет спорные предлоги."""
        from core.extract import AMBIGUOUS_CAUSE_PREPS
        self.assertEqual(AMBIGUOUS_CAUSE_PREPS, frozenset({"из-за", "от"}))
        for p in AMBIGUOUS_CAUSE_PREPS:
            self.assertEqual(PREP_CASES[p][1], "LOCATED_IN",
                             "дефолт спорных предлогов — предмет решения, "
                             "менять можно только вместе с memory.txt")


class TestPastTenseCauseIsStillRecorded(unittest.TestCase):
    """
    ОСТАТОК, найденный СКВОЗНЫМ прогоном, а не чтением экстрактора.

    Прошедшее время («погас») разводится как СОБЫТИЕ (predicate='event'), и
    обучающий не пускает событие в онтологию — принцип «из одного случая
    знание не рождается». Это ВЕРНО, но событийная запись хранит только
    subject/action/obj, поэтому причина в записи не помещалась НИКУДА: ни в
    observations, ни в events. Предложение с причиной проходило урок и мир о
    причине молчал.

    ЗДЕСЬ ЗАПЕРТО то, что причина событие ПЕРЕЖИВАЕТ (поле "cause"), а НЕ то,
    что она стала знанием: перевести событийную причину в CAUSES нельзя без
    нарушения принципа выше — это отдельное решение хозяина (memory.txt, ЖДЁТ).
    """

    @staticmethod
    def _learn(text):
        """Без диска и без живой памяти: load_world/save_world в памяти теста."""
        from brain.learner import Learner
        world = {"entities": {}, "observations": {}, "unknown_words": {}}
        lr = Learner(load_world=lambda: dict(world),
                     save_world=lambda w: world.update(w), teacher=None)
        lr.learn_from_text(text, source="text", allow_teacher=False)
        return world

    def test_event_keeps_its_cause(self):
        world = self._learn("Пожар погас вследствие дождя.")
        events = world.get("events") or []
        self.assertTrue(events, "событие не записано — проверять нечего")
        self.assertEqual(events[0].get("cause"), "дождь",
                         f"причина события потеряна: {events[0]}")

    def test_ontology_still_not_polluted_by_single_event(self):
        """Причина ОДНОКРАТНОГО события не обязана становиться законом мира."""
        world = self._learn("Пожар погас вследствие дождя.")
        rels = [v.get("relation") for v in (world.get("observations") or {}).values()
                if isinstance(v, dict)]
        self.assertNotIn("CAUSES", rels,
                         "однократное событие записано как знание — нарушение "
                         "принципа «из одного случая закон не рождается»")

    def test_generic_event_without_cause_got_no_empty_field(self):
        """Прошедшее время БЕЗ причины: событие есть, пустого поля нет.
        Настоящее время («спит») событием не считается — подставил бы его,
        тест skipped-омся и ничего бы не запирал."""
        world = self._learn("Пёс спал на подоконнике.")
        events = world.get("events") or []
        self.assertTrue(events, "обычное событие не записано — замок пуст")
        self.assertNotEqual(events[0].get("cause"), "",
                            "пустой cause раздувает формат без нужды")
        self.assertNotIn("cause", events[0],
                         f"пустое поле всё же записано: {events[0]}")


class TestConsumerActuallyReadsIt(unittest.TestCase):
    """CAUSES не самоцель: симулятор обязан до него дотянуться."""

    def test_simulator_reads_produced_causes(self):
        facts = ex.extract("Пожар погас вследствие дождя.", order=1).facts
        world = {"observations": {}}
        for i, f in enumerate(facts):
            world["observations"][f"{i}"] = {"head": f.head,
                                             "relation": f.relation,
                                             "tail": f.tail}
        sk = StaticKnowledge(world)
        self.assertEqual(sk.causes("дождь"), ["погаснуть"],
                         "порождённый CAUSES не читается StaticKnowledge — "
                         "значит правка производит мусор, а не знание")


class TestMeasurementIsReproducible(unittest.TestCase):
    """
    Числа замороженного замера обязан воспроизводить скрипт, а не этот файл
    (урок 16). Строка ИТОГ пересобирается командой.
    """

    @classmethod
    def setUpClass(cls):
        env = dict(os.environ, LIPID2_STATUS="0")
        r = subprocess.run([sys.executable, MEASURE], capture_output=True,
                           text=True, cwd=ROOT, env=env, timeout=300)
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("ИТОГ ")]
        if not line:
            raise AssertionError(
                f"замер не напечатал ИТОГ (rc={r.returncode}):\n{r.stdout[-400:]}\n"
                f"{r.stderr[-400:]}")
        cls.s = dict(kv.split("=", 1) for kv in line[-1].split()[1:])

    def test_ambiguous_causes_are_still_zero(self):
        """Ни одного CAUSES из спорных предлогов — ровно 0 (замер корпуса)."""
        self.assertEqual(int(self.s["poly_causes"]), 0)

    def test_cause_prepositions_now_produce_causes(self):
        """Три однозначных предлога дают ровно три CAUSES (после правки)."""
        self.assertEqual(int(self.s["unambig_causes"]), 3)

    def test_relations_by_name(self):
        for key, want in (("vseledstvie", "CAUSES"), ("vvidu", "CAUSES"),
                          ("blagodarya", "CAUSES"),
                          ("izza", "LOCATED_IN"), ("ugol", "LOCATED_IN")):
            self.assertEqual(self.s[key], want,
                             f"{key}={self.s[key]}, ожидалось {want}")

    def test_place_corpus_untouched(self):
        """Корпус места: 7 связей LOCATED_IN (замер), причина: 6 — статус-кво."""
        self.assertEqual(int(self.s["place_located"]), 7)
        self.assertEqual(int(self.s["cause_poly_located"]), 6)

    def test_manual_world_count(self):
        """Миров с ручным CAUSES ровно 1 — от него проверяется ориентация."""
        self.assertEqual(int(self.s["manual_worlds"]), 1)

    def test_manual_world_detector_is_format_not_substring(self):
        """
        Счётчик «мир с CAUSES» обязан смотреть ФОРМАТ, а не искать подстроку.

        ЗАМЕРЕНО: прежний вариант («любой .json со строкой CAUSES») поднял
        manual_worlds с 1 до 5 от одного соседнего файла — отчёта метрики,
        где CAUSES всего лишь имя отношения. Тест обязан упасть и на grep'e,
        и на пустом корпусе: для этого один файл с CAUSES в ТЕКСТЕ фразы и
        один мир с CAUSES в observations.
        """
        import importlib.util
        import shutil
        spec = importlib.util.spec_from_file_location("pGm", MEASURE)
        # модуль измерителя при импорте выполняет замер, поэтому функцию
        # достаём текстом и исполняем в чистом пространстве имён
        with open(MEASURE, encoding="utf-8") as fh:
            src = fh.read()
        start = src.index("def manual_cause_worlds(")
        end = src.index("worlds = manual_cause_worlds(")
        ns = {"json": json, "_os": os}
        exec(src[start:end], ns)
        detect = ns["manual_cause_worlds"]

        tmp = tempfile.mkdtemp(prefix="pG_detect_")
        try:
            # мир в формате мира: observations + relation == CAUSES
            with open(os.path.join(tmp, "real_world.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"observations": {"a|CAUSES|b": {"head": "a",
                                                           "relation": "CAUSES",
                                                           "tail": "b"}}}, fh)
            # НЕ мир: CAUSES упоминается как текст/имя отношения вне формата
            with open(os.path.join(tmp, "report.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"by_rel": {"CAUSES": "0/1"},
                           "cases": [{"text": "A CAUSES B"}]}, fh)
            self.assertEqual(detect(tmp), ["real_world.json"],
                             "счётчик считает миром любой файл со словом "
                             "CAUSES — метрика зависит от соседних файлов")
            # и mutation-guard: пустой каталог обязан давать пусто, а не «1»
            self.assertEqual(detect(tempfile.mkdtemp(prefix="pG_empty_")), [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_past_tense_cause_survives(self):
        """Причина прошедшего времени переживает урок во всех 3 случаях.
        До правки замер давал 0 из 3: событие теряло причину бесследно."""
        self.assertEqual(int(self.s["past_total"]), 3)
        self.assertEqual(int(self.s["past_kept"]), 3,
                         "событие снова теряет причину")


if __name__ == "__main__":
    unittest.main()
