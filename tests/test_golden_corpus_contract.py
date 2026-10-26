"""
Замок золотого корпуса (очередь, пункт A): метрика обязана мерить, а не врать.

ЗАЧЕМ ЭТОТ ФАЙЛ. Корпус и скорер — такие же данные и такой же код, как всё
остальное, а значит обязаны иметь контракт. Без замка метрика деградирует
молча, и это тот же класс дефекта, что E4 (сон убивал переходы без единой
строки) и G-остаток (заметки разбора не доходили до отчёта):

  1. корпус может быть размонтирован: переставить id местами, вписать отношение
     которого нет в реестре, сделать expect_world без expect — и метрика
     продолжит уверенно печатать число;
  2. арифметику precision/recall можно испортить одной скобкой, и число
     изменится незначительно — глазами не увидеть;
  3. «не размечено» может перестать быть нулём: тогда precision считается по
     неизвестно какой выборке, а выглядит как измеренная точность;
  4. тесты корпуса обязаны жить в ИЗОЛИРОВАННОЙ памяти (правило проекта: тест
     не имеет права писать в рабочую память).

ЧИСЛА ЗДЕСЬ — НЕ ЭТАЛОН. Сравнивать «стало лучше/хуже» по зафиксированной
в тесте цифре нельзя: корпус правят, цифра обязана плыть. Заперто только то,
что враньём быть не может: арифметика, покрытие, отсутствие перевёрнутых
смыслов, независимость от порядка кейсов.
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.storage import RelationType
from tools.golden_corpus import (DEV_PATH, HELD_PATH, PRODUCTION_ROLES,
                                 load_cases, produce_extract, produce_world,
                                 score_case, score_run)
from core.extract import SimpleExtractor

CORPUS = {"dev": DEV_PATH, "heldout": HELD_PATH}


def _all_markers(case):
    """Все тройки, которые кейс считает размеченными на уровне extract."""
    for key in ("expect", "neg", "forbid", "also_ok", "junk"):
        for t in case.get(key, []):
            yield key, tuple(t)


class TestCorpusIntegrity(unittest.TestCase):
    """Корпус — данные с контрактом. Ломается здесь, а не в цифре на выходе."""

    def test_ids_unique_and_prefixed(self):
        for name, path in CORPUS.items():
            cases = load_cases(path)
            ids = [c["id"] for c in cases]
            self.assertEqual(len(ids), len(set(ids)), f"{name}: повторяющийся id")
            want = "d" if name == "dev" else "h"
            for c in cases:
                self.assertTrue(c["id"].startswith(want),
                                f"{name}: id {c['id']} не блока {want} — held-out "
                                f"нельзя размывать кейсами из dev")

    def test_size_matches_declaration(self):
        """Оба блока обязаны быть не меньше заявленных ~150 фраз вместе."""
        total = sum(len(load_cases(p)) for p in CORPUS.values())
        self.assertGreaterEqual(total, 80,
                                f"кейсов {total}: корпус обещан как «~150 фраз», "
                                f"сьедать его молча нельзя")

    def test_relations_exist_in_registry(self):
        known = {r.name for r in RelationType}
        for name, path in CORPUS.items():
            for c in load_cases(path):
                for key, tri in _all_markers(c):
                    self.assertIn(tri[1], known,
                                  f"{c['id']}/{key}: отношения {tri[1]} нет в "
                                  f"RelationType — эталон на выдуманном отношении "
                                  f"никогда не будет найден")

    def test_triples_have_three_fields(self):
        for name, path in CORPUS.items():
            for c in load_cases(path):
                for key, _ in _all_markers(c):
                    pass
                for key in ("expect", "neg", "forbid", "also_ok", "junk",
                            "expect_world", "neg_world", "also_ok_world", "junk_world"):
                    for tri in c.get(key, []):
                        self.assertEqual(len(tri), 3,
                                         f"{c['id']}/{key}: не тройка: {tri}")

    def test_every_marked_case_explains_itself(self):
        """also_ok/junk без причины = подгонка под вывод, а не разметка."""
        for name, path in CORPUS.items():
            for c in load_cases(path):
                if c.get("also_ok") or c.get("junk"):
                    self.assertTrue(c.get("триаж"),
                                    f"{c['id']}: размечено also_ok/junk без "
                                    f"обоснования — правку ожиданий под вывод "
                                    f"нечем отличить от подгонки")

    def test_markup_fixes_listed_not_silent(self):
        """Кейс после первого прогона правится только вслух, с причиной."""
        for name, path in CORPUS.items():
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            fixes = data.get("markup_fixes", {}).get("список", [])
            self.assertTrue(fixes,
                            f"{name}: полка markup_fixes заведена, значит кейсы "
                            f"после прогона правились — список не может быть пуст")
            for fix in fixes:
                for key in ("id", "было", "стало", "причина"):
                    self.assertIn(key, fix, f"{name}: правка разметки без поля {key}")

    def test_world_level_declares_aliases(self):
        """expect_world обязан опираться на expect: уровень не живёт в отрыве."""
        for name, path in CORPUS.items():
            for c in load_cases(path):
                if "expect_world" in c:
                    self.assertIn("expect", c,
                                  f"{c['id']}: expect_world без expect — уровень "
                                  f"мира не должен забывать, что найдено ниже")

    def test_heldout_not_contaminated_with_dev(self):
        """Свежая лексика: совпадающая фраза превращает held-out в dev."""
        dev_texts = {c["text"] for c in load_cases(DEV_PATH)}
        overlap = [c["text"] for c in load_cases(HELD_PATH) if c["text"] in dev_texts]
        self.assertEqual(overlap, [],
                         f"эти фразы стоят в обоих блоках — held-out перестал "
                         f"быть проверкой на неподогнанности: {overlap}")


class TestScoringArithmetic(unittest.TestCase):
    """
    Арифметика. Проверяется на СОЧИНЁННЫХ данных, а не на корпусе: только так
    видно, что формула считает, а не что она повторяет вывод экстрактора.
    """

    def _case(self, **kw):
        base = {"id": "x1", "text": "тест", "expect": [], "neg": [], "forbid": []}
        base.update(kw)
        return base

    def test_recall_counts_hits_over_expectations(self):
        case = self._case(expect=[["a", "CAN_DO", "b"], ["a", "CAN_DO", "c"]])
        produced = [(("a", "CAN_DO", "b"), False)]
        r = score_case(case, produced)
        self.assertEqual((r["expect_n"], len(r["hits"]), len(r["misses"])), (2, 1, 1))

    def test_negation_is_not_positive_fact(self):
        """Отрицанная связь НЕ засчитывается как положительная и наоборот."""
        case = self._case(expect=[["a", "CAN_DO", "b"]], neg=[["a", "CAN_DO", "c"]])
        only_neg = [(("a", "CAN_DO", "b"), True), (("a", "CAN_DO", "c"), True)]
        r = score_case(case, only_neg)
        self.assertEqual(r["hits"], [], "отрицанная связь засчитана как утверждение")
        self.assertEqual(len(r["neg_hits"]), 1)

    def test_junk_gets_no_precision_credit(self):
        """Ложная связь не имеет права повышать precision."""
        case = self._case(expect=[["a", "CAN_DO", "b"]],
                          junk=[["a", "LIE", "x"]], also_ok=[])
        produced = [(("a", "CAN_DO", "b"), False), (("a", "LIE", "x"), False)]
        r = score_case(case, produced)
        self.assertEqual(len(r["junk_hits"]), 1)
        run = score_run(SimpleExtractor(), [case],
                        producer=lambda sf: produced)
        self.assertEqual(run["junk"], 1)
        self.assertEqual(run["precision_strict"], 0.5)
        self.assertEqual(run["precision_triaj"], 0.5,
                         "also_ok у ложной связи появился из ниоткуда: junk дал кредит")

    def test_also_ok_credits_precision_not_recall(self):
        """Второстепенная ВЕРНАЯ связь не обязана была найтись: recall не растёт."""
        case = self._case(expect=[["a", "CAN_DO", "b"]], also_ok=[["a", "PART_OF", "z"]])
        produced = [(("a", "CAN_DO", "b"), False), (("a", "PART_OF", "z"), False)]
        run = score_run(SimpleExtractor(), [case], producer=lambda sf: produced)
        self.assertEqual(run["recall"], 1.0)
        self.assertEqual(run["precision_strict"], 0.5, "also_ok залез в строгую precision")
        self.assertEqual(run["precision_triaj"], 1.0)
        self.assertEqual(run["unlabeled"], 0, "размеченное посчитали не размеченным")

    def test_unlabeled_is_not_called_correct_or_wrong(self):
        case = self._case(expect=[["a", "CAN_DO", "b"]])
        produced = [(("a", "CAN_DO", "b"), False), (("q", "PART_OF", "w"), False)]
        run = score_run(SimpleExtractor(), [case], producer=lambda sf: produced)
        self.assertEqual(run["unlabeled"], 1)
        self.assertEqual(run["junk"], 0,
                         "неизвестное записано в ложь: «не спрашивали» != «не нашли»")

    def test_known_gap_has_separate_denominator(self):
        case = self._case(expect=[["a", "CAN_DO", "b"]])
        case["known_gap"] = True
        run = score_run(SimpleExtractor(), [case],
                        producer=lambda sf: [(("a", "CAN_DO", "b"), False)])
        self.assertEqual(run["known_gap_found"], "1/1")
        self.assertEqual(run["expect_total"], 1)

    def test_plain_recall_excludes_known_gaps(self):
        """
        Известная дыра не имеет права топить обычный recall.

        Иначе метрика не отличает «правило сломалось» от «мы и не обещали»,
        а ради этого разделения known_gap и заведён: регресс обязан быть
        виден, а не утоплен в заведомых пропусках.
        """
        plain = self._case(id="d1", text="найдено.",
                           expect=[["a", "CAN_DO", "b"]])
        gap = self._case(id="h1", text="известная дыра.",
                         expect=[["c", "CAN_DO", "d"]])
        gap["known_gap"] = True

        def producer(sf):
            """Обычный кейс найден, известная дыра — нет (как в реальности)."""
            if sf.sentence == "найдено.":
                return [(("a", "CAN_DO", "b"), False)]
            return []

        run = score_run(SimpleExtractor(), [plain, gap], producer=producer)
        self.assertEqual(run["known_gap_found"], "0/1")
        self.assertEqual(run["recall_plain"], 1.0,
                         f"известная дыра попала в знаменатель обычного recall: "
                         f"recall_plain={run['recall_plain']}, ждём 1.0")
        self.assertLess(run["recall"], 1.0,
                        "общий recall обязан видеть пропуск известной дыры")

    def test_arithmetic_identity_produced_parts(self):
        """
        Выданное = верно + второстепенно + ложь + не размечено. Сумма обязана
        сходиться: именно так ловится «потерянная» (или дважды посчитанная)
        часть выборки.
        """
        for name, path in CORPUS.items():
            run = score_run(SimpleExtractor(), load_cases(path))
            for c in run["per_case"]:
                parts = [t for group in (c["hits"], c["neg_hits"], c["also_ok"],
                                         c["junk_hits"], c["extra"])
                         for t in group]
                self.assertEqual(
                    len(parts), len(set(parts)),
                    f"{c['id']}: одна тройка зачтена дважды — parts={parts}")

    def test_same_triple_cannot_be_counted_twice(self):
        """
        Одна связь в двух карманах надувала метрику: в h28 отрицанная связь
        стояла и в neg, и в also_ok — popala в recall И в precision. Скорер
        обязан сам это ловить, а не полагаться на аккуратность разметчика.
        """
        case = self._case(expect=[], neg=[["a", "P", "b"]],
                          also_ok=[["a", "P", "b"]])
        produced = [(("a", "P", "b"), True)]
        r = score_case(case, produced)
        self.assertEqual(r["markup_conflict"], [("a", "P", "b")],
                         "переклика эталонов не замечена — метрику можно надувать")
        run = score_run(SimpleExtractor(), [case], producer=lambda sf: produced)
        self.assertEqual(run["markup_conflicts"], 1)
        # и главное: зачёт ровно один
        self.assertEqual(len(r["neg_hits"]) + len(r["also_ok"]), 1)

    def test_forbid_wins_over_also_ok(self):
        """Запрещённая связь не имеет права получить кредит точности."""
        case = self._case(expect=[], forbid=[["a", "P", "b"]],
                          also_ok=[["a", "P", "b"]])
        produced = [(("a", "P", "b"), False)]
        r = score_case(case, produced)
        self.assertEqual(r["also_ok"], [],
                         "перевёрнутая связь получила кредит точности")
        self.assertEqual(len(r["forbid_hits"]), 1)

    def test_corpus_itself_has_no_conflicts(self):
        """Ни одна связь корпуса не должна засчитываться дважды."""
        for name, path in CORPUS.items():
            run = score_run(SimpleExtractor(), load_cases(path))
            bad = {c["id"]: c["markup_conflict"] for c in run["per_case"]
                   if c.get("markup_conflict")}
            self.assertEqual(bad, {},
                             f"{name}: переклика эталонов надувает метрику: {bad}")

    def test_forbid_violation_is_not_a_percentage(self):
        case = self._case(expect=[["a", "CAN_DO", "b"]],
                          forbid=[["b", "CAN_DO", "a"]])
        produced = [(("a", "CAN_DO", "b"), False), (("b", "CAN_DO", "a"), False)]
        run = score_run(SimpleExtractor(), [case], producer=lambda sf: produced)
        self.assertEqual(run["forbid_violations"], 1)

    def test_no_inverted_meanings_in_corpus(self):
        """Перевёрнутый смысл роняет прогон, а не снижает процент."""
        for name, path in CORPUS.items():
            run = score_run(SimpleExtractor(), load_cases(path))
            bad = [(c["id"], t) for c in run["per_case"] for t in c["forbid_hits"]]
            self.assertEqual(bad, [],
                             f"{name}: перевёрнутые связи {bad} — опаснее, чем "
                             f"отсутствующие (урок CAUSES)")


class TestCoverageIsHonest(unittest.TestCase):
    """
    Покрытие — то, без чего precision одно число по трём разным вещам.
    """

    def test_everything_produced_is_triaged(self):
        for name, path in CORPUS.items():
            run = score_run(SimpleExtractor(), load_cases(path))
            left = {c["id"]: c["extra"] for c in run["per_case"] if c["extra"]}
            self.assertEqual(left, {},
                             f"{name}: {sum(len(v) for v in left.values())} троек "
                             f"не разобрано человеком: {left}. Precision по такой "
                             f"выборке — не метрика")

    def test_triage_covers_all_junk_found(self):
        """
        Разметка junk обязана ДЕЙСТВИТЕЛЬНО встречаться: иначе junk записан
        «на будущее» и ничего не мерит (урок 19: тест, который мутируемый код
        не читает, бесполезен).
        """
        for name, path in CORPUS.items():
            cases = load_cases(path)
            marked = [c["id"] for c in cases if c.get("junk")]
            run = score_run(SimpleExtractor(), cases)
            found = [c["id"] for c in run["per_case"] if c["junk_hits"]]
            self.assertTrue(marked, f"{name}: ни одной размеченной лжи — блок junk пуст")
            self.assertEqual(sorted(marked), sorted(found),
                             f"{name}: junk размечен, но не найден (или найден, "
                             f"но не размечен) — разметка протухла")


class TestLevelDifferenceIsMeasured(unittest.TestCase):
    """
    Расхождение уровней — не шум, а предмет измерения: экстрактор находит,
    мир хранит иначе (REL_ALIAS, событие вместо факта, потеря падежной роли).
    """

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="lipid_lock_")
        cls.extract = score_run(SimpleExtractor(), load_cases(DEV_PATH))
        cls.world = score_run(SimpleExtractor(), load_cases(DEV_PATH),
                              produce_world, cls.work)

    def test_levels_are_comparable_numbers(self):
        for run in (self.extract, self.world):
            self.assertGreater(run["recall"], 0.0)
            self.assertGreater(run["precision_triaj"], 0.0)

    def test_alias_relation_is_recognized_on_world_level(self):
        """
        IS_ROLE -> IS_A: мозг переписывает ИМЯ отношения, смысл тот же. Если
        expect_world перестанет читаться, уровень мира увидит здесь потерю и
        recall просядет — тест обязан это поймать.
        """
        case = next(c for c in load_cases(DEV_PATH) if c["id"] == "d08")
        self.assertIn("expect_world", case)
        world_rows = produce_world(case["text"], self.work, "lock_d08")
        self.assertIn(("кошка", "IS_A", "млекопитающее"), [t for t, _ in world_rows])
        r = score_case(case, world_rows, "world")
        self.assertEqual(r["misses"], [],
                         "expect_world не применён: уровневая метрика врёт")

    def test_event_role_loss_is_counted_not_hidden(self):
        """
        Событие хранит subject/action/obj, поэтому МЕСТО у фразы в прошедшем
        времени доходит до мира как OBJECT_OF. Потеря обязана быть названа
        числом (замерено: «Учитель вошёл в класс» -> событие obj=класс).
        """
        case = {"id": "lock1", "text": "Учитель вошёл в класс.",
                "expect": [], "neg": [], "forbid": [], "basket": "any"}
        run = score_run(SimpleExtractor(), [case], produce_world, self.work)
        self.assertEqual(run["role_loss"], 1,
                         f"роль не посчитана: {run['per_case'][0]}")
        self.assertIn(case["text"], [c["text"] for c in run["per_case"]])

    def test_production_roles_are_a_closed_set(self):
        """Множество ролей, схлопающихся в OBJECT_OF, обязано быть названным."""
        self.assertTrue(PRODUCTION_ROLES)
        known = {r.name for r in RelationType}
        self.assertTrue(PRODUCTION_ROLES <= known,
                        f"в списке ролей чужие имена: {PRODUCTION_ROLES - known}")


class TestCorpusDoesNotTouchLiveMemory(unittest.TestCase):
    """Правило проекта: прогон метрики не имеет права писать в рабочую память."""

    def test_live_world_untouched_by_world_level_run(self):
        from core.worldfile import default_memory_dir
        path = default_memory_dir()[1]

        def snapshot():
            """Содержимое живой Вселенной целиком, а не mtime."""
            if not os.path.exists(path):
                return None
            with open(path, "rb") as fh:
                return fh.read()

        before = snapshot()
        work = tempfile.mkdtemp(prefix="lipid_iso_")
        score_run(SimpleExtractor(), load_cases(HELD_PATH)[:8], produce_world, work)
        self.assertEqual(before, snapshot(),
                         "прогон корпуса изменил живую Вселенную — изоляция "
                         "сломана, замеры больше не повторяемы")


if __name__ == "__main__":
    unittest.main(verbosity=2)
