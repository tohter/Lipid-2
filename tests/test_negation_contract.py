"""
КОНТРАКТ: ОБЛАСТЬ ДЕЙСТВИЯ «НЕ» И НУЛЕВАЯ СВЯЗКА (очередь: баг «не A, а B»).

БАГ, ЗАПРЕЩЁННЫЙ ЭТИМ ФАЙЛОМ (найден задачей 35, записан в memory.txt как
«НАЙДЕН И НЕ ИСПРАВЛЕН»):

    «Кит — это не рыба, а млекопитающее»  давало  ДВА факта с negated=True:
        кит --IS_ROLE /НЕ/--> рыба
        кит --IS_ROLE /НЕ/--> млекопитающее      <-- «кит НЕ млекопитающее»

    Ложное отрицание хуже пропущенного: отрицанная связь хранится и в
    observations (negated=True), и опыта ей хватает — значит потом её НЕ
    поднимет ни консольдация, ни инспектор конфликтов (оба пропускают
    negated). Ошибка осела бы в базе навсегда.

ПРИЧИНА (одна строка): `neg = any(t["word"] in NEGATIONS for t in toks)` —
отрицание считалось по всему предложению и предъявлялось каждой связи.

ВТОРОЙ ДЕФЕКТ, пойманный попутно (замер, а не теория): в «Кит не рыба,
а млекопитающее» сказуемого нет вовсе, вершиной становилось подлежащее, и
именная часть уезжала как OBJECT_OF:

        кит --OBJECT_OF--> млекопитающее    («кит — объект действия»)

хотя при наличии тире/«это» та же связь выходила IS_ROLE. То есть ОТНОШЕНИЕ
выбирала ПУНКТУАЦИЯ, а не разбор.

Тесты НЕ запускают LLM и не пишут в рабочую память.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.extract import (  # noqa: E402
    SimpleExtractor, NegationScope, ZERO_COPULA, analyze,
)


def facts_of(sentence):
    return SimpleExtractor().extract(sentence)


def triples(sentence):
    return {(f.head, f.relation, f.tail) for f in facts_of(sentence).facts}


def by_tail(sentence, tail):
    return [f for f in facts_of(sentence).facts if f.tail == tail]


class TestNegativeContrast(unittest.TestCase):
    """«не A, а B»: отрицается ТОЛЬКО A."""

    def test_contrast_denies_only_first_term(self):
        sf = facts_of("Кит — это не рыба, а млекопитающее.")
        fish = by_tail("Кит — это не рыба, а млекопитающее.", "рыба")
        mamm = by_tail("Кит — это не рыба, а млекопитающее.", "млекопитающее")
        self.assertTrue(fish, "отрицаемый член обязан остаться в разборе")
        self.assertTrue(fish[0].negated, "«не рыба» — рыба отрицается")
        self.assertTrue(mamm, "утверждаемый член обязан быть в разборе")
        self.assertFalse(mamm[0].negated,
                         "ГЛАВНЫЙ ЗАПРЕТ: «а млекопитающее» — УТВЕРЖДЕНИЕ, "
                         "отрицание к нему переходить не имеет права")
        self.assertEqual(mamm[0].relation, "IS_ROLE")
        self.assertEqual(sf.notes and any("УТВЕРЖДАЮТСЯ" in n for n in sf.notes),
                         True, "уточнение области обязано быть ВИДНО в отчёте")

    def test_contrast_without_dash_and_copula(self):
        """Тот же смысл без тире и «это»: отношение не должно зависеть от
        пунктуации (прежний дефект: без связки выходил OBJECT_OF)."""
        t = triples("Кит не рыба, а млекопитающее.")
        self.assertIn(("кит", "IS_ROLE", "млекопитающее"), t)
        self.assertNotIn(("кит", "OBJECT_OF", "млекопитающее"), t,
                         "именная часть НЕ является объектом действия")
        self.assertFalse(by_tail("Кит не рыба, а млекопитающее.",
                                 "млекопитающее")[0].negated)

    def test_contrast_with_adjective_predicate(self):
        t = triples("Кошка — не собака, а хищник.")
        self.assertIn(("кошка", "IS_ROLE", "хищник"), t)
        self.assertFalse(by_tail("Кошка — не собака, а хищник.",
                                 "хищник")[0].negated)

    def test_prologue_outside_scope_stays_positive(self):
        """«Столица Франции — не Париж, а Лион»: пролог ДО частицы не
        отрицается (PART_OF остаётся знанием, а не отрицанием)."""
        s = "Столица Франции — не Париж, а Лион."
        of = by_tail(s, "франция")
        self.assertTrue(of)
        self.assertFalse(of[0].negated,
                         "«столица Франции» — не то, что отрицает «не»")
        self.assertTrue(by_tail(s, "париж")[0].negated)
        self.assertFalse(by_tail(s, "лион")[0].negated)

    def test_inserted_contrast_does_not_negate_main_clause(self):
        """«Кошка, а не собака, ловит мышь»: вставной член закрыт, действие
        утверждается. Иначе записалось бы «кошка не ловит мышь» — наоборот."""
        s = "Кошка, а не собака, ловит мышь."
        self.assertFalse(by_tail(s, "ловить")[0].negated)
        self.assertFalse(by_tail(s, "мышь")[0].negated)

    def test_fronted_negation_denies_whole_first_clause(self):
        """«Не кошка поймала мышь, а собака» — событие с кошкой ОТРИЦАЕТСЯ."""
        sf = facts_of("Не кошка поймала мышь, а собака.")
        self.assertTrue(any(f.negated for f in sf.facts),
                        "без частицы в базе осталось бы «кошка поймала мышь»")
        self.assertTrue(sf.events and sf.events[0].negated,
                        "отрицанное событие обязано нести флаг")


class TestPlainNegationUnchanged(unittest.TestCase):
    """БЕЗ противопоставления область = всё предложение (прежнее поведение)."""

    def test_plain_negation_still_covers_dependents(self):
        s = "Кошка не ест рыбу."
        self.assertTrue(by_tail(s, "есть")[0].negated)
        self.assertTrue(by_tail(s, "рыба")[0].negated,
                        "отрицание глагола отрицает и его объект: иначе в граф "
                        "попало бы «кошка ест рыбу»")

    def test_no_contrast_no_scope_note(self):
        sf = facts_of("Кошка не спит на диване.")
        self.assertFalse(any("УТВЕРЖДАЮТСЯ" in n for n in sf.notes),
                         "заметка про область положена только там, где она "
                         "что-то изменила")

    def test_positive_sentence_untouched(self):
        for s in ("Кит — это млекопитающее.", "Вода гасит огонь.",
                  "Хвост собаки мокрый.", "Кошки пушистые."):
            for f in facts_of(s).facts:
                self.assertFalse(f.negated, f"{s}: ложное отрицание {f}")


class TestZeroCopula(unittest.TestCase):
    """Именная часть без сказуемого: быть не может, узла «быть» не заводим."""

    def test_zero_copula_name_is_not_a_node(self):
        """Служебный символ НЕ должен уезжать в хвосты/вершины связей."""
        for s in ("Кит не рыба, а млекопитающее.",
                  "Возраст шкафа не 5, а 50 лет."):
            for f in facts_of(s).facts:
                self.assertNotIn(ZERO_COPULA, (f.head, f.tail),
                                 f"служебная метка просочилась в знание: {f}")

    def test_predicate_number_agreement_required(self):
        """Именной частью признаётся только согласованное по числу ИМ.п."""
        sf = facts_of("Киты не рыба.")
        self.assertTrue(all(f.relation != "OBJECT_OF" for f in sf.facts),
                        "несогласованное ИМ.п. не должно становиться объектом")


class TestPossessionGuard(unittest.TestCase):
    """Регрессия: бытийное владение — НЕ именное предложение."""

    def test_possession_still_only_has_part(self):
        t = triples("У кошки есть хвост.")
        self.assertIn(("кошка", "HAS_PART", "хвост"), t)
        self.assertNotIn(("хвост", "IS_ROLE", "кошка"), t,
                         "предложное слово не может быть именной частью")

    def test_regres_possession_relation(self):
        for s in ("У кошки есть хвост.", "У дерева много веток."):
            for head, rel, tail in triples(s):
                self.assertNotEqual(rel, "IS_ROLE",
                                    f"{s}: владение уехало в IS_ROLE")


class TestNegationScopeUnit(unittest.TestCase):
    """Юнит на сам класс области — чтобы правила было видно по одному признаку."""

    def _scope(self, sentence):
        return NegationScope(analyze(sentence))

    def test_no_adversative_means_wide_scope(self):
        sc = self._scope("Кошка не ест рыбу.")
        self.assertTrue(sc.any)
        self.assertFalse(sc.narrow, "без «а»/«но» область — всё предложение")
        self.assertEqual(sc.rescued, 0)

    def test_adversative_makes_scope_narrow_and_counts(self):
        sc = self._scope("Кит — это не рыба, а млекопитающее.")
        self.assertTrue(sc.narrow)
        self.assertEqual(sc.denied, ["рыба"])
        self.assertGreater(sc.rescued, 0,
                           "спасённые от отрицания слова обязаны быть числом")

    def test_negates_none_is_false_when_narrow(self):
        sc = self._scope("Кит — это не рыба, а млекопитающее.")
        self.assertFalse(sc.negates(None),
                         "для синтетической записи выдумывать отрицание нельзя")

    def test_scopes_never_cover_words_before_particle(self):
        sc = self._scope("Столица Франции — не Париж, а Лион.")
        self.assertNotIn("франция", sc.denied)
        self.assertIn("париж", sc.denied)


class TestLearnerKeepsFix(unittest.TestCase):
    """Через Learner: ложное отрицание не должно осесть в observations."""

    def _learn(self, text):
        import json
        import tempfile
        from brain.learner import Learner
        path = os.path.join(tempfile.mkdtemp(prefix="neg_contract_"),
                            "world.json")

        def load_world():
            try:
                with open(path, encoding="utf-8") as fh:
                    return json.load(fh)
            except FileNotFoundError:
                return {"entities": {}, "observations": {},
                        "unknown_words": {}}

        def save_world(w):
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(w, fh, ensure_ascii=False)

        save_world(load_world())
        # Учитель выключен НАМЕРЕННО: ложное отрицание — свойство разборки,
        # а не канала знания. Так тест не тянет LLM и не трогает рабочую память.
        lr = Learner(load_world=load_world, save_world=save_world,
                     teacher=None, max_teacher_calls=0)
        lr.learn_from_text(text, allow_teacher=False)
        return load_world()

    def test_negated_false_pair_reaches_observations(self):
        world = self._learn("Кит — это не рыба, а млекопитающее.")
        obs = world.get("observations", {})
        rows = {k: bool(v.get("negated")) for k, v in obs.items()}
        pos = [k for k in rows if "млекопитающее" in k]
        self.assertTrue(pos, f"утверждаемая связь обязана быть записана: {rows}")
        for k in pos:
            self.assertFalse(rows[k],
                             f"{k}: в базе лежит отрицание там, где утверждение")
        neg = [k for k in rows if "рыба" in k]
        for k in neg:
            self.assertTrue(rows[k], f"{k}: «не рыба» обязано остаться отрицанием")


if __name__ == "__main__":
    unittest.main(verbosity=2)
