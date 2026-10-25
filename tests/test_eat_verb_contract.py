"""
КОНТРАКТ, ОЧЕРЕДЬ C: «ЕСТЬ» ВЫПАДАЕТ ИЗ ГРАФА (SUBORDINATE_TASK.md, п.9.3).

ЗАМЕР ДО ПРАВОК (записан ДО первой строчки правки; LIPID2_STATUS=0, LLM
выключен, живая память не тронута):

  «Кошка спрятала мышь, чтобы не есть её.»
      [('кошка','CAN_DO','спрятать'), ('спрятать','OBJECT_OF','мышь')]
      ЦЕЛИ НЕТ. Рядом эталон той же конструкции с другим инфинитивом:
  «Кошка спрятала мышь, чтобы не ловить её.»
      ... ('спрятать','PURPOSE','ловить')                  <-- есть
  «Кошка спрятала мышь, чтобы съесть её.»
      ... ('спрятать','PURPOSE','съесть')                  <-- есть
  «Кошка спрятала мышь, чтобы не жрать её.»
      ... ('спрятать','PURPOSE','жрать')                   <-- есть
  Одна лексема невидима в одной форме: «съесть» и «жрать» дают цель, «есть» —
  нет. Заперто тестом test_stopword_verb_makes_purpose_invisible (будильник,
  обязан был упасть при починке — см. ниже, он заменён позитивным замком).

  «Кошка будет есть рыбу.»
      [('кошка','IS_ROLE','рыба')]        <-- САМЫЙ ДОРОГОЙ ДЕФЕКТ: смысл
        записан ОБРАТНО. Кошка НЕ является рыбой; кошка будет ЕСТЬ рыбу.
        Эталон: «Кошка будет ловить рыбу.» ->
        [('кошка','CAN_DO','ловить'), ('ловить','OBJECT_OF','рыба')]

  «Кошка начала есть рыбу.»
      [('кошка','IS_ROLE','начало'), ('кошка','IS_ROLE','рыба')]
      (у эталона «начала ловить» есть своё мусор-«начало» — тот дефект
      относится к «начала» как омониму сущестивительного, не к «есть»)

ПРИЧИНЫ (их ДВЕ, независимые; обе замерены):

  1. Словоформа «есть» стоит в STOP (core/morph.py, строка 82 — ВНИМАНИЕ:
     очередь указывала STOP_LEMMAS, это НЕТАК: в STOP_LEMMAS «есть» нет).
     Токен получает content=False и исчезает из разбора.

  2. Даже с снятым STOP цель НЕ появляется. _purpose_part отвергает часть
     проверкой «есть ли своё сказуемое»: она смотрит person/tense по ВСЕМ
     вариантам слова, а у «есть» рядом с инфинитивом (0.893) висит вариант
     «быть» VERB 3per pres с весом 0.036 — в ~25 раз тише. Тот же класс
     дефекта, что уже чинили в очереди Б («котором» и «котор» при 0.003).

ПОЧЕМУ НЕЛЬЗЯ ПРОСТО ВЫБРОСИТЬ «есть» ИЗ STOP (замер того эксперимента):
      «У кошки есть хвост.»    -> + ложный ('хвост','CAN_DO','есть')
      «Кошка есть животное.»   -> ('есть','OBJECT_OF','животное') вместо
                                  ('кошка','IS_ROLE','животное')  ДЕГРАДАЦИЯ
      «Кошка есть рыба.»       -> IS_ROLE ПОТЕРЯН                 ДЕГРАДАЦИЯ
      «У реки есть берега.»    -> [] вместо ('река','HAS_PART','берег')
      «Вода есть.»             -> ('вода','CAN_DO','есть')  (было [])
    Слово «есть» ОДНОВРЕМЕННО глагол и связка, и pymorphy их НЕ РАЗЛИЧАЕТ:
    у формы «есть» инфинитив 0.893, а «быть» 3per — 0.036, но это веса
    ВАРИАНТОВ одного слова, а не Different constructions. Различает их только
    ОКРУЖЕНИЕ, и решение принимает экстрактор, а не морфология.

ЧЕГО ЭТОТ КОНТРАКТ НЕ ТРЕБУЕТ
  НЕ требует считать «Вода есть» действием (вода не ест). НЕ требует
  разбирать «есть» как связку лучше, чем сейчас, — поведение связки
  ЗАПРЕЩЕНО МЕНЯТЬ (тесты на IS_ROLE и HAS_PART ниже). Требует одного:
  перестать терять глагол там, где он явно назван глаголом окружением
  (вспомогательный глагол, союз цели, частица), и не испортить при этом
  связку и владение.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.extract import SimpleExtractor          # noqa: E402
from core.morph import Morph                      # noqa: E402

ex = SimpleExtractor()

SENT_EAT = "Кошка спрятала мышь, чтобы не есть её."
SENT_EAT_POS = "Кошка спрятала мышь, чтобы есть её."
SENT_CATCH = "Кошка спрятала мышь, чтобы не ловить её."
SENT_FUTURE = "Кошка будет есть рыбу."
SENT_EAT_FLAT = "Кошка ест рыбу."
SENT_COPULA = "Кошка есть животное."
SENT_HAVE = "У кошки есть хвост."
SENT_HAVE_RIVER = "У реки есть берега."


def triples(sentence):
    return {(f.head, f.relation, f.tail) for f in ex.extract(sentence).facts}


class TestEatIsVisibleAsVerb(unittest.TestCase):
    """Глагол «есть» обязан доходить до графа в тех же формах, что и любой другой."""

    def test_purpose_eat_matches_other_infinitives(self):
        """
        Ряд должен быть ОДИНАКОВЫЙ: с «ловить» цель пишется, с «есть» — нет
        (замер до правки). Требуем цели по эталону соседней конструкции.
        """
        self.assertIn(("спрятать", "PURPOSE", "ловить"), triples(SENT_CATCH))
        got = triples(SENT_EAT)
        self.assertIn(("спрятать", "PURPOSE", "есть"), got,
                      f"цель с «есть» потеряна: {got}")

    def test_purpose_eat_negated_keeps_main_verb_positive(self):
        """
        «не» отрицает ЦЕЛЬ, а не главное действие (NEGATION/SUBORDINATE:
        ровно этот дефект стоил отдельного файла).
        """
        for f in ex.extract(SENT_EAT).facts:
            if f.head == "кошка" and f.relation == "CAN_DO":
                self.assertFalse(f.negated,
                                 f"«не» добралось до главного: {f}")

    def test_purpose_without_negation(self):
        got = triples(SENT_EAT_POS)
        self.assertIn(("спрятать", "PURPOSE", "есть"), got,
                      f"цель с «есть» потеряна: {got}")

    def test_future_eat_is_not_identity(self):
        """
        «Кошка будет есть рыбу» ЗАМЕРЕНО как ('кошка','IS_ROLE','рыба'):
        в граф лёг смысл «кошка является рыбой». Требуем действие и объект.
        """
        got = triples(SENT_FUTURE)
        self.assertNotIn(("кошка", "IS_ROLE", "рыба"), got,
                         f"кошка не является рыбой: {got}")
        self.assertIn(("кошка", "CAN_DO", "есть"), got,
                      f"действие потеряно: {got}")
        self.assertIn(("есть", "OBJECT_OF", "рыба"), got,
                      f"объект действия потерян: {got}")


class TestGovernedByWhom(unittest.TestCase):
    """
    Управляющее слово бывает разное, и мутации это проверены по отдельности.
    Список слов-управляющих не пишется (форм слишком много) — признак берётся
    из морфологии варианта, и каждый класс признака обязан быть заперт.
    """

    def test_modal_short_adjective(self):
        """«должна есть» — ADJS. Мутация MODAL_POS=() обязана уронить тест."""
        got = triples("Кошка должна есть мясо.")
        self.assertIn(("кошка", "CAN_DO", "есть"), got, f"{got}")
        self.assertIn(("есть", "OBJECT_OF", "мясо"), got, f"{got}")

    def test_predicate_word(self):
        """
        «надо есть» — PRED, лицо/время у него отсутствуют вовсе, признаку
        «глагол с лицом» оно не удовлетворяет. ЗАПРЕЩЕНО терять инфинитив
        после модалки.

        ЗАМЕРЕНО (и это НЕ регрессия, а отдельный старый дефект ВНЕ очереди C):
        безличная конструкция не даёт связей НИ для какого инфинитива —
        «Надо ловить рыбу.» в базовом коде тоже []. Субъекта в таком предложении
        нет, заводить узел «рыба -> ловить» экстрактор не умеет. Поэтому здесь
        проверяется только то, что «есть» не стало хуже соседа.
        """
        self.assertEqual(triples("Надо ловить рыбу."), triples("Надо есть рыбу."),
                         "«есть» обращается с безличной конструкцией иначе, "
                         "чем любой другой инфинитив")
        # Там, где субъект есть, модалка обязана пропускать инфинитив.
        got = triples("Кошка должна есть мясо.")
        self.assertIn(("кошка", "CAN_DO", "есть"), got, f"{got}")

    def test_purpose_conjunction(self):
        """«чтобы» — не глагол; его право требовать инфинитив задано явно."""
        self.assertIn(("спрятать", "PURPOSE", "есть"), triples(SENT_EAT))

    def test_noise_variant_does_not_make_a_noun_a_governor(self):
        """
        Порог NOISE_SCORE в governs_infinitive — не косметика, он отобран по
        корпусу experiments/ru_words.txt (108517 форм, 191 слово, где порог
        меняет решение, и 21 из них — уверенный не-глагол с ТИХИМ глагольным
        вариантом). Без порога такие слова притворяются управляющими и
        превращают связку «есть» в поедание. ЗАМЕРЕНО: у «правило» топ NOUN
        0.957, а рядом «править» VERB 3per с весом 0.0108.
        """
        from core.morph import governs_infinitive
        for tok_word in ("правило", "правил", "скоро", "далеко", "легко",
                         "стекла", "минут", "кучу"):
            vs = Morph.parse_all(tok_word)
            tok = {"lemma": vs[0]["lemma"], "variants": vs}
            self.assertFalse(governs_infinitive(tok),
                             f"«{tok_word}» выдало себя за управляющий глагол")

    def test_real_governors_are_recognised(self):
        """Обратная сторона того же признака: настоящие управляющие слова."""
        from core.morph import governs_infinitive
        for w in ("будет", "может", "хочет", "начал", "перестал", "должен",
                  "надо", "чтобы"):
            vs = Morph.parse_all(w)
            self.assertTrue(governs_infinitive({"lemma": vs[0]["lemma"],
                                                "variants": vs}),
                            f"«{w}» перестало управлять инфинитивом")

    def test_noun_before_eat_does_not_become_verb(self):
        """
        Существительное инфинитивом управлять НЕ может, поэтому после него
        «есть» остаётся связкой. Мутация «считать инфинитивом любое «есть»»
        обязана уронить этот тест.
        """
        for s in ("Рыба есть.", "Кошка есть."):
            got = triples(s)
            self.assertNotIn((s.split()[0].lower(), "CAN_DO", "есть"), got,
                             f"«{s}» прочитано как поедание: {got}")


class TestCopulaAndPossessionPreserved(unittest.TestCase):
    """
    Обратная сторона: «есть» ОСТАЁТСЯ связкой и там, где он связка. Правка
    стоп-листа наивным removal'ом это ломает (замерено, докstring сверху).
    """

    def test_copula_stays_identity_not_eating(self):
        got = triples(SENT_COPULA)
        self.assertIn(("кошка", "IS_ROLE", "животное"), got,
                      f"связка перестала работать: {got}")
        self.assertNotIn(("кошка", "CAN_DO", "есть"), got,
                         f"«кошка есть животное» прочитано как поедание: {got}")

    def test_copula_with_nominative_predicate(self):
        self.assertIn(("кошка", "IS_ROLE", "рыба"), triples("Кошка есть рыба."))

    def test_possession_stays_clean(self):
        got = triples(SENT_HAVE)
        self.assertIn(("кошка", "HAS_PART", "хвост"), got,
                      f"владение потеряно: {got}")
        self.assertNotIn(("хвост", "CAN_DO", "есть"), got,
                         f"хвост не может «есть»: {got}")

    def test_possession_plural(self):
        got = triples(SENT_HAVE_RIVER)
        self.assertIn(("река", "HAS_PART", "берег"), got,
                      f"владение потеряно: {got}")

    def test_existential_is_not_eating(self):
        """«Вода есть» — это «существует», не «поедает»."""
        got = triples("Вода есть.")
        self.assertNotIn(("вода", "CAN_DO", "есть"), got,
                         f"бытийное «есть» стало поеданием: {got}")


class TestNoRegressionOnOtherForms(unittest.TestCase):
    """Личные формы никогда в STOP не сидели — они и после правки как были."""

    def test_personal_form_unchanged(self):
        self.assertEqual(
            {("кошка", "CAN_DO", "есть"), ("есть", "OBJECT_OF", "рыба")},
            triples(SENT_EAT_FLAT), f"личный глагол разъехался: "
                                    f"{triples(SENT_EAT_FLAT)}")

    def test_other_infinitives_unchanged(self):
        self.assertEqual(
            {("кошка", "CAN_DO", "спрятать"), ("спрятать", "OBJECT_OF", "мышь"),
             ("спрятать", "PURPOSE", "ловить")},
            triples(SENT_CATCH))

    def test_morphology_stop_list_other_words_intact(self):
        """Снимать надо ОДНУ словоформу, а не половину стоп-листа."""
        for w in ("был", "было", "будет", "может", "там", "очень"):
            self.assertTrue(Morph.is_stop(w), f"«{w}» выпало из стоп-листа")


if __name__ == "__main__":
    unittest.main()
