"""
КОНТРАКТ, ОЧЕРЕДЬ B: СВЯЗЬ «ГДЕ» С МЕСТОМ НЕ ЗАПИСАНА (SUBORDINATE_TASK.md, п.9).

ЗАМЕР ДО ПРАВОК (записан ДО первой строчки правки; экстрактор плоский,
LIPID2_STATUS=0, LLM выключен; команды и вывод полностью):

  «Дом, где жил пёс, сгорел.»
      [('дом', 'CAN_DO', 'сгореть'), ('пёс', 'CAN_DO', 'жить')]
      notes=['придаточное: разобрано по частям (3 ч.)']
      СВЯЗИ С МЕСТОМ НЕТ. Отношение LOCATED_IN в реестре есть
      (core/storage.py, индекс 6) и плоский путь его пишет:
      «Пёс живёт в доме.» -> [('жить','LOCATED_IN','дом'), ('пёс','CAN_DO','жить')]

  «Дом, в котором жил пёс, сгорел.»
      [('дом', 'HAS_PROPERTY', 'который')]
      notes=['несколько существительных в ИМ.п. согласуются со сказуемым
              — субъект спорный']
      ХУЖЕ, ЧЕМ «ГДЕ»: союзное слово осело СВОЙСТВОМ сущности (тот же дефект,
      ради которого написан is_relative_pronoun), а сказуемое главного
      предложения («сгорел») ПОТЕРЯНО целиком.
      Причина измерена: Morph.is_relative_pronoun("котором") == False, хотя у
      top-вариантаlemma «который» и полный набор Subx+Anph+Apro. Причина —
      правило all() по ВСЕМ вариантам, а pymorphy добавляет мусорный вариант
      «котор» (NOUN) с вероятностью 0.003.

  «Стул, на котором сидит пёс, сломан.»
      [('стул', 'HAS_PROPERTY', 'который')]   — тот же дефект.

  «Город, где живут псы, большой.»
      [('город', 'HAS_PROPERTY', 'большой'), ('пёс', 'CAN_DO', 'жить')]
      — связь с местом тоже пропала.

  «Я был там, где жил пёс.»  /  «Он живёт там, где текла река.»
      _antecedent возвращает '' (замер) — антецедента-существительного нет,
      «там» существительным не является. Здесь связь НЕ должна появляться.

НАПРАВЛЕНИЕ ОТНОШЕНИЯ — ПО КОНВЕНЦИИ РЕЕСТРА, А НЕ ПО ТЕКСТУ ОЧЕРЕДИ.
Очередь написала «дом --LOCATED_IN--> жить». ЗАМЕРЕНО: ВСЕ существующие пути
пишут НАПРАВЛЕНИЕ «действие -> место», и оно заперто тестом
tests/test_learning_contract.py:99 — ("сидеть","LOCATED_IN","подоконник"):
  «Кошка спит на подоконнике.» -> ('спать','LOCATED_IN','подоконник')
  «Пёс живёт в доме.»          -> ('жить','LOCATED_IN','дом')
  «В доме живёт пёс.»          -> ('жить','LOCATED_IN','дом')
Писать обратное значило бы рассорить одно отношение в двух направлениях.
Поэтому контракт требует ('жить','LOCATED_IN','дом').

ЧЕГО ЭТОТ КОНТРАКТ НЕ ТРЕБУЕТ
  Не требует разбирать «там, где» без антецедента — там связь не из чего
  строить, и правдоподобная ложь дороже отсутствующей связи (тот же
  принцип, что в TestHonestGapStillHonest). Требует только НЕ ТЕРЯТЬ то, что
  плоский разбор уже умел («пёс CAN_DO жить»).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.extract import SimpleExtractor  # noqa: E402
from core.morph import Morph  # noqa: E402

ex = SimpleExtractor()

SENT_HOUSE = "Дом, где жил пёс, сгорел."
SENT_WHICH = "Дом, в котором жил пёс, сгорел."
SENT_CHAIR = "Стул, на котором сидит пёс, сломан."
SENT_CITY = "Город, где живут псы, большой."
SENT_THERE = "Я был там, где жил пёс."


def triples(sentence):
    return {(f.head, f.relation, f.tail) for f in ex.extract(sentence).facts}


class TestWhereLinksPlace(unittest.TestCase):
    """«Где» обязано связать место с действием придаточного."""

    def test_gde_records_located_in(self):
        got = triples(SENT_HOUSE)
        self.assertIn(("жить", "LOCATED_IN", "дом"), got,
                      f"«где» не связало дом с действие: {got}")

    def test_gde_keeps_can_do_facts(self):
        got = triples(SENT_HOUSE)
        self.assertIn(("пёс", "CAN_DO", "жить"), got,
                      f"агент придаточного потерян: {got}")
        self.assertIn(("дом", "CAN_DO", "сгореть"), got,
                      f"сказуемое главного потеряно: {got}")

    def test_gde_with_plural_subject(self):
        got = triples(SENT_CITY)
        self.assertIn(("жить", "LOCATED_IN", "город"), got,
                      f"«где» с мн.ч. не дало связи с местом: {got}")
        self.assertIn(("пёс", "CAN_DO", "жить"), got, f"{got}")


class TestKotoromIsARelativeWord(unittest.TestCase):
    """
    Предложная форма союзного слова обязана ПОЗНАВАТЬСЯ.

    Замер: all() по вариантам отвергает «котором» из-за варианта «котор»
    (NOUN, score=0.003) — шума в 300 раз тише правильного разбора.
    """

    def test_prepositional_form_recognized(self):
        self.assertTrue(Morph.is_relative_pronoun("котором"),
                        "«в котором» перестало быть союзным словом")

    def test_demonstratives_do_not_become_relative_by_accident(self):
        """
        Ослаблять all() можно НЕ НА ВСЮ СЛОВОФОРМУ: «том» и «котором»
        морфемно неразличимы (замер: у обоих ADJF+Anph+Apro+Subx+loct,
        различает только лемма). Указательное «том/того/той/теми» не должно
        приобрести статус союзного слова — иначе «в том доме» объявится
        придаточным.
        """
        for w in ("том", "того", "той", "теми", "каком", "какая"):
            self.assertFalse(Morph.is_relative_pronoun(w),
                             f"указательное/вопросительное «{w}» стало союзным")

    def test_kotorom_clause_links_place(self):
        got = triples(SENT_WHICH)
        self.assertIn(("жить", "LOCATED_IN", "дом"), got,
                      f"«в котором» не дало связи с местом: {got}")
        self.assertIn(("дом", "CAN_DO", "сгореть"), got,
                      f"сказуемое главного потеряно: {got}")
        self.assertIn(("пёс", "CAN_DO", "жить"), got, f"{got}")

    def test_service_word_never_becomes_property(self):
        """
        Тот же дефект п.18 в новой форме: «дом HAS_PROPERTY который» —
        служебное слово осело свойством сущности.
        """
        for sentence in (SENT_WHICH, SENT_CHAIR):
            for head, rel, tail in triples(sentence):
                self.assertNotEqual(
                    (rel, tail), ("HAS_PROPERTY", "который"),
                    f"служебное слово стало свойством: {(head, rel, tail)} "
                    f"({sentence})")

    def test_chair_with_preposition_links_place(self):
        got = triples(SENT_CHAIR)
        self.assertIn(("сидеть", "LOCATED_IN", "стул"), got,
                      f"«на котором» не связало стул с сидеть: {got}")
        self.assertIn(("пёс", "CAN_DO", "сидеть"), got, f"{got}")


    def test_about_preposition_is_not_a_place(self):
        """
        Предложный падеж — НЕ всегда место.

        «ДОМ, О КОТОРОМ мечтал пёс» — это ABOUT (таблица PREP_CASES:
        о/об -> ABOUT), тем же путём это трактует и плоский разбор. Мутация
        (счесть всякий предложный союз местом) дала
        «мечтать --LOCATED_IN--> дом»: пёс мечтал НЕ внутри дома.
        До правки связи не было вовсе — вернуть её ложной нельзя.

        ФОРМА ВАЖНА: берётся именно «о КОТОРОМ» (loct), а не «о которой» —
        у последней pymorphy первым variant даёт род.п., и она в loc-путь
        не заходит, то есть защиту не проверяет (замер).
        """
        for sent in ("Дом, о котором мечтал пёс, сгорел.",
                     "Книга, о которой думал пёс, лежала."):
            got = triples(sent)
            self.assertFalse([t for t in got if t[1] == "LOCATED_IN"],
                             f"«о котором» записано местом: {sent} {got}")

    def test_place_prepositions_do_become_located_in(self):
        """
        Обратная сторона того же признака: в/на — место, и их отсекать
        нельзя (иначе тест выше прошёл бы просто на выключенном механизме).
        """
        self.assertIn(("жить", "LOCATED_IN", "дом"),
                      triples("Дом, в котором жил пёс, сгорел."))
        self.assertIn(("сидеть", "LOCATED_IN", "стул"),
                      triples("Стул, на котором сидит пёс, сломан."))


class TestNoInventionWithoutAntecedent(unittest.TestCase):
    """
    «Там, где» без существительного-антецедента: связь НЕ из чего строить.
    Плоский разбор до правки давал («пёс»,'CAN_DO','жить') — потерять это
    было бы регрессией.
    """

    def test_there_where_keeps_action_but_invents_no_place(self):
        got = triples(SENT_THERE)
        self.assertIn(("пёс", "CAN_DO", "жить"), got,
                      f"факт придаточного потерян: {got}")
        self.assertNotIn(("там", "LOCATED_IN", "жить"), got,
                         f"местом стало указательное «там»: {got}")
        self.assertFalse([t for t in got if t[1] == "LOCATED_IN"],
                         f"связь с местом сочинена без антецедента: {got}")


class TestNoRegressionOnSimpleSentences(unittest.TestCase):
    """Простые предложения НЕ должны разъехаться: «том» не стал придаточным."""

    def test_demo_pronoun_sentence_unchanged(self):
        got = triples("В том доме жили псы.")
        self.assertIn(("жить", "LOCATED_IN", "дом"), got,
                      f"плоский предложный путь сломан: {got}")
        self.assertIn(("пёс", "CAN_DO", "жить"), got, f"{got}")

    def test_flat_place_sentence_unchanged(self):
        got = triples("Пёс живёт в доме.")
        self.assertIn(("жить", "LOCATED_IN", "дом"), got, f"{got}")

    def test_no_subordinate_note_on_simple(self):
        sf = ex.extract("В том доме жили псы.")
        self.assertFalse([n for n in sf.notes if "придаточн" in n.lower()],
                         f"простое предложение помечено придаточным: {sf.notes}")


if __name__ == "__main__":
    unittest.main()
