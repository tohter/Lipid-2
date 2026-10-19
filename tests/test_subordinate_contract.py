"""
КОНТРАКТ: ПРИДАТОЧНЫЕ (очередь, п.18).

ЗАМЕР ДО ПРАВКИ (замерен ДО первой строчки правки, скрипт /tmp/amanda/p18_baseline.py,
экстрактор плоский, Учитель выключен — то есть всё ниже получено без LLM):

  «Мяч, который бросил мальчик, укатился.»
      мяч --CAN_DO--> бросить            conf=0.85   <-- Инверсия агент/объект:
                                                        бросал МАЛЬЧИК, не мяч
      мяч --HAS_PROPERTY--> который      conf=0.63   <-- служебное слово стало
                                                        СВОЙСТВОМ сущности
      СОБЫТИЕ subject='мяч' action='бросить' obj=''  <-- агент потерян целиком
      (сказуемое главного предложения «укатился» не попало ни в один факт)

  «Дом, где жил пёс, сгорел.»
      дом --CAN_DO--> жила               conf=0.36   <-- и узел «жила» НЕ
                                                        лемматизирован: в граф
                                                        легла форма, а не «жить»
      (пёс — тот, кто ЖИЛ; дом — тот, что СГОРЕЛ; оба отношения неверны)

  «Река, берега которой заросли, замёрзла.»
      берег --HAS_PROPERTY--> который    conf=0.48
      замёрзнуть --OBJECT_OF--> берег    conf=0.42   <-- река замёрзла НЕ «в берег»
      замёрзнуть --OBJECT_OF--> заросль  conf=0.41   <-- «заросль» — глагол «зарасти»

    «Кошка спрятала мышь, чтобы не есть её.»
      кошка --CAN_DO /НЕ/--> спрятать               <-- САМЫЙ ДОРОГОЙ ДЕФЕКТ:
                                                        «не» из придаточного цели
                                                        отрицает ГЛАВНОЕ. Смысл
                                                        записи обратный исходному.

ОБЩАЯ ПРИЧИНА (одна, а не четыре): у сложноподчинённого предложения ищутся ОДИН
субъект и ОДНО сказуемое на весь текст. Второе сказуемое остаётся без пары, и
главное предложение либо теряет сказуемое, либо получает чужое. Отрицание, выбираемое
по всему предложению, достает не тому придаточному (ср. NEGATION_TASK.md: там починили
область внутри ПРОСТОГО предложения, здесь нужна область по СЕГМЕНТАМ).

ЧЕГО ЭТИ ТЕСТЫ НЕ ТРЕБУЮТ
  Никакой семантики и никаких моделей: зависимость строится по ПУНКТУАЦИИ и
  ГРАММЕМАМ (замер: у всех форм «который» есть Subx+Anph+Apro, у «какой», «чей»
  и «пушистый» их нет — см. TestWhichIsNotAWord). Придаточное, граница которого
  не видна, обязано остаться с честной заметкой, а не с правдоподобной ложью.

Тесты не запускают LLM и не пишут в рабочую память.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.extract import SimpleExtractor, analyze  # noqa: E402
from core.morph import Morph  # noqa: E402

ex = SimpleExtractor()

SENT_BALL = "Мяч, который бросил мальчик, укатился."
SENT_DOG = "Пёс, который охранял дом, убежал."
SENT_HOUSE = "Дом, где жил пёс, сгорел."
SENT_RIVER = "Река, берега которой заросли, замёрзла."
SENT_PURPOSE = "Кошка спрятала мышь, чтобы не ловить её."
# Отдельный замер, не входящий в п.18: та же конструкция с глаголом «есть».
SENT_EAT = "Кошка спрятала мышь, чтобы не есть её."


def parse(sentence):
    return ex.extract(sentence)


def triples(sentence):
    return {(f.head, f.relation, f.tail) for f in parse(sentence).facts}


def rels_of(sentence, head):
    return {f.relation for f in parse(sentence).facts if f.head == head}


class TestWhichIsNotAWord(unittest.TestCase):
    """
    Союзное слово придаточного — НЕ узел графа и НЕ свойство.

    ЗАМЕРЕНО: «пёс --HAS_PROPERTY--> который», «мяч --HAS_PROPERTY--> который»,
    «берег --HAS_PROPERTY--> который». В графе это выглядело как знание о слове.
    """

    def test_which_family_is_recognized_by_grammemes_not_by_list(self):
        """
        Признак берётся из морфологии, а не из списка слов: список не покрывает
        падежные формы и молча пропускает новую.
        """
        for form in ("который", "которая", "которые", "которого", "которой",
                     "которым", "которых", "которыми"):
            self.assertTrue(Morph.is_relative_pronoun(form),
                            f"форма «{form}» не распознана как союзное слово")

    def test_ordinary_adjectives_are_not_relative(self):
        """
        Обратная сторона: расширять нельзя вслепую. «какой»/«чей» — вопросительные
        (Subx нет), «пушистый» — обычное прилагательное (Apro нет).
        """
        for word in ("пушистый", "какой", "чей", "дом", "бросить"):
            self.assertFalse(Morph.is_relative_pronoun(word),
                             f"«{word}» не имеет права считаться союзным словом")

    def test_relative_never_becomes_entity_or_property(self):
        for sentence in (SENT_BALL, SENT_DOG, SENT_RIVER):
            sf = parse(sentence)
            self.assertNotIn("который", sf.content_lemmas,
                             f"союзное слово стало значимой леммой: {sentence}")
            for f in sf.facts:
                self.assertNotIn("который", (f.head, f.tail),
                                 f"союзное слово стало узлом: {f} в {sentence}")
                self.assertNotEqual(f.tail, "который")


class TestClauseSegmentation(unittest.TestCase):
    """
    Каждое придаточное получает СВОЮ пару (субъект, сказуемое).

    Без этого главное предложение теряет сказуемое: замерено, что «укатился»,
    «убежал» и «сгорел» не входили ни в один факт.
    """

    def test_main_clause_predicate_survives(self):
        heads = {f.tail for f in parse(SENT_BALL).facts}
        self.assertIn("укатиться", heads,
                      "сказуемое главной части потеряно: «мяч укатился» не записано")

    def test_subject_of_main_clause_is_not_robbed_by_relative(self):
        """«Пёс, который охранял дом, убежал» — убежал ПЁС (он же и охранял)."""
        sf = parse(SENT_DOG)
        escaped = [f for f in sf.facts if f.tail == "убежать"]
        self.assertTrue(escaped, f"«убежать» не записан: {sf.facts}")
        self.assertEqual({f.head for f in escaped}, {"пёс"},
                         f"кто убежал — определено неверно: {escaped}")

    def test_house_did_not_live(self):
        """
        «Дом, где жил пёс, сгорел» — жил ПЁС, сгорел ДОМ.

        ЗАМЕРЕНО ДО ПРАВКИ: дом --CAN_DO--> жила (форма, не лемма!).
        """
        sf = parse(SENT_HOUSE)
        for f in sf.facts:
            self.assertNotEqual((f.head, f.tail), ("дом", "жила"),
                                "не лемматизованная форма «жила» легла в граф")
            if f.relation == "CAN_DO" and f.tail in ("жить", "жила"):
                self.assertEqual(f.head, "пёс",
                                 f"жит не дом: {f}")
        burned = [f for f in sf.facts if f.tail == "сгореть"]
        self.assertTrue(burned, f"«сгореть» потерян: {sf.facts}")
        self.assertEqual({f.head for f in burned}, {"дом"})

    def test_verb_tails_are_lemmas_not_forms(self):
        """
        Узел графа — лемма. Форма «жила» в графе означает, что у одного глагола
        появится два узла («жить» и «жила») и знание о них разведётся.
        """
        for sentence in (SENT_BALL, SENT_DOG, SENT_HOUSE, SENT_RIVER):
            for f in parse(sentence).facts:
                if f.relation in ("CAN_DO", "OBJECT_OF", "PURPOSE"):
                    self.assertEqual(
                        Morph.lemma(f.tail), f.tail,
                        f"хвост связи не лемма: {f} ({sentence})")


class TestRoleAssignment(unittest.TestCase):
    """
    Падеж союзного слова решает, КТО агент.

    «который» в Вин.п. = объект придаточного, агент — существительное в ИМ.п.
    («мяч, который бросил мальчик»): бросал мальчик.
    «который» в ИМ.п. = субъект придаточного («пёс, который охранял дом»).
    """

    def test_accusative_relative_keeps_human_agent(self):
        sf = parse(SENT_BALL)
        threw = [f for f in sf.facts if f.tail == "бросить"]
        self.assertTrue(threw, f"«бросить» не записан: {sf.facts}")
        self.assertEqual({f.head for f in threw}, {"мальчик"},
                         f"инверсия агент/объект: {threw}")

    def test_accusative_relative_becomes_patient(self):
        """Объект придаточного обязан дойти до графа: бросить -> мяч."""
        sf = parse(SENT_BALL)
        got = {(f.head, f.relation, f.tail) for f in sf.facts}
        self.assertIn(("бросить", "OBJECT_OF", "мяч"), got,
                      f"объект действия потерян: {sorted(got)}")

    def test_nominative_relative_is_agent_of_its_clause(self):
        got = triples(SENT_DOG)
        self.assertIn(("пёс", "CAN_DO", "охранять"), got)
        self.assertIn(("охранять", "OBJECT_OF", "дом"), got)
        self.assertNotIn(("который", "HAS_PROPERTY", "дом"), got)

    def test_event_records_agent_not_patient(self):
        """
        Событие — носитель динамики. Если в событии subject='мяч', действие
        «бросить» приписано предмету, и последующий вывод строится на лжи.
        """
        events = parse(SENT_BALL).events
        throw = [e for e in events if e.action == "бросить"]
        self.assertTrue(throw, f"событие «бросить» не собрано: {events}")
        self.assertEqual(throw[0].subject, "мальчик",
                         f"агент события потерян: {throw[0]}")


class TestNegationStaysInItsClause(unittest.TestCase):
    """
    «не» отрицает СВОЁ придаточное, а не всё сложное предложение.

    Подзадача та же, что в NEGATION_TASK.md, но граница теперь — СЕГМЕНТ,
    а не часть простого предложения.
    """

    def test_purpose_negation_does_not_negate_main_action(self):
        sf = parse(SENT_PURPOSE)
        hid = [f for f in sf.facts if f.tail == "спрятать"]
        self.assertTrue(hid, f"«спрятать» не записан: {sf.facts}")
        for f in hid:
            self.assertFalse(
                f.negated,
                f"«не» из придаточного цели отрицает главное предложение: {f}. "
                "Запись «кошка НЕ прятала мышь» обратна смыслу фразы.")

    def test_negation_of_purpose_is_preserved(self):
        """Отрицание внутри придаточного не должно исчезнуть вместе с правкой."""
        sf = parse(SENT_PURPOSE)
        caught = [f for f in sf.facts if f.tail == "ловить"]
        self.assertTrue(caught, f"цель не записана вовсе: {sf.facts}")
        self.assertTrue(any(f.negated for f in caught),
                        f"отрицание цели потеряно: {caught}")

    def test_stopword_verb_makes_purpose_invisible(self):
        """
        ЗАМЕР ДЕФЕКТА, НЕ входящего в п.18 (записан в очередь как найденный).

        «есть» сидит в STOP и STOP_LEMMAS как СВЯЗКА («кошка есть животное»),
        и из-за этого из графа исчезает и ГЛАГОЛ «есть». Замер:
            «Кошка ест рыбу.»          -> кошка CAN_DO есть (узел создан!)
            «...чтобы не есть её.»     -> ни одной связи с «есть»
        Одна и та же лексема видна в одной форме и невидна в другой: «ест»
        создаёт узел, «есть» — нет. Правится НЕ здесь (это стоп-лист
        морфологии, он влияет на весь корпус), а снятием «есть» из STOP_LEMMAS
        с различением связки и глагола. Тест нужен, чтобы починка была
        ОСОЗНАННОЙ: она обязана уронить этот тест.
        """
        eaten = [f for f in parse(SENT_EAT).facts if "есть" in (f.head, f.tail)]
        self.assertEqual([], eaten,
                         "«есть» перестал выпадать из разбора — убери этот тест "
                         "и проверь, что связка «кошка есть животное» не стала "
                         "узлом CAN_DO")


class TestHonestyWhenParsingFails(unittest.TestCase):
    """
    Там, где придаточное НЕ разобрано, обязана быть заметка.

    Запрет на тихую деградацию: без строки в отчёте нельзя отличить
    «придаточного не было» от «придаточное не размечено».
    """

    def test_subordinate_is_announced(self):
        for sentence in (SENT_BALL, SENT_DOG, SENT_HOUSE, SENT_PURPOSE):
            notes = " ".join(parse(sentence).notes)
            self.assertIn("придаточн", notes.lower(),
                          f"придаточное не объявлено в отчёте: {sentence} "
                          f"notes={parse(sentence).notes}")

class TestHonestGapStillHonest(unittest.TestCase):
    """
    Там, где придаточное НЕ разобрано, связь обязана НЕ записываться.

    Правдоподобная ложь дороже отсутствующей связи: отсутствующую можно
    добыть Учителем, а ложная уже лежит в онтологии и опыта ей хватает.
    """

    def test_ambiguous_clause_loses_facts_not_truth(self):
        """
        «Река, берега которой заросли, замёрзла» — ЗАМЕРЕНО: «заросли»
        омонимична (СУЩЕСТВИТЕЛЬНОЕ «заросль» мн.ч. им.п. вероятнее ГЛАГОЛА
        «зарасти» прош.вр. мн.ч.), поэтому придаточное остаётся неразобранным.

        Требование здесь не «разобрать любой ценой», а ГРОМКО сказать об
        отказе и НЕ сочинять связи. Замер до правки давал ровно обратное:
        «замёрзнуть --OBJECT_OF--> берег» и «замёрзнуть --OBJECT_OF--> заросль».
        """
        got = triples(SENT_RIVER)
        for head, rel, tail in got:
            self.assertNotEqual((head, rel), ("замёрзнуть", "OBJECT_OF"),
                                f"река не может замерзнуть «в берег»: {got}")
            self.assertNotIn(tail, ("заросль", "который"),
                             f"в граф попала служебная/омонимная форма: {got}")
        notes = " ".join(parse(SENT_RIVER).notes)
        self.assertTrue(
            "не записаны" in notes or "не найдено" in notes,
            f"отказ обязан быть громким, получены notes: {notes}")

    def test_clause_facts_are_dynamics_not_ontology(self):
        """
        Произошедшее в придаточном — СОБЫТИЕ, а не знание о виде.

        Плоский разбор помечает факты события целиком (_collect_events), и
        факты придаточного обязаны помечаться так же. Замер сразу после первой
        правки: «пёс CAN_DO охранять» осталось predicate=fact, то есть один
        рассказ осел бы как вечное свойство пса — тот самый дефект, из-за
        которого события вынесли в отдельную секцию (EVENTS_TASK.md).
        """
        for f in parse(SENT_DOG).facts:
            self.assertEqual(
                f.predicate, "event",
                f"факт придаточного записан как вечное знание: {f}")

class TestNoRegressionOnSimpleSentences(unittest.TestCase):
    """
    Плоские предложения обязаны НЕ измениться: правка касается только
    сложноподчинённых, иначе цена ремонта выше ценности.
    """

    def test_participial_and_no_comma_stays_clean(self):
        """
        Обратная сторона: простое предложение НЕ должно обрастать заметками о
        придаточном — иначе предупреждение обесценивается (тот же приём, что
        в test_negation_contract про ложные конфликты ИМ.п.).
        """
        sf = parse("Мальчик бросил мяч.")
        self.assertFalse([n for n in sf.notes if "придаточн" in n.lower()],
                         f"простое предложение помечено как придаточное: {sf.notes}")

    def test_simple_sentences_unchanged(self):
        cases = {
            "Мальчик бросил мяч.": {("мальчик", "CAN_DO", "бросить"),
                                    ("бросить", "OBJECT_OF", "мяч")},
            "Кошка поймала мышь.": {("кошка", "CAN_DO", "поймать"),
                                    ("поймать", "OBJECT_OF", "мышь")},
            "Хвост собаки мокрый.": {("хвост", "PART_OF", "собака"),
                                     ("хвост", "HAS_PROPERTY", "мокрый")},
            "Кошка боится грозы.": {("кошка", "CAN_DO", "бояться")},
        }
        for sentence, expected in cases.items():
            got = triples(sentence)
            for trio in expected:
                self.assertIn(trio, got,
                              f"регрессия на простом предложении {sentence!r}: "
                              f"пропало {trio}, получено {sorted(got)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
