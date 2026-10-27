"""
Контракт записи об entity: что вообще имеет право называться знанием.

Два класса дефектов, оба найдены в ЖИВОМ lipid_world.json, а не в теории:

  1. Тавтологическая иерархия. 4 из 6 сущностей имели в hierarchy форму самого
     себя: 'экосистема' -> ['экологические термины', 'экосистемы'], 'луга' ->
     ['луга', ...]. Источник — категории Википедии, где 'Экосистемы' это
     множественное число того же понятия. Ответ на вопрос превращался в
     «Экосистема — это экосистемы»: строка правдоподобная, информации ноль,
     а выглядит как успешный поиск знания.

  2. Пустая запись, неотличимая от знания. Сущность «в графе есть» и «о ней
     что-то известно» — разные утверждения; первая версия путала их и позволяла
     отчитаться об успехе, записав одно только слово.

Отдельно проверяется, что нормализация НЕ ломает полезные категории: 'кошачьи'
для 'кошки' — другое слово и настоящая категория, а 'Луга Ленинградской
области' для 'луга' — перечисление населённых пунктов, не форма слова.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.records import is_empty_entity, meaningful_categories, merge_entity

# См. tests/test_lemma_service_contract.py: тест, сверяющий русские формы,
# обязан внедрить лемматизатор ЯВНО. До A2.2 морфология включалась сама собой
# (records сам звал pymorphy3), теперь она приходит извне — и молча остаться
# выключенной значит проверять на эвристике, думая, что проверяешь на языке.
from lang.ru.lemma import install as _install_lemma_service

_install_lemma_service()


class TestMeaningfulCategories(unittest.TestCase):

    def test_self_form_is_dropped(self):
        """Замерено: 4 из 6 реальных записей содержали форму самого себя."""
        cases = [
            ("экосистема", ["экологические термины", "экосистемы"],
             ["экологические термины"]),
            ("флюгер", ["флюгеры", "измерительные приборы"], ["измерительные приборы"]),
            ("луга", ["луга", "районные центры ленинградской области"],
             ["районные центры ленинградской области"]),
            ("сеть", ["сети", "компьютерные сети"], ["компьютерные сети"]),
            ("кошки", ["кошки", "кошачьи"], ["кошачьи"]),
        ]
        for concept, hier, want in cases:
            with self.subTest(concept=concept):
                self.assertEqual(meaningful_categories(concept, hier), want)

    def test_whose_category_all_by_the_same_name_is_dropped(self):
        self.assertEqual(meaningful_categories("джойстик", ["джойстики"]), [])

    def test_service_categories_are_dropped(self):
        for cat in ("Страницы значений по алфавиту",
                    "Статьи, требующие викификации",
                    "Перенаправления",
                    "Заготовки статей"):
            with self.subTest(cat=cat):
                self.assertEqual(meaningful_categories("что-то", [cat]), [])

    def test_different_words_survive(self):
        """'кошачьи' — ДРУГОЕ слово, и выкинуть его = потерять знание."""
        self.assertEqual(meaningful_categories("кошка", ["кошачьи"]), ["кошачьи"])
        self.assertEqual(
            meaningful_categories("луга", ["Луга Ленинградской области"]),
            ["Луга Ленинградской области"])

    def test_duplicates_after_normalization_collapse(self):
        out = meaningful_categories("сеть", ["компьютерные сети", "Компьютерные сети"])
        self.assertEqual(len(out), 1)

    def test_non_strings_ignored(self):
        self.assertEqual(meaningful_categories("а", [None, 1, "", "  б "]), ["б"])

    def test_nothing_is_not_tautology(self):
        """
        Пусто — лучше тавтологии: пустая иерархия уводит обработчик в ветку
        «данных нет», тогда как тавтология притворяется ответом.
        """
        self.assertEqual(meaningful_categories("неттакого", []), [])


class TestEmptyEntity(unittest.TestCase):

    def test_name_alone_is_empty(self):
        self.assertTrue(is_empty_entity({"concept": "флюгер"}))
        self.assertTrue(is_empty_entity({"concept": "флюгер",
                                        "hierarchy": ["флюгеры"]}))
        self.assertTrue(is_empty_entity({"concept": "флюгер", "properties": [],
                                        "actions": []}))

    def test_any_content_makes_it_not_empty(self):
        for rec in ({"concept": "ф", "properties": ["крутится"]},
                    {"concept": "ф", "actions": ["указывает"]},
                    {"concept": "ф", "definition": "это прибор"},
                    {"concept": "ф", "hierarchy": ["приборы"]}):
            with self.subTest(keys=list(rec)):
                self.assertFalse(is_empty_entity(rec))

    def test_garbage_is_empty_not_crash(self):
        self.assertTrue(is_empty_entity(None))
        self.assertTrue(is_empty_entity("флюгер"))
        self.assertFalse(is_empty_entity({"definition": "есть определение"}))


class TestMergeStillMerges(unittest.TestCase):
    """Слияние не имеет права деградировать: оно было единственным местом,
    где решается, что имеет право перезаписать известное."""

    def test_lists_union_scalars_not_overwritten(self):
        merged = merge_entity(
            {"concept": "флюгер", "properties": ["крутится"], "hierarchy": ["приборы"]},
            {"concept": "флюгер", "properties": ["стоит на крыше"],
             "hierarchy": ["флюгеры"], "definition": "прибор для направления ветра"})
        self.assertEqual(merged["properties"], ["крутится", "стоит на крыше"])
        self.assertEqual(merged["hierarchy"], ["приборы"],
                         "тавтология не должна проникнуть через merge")
        self.assertEqual(merged["definition"], "прибор для направления ветра")

    def test_cleaning_applies_to_saved_side_too(self):
        """
        Чистка обязана работать в обе стороны. Если чистить только «новое»,
        тавтология, уже лежащая в файле, переживёт любое количество правок.
        """
        merged = merge_entity(
            {"concept": "экосистема", "hierarchy": ["экосистемы"]},
            {"concept": "экосистема", "hierarchy": ["экология"]})
        self.assertEqual(merged["hierarchy"], ["экология"])

    def test_empty_new_does_not_erase_known(self):
        prev = {"concept": "ф", "properties": ["а"]}
        self.assertEqual(merge_entity(prev, {})["properties"], ["а"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestStripSelfName(unittest.TestCase):
    """
    Пояснение не обязано начинаться с того, что оно поясняет.

    ЗАМЕРЕНО в GUI-прогоне: ответ пользователя «Пчела — летаемое насекомое;
    опыляет цветы» попал в hierarchy целиком, а ответ склеился из двух имён:
    «Пчела — это Пчела — летаемое насекомое». Тот же дефект в Википедии чинился
    отдельной функцией; правило живёт в ОДНОМ месте (core/records.py).
    """

    def test_answer_category_without_headword(self):
        from core.records import strip_self_name
        self.assertEqual(strip_self_name("пчела", "Пчела — летаемое насекомое"),
                         "летаемое насекомое")

    def test_short_category_still_cut(self):
        """Порог для ответа пользователя мал: «прибор» — полноценная категория."""
        from core.records import strip_self_name
        self.assertEqual(strip_self_name("флюгер", "Флюгер — прибор"), "прибор")

    def test_definition_needs_substantial_remainder(self):
        """Для определения Вики остаток обязан быть фразой, а не огрызком."""
        from core.records import strip_self_name
        self.assertEqual(strip_self_name("кошка", "Кошка — это", min_rest=15),
                         "Кошка — это",
                         "определение срезано до пустоты")

    def test_tautology_is_not_hidden(self):
        from core.records import strip_self_name
        self.assertEqual(strip_self_name("флюгер", "Флюгер — это флюгер"),
                         "Флюгер — это флюгер",
                         "срез бессмысленен: остатки тоже тавтология")

    def test_other_words_untouched(self):
        from core.records import strip_self_name
        self.assertEqual(strip_self_name("пчела", "летаемое насекомое"),
                         "летаемое насекомое")
        self.assertEqual(strip_self_name("пчела", ""), "")

    def test_user_answer_record_has_clean_hierarchy_and_definition(self):
        """Запись с слов пользователя: категория чистая, определение сохранено."""
        import json, os, tempfile
        d = tempfile.mkdtemp(prefix="l2user-cat-")
        path = os.path.join(d, "lipid_world.json")
        os.environ["LIPID2_MEMORY_DIR"] = d
        from core.worldfile import write_world
        write_world(path, {"entities": {}, "unknown_words": {}})
        from brain.lipid2_brain import Lipid2Brain
        brain = Lipid2Brain(memory_dir=d)
        brain._learn_from_user("пчела",
                               "Пчела — летаемое насекомое; опыляет цветы")
        ent = json.load(open(path, encoding="utf-8"))["entities"]["пчела"]
        self.assertEqual(ent["hierarchy"], ["летаемое насекомое"],
                         "в иерархию попала фраза с самим словом")
        self.assertEqual(ent["properties"], ["опыляет цветы"])
        self.assertIn("опыляет цветы", ent["definition"])
        os.environ.pop("LIPID2_MEMORY_DIR", None)

    def test_answer_of_user_record_is_not_answered_twice(self):
        """Ответ на вопрос не должен быть «X — это X, ...» — проверка всего пути."""
        import os, tempfile
        d = tempfile.mkdtemp(prefix="l2user-ans-")
        os.environ["LIPID2_MEMORY_DIR"] = d
        from core.worldfile import write_world
        write_world(os.path.join(d, "lipid_world.json"),
                    {"entities": {}, "unknown_words": {}})
        from brain.lipid2_brain import Lipid2Brain
        brain = Lipid2Brain(memory_dir=d)
        brain._learn_from_user("пчела", "Пчела — летаемое насекомое; опыляет цветы")
        answer = brain._handle_definition("пчела")
        low = answer.lower()
        self.assertFalse(low.count("пчел") > 1,
                         f"слово повторено в ответе: {answer!r}")
        self.assertIn("летаемое насекомое", answer)
        os.environ.pop("LIPID2_MEMORY_DIR", None)
