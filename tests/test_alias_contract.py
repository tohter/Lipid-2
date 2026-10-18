"""
tests/test_alias_contract.py — ВАРИАНТ ИМЕНИ НЕ ЗАВОДИТ ВТОРУЮ СУЩНОСТЬ.

ЧТО ЭТОТ ТЕСТ ЗАПИРАЕТ
----------------------
Хозяин поставил модель явно: «бегемот» и «гиппопотам» — ЭТО ОДНА СУЩНОСТЬ с
двумя именами, а не две сущности. Раньше `entities` индексировался по лемме,
поэтому второе написание рождало второй узел со своим набором связей, и
разлепить их было нельзя.

Запирается и СПОСОБ получения синонима. Замер показал, что косинус эмбеддингов
его НЕ даёт: «бегемот/гиппопотам» = 0.218 против 0.134 у заведомо чужих
«стол/небо» — диапазоны перекрываются. Значит синонима можно УЗНАТЬ из
грамматики текста, но нельзя УГАДАТЬ. Отсюда два класса тестов:

  1. конструкция «X, или Y» + единственное число сказуемого -> ОДИН канон;
  2. конструкции, на которые это правило НЕ должно позариться (перечисление,
     винительный падеж, множественное число) -> ДВЕ разные сущности.

Ложное объединение необратимо и отравляет весь граф, поэтому проверок «не
слиять» здесь больше, чем проверок «слиять».

ПОЧЕМУ ТЕСТЫ УЧАТ ЧЕРЕЗ ФЕЙК-УЧИТЕЛЯ
------------------------------------
Замер до написания теста: без Учителя (`max_teacher_calls=0`) существительное
вообще НЕ становится сущностью — оно остаётся в unknown_words, и проверка
«во Вселенной одна сущность» была бы проверкой пустого словаря. Учитель здесь
подменяется ДЕТЕРМИНИРОВАННОЙ функцией (без сети и LLM), при этом её ответ
никак не про «бегемота»: вариант имени приходит из грамматики предложения, а
не из подсказки учителя.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from brain.learner import Learner
from core.extract import SimpleExtractor
from core.records import merge_entity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _teacher(word):
    """Учитель, который ничего не знает про бегемотов: только категория."""
    return {"concept": word, "hierarchy": ["животные"],
            "properties": [], "actions": []}


def _learner(tmp, teacher=_teacher, max_calls=6):
    """Мир в temporaryDirectory: тест не имеет права трогать живую память."""
    path = os.path.join(tmp, "world.json")

    def load_world():
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return {"entities": {}, "observations": {}, "unknown_words": {}}

    def save_world(w):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(w, fh, ensure_ascii=False)

    save_world(load_world())
    lr = Learner(load_world=load_world, save_world=save_world,
                 teacher=teacher, max_teacher_calls=max_calls)
    return lr, load_world, path


APPRESSION = "Бегемот, или гиппопотам, живёт в реке."


class TestAliasExtraction(unittest.TestCase):
    """Синтаксис: что считается вариантом имени, а что нет."""

    def setUp(self):
        self.ex = SimpleExtractor()

    def aliases(self, sentence):
        return [tuple(p) for p in self.ex.extract(sentence, 1).aliases]

    def test_apposition_gives_alias(self):
        self.assertEqual(self.aliases(APPRESSION), [("бегемот", "гиппопотам")])

    def test_apposition_not_limited_to_one_pair(self):
        self.assertEqual(self.aliases("Собака, или пёс, вернулась домой."),
                         [("собака", "пёс")])

    def test_plural_verb_is_enumeration_not_alias(self):
        # ЛОВУШКА. На уровне токенов «Кот, или собака, спят» неотличимо от
        # аппозиции. Если это правило отвалится, кто-нибудь выучит «кот=собака».
        self.assertEqual(self.aliases("Кот, или собака, спят."), [])

    def test_no_commas_is_enumeration(self):
        self.assertEqual(self.aliases("Кот или собака спят."), [])

    def test_nouns_after_verb_are_objects(self):
        self.assertEqual(self.aliases("Он выбрал стол, или стул."), [])

    def test_copula_is_not_alias(self):
        # «Гиппопотам — это бегемот» и «Кит — это млекопитающее» дают
        # ОДИНАКОВЫЙ разбор, поэтому тождество от роли здесь неотличимо.
        # Придумывать синонимию на таком синтаксисе нельзя.
        self.assertEqual(self.aliases("Гиппопотам — это бегемот."), [])

    def test_plain_sentence_untouched(self):
        self.assertEqual(self.aliases("Кошка сидит на подоконнике."), [])


class TestAliasStorage(unittest.TestCase):
    """Модель данных: одна сущность, список имён, связь SYNONYM_OF."""

    def test_one_entity_two_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            lr, load_world, _ = _learner(tmp)
            rep = lr.learn_from_text(APPRESSION, source="test")
            world = load_world()
            ents = world["entities"]
            self.assertIn("бегемот", ents)
            self.assertNotIn("гиппопотам", ents,
                             "вариант имени завёл ВТОРУЮ сущность")
            self.assertEqual(ents["бегемот"]["aliases"], ["гиппопотам"])
            self.assertIn("гиппопотам|SYNONYM_OF|бегемот",
                          world["observations"])
            self.assertGreaterEqual(rep.aliases_new, 1)
            self.assertGreaterEqual(rep.aliases_written, 1,
                                    "пара выведена, но в сущность не записана")

    def test_repeat_lesson_does_not_add_entities_or_aliases(self):
        # Дедупликация по КОЛИЧЕСТВУ, а не по побайтовому снимку файла:
        # переизучение УВЕЛИЧИВАЕТ support наблюдения (подкрепление), и это
        # правильное изменение, а не баг. Замерено: support = 1, 2, 3.
        with tempfile.TemporaryDirectory() as tmp:
            lr, load_world, _ = _learner(tmp)
            for _ in range(3):
                lr.learn_from_text(APPRESSION, source="test")
            world = load_world()
            self.assertEqual(sorted(world["entities"]),
                             ["бегемот", "жить", "река"],
                             "повторный урок наплодил сущностей")
            self.assertEqual(world["entities"]["бегемот"]["aliases"],
                             ["гиппопотам"],
                             "повторный урок удвоил вариант имени")
            obs = world["observations"]["гиппопотам|SYNONYM_OF|бегемот"]
            self.assertEqual(obs["relation"], "SYNONYM_OF")

    def test_reversed_word_order_does_not_add_entity(self):
        with tempfile.TemporaryDirectory() as tmp:
            lr, load_world, _ = _learner(tmp)
            lr.learn_from_text(APPRESSION, source="test")
            lr.learn_from_text(
                "Гиппопотам, или бегемот, выходит на берег.", source="test")
            ents = load_world()["entities"]
            names = [k for k in ents if "гиппопотам" in k or "бегемот" in k]
            self.assertEqual(names, ["бегемот"],
                             f"разбросалось по-разному: {sorted(ents)}")

    def test_alias_survives_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            lr, _, path = _learner(tmp)
            lr.learn_from_text(APPRESSION, source="test")
            with open(path, encoding="utf-8") as fh:
                back = json.load(fh)
            self.assertIn("гиппопотам", back["entities"]["бегемот"]["aliases"],
                          "алиас потерялся при сохранении/перечитывании")

    def test_merge_entity_keeps_aliases(self):
        # Слияние записей объединяет только списки из внутреннего кортежа
        # LISTS. Без "aliases" там второе имя исчезало бы молча: правило
        # «новое имеет право ДОБАВИТЬ, но не СТЕРЕТЬ» оставило бы в файле то
        # написание, которое попало раньше.
        a = {"name": "бегемот", "aliases": ["гиппопотам"]}
        b = {"name": "бегемот", "aliases": ["hippopotamus"]}
        for x, y in ((a, b), (b, a)):        # порядок не должен терять имя
            merged = merge_entity(x, y)
            self.assertEqual(set(merged["aliases"]),
                             {"гиппопотам", "hippopotamus"},
                             "merge_entity потерял вариант имени")

    def test_conflicting_canon_is_refused_and_loud(self):
        # Одно имя не может быть вариантом двух разных сущностей. Молча
        # переприсвоить — значит тихо переписать смысл уже выученного слова.
        with tempfile.TemporaryDirectory() as tmp:
            lr, load_world, _ = _learner(tmp)
            lr.learn_from_text(APPRESSION, source="test")
            rep = lr.learn_from_text(
                "Носорог, или гиппопотам, стоит в грязи.", source="test")
            world = load_world()
            self.assertIn("гиппопотам", world["entities"]["бегемот"]["aliases"])
            self.assertNotIn(
                "гиппопотам",
                world["entities"].get("носорог", {}).get("aliases") or [],
                "конфликтный вариант всё-таки записан вторым каноном")
            self.assertTrue(any("гиппопотам" in n for n in rep.notes),
                            f"конфликт не озвучен: {rep.notes}")

    def test_second_name_resolves_to_canon(self):
        # Прямое следствие модели: во ВТОРОМ тексте второе имя не рождает
        # новую сущность и не уходит к Учителя.
        with tempfile.TemporaryDirectory() as tmp:
            lr, load_world, _ = _learner(tmp)
            lr.learn_from_text(APPRESSION, source="test")
            lr.learn_from_text("Гиппопотам ест водоросли.", source="test")
            ents = load_world()["entities"]
            self.assertNotIn("гиппопотам", ents,
                             "второе имя стало отдельной сущностью")
            self.assertIn("бегемот", ents)

    def test_participant_links_point_to_canon(self):
        # БЕЗ этого шага текст «Бегемот, или гиппопотам, живёт в реке.
        # Гиппопотам ест водоросли.» даёт действие «есть» с участником
        # «гиппопотам», которого во Вселенной нет. Ссылка выглядит как
        # заполненное поле, а по факту ведёт в никуда — и молчит.
        with tempfile.TemporaryDirectory() as tmp:
            lr, load_world, _ = _learner(tmp)
            lr.learn_from_text(APPRESSION, source="test")
            rep = lr.learn_from_text("Гиппопотам ест водоросли.", source="test")
            parts = load_world()["entities"]["есть"]["participants"]
            self.assertEqual(parts["agent"], ["бегемот"],
                             "участник указывает на несуществующую сущность")
            self.assertGreaterEqual(rep.aliases_canonicalized, 1)

    def test_no_entity_without_teacher(self):
        # Рамка честности: без Учителя существительное НЕ становится сущностью
        # (замерено). Если этот тест упадёт — значит, путь «молча придумали
        # сущность без учителя» появился сам собой.
        with tempfile.TemporaryDirectory() as tmp:
            lr, load_world, _ = _learner(tmp, teacher=None, max_calls=0)
            lr.learn_from_text(APPRESSION, source="test")
            world = load_world()
            self.assertNotIn("бегемот", world["entities"])
            # не требовало
            self.assertIn("гиппопотам|SYNONYM_OF|бегемот",
                          world["observations"])


class TestNoCosineMerging(unittest.TestCase):
    """
    Слияние по косинусу запрещено, и это зафиксировано тестом, а не устной
    договорённостью: ложное объединение двух сущностей необратимо.
    """

    def test_synonyms_are_not_resolvable_by_similarity(self):
        try:
            from core.encoder import get_encoder
        except Exception:                                    # нет модели — ок
            self.skipTest("нет SDR-кодировщика: косинус не измерить")
        enc = get_encoder()
        if not getattr(enc, "semantic", False):
            self.skipTest("SDR не семантический: косинус не измерить")

        import numpy as np

        def cos(a, b):
            va = np.asarray(enc.encode(a), float)
            vb = np.asarray(enc.encode(b), float)
            d = float(np.linalg.norm(va) * np.linalg.norm(vb))
            return float(va @ vb / d) if d else 0.0

        far = cos("бегемот", "гиппопотам")
        stranger = cos("стол", "небо")
        self.assertLess(
            far, 0.6,
            "если косинус книжных синонимов вырос, переоценивать метод надо "
            "по всему замеру из experiments/MERGE_TASK.md, а не по паре")
        # Разрыв между синонимами и заведомо чужими словами мал: автоматом по
        # порогу такие пары не развести. Синонима обязан назвать текст,
        # Учитель или данные — но не мера близости векторов.
        self.assertLess(
            far - stranger, 0.35,
            "разрыв синонимов и чужих пар вырос — всё равно НЕ повод сливать "
            "автоматически: порог обязан быть явным и проверенным на всём")


if __name__ == "__main__":
    unittest.main(verbosity=2)
