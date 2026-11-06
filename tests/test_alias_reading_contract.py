"""
ВАРИАНТ ИМЕНИ ОБЯЗАН НАХОДИТЬ ЗНАНИЕ ПРИ ЧТЕНИИ.

ЗАЧЕМ ЭТОТ ФАЙЛ. Запертый test_alias_contract закрывает ЗАПИСЬ: «X, или Y»
не заводит вторую сущность. Этот файл закрывает вторую половину той же
модели хозяина («бегемот и гиппопотам — ЭТО ОДНА СУЩНОСТЬ») — ЧТЕНИЕ.

Замер 04.11, живой мир из одной записи «бегемот» c aliases:["гиппопотам"]:

    запрос «бегемот»     -> НАЙДЕНО
    запрос «гиппопотам»  -> НЕ НАЙДЕНО
    то же + наблюдение SYNONYM_OF рядом -> ВСЁ РАВНО НЕ НАЙДЕНО

Поле aliases при поиске не читает НИ ОДЕН модуль мозга (grep по
brain/lipid2_brain.py на "aliases" пуст). Синонимический резолв в ядре есть
(GraphMemory.resolve_canonical_node, core/graph.py:274), но мозг не вызывает
его ни разу. Наблюдение SYNONYM_OF помогает только learner-у на записи
(_world_aliases), то есть второй клад с тем же знанием читается в другом
месте и не читается там, где спрашивает человек.

Замер же многочленных имён (пример хозяина про «падшая женщина»):
двухсловная сущность НАХОДИТСЯ (и в косвенных падежах — сведение форм
работает), а её aliases «блудница», «шлюха» — нет. Одна и та же дыра
закрывает и бегемота, и все двадцать слов: алиас должен работать в чтении,
отдельного механизма не просим.

ЗАМОК ОДНОСТОРООННИЙ НАПРЯВЛЕННОСТЬЮ, А НЕ СИЛОЙ: здесь проверяется, что
знание ОДНО и доступ по любому имени. «Не слиять лишнее» заперто там же,
в test_alias_contract — дублировать не нужно, там проверок «не слить» больше.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def write_world(dirpath: str, entities: dict, observations: dict = None):
    with open(os.path.join(dirpath, "lipid_world.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"entities": entities,
                   "observations": observations or {}}, fh, ensure_ascii=False)


class TestAliasFindsKnowledge(unittest.TestCase):
    """Имя из aliases — это то же знание, а не неизвестное слово."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="l2alias_")
        os.environ["LIPID2_STATUS"] = "0"

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _brain(self, entities: dict, observations: dict = None):
        write_world(self.dir, entities, observations)
        from brain.lipid2_brain import Lipid2Brain
        b = Lipid2Brain(memory_dir=self.dir)
        b.use_llm_teacher = False
        b.use_wiki_teacher = False
        b.ask_user_enabled = False
        return b

    def test_alias_finds_entity(self):
        """Замер дыры: «гиппопотам» не находил «бегемота»."""
        b = self._brain({
            "бегемот": {"concept": "бегемот", "aliases": ["гиппопотам"],
                        "hierarchy": ["млекопитающие"],
                        "definition": "травоядное животное"},
        })
        data = b.find_in_memory("гиппопотам")
        self.assertIsNotNone(
            data, "вариант имени из aliases не нашёл знание — мозг пойдёт "
                  "к LLM за копией того, что уже лежит в файле")
        self.assertEqual(data.get("matched_concept"), "бегемот",
                         "найдено не то, что должно: алиас обязан вести К "
                         "канону, а не создавать видимость ответа")

    def test_answer_by_alias(self):
        """Сквозная проверка: вопрос по варианту имени получает определение
        канона, а не «не знаю»."""
        b = self._brain({
            "бегемот": {"concept": "бегемот", "aliases": ["гиппопотам"],
                        "definition": "травоядное животное"},
        })
        out = b.process_input("что такое гиппопотам")
        self.assertIn("травоядное животное", out,
                      f"вопрос по варианту имени не получил знания: {out!r}")

    def test_multiword_synonym_chain(self):
        """Многословный канон и десятки вариантов (пример хозяина про
        «падшая женщина»): любой alias обязан вести к одной записи."""
        b = self._brain({
            "падшая женщина": {
                "concept": "падшая женщина",
                "aliases": ["блудница", "распутница", "шлюха", "потаскуха"],
                "definition": "женщина, ведущая распутный образ жизни"},
        })
        for alias in ("блудница", "шлюха", "потаскуха"):
            with self.subTest(alias=alias):
                data = b.find_in_memory(alias)
                self.assertIsNotNone(
                    data, f"вариант «{alias}» не нашёл двухсловный канон")
                self.assertEqual(data.get("matched_concept"),
                                 "падшая женщина")

    def test_alias_of_declined_form(self):
        """Вариант имени приходит в тексте в падеже: «про гиппопотама»."""
        b = self._brain({
            "бегемот": {"concept": "бегемот", "aliases": ["гиппопотам"],
                        "definition": "травоядное животное"},
        })
        self.assertIsNotNone(b.find_in_memory("гиппопотам"),
                             "базовая форма алиаса не находит — производные "
                             "формы проверять бессмысленно")

    def test_no_second_curiosity_for_alias(self):
        """ЦЕНА: вопрос по варианту имени не имеет права звать обучение — иначе
        вторая сущность родится на ЧТЕНИИ, хотя запись осталась одной."""
        calls = []
        b = self._brain({
            "бегемот": {"concept": "бегемот", "aliases": ["гиппопотам"],
                        "definition": "травоядное животное"},
        })
        orig = b._trigger_curiosity_and_learn
        b._trigger_curiosity_and_learn = lambda c: (calls.append(c),
                                                    orig(c))[1]
        b.process_input("что такое гиппопотам")
        self.assertEqual(calls, [],
                         f"знание найдено под именем канона, а учёба вызвана "
                         f"для «{calls}» — так и рождается вторая сущность")

    def test_alias_does_not_invent_target(self):
        """ЖИВОСТЬ В СТОРОНУ ОТКАЗА: чужое слово обязано остаться
        неизвестным. Мутация «алиасы = искать по любому совпадению» или
        «при промахе заводить запись» поймается здесь."""
        b = self._brain({
            "бегемот": {"concept": "бегемот", "aliases": ["гиппопотам"],
                        "definition": "травоядное животное"},
        })
        self.assertIsNone(b.find_in_memory("носорог"),
                          "резолвер по алиасам начал находить то, чего нет")

    def test_conflicting_alias_does_not_steal(self):
        """Конфликт данных (тот же закон, что в _record_aliases): вариант,
        уже привязанный к другому канону, не имеет права перетянуть чтение.
        Здесь: у двух записей один и тот же alias — ответ обязан быть
        ДЕТЕРМИНИРОВАН и вести к одному канону, а не «последним найденным»."""
        b = self._brain({
            "бегемот": {"concept": "бегемот", "aliases": ["зверь"],
                        "definition": "травоядное животное"},
            "носок": {"concept": "носок", "aliases": ["зверь"],
                      "definition": "часть одежды"},
        })
        data = b.find_in_memory("зверь")
        self.assertIsNotNone(data)
        self.assertIn(data.get("matched_concept"), ("бегемот", "носок"),
                      "конфликтующий алиас выдал выдуманный канон")
        # главный инвариант: один и тот же запрос даёт ОДИН и тот же ответ
        again = b.find_in_memory("зверь")
        self.assertEqual(again.get("matched_concept"),
                         data.get("matched_concept"),
                         "один запрос — два разных ответа: чтение по алиасу "
                         "недетерминировано")


if __name__ == "__main__":
    unittest.main()
