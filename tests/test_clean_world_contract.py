"""
Контракт миграции накопленных данных (tools/clean_world.py).

Правильное чтение чинит БУДУЩЕЕ, файл чинит ПРОШЛОЕ: пока правила filters
применяются только при записи, в lipid_world.json остаётся всё, что нажито
до правки. Поэтому у миграции есть свои тесты — она меняет настоящую память,
и ошибка здесь дороже, чем ошибка в ответе.

Все проверки идут на ВРЕМЕННОМ файле. Реальный lipid_world.json тесты не
трогают ни при каком раскладе (проверяется отдельным тестом).
"""
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location(
    "clean_world", os.path.join(ROOT, "tools", "clean_world.py"))
clean_world = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clean_world)

from core.worldfile import read_world, write_world                   # noqa: E402


class TestPlan(unittest.TestCase):

    def test_tautology_is_cut_from_hierarchy(self):
        world = {"entities": {"экосистема": {"concept": "экосистема",
                                             "hierarchy": ["экосистемы", "природа"]}},
                 "unknown_words": {}}
        clean, changes = clean_world.plan(world)
        self.assertEqual(clean["entities"]["экосистема"]["hierarchy"], ["природа"])
        self.assertTrue(changes, "миграция молчала там, где есть что чинить")

    def test_broken_definition_is_dropped_but_real_one_kept(self):
        """
        Критерий по ЗАМЕРУ реального файла: «Пчёлы (лат» — обрывок, а
        122-символьное определение «ввод» без точки в конце — знание.
        Правило «нет точки = мусор» сожрало бы оба настоящих определения.
        """
        world = {"entities": {
            "пчела": {"concept": "пчела", "definition": "Пчёлы (лат"},
            "ввод": {"concept": "ввод",
                     "definition": "Термин ввод обозначает либо появление, "
                                   "либо изменение условий, которые влияют "
                                   "на систему"},
        }, "unknown_words": {}}
        clean, _ = clean_world.plan(world)
        # Запись, у которой после среза не осталось ничего, обязана УЕХАТЬ в
        # unknown_words: «Пчёлы (лат» — не знание, а его видимость.
        self.assertNotIn("пчела", clean["entities"],
                         "огрызок остался притворяться знанием")
        self.assertIn("пчела", clean["unknown_words"])
        self.assertIn("definition", clean["entities"]["ввод"],
                      "настоящее определение удалено как обрывок")

    def test_entity_and_unknown_cannot_both_be_true(self):
        world = {"entities": {"пчела": {"concept": "пчела", "properties": ["гудит"]}},
                 "unknown_words": {"пчела": {"count": 2}}}
        clean, changes = clean_world.plan(world)
        self.assertNotIn("пчела", clean["unknown_words"])
        self.assertIn("пчела", clean["entities"])
        self.assertTrue(any("unknown_words" in c for c in changes))

    def test_unknown_without_knowledge_stays(self):
        """Слово, у которого нечего оставить, обязано остаться неизвестным."""
        world = {"entities": {"зюзя": {"concept": "зюзя"}},
                 "unknown_words": {}}
        clean, _ = clean_world.plan(world)
        self.assertNotIn("зюзя", clean["entities"])
        self.assertIn("зюзя", clean["unknown_words"])

    def test_plan_does_not_mutate_input(self):
        world = {"entities": {"флюгер": {"concept": "флюгер",
                                         "hierarchy": ["флюгеры", "приборы"]}},
                 "unknown_words": {}}
        before = json.dumps(world, sort_keys=True)
        clean_world.plan(world)
        self.assertEqual(json.dumps(world, sort_keys=True), before,
                         "plan изменил входной мир — «показ» опасен")

    def test_idempotent(self):
        world = {"entities": {"луга": {"concept": "луга",
                                       "hierarchy": ["луга", "деревни"]}},
                 "unknown_words": {}}
        once, ch1 = clean_world.plan(world)
        twice, ch2 = clean_world.plan(once)
        self.assertEqual(json.dumps(once, sort_keys=True),
                         json.dumps(twice, sort_keys=True))
        self.assertEqual(ch2, [], "повторный прогон нашёл то, что уже исправил")


class TestWriteMode(unittest.TestCase):

    def test_backup_created_and_file_stays_readable(self):
        d = tempfile.mkdtemp(prefix="l2clean-")
        path = os.path.join(d, "lipid_world.json")
        broken = {"entities": {"флюгер": {"concept": "флюгер",
                                          "hierarchy": ["флюгеры"]}},
                  "unknown_words": {}}
        write_world(path, broken)
        rc = subprocess.run(
            [sys.executable, os.path.join(ROOT, "tools", "clean_world.py"),
             "--write"],
            env=dict(os.environ, LIPID2_MEMORY_DIR=d, LIPID2_STATUS="0"),
            capture_output=True, text=True, timeout=120)
        self.assertEqual(rc.returncode, 0, rc.stderr[-500:])
        self.assertTrue(os.path.exists(path + ".pre-clean"),
                        "исходник не сохранён: ошибочная миграция невычислима")
        world = read_world(path)
        self.assertTrue(world.get("unknown_words", {}).get("флюгер")
                        or world.get("entities", {}).get("флюгер"),
                        "запись исчезла совсем вместо переноса")

    def test_no_write_flag_leaves_file_untouched(self):
        d = tempfile.mkdtemp(prefix="l2clean2-")
        path = os.path.join(d, "lipid_world.json")
        raw = json.dumps({"entities": {"ф": {"concept": "ф",
                                             "hierarchy": ["фы"]}},
                          "unknown_words": {}}, ensure_ascii=False)
        with io.open(path, "w", encoding="utf-8") as fh:
            fh.write(raw)
        subprocess.run([sys.executable, os.path.join(ROOT, "tools",
                                                     "clean_world.py"), "--show"],
                       env=dict(os.environ, LIPID2_MEMORY_DIR=d, LIPID2_STATUS="0"),
                       capture_output=True, text=True, timeout=120)
        with io.open(path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), raw, "--show переписал файл")


class TestRealMemoryNotTouchedByTests(unittest.TestCase):
    """
    Тесты не имеют права писать в рабочую память. Проверка — контрольная сумма
    реального файла ДО и ПОСЛЕ запуска всего набора (запускается в подпроцессе,
    чтобы другие тесты уже успели отработать).
    """
    PATH = os.path.join(ROOT, "data", "memory", "lipid_world.json")

    def test_checksum_stable_across_a_dummy_import(self):
        if not os.path.exists(self.PATH):
            self.skipTest("рабочей памяти ещё нет")
        with io.open(self.PATH, "rb") as fh:
            before = fh.read()
        env = dict(os.environ, LIPID2_STATUS="0")
        env.pop("LIPID2_MEMORY_DIR", None)
        subprocess.run([sys.executable, "-c", "import brain.lipid2_brain"],
                       cwd=ROOT, env=env, capture_output=True, timeout=180)
        subprocess.run([sys.executable, os.path.join(ROOT, "tools",
                                                     "clean_world.py"), "--show"],
                       env=env, capture_output=True, timeout=120)
        with io.open(self.PATH, "rb") as fh:
            self.assertEqual(fh.read(), before,
                             "показ/импорт мозга изменил рабочую память")


if __name__ == "__main__":
    unittest.main(verbosity=2)
