"""
Контракт кэша мира (очередь, п.13).

Кэш обязан: (1) отдавать мир без повторного разбора, (2) видеть ЧУЖУЮ
запись в файл (иначе «что такое кошка» отвечало бы по устаревшей памяти).
Проверяем оба свойства — второе важнее первого.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.lipid2_brain import Lipid2Brain


class TestWorldCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lipid_cache_")
        self.path = os.path.join(self.tmp, "lipid_world.json")
        self._write({"entities": {"кошка": {"concept": "кошка"}},
                     "relations": [], "observations": {},
                     "unknown_words": {}, "transitions": {}})
        self.brain = Lipid2Brain(memory_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, world):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(world, f, ensure_ascii=False)

    def test_second_read_returns_cached_object(self):
        w1 = self.brain._load_world()
        w2 = self.brain._load_world()
        self.assertIs(w1, w2, "повторное чтение не попало в кэш")

    def test_external_write_invalidates_cache(self):
        self.brain._load_world()          # прогреваем кэш
        world = {"entities": {"кошка": {"concept": "кошка"},
                              "собака": {"concept": "собака"}},
                 "relations": [], "observations": {},
                 "unknown_words": {}, "transitions": {}}
        self._write(world)                # «чужой» писатель
        again = self.brain._load_world()
        self.assertIn("собака", again["entities"],
                          "кэш не увидел чужую запись — устаревшая память")

    def test_save_world_keeps_cache_fresh(self):
        w = self.brain._load_world()
        w["entities"]["птица"] = {"concept": "птица"}
        self.brain._save_world(w)
        again = self.brain._load_world()
        self.assertIn("птица", again["entities"])


if __name__ == "__main__":
    unittest.main(verbosity=2)