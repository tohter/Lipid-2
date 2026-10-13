"""
Контракт транзакций мира (очередь, п.11).

Проверяем ОДНО свойство, ради которого добавлен world_transaction:
параллельные писатели (фоновое обучение + Учитель) НЕ ТЕРЯЮТ правки
друг друга. Без замка read-modify-write теряет обновления — и потеря
неотличима от «такого знания не было».
"""
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.worldfile import read_world, write_world, world_transaction


EMPTY = {"entities": {}, "relations": [], "observations": {},
         "unknown_words": {}, "transitions": {}}


class WorldTransactionTest(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_transaction_saves_on_exit(self):
        with world_transaction(self.path) as world:
            world["entities"]["кошка"] = {"concept": "кошка"}
        again = read_world(self.path)
        self.assertIn("кошка", again["entities"])

    def test_transaction_skips_save_on_error(self):
        try:
            with world_transaction(self.path) as world:
                world["entities"]["кошка"] = {"concept": "кошка"}
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        # Файла не было — исключение внутри блока не должно его создавать
        self.assertFalse(os.path.exists(self.path))

    def test_parallel_writers_do_not_lose_updates(self):
        """
        50 потоков, каждый добавляет СВОЮ сущность через транзакцию.
        После завершения в мире обязаны быть ВСЕ 50.
        Без замка часть правок теряется на read-modify-write.
        """
        write_world(self.path, dict(EMPTY))
        n = 50
        barrier = threading.Barrier(n)

        def worker(idx):
            barrier.wait()
            with world_transaction(self.path) as world:
                world["entities"][f"word{idx}"] = {"concept": f"word{idx}"}

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        final = read_world(self.path)
        missing = [f"word{i}" for i in range(n)
                   if f"word{i}" not in final["entities"]]
        self.assertEqual(missing, [],
                         f"потеряно правок: {len(missing)} из {n}")


if __name__ == "__main__":
    unittest.main()