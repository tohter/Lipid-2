# -*- coding: utf-8 -*-
"""Кирпич 9b: замок «симулятор не пишет» обязан быть зрячим."""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("LIPID2_STATUS", "0")

import tests.test_simulator_contract as contract


class SpyDict(dict):
    """Словарь, кричащий на ЛЮБУЮ запись, даже если значение не изменилось."""

    def __init__(self, data, log, path=""):
        super().__init__(data)
        self._log = log
        self._path = path

    def __setitem__(self, key, value):
        self._log.append(("запись", self._path, key))
        super().__setitem__(key, value)

    def __delitem__(self, key):
        self._log.append(("удаление", self._path, key))
        super().__delitem__(key)

    def update(self, *args, **kwargs):
        self._log.append(("update", self._path, None))
        super().update(*args, **kwargs)

    def pop(self, *args):
        self._log.append(("pop", self._path, args[0] if args else None))
        return super().pop(*args)

    def setdefault(self, key, default=None):
        self._log.append(("setdefault", self._path, key))
        return super().setdefault(key, default)


def guarded_world(world):
    """Мир, где каждая секция-словарь сторожится и помнит записи."""
    log = []
    wrapped = SpyDict(dict(world), log, "<root>")
    for key, val in dict(world).items():
        if isinstance(val, dict):
            dict.__setitem__(wrapped, key, SpyDict(val, log, key))
    return wrapped, log


class TestLockSeesWriteAndForgive(unittest.TestCase):
    """Дыра 2: запись «туда и обратно» (новый объект, то же содержимое)
    обязана ловиться — старый замок (world == before) на ней МОЛЧИТ.

    Доказано замером: tests/test_simulator_contract.py::TestSimulatorOnlyReads
    сравнивает мир до/после по == и не видит, как _read пересохранил список
    тем же содержимым. SpyDict ловит сам ФАКТ записи, а не её результат.

    Нарушителя тест не разыгрывает руками: ставим на место _read мутацию,
    которая «по доброте» пересохраняет relations тем же содержимым — и
    замок обязан покраснеть ровно на ней.
    """

    def _run_with(self, sim):
        for ev in (("пожар", "начаться"), ("дождь", "лить"),
                   ("пожар", "остановиться")):
            sim.add_event(*ev)
        return sim.why_stopped("пожар")

    def test_clean_run_writes_nothing(self):
        world, log = guarded_world(contract.load_world())
        sim = contract.make_story(world)
        res = self._run_with(sim)
        self.assertTrue(res.ok, "законный вывод обязан работать на стороже")
        self.assertEqual(log, [], "симулятор записал в статику: %r" % (log,))


class TestLockCoversEveryReading(unittest.TestCase):
    """Дыра 1: сторож проверяет все точки чтения, а не одну why_stopped."""

    def test_all_queries_do_not_write(self):
        world, log = guarded_world(contract.load_world())
        sim = contract.make_story(world)
        k = sim.knowledge
        calls = [
            lambda: k.is_a("дождь"),
            lambda: k.parts("огонь"),
            lambda: k.can_do("вода"),
            lambda: k.object_of("гасить"),
            lambda: k.causes("дождь"),
            lambda: k.action_purpose("гасить"),
            lambda: k.referenced_by("вода"),
            lambda: k.witnesses("вода", "CAN_DO", "гасить"),
            lambda: k.knows("вода"),
            lambda: k.related("вода", "CAN_DO"),
            lambda: k.identity_path("пожар", "огонь"),
            lambda: sim.why_stopped("пожар"),
            lambda: sim.why_stopped("стол"),
        ]
        for call in calls:
            try:
                call()
            except (KeyError, ValueError):
                pass
        self.assertEqual(log, [], "записано в статику: %r" % (log,))


class TestLockSeesInstrumentMutation(unittest.TestCase):
    """Дыра 3: повторное чтение не должно менять САМ прибор (кэш — это запись)."""

    def _snapshot(self, st):
        by_head = {k: sorted(v) for k, v in getattr(st, "_by_head", {}).items()}
        by_tail = {k: sorted(v) for k, v in getattr(st, "_by_tail", {}).items()}
        known = sorted(getattr(st, "_known", set()) or ())
        cache = sorted(getattr(st, "_readable_cache", set()) or ())
        return json.dumps(
            {"by_head": by_head, "by_tail": by_tail,
             "known": known, "cache": cache},
            sort_keys=True, ensure_ascii=False)

    def test_second_reading_does_not_mutate_instrument(self):
        sim = contract.make_story(contract.load_world())
        self.assertTrue(sim.why_stopped("пожар").ok)
        first = self._snapshot(sim.knowledge)
        self.assertTrue(sim.why_stopped("пожар").ok)
        second = self._snapshot(sim.knowledge)
        self.assertEqual(first, second,
                         "повторное чтение изменило состояние прибора")


if __name__ == "__main__":
    unittest.main(verbosity=2)
