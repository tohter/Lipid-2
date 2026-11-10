"""
ЗАМЕР: пишет ли симулятор в статическую базу во время вывода?

Заявление в коде (brain/simulator.py, класс Simulator):
    "Знания берёт ТОЛЬКО из статической базы (StaticKnowledge) и туда не пишет."

Заявление - не замок. Этот файл замеряет его:
  1) снимок мира ДО и ПОСЛЕ успешного вывода (deep-копия + json);
  2) сравнение ПО СЕКЦИЯМ (общее "нет" не должно прятать правку одной секции);
  3) капкан: подменяем секции мира dict'ом, который КРИЧИТ на запись.

Запуск:  LIPID2_STATUS=0 .venv/bin/python experiments/measure_sim_writes.py
"""

import copy
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
os.environ.setdefault("LIPID2_STATUS", "0")

import tests.test_simulator_contract as contract


class TripwireDict(dict):
    """dict, который запоминает попытки записи и не мешает выводу."""

    def __init__(self, data, log, path=""):
        super().__init__(data)
        self._log = log
        self._path = path

    def __setitem__(self, k, v):
        self._log.append(("%s[%r]" % (self._path, k), "setitem"))
        super().__setitem__(k, v)

    def __delitem__(self, k):
        self._log.append(("%s[%r]" % (self._path, k), "delitem"))
        super().__delitem__(k)

    def update(self, *a, **kw):
        self._log.append((self._path or "<root>", "update"))
        super().update(*a, **kw)

    def pop(self, *a):
        self._log.append((self._path or "<root>", "pop"))
        return super().pop(*a)

    def setdefault(self, k, d=None):
        self._log.append(("%s[%r]" % (self._path, k), "setdefault"))
        return super().setdefault(k, d)


def run():
    sim = contract.build_fire_rain_simulator()
    world = sim.knowledge.world
    deep, text = copy.deepcopy(world), json.dumps(
        world, sort_keys=True, ensure_ascii=False, default=str)
    print("мир: %d секций - %s" % (len(deep), ", ".join(sorted(deep))))

    r1 = sim.why_stopped("пожар")
    print("вывод №1: ok=%s  ответ=%r" % (r1.ok, r1.answer))
    after1 = json.dumps(world, sort_keys=True, ensure_ascii=False, default=str)
    print("статика после вывода №1 изменена:", "ДА" if after1 != text else "нет")

    r2 = sim.why_stopped("пожар")
    after2 = json.dumps(world, sort_keys=True, ensure_ascii=False, default=str)
    print("вывод №2 (кэши): ok=%s  статика изменена: %s"
          % (r2.ok, "ДА" if after2 != after1 else "нет"))

    base, now = json.loads(text), json.loads(after2)
    diff = [k for k in set(base) | set(now) if base.get(k) != now.get(k)]
    print("секции, разошедшиеся с исходным снимком:", diff or "ни одной")
    print("events в мире StaticKnowledge:", "events" in world)
    print("события в симуляторе (своя память):", len(sim.events))


def tripwire():
    """Ловит запись, которая случайно записала ПРЕЖНЕЕ значение (снимок её не видит)."""
    sim = contract.build_fire_rain_simulator()
    world = sim.knowledge.world
    log = []
    wrapped = {}
    for k, v in world.items():
        wrapped[k] = TripwireDict(v, log, k) if isinstance(v, dict) else v
    sim.knowledge.world = wrapped
    try:
        r = sim.why_stopped("пожар")
        print("выход через капкан: ok=%s  ответ=%r" % (r.ok, r.answer))
    except Exception as exc:  # noqa: BLE001
        print("выход через капкан падает: %s" % type(exc).__name__)
    print("записей в мир за вывод:", len(log))
    for item in log[:10]:
        print("   ", item)


if __name__ == "__main__":
    run()
    tripwire()
