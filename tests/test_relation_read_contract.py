"""
КОНТРАКТ ДЫРЫ F: ОТНОШЕНИЕ, КОТОРОЕ МИР ПИШЕТ, А СПРОСИТЬ НЕЧЕМ.

ЧТО ЗАМЕРЕНО ДО ПРАВКИ (experiments/F_SEMANTICS.txt, учителя выключены,
живая память не тронута). Рассказ про Зинаиду + два урока дают 15 рёбер с
головой. Публичные методы симулятора покрывают ДВА отношения из шести,
которые мир реально пишет:

    отношение      рёбер   сколько связей вернул публичный метод
    CAN_DO             4                                  4
    OBJECT_OF          5                                  5
    IS_ROLE            2                          НЕТ МЕТОДА
    PART_OF            2                          НЕТ МЕТОДА
    HAS_PROPERTY       1                          НЕТ МЕТОДА
    LOCATED_IN         1                          НЕТ МЕТОДА

40% рёбер рассказа не читаются НИ ОДНИМ публичным методом и НИ ОДНОЙ
схемой. Путь загадки я теперь НАХОЖУ (П2, referenced_by), но проецировать
его на вывод нечем: шаги схемы, просящие IS_ROLE, получают пустоту.

ТРИ ПРИЧИНЫ, И ВСЕ ТРИ ИЗМЕРЕНЫ — они РАЗНЫЕ, поэтому и тесты разные.

1) ОБЩЕГО ПУБЛИЧНОГО ЧТЕНИЯ НЕТ. В StaticKnowledge есть _tails(x, rel) —
   он принимает ЛЮБОЕ имя отношения и отвечает правильно, в том числе для
   IS_ROLE и PART_OF. Он ПРИВАТНЫЙ. Публичных методов восемь: шесть
   специализированных по одному отношению, knows и referenced_by. То есть
   любое НОВОЕ отношение реестра становится нечитаемым, пока кто-то не
   допишет седьмой метод. Это шире, чем «не хватает шести методов»:
   дыра в ФОРМЕ доступа, а не в списке имён.

2) СХЕМЫ ПРОСЯТ ОТНОШЕНИЕ ПО ТАБЛИЦЕ ИЗ ШЕСТИ ИМЁН
   (core/schemes.py::_direct). За пределом таблицы — молчаливый пустой
   список. Проверено: шаг схемы с relation="IS_ROLE" на живом ребре
   «невестка IS_ROLE жена» даёт found=[] и в res.missing пишется
   «нет ребра IS_ROLE» — фраза, которая ВРАНЬЁ: ребро есть. Отказ обязан
   быть ОТЛИЧИМ от «ребра нет» (то же правило, что заперто для П2:
   ложная связь хуже пропущенной, а молчаливый ноль хуже отказа).

3) ПИШУЩИЙ КОД МЕСТОМ ПИШЕТ НЕ ТО ОТНОШЕНИЕ — ЭТО ДРУГАЯ ДЫРА, и в этом
   файле её НЕТ. Два ребра из шести нечитаемых семантически неверны:
   «жена PART_OF сын» и «сын PART_OF зинаида» читаются как «жена — составная
   часть сына» и «сын — составная часть Зинаиды», тогда как факт —
   семейная принадлежность. Правка читателя СДЕЛАЛА БЫ ЭТУ ЛОЖЬ ВИДИМОЙ
   схемам, поэтому писатель лечится ОТДЕЛЬНЫМ коммитом и отдельным
   решением о том, каким отношением вообще кодировать родство. См.
   experiments/F_SEMANTICS.txt.

Граница этого файла: ЧТЕНИЕ. Ни разу не про «кто прав», только про «чем
спросить». Поэтому тесты ниже обязаны починиться одной правкой формы
доступа и не должны зависеть от того, какое отношение писатель выберет.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from brain.simulator import StaticKnowledge          # noqa: E402
from core.schemes import Scheme, SchemeInterpreter, Step   # noqa: E402


def world_of(edges):
    """Мир из одних наблюдений — чтобы тест не зависел от онтологии."""
    obs = {}
    for h, r, t in edges:
        obs[f"{h}|{r}|{t}"] = {"head": h, "relation": r, "tail": t}
    return {"observations": obs, "entities": {}}


# Кирпичи загадки в том виде, в каком их пишет разбор (замер).
RIDDLE = [("невестка", "IS_ROLE", "жена"),
          ("жена", "PART_OF", "сын"),
          ("егор", "IS_ROLE", "сын"),
          ("сын", "PART_OF", "зинаида"),
          ("зинаида", "CAN_DO", "гостить"),
          ("гостить", "LOCATED_IN", "семья"),
          ("сын", "HAS_PROPERTY", "свой")]

# Отношения, которые мир ПИШЕТ в рассказе. Ровно те, где болит.
WRITTEN = ("IS_ROLE", "PART_OF", "HAS_PROPERTY", "LOCATED_IN")
# Те же самые, уже читаемые двумя методами.
WORKING = ("CAN_DO", "OBJECT_OF")


class TestGeneralReadExists(unittest.TestCase):
    """ПРИЧИНА 1: публичного чтения «по имени отношения» нет."""

    def test_public_read_by_relation_name(self):
        """Должен существовать ОТКРЫТЫЙ метод related(узел, отношение),
        принимающий ЛЮБОЕ имя отношения, а не шесть закрытованных по одному.
        Сейчас единственный такой метод — приватный _tails."""
        sk = StaticKnowledge(world_of(RIDDLE))
        self.assertTrue(
            hasattr(sk, "related") and callable(sk.related),
            "публичного обхода по имени отношения нет: чтение зашито в шесть "
            "методов по одному отношению, и каждое следующее отношение "
            "реестра (а их 17) родится нечитаемым")

    def test_reads_every_relation_the_world_writes(self):
        """Все четыре «немых» отношения отвечают через ЭТОТ метод — и
        отвечают ПРАВИЛЬНО, а не пустотой."""
        sk = StaticKnowledge(world_of(RIDDLE))
        want = {
            ("невестка", "IS_ROLE"): ["жена"],
            ("жена", "PART_OF"): ["сын"],
            ("сын", "HAS_PROPERTY"): ["свой"],
            ("гостить", "LOCATED_IN"): ["семья"],
        }
        bad = []
        for (node, rel), exp in want.items():
            got = sk.related(node, rel)
            if sorted(got) != sorted(exp):
                bad.append(f"{node} --{rel}--> получил {list(got)}, ждал {exp}")
        self.assertEqual(bad, [], "отношения мира нечитаемы:\n  " +
                         "\n  ".join(bad))

    def test_unknown_relation_is_not_silently_empty(self):
        """Отношение вне реестра — отказ, а не пустой список. Иначе
        опечатка в имени отношения (IS_ROL) станет «знания нет», и вывод
        замолчит, а не сломается."""
        sk = StaticKnowledge(world_of(RIDDLE))
        with self.assertRaises(KeyError):
            sk.related("невестка", "IS_ROL")

    def test_known_node_without_such_edge_is_empty_not_error(self):
        """Обратная сторона того же: знакомый узел, знакомое отношение,
        просто нет ребра -> пустой список. Пусто и отказ — РАЗНЫЕ ответы."""
        sk = StaticKnowledge(world_of(RIDDLE))
        self.assertEqual(list(sk.related("невестка", "LOCATED_IN")), [])


class TestSpecializedAgreeWithGeneral(unittest.TestCase):
    """Шесть специализированных методов обязаны стать ОБЁРТКАМИ над общим,
    а не жить второй копией чтения: две копии расходятся тихо."""

    def test_wrappers_do_not_disagree(self):
        sk = StaticKnowledge(world_of(RIDDLE))
        pairs = [("зинаида", "CAN_DO", sk.can_do),
                 ("гостить", "OBJECT_OF", sk.object_of),
                 ("гостить", "LOCATED_IN", None)]
        bad = []
        for node, rel, fn in pairs:
            if fn is None:
                continue
            if sorted(fn(node)) != sorted(sk.related(node, rel)):
                bad.append(f"{node}/{rel}: обёртка {sorted(fn(node))} против "
                           f"общего {sorted(sk.related(node, rel))}")
        self.assertEqual(bad, [], "специализированный метод разошёлся с "
                                  "общим:\n  " + "\n  ".join(bad))

    def test_registry_has_no_unreadable_relation(self):
        """ГЛАВНЫЙ ЗАМОК. Для каждого отношения, которое экстрактор способен
        записать, существует путь чтения. Сейчас их 17 в реестре, и девять
        недоступны ни публичному коду, ни схемам."""
        from core.storage import RelationType
        sk = StaticKnowledge(world_of(RIDDLE))
        unreadable = []
        for rel in [e.name for e in RelationType]:
            try:
                sk.related("невестка", rel)
            except KeyError:
                unreadable.append(rel)
        self.assertEqual(unreadable, [],
                         "отношений, которых нельзя спросить: "
                         + ", ".join(unreadable))


class TestSchemeCanAskAnyRelation(unittest.TestCase):
    """ПРИЧИНА 2: таблица из шести имён в core/schemes.py::_direct."""

    def _run(self, edges, relation, node="невестка"):
        sk = StaticKnowledge(world_of(edges))
        sch = Scheme(name="проверка", question="?",
                     steps=[Step(kind="edge", role="кто", frm="$subject",
                                 relation=relation)])
        return SchemeInterpreter(sk).run(sch, {"subject": node})

    def test_scheme_reads_is_role(self):
        """Шаг схемы с relation=IS_ROLE на ЖИВОМ ребре обязан привязать
        роль. Сейчас — пустота, и цепочка загадки обрывается на первом шаге."""
        res = self._run(RIDDLE, "IS_ROLE")
        self.assertEqual(res.bindings.get("кто"), ["жена"],
                         f"схема не прочитала IS_ROLE; missing={res.missing}")

    def test_silent_zero_is_not_an_answer(self):
        """САМЫЙ ВАЖНЫЙ ТЕСТ ФАЙЛА, и он переписан ПОСЛЕ починки — специально,
        чтобы не остаться мёртвым (урок 74). До правки он проверял «при
        живом IS_ROLE не писать «ребра нет»». После правки IS_ROLE читается,
        res.ok=True, и та же проверка зелёная НИЧЕГО НЕ МЕРЯ.
        Проверяемое состояние перенесено туда, где оно ЖИВЁТ и теперь:
        нечитаемое отношение. Обязано называться нечитаемым, а не пустотой."""
        walk = self._run(RIDDLE, "IS_ROL")   # опечатка в имени отношения
        self.assertFalse(walk.ok,
                         "схема «прошла» с отношением, которого нет ни в "
                         "реестре, ни в мире — значит нечитаемое отношение "
                         "считается пустотой")

        # ЯДРО: отказ — структура (роль, отношение), а не сочинённая фраза.
        self.assertTrue(walk.unreadable,
                        msg=f"ядро потеряло отказ: unreadable="
                            f"{walk.unreadable} missing={walk.missing}")
        self.assertEqual([], list(walk.missing),
                          msg=f"ядро сочиняет текст вместо структуры: "
                              f"{walk.missing}")

        # МОЗГ: ту же структуру он называет человеку ОТДЕЛЬНО от «не хватило».
        # Проверка на двух концах, а не на строке: смена формулировки не
        # должна делать проверку мёртвой, а удаление отказа — ловиться.
        from brain.simulator import SimulationResult, absorb_walk
        res = SimulationResult("в")
        absorb_walk(res, walk)
        text = " ".join(res.missing).lower()
        self.assertIn("не читается", text,
                      msg=f"отчёт враньё: {text!r} — отношение нельзя "
                          f"прочитать, а написано как будто связью обделено")
        self.assertNotIn("нет ребра", text,
                         f"нечитаемое отношение не имеет права называться "
                         f"отсутствием ребра: {text!r}")

    def test_brain_names_refusal_for_both_worlds(self):
        """
        ОДИН поглощатель на ДВА мира, и это ПРОВЕРКА, а не пожелание.

        brain/simulator.py и brain/social_sim.py сводят ход схемы в ответ
        РАЗНЫМИ кодами. Если второй скажет своими строками «walk.missing»,
        social-мир потеряет различие «прибор сломан» / «связи нет», и ни
        одна проверка про статику этого не увидит — она про другой модуль.

        ИЩЕМ ВЫЗОВ В АСТ, а не вхождение слова: имя живёт и в import, и
        поиск по тексту файла остаётся зелёным при удалённом вызове (это
        было измерено мутацией — урок 74).
        """
        import ast as _ast
        import brain.social_sim as ss
        tree = _ast.parse(open(ss.__file__, encoding="utf-8").read())
        calls = [n for n in _ast.walk(tree)
                 if isinstance(n, _ast.Call)
                 and getattr(n.func, "id", None) == "absorb_walk"]
        self.assertTrue(calls,
                        msg="social_sim не вызывает absorb_walk:social-мир "
                            "сводит отказ схемы своими строками")
        # и не сводит его по-старому руками
        hand = [n for n in _ast.walk(tree)
                if isinstance(n, _ast.Attribute) and n.attr == "missing"
                and isinstance(n.value, _ast.Attribute)
                and n.value.attr == "chain"]
        self.assertEqual([], hand,
                         msg="social_sim кладёт walk.missing руками — "
                             "unreadable туда не попадёт")

    def test_table_of_six_names_is_gone(self):
        """Схема обязана уметь прочитать ЛЮБОЕ имя реестра, а не шесть
        записанных в Python. Это и есть форма доступа: иначе следующее
        отношение (INSTRUMENT, RECIPIENT, ABOUT, VALUE) снова станет невидимым
        при полностью рабочих тестах на старых шести.

        ПОСТРОЕНО НАПРЯМУЮ, а не «пройди по реестру и посмотри, что не
        сломалось»: первая моя версия этого теста кидала в пустой мир и
        злилась на «нет ребра IS_A» — а это ЧЕСТНЫЙ ответ об отсутствии
        связи, не дефект. Мера обязана давать каждому имени РЕБРО, иначе
        она измеряет пустоту мира, а не силу читателя."""
        from core.storage import RelationType
        names = [e.name for e in RelationType]
        edges = [("невестка", rel, f"метка_{i}")
                 for i, rel in enumerate(names)]
        sk = StaticKnowledge(world_of(edges))
        blind = []
        for rel in names:
            sch = Scheme(name="п", question="?",
                         steps=[Step(kind="edge", role="кто",
                                     frm="$subject", relation=rel)])
            got = SchemeInterpreter(sk).run(sch, {"subject": "невестка"})
            if got.bindings.get("кто") != [f"метка_{names.index(rel)}"]:
                blind.append(f"{rel}: ok={got.ok} missing={' '.join(got.missing)}")
        self.assertEqual(blind, [],
                         "схема не прочитала (и не отвергла честно) эти "
                         "имена реестра:\n  " + "\n  ".join(blind))

    def test_working_relation_still_reads(self):
        """РЕГРЕССИЯ. CAN_DO читался и обязан читаться: правка формы
        доступа не имеет права сломать то, что работало (схемы «пожар» и
        «предательство» стоят именно на нём)."""
        res = self._run([("зинаида", "CAN_DO", "гостить")], "CAN_DO",
                        node="зинаида")
        self.assertEqual(res.bindings.get("кто"), ["гостить"],
                         f"CAN_DO сломался при правке: missing={res.missing}")

    def test_absent_edge_reports_absent_edge(self):
        """И ОБРАТНОЕ: если ребра ПРАВДА нет, отказ обязан называть
        ОТСУТСТВИЕ связи, а не невозможность прочесть. Иначе два разных
        диагноза поменяются местами и база будет чиниться не туда."""
        res = self._run([("невестка", "IS_ROLE", "жена")], "IS_ROLE",
                        node="зинаида")
        self.assertFalse(res.ok)
        text = " ".join(res.missing)
        self.assertIn("нет ребра", text,
                     f"при ОТСУТСТВУЮЩЕМ ребре отчёт должен говорить именно "
                     f"об отсутствии, а он говорит: {text!r}")
        self.assertNotIn("не читается", text,
                         f"при отсутствующем ребре нельзя ссылаться на "
                         f"нечитаемость отношения: {text!r}")


if __name__ == "__main__":
    unittest.main()
