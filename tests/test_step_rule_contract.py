"""
КОНТРАКТ П4: ОБХОД ОБЯЗАН УМЕТЬ ОТКАЗАТЬ (правило шага).

Задача, ради которой всё это: «кем приходятся невестка Зинаиды и жена Егора».
Для неё нужен обход графа. Замер (experiments/P4_STEP_RULE.txt,
tools/story_without_lesson.py) показал, что НЕОГРАНИЧЕННЫЙ обход отвечает на
нее ЛОЖЬЮ, ни одного выдуманного ребра не потребовав:

    невестка <-OBJECT_OF- недолюбливать -CAN_DO-> зинаида
             -CAN_DO-> гостить -OBJECT_OF-> егор

Каждое ребро честно взято из рассказа. Ложь в ШАГЕ: обход вошёл в СОБЫТИЕ
(«гостить») и вышел из него, перенеся тождество лица сквозь действие. Так
два несвязанных лица оказались «связанными». Проект считает ложную связь
хуже отсутствующей — значит обход обязан уметь ОТКАЗАТЬ.

ПОЧЕМУ ЗАМОК ЖИВЁТ РЯДОМ С related(), а не в звене 4. Отказ — свойство
обхода, а не свойства вопроса. Если запретить врать только в одном
обработчике, следующий вызовивший обход (социальный мир, «а что ты знаешь
о X», звено 5) получит ту же ложь.

ЗАПРЕТОМ ПРОХОДА, а не ТРЕБОВАНИЕМ ЛИЦА — по замеру типов: у лиц типа НЕТ
(«зинаида», «жена», «сын» не записи сущностей, а хвосты наблюдений), так что
правило «разрешены только лица» убило бы и честный путь.

ТИП ЧИТАЕТСЯ ИЗ МИРА, А НЕ ИЗ ЯДРА. Замер: узел ядра на «гостить» отвечает
NodeType.ENTITY (ACTION не у одного из двадцати узлов), а node_type в
проекте как решение не читает никто. Полю мира kind == "action" доверяют
четыре читателя мозга. Правило, прочитавшее тип у ядра, молча не запретило бы
РОВНО НИЧЕГО и осталось бы зелёным над ложью.

ЖИВОСТЬ. Замок «ложного пути нет» тривиально выполняется обходом, который
вообще ничего не находит. Поэтому рядом ЗАПЕРТО, что честный путь из тех же
методов СТАВИТСЯ (test_honest_chain_still_builds), и что отказ различим по
причине (test_refusal_names_the_reason). Без этих двух замок охранял бы
пустоту.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.simulator import StaticKnowledge          # noqa: E402


def world(edges, kinds=None):
    """Мир из одних наблюдений + записей сущностей с их kind.

    kind передаётся ОТДЕЛЬНЫМ аргументом намеренно: именно поле kind — то
    место, где мир пишет «это действие» (замер: читает его мозг в четырёх
    местах). Без записей с kind мир был бы плоским и проверка ничего не
    проверяла.
    """
    obs = {f"{h}|{r}|{t}": {"head": h, "relation": r, "tail": t}
           for h, r, t in edges}
    ents = {k: {"concept": k, "kind": v} for k, v in (kinds or {}).items()}
    return {"observations": obs, "entities": ents}


# ТОЧНАЯ копия ложного пути из рассказа (имена узлов и отношений НЕ выдуманы:
# сняты с дампа experiments/STORY_WITHOUT_LESSON.txt).
FALSE_STORY = world(
    [("недолюбливать", "OBJECT_OF", "невестка"),
     ("зинаида", "CAN_DO", "недолюбливать"),
     ("зинаида", "CAN_DO", "гостить"),
     ("гостить", "OBJECT_OF", "егор")],
    kinds={"недолюбливать": "action", "гостить": "action",
           "зинаида": "entity", "егор": "entity", "невестка": "entity"})

# ЧЕСТНАЯ цепочка: та, ради которой затеяно всё (уроки определений + П2).
HONEST = world(
    [("невестка", "IS_ROLE", "жена"),
     ("егор", "IS_ROLE", "сын"),
     ("жена", "PART_OF", "сын"),
     ("сын", "PART_OF", "зинаида")])

# ДРУГОЙ выдуманный рассказ, ДРУГИЕ слова, ДРУГИЕ отношения (ABOUT и
# RECIPIENT вместо CAN_DO и OBJECT_OF), то же устройство: лицо -> событие ->
# лицо. Замок обязан ловить КЛАСС, а не замеренный случай.
FALSE_OTHER = world(
    [("спор", "ABOUT", "маша"),
     ("спор", "RECIPIENT", "петр")],
    kinds={"спор": "action"})


class TestFalsePathRefused(unittest.TestCase):
    """Через событие тождество лица НЕ переносится."""

    def test_story_false_path_is_not_an_answer(self):
        sk = StaticKnowledge(FALSE_STORY)
        res = sk.identity_path("невестка", "егор")
        self.assertFalse(res.path,
                         f"обход выдал ложный путь как ответ: {res.path}. "
                         f"Он прошёл сквозь событие и объявил тождество там, "
                         f"где рассказа про то нет")
        self.assertTrue(res.blocked,
                        "отказ обязан НАЗЫВАТЬ узлы, через которые путь был "
                        "возможен, но не заслуживает доверия: иначе пустой "
                        "список неотличим от пустого графа")
        self.assertIn("гостить", res.blocked,
                      f"в отказе не помянуто событие-виновник: {res.blocked}")

    def test_rule_covers_a_class_not_the_measured_case(self):
        """Другой рассказ, другие отношения. Если правило зашито под CAN_DO/
        OBJECT_OF или под слово «гостить», этот тест его разоблачит.
        """
        sk = StaticKnowledge(FALSE_OTHER)
        res = sk.identity_path("маша", "петр")
        self.assertFalse(res.path,
                         f"правило сработало только на замеренном случае: "
                         f"{res.path}")
        self.assertIn("спор", res.blocked)

    def test_event_blocks_no_matter_which_direction(self):
        """Ложь не зависит от направления: обход идёт в обе стороны, и в обе
        стороны событие остаётся событием.
        """
        sk = StaticKnowledge(FALSE_STORY)
        self.assertFalse(sk.identity_path("егор", "невестка").path)


class TestHonestChainStillBuilds(unittest.TestCase):
    """ЖИВОСТЬ ЗАМКА: запретив ложь, обход обязан оставить правду.

    Без этого теста зелёным был бы и обход, возвращающий пустоту всегда.
    """

    def test_honest_chain_still_builds(self):
        sk = StaticKnowledge(HONEST)
        res = sk.identity_path("невестка", "егор")
        self.assertTrue(res.path,
                        "правило шага убило честный путь: "
                        f"невестка -IS_ROLE-> жена -PART_OF-> сын <-IS_ROLE- "
                        f"егор. Лица типа не имеют (замер), поэтому правило "
                        f"обязано запрещать ПРОХОД ЧЕРЕЗ СОБЫТИЕ, а не "
                        f"требовать известных лиц")
        self.assertEqual(res.blocked, [],
                         "честному пути нечего запрещать — отказ протёк туда, "
                         "где его быть не должно")

    def test_nodes_without_type_are_passable(self):
        """Лица в мире НЕ имеют записи kind (их нет в entities). Обход обязан
        пройти через них: иначе отказ стал бы отказом от рассуждения.
        """
        w = world([("невестка", "IS_ROLE", "жена"),
                   ("жена", "PART_OF", "сын"),
                   ("егор", "IS_ROLE", "сын")])
        self.assertTrue(StaticKnowledge(w).identity_path("невестка", "егор").path)


class TestRefusalNamesTheReason(unittest.TestCase):
    """Отказ обязан быть РАЗЛИЧИМ по причине (структурой, не текстом)."""

    def test_four_different_situations_four_codes(self):
        """Четыре разных положения дел обязаны читаться ИЗ ОТВЕТА, а не
        угадываться по длине списка."""
        sk_false = StaticKnowledge(FALSE_STORY)
        sk_disconnected = StaticKnowledge(world([
            ("а", "IS_ROLE", "б"), ("в", "IS_ROLE", "г")]))
        r_blocked = sk_false.identity_path("невестка", "егор")
        r_nopath = sk_disconnected.identity_path("а", "в")
        r_unknown = sk_false.identity_path("невестка", "такого_слова_нет")
        codes = {r_blocked.code, r_nopath.code, r_unknown.code}
        self.assertEqual(len(codes), 3,
                         f"разные положения дел слились в один ноль: "
                         f"{[(r.code, r.path, r.blocked) for r in (r_blocked, r_nopath, r_unknown)]}")
        self.assertEqual(r_blocked.code, "BLOCKED_BY_EVENT")
        self.assertEqual(r_unknown.code, "UNKNOWN_NODE")

    def test_codes_are_machine_readable(self):
        """Коды — IR, а не русская фраза: их читает вывод, а не человек."""
        r = StaticKnowledge(FALSE_STORY).identity_path("невестка", "егор")
        self.assertEqual(r.code, r.code.upper().replace(" ", "_"))
        self.assertTrue(r.code.isascii(),
                        f"код отказа нечитабелен машиной: {r.code!r}")


class TestTypeComesFromWorld(unittest.TestCase):
    """Правило читает тип ИЗ МИРА. Замер: у ядра на «гостить» — ENTITY."""

    def test_action_kind_is_what_makes_a_node_an_event(self):
        sk = StaticKnowledge(FALSE_STORY)
        self.assertTrue(sk.is_event("гостить"))
        self.assertFalse(sk.is_event("зинаида"),
                         "лицо записано событием — правило запретит всё")

    def test_record_without_kind_is_not_an_event(self):
        """Отсутствие типа НЕ событие. Иначе запрет стал бы всеобщим."""
        sk = StaticKnowledge(HONEST)
        self.assertFalse(sk.is_event("жена"))
        self.assertFalse(sk.is_event("совсем_неизвестное_слово"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
