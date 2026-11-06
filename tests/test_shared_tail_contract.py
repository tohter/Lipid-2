"""
ОБЩИЙ ХВОСТ — НЕ ТОЖДЕСТВО.

ЗАЧЕМ ЭТОТ ФАЙЛ. identity_path (brain/simulator.py) отвечает на вопрос
«можно ли от одного лица осмысленно заключить про другое». Единственный
запрет, который в нём был (замер 04.11), — проход ЧЕРЕЗ СОБЫТИЕ. Замок от
ложных событий стоит, он заперт в test_step_rule_contract, и этот файл его
трогать не смеет.

Но сам обход ходит в ОБЕ стороны ЛЮБЫМ отношением, и из этого следует
вывод, который никто не проверял: две СУЩНОСТИ, у которых есть общий хвост,
оказываются связанными «тождеством». Замер 04.11 (миры из одних наблюдений,
kind ни у одного узла не задан):

    хвост/лапа  -> мокрый      ПУТЬ:  хвост-HAS_PROPERTY->мокрый<-лапа
    трава/попугай -> зелёный   ПУСТЬ: через общий цвет
    кошка/собака -> животное   ПУСТЬ: через общий класс (IS_A)
    маша/петр   -> врач        ПУТЬ:  через общую роль
    колесо/мотор -> машина     ПУСТЬ: у одной машины

Ни одно ребро не выдумано. Вывод ЛОЖЕН во всех пяти: «хвост и лапа — одно»,
«кошка — это собака», «колесо — это мотор». Это тот же класс дефекта, что
ложный путь через событие (урок: обход, которому можно ходить ЛЮБЫМ
отношением, объявляет тождество там, где рассказа про то нет), и та же
невидимость: золотой корпус сверяет ТРОЙКИ, а лжёт ЦЕПОЧКА.

ПРАВИЛО, А НЕ СПИСОК. Запрет формулируется структурно: шаг в узел и шаг ИЗ
узла по ОДНОМУ И ТОМУ ЖЕ отношению означает «оба висят на одном хвосте»
(звезда), а не «одно продолжается в другое». Поэтому тесты обязаны покрыть
и класс (IS_A), и роль (IS_ROLE), и меронию (PART_OF), и свойство
(HAS_PROPERTY), и направление — иначе замок окажется зашитым под замеренный
случай (тот же грех, против которого написан test_rule_covers_a_class...).

ЖИВОСТЬ — ГЛАВНАЯ ЧАСТЬ ФАЙЛА. Простейший способ сделать файл зелёным —
запретить обход совсем. Поэтому каждый отказ сопровождён честной цепочкой,
которая обязана пройти: невестка->жена<-егор (разные отношения), машина ->
мотор -> коленвал (одно отношение, НО цепочкой, а не звездой), транзитивный
IS_A (заперт в test_schemes_contract — трогать нельзя).
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from brain.simulator import StaticKnowledge  # noqa: E402


def world(edges, kinds=None):
    obs = {f"{h}|{r}|{t}": {"head": h, "relation": r, "tail": t}
           for h, r, t in edges}
    ents = {k: {"concept": k, "kind": v} for k, v in (kinds or {}).items()}
    return {"observations": obs, "entities": ents}


# Измеренные звёзды: две сущности висят на одном хвосте.
STARS = [
    ("общее свойство", [("хвост", "HAS_PROPERTY", "мокрый"),
                        ("лапа", "HAS_PROPERTY", "мокрый")], "хвост", "лапа"),
    ("общий цвет", [("трава", "HAS_PROPERTY", "зелёный"),
                    ("попугай", "HAS_PROPERTY", "зелёный")], "трава", "попугай"),
    ("общий класс", [("кошка", "IS_A", "животное"),
                     ("собака", "IS_A", "животное")], "кошка", "собака"),
    ("общая роль", [("маша", "IS_ROLE", "врач"),
                    ("петр", "IS_ROLE", "врач")], "маша", "петр"),
    ("общая часть целого", [("колесо", "PART_OF", "машина"),
                            ("мотор", "PART_OF", "машина")], "колесо", "мотор"),
]

# Честные цепочки: обязанность пройти (живость замка).
HONEST = [
    ("роль -> часть <- роль", [("невестка", "IS_ROLE", "жена"),
                               ("жена", "PART_OF", "сын"),
                               ("егор", "IS_ROLE", "сын")], "невестка", "егор"),
    ("цепочка одним отношением", [("машина", "PART_OF", "мотор"),
                                  ("мотор", "PART_OF", "коленвал")],
     "машина", "коленвал"),
    ("транзитивный IS_A", [("дождь", "IS_A", "вода"),
                           ("вода", "IS_A", "вещество")], "дождь", "вещество"),
    ("цепочка ролей", [("иван", "IS_ROLE", "сын"),
                       ("сын", "IS_ROLE", "наследник")], "иван", "наследник"),
]


class TestSharedTailIsNotIdentity(unittest.TestCase):
    """Звезда не есть путь тождества — ни одним из четырёх измеренных
    отношений."""

    def test_star_is_not_an_answer(self):
        for tag, edges, a, b in STARS:
            with self.subTest(case=tag):
                res = StaticKnowledge(world(edges)).identity_path(a, b)
                self.assertFalse(
                    res.path,
                    f"{tag}: обход объявил «{a}» и «{b}» связанными "
                    f"тождеством: {res.path}. Ребра верны, лжена ЦЕПОЧКА: "
                    f"шаг В хвост и шаг ИЗ хвоста по одному отношению — это "
                    f"«оба про хвост», а не «одно продолжается в другое»")

    def test_refusal_is_visible_as_refusal(self):
        """Отказ обязан читаться ИЗ СТРУКТУРЫ (код), а не быть пустым
        списком-невидимкой: «нет пути» и «путь был, но он лжёт» — разные
        починки (база против вывода), тот же закон, что в П4."""
        for tag, edges, a, b in STARS:
            with self.subTest(case=tag):
                res = StaticKnowledge(world(edges)).identity_path(a, b)
                self.assertNotEqual(
                    res.code, "OK",
                    f"{tag}: путь отвергнут молча — код остался OK")
                self.assertTrue(res.code and res.code.isascii(),
                                f"код отказа нечитабелен машиной: {res.code!r}")

    def test_direction_does_not_matter(self):
        """Ложь не зависит от направления вопроса: обход двусторонний."""
        for tag, edges, a, b in STARS:
            with self.subTest(case=tag):
                sk = StaticKnowledge(world(edges))
                self.assertFalse(sk.identity_path(b, a).path,
                                 f"{tag}: в обратную сторону звезда прошла")

    def test_rule_is_not_about_has_property_only(self):
        """Замок обязан ловить КЛАСС. Если бы запрет зашили под
        HAS_PROPERTY (первый замеренный случай), IS_A/PART_OF/IS_ROLE
        прошли бы. Проверено выше на четырёх отношениях; здесь — прямое
        утверждение, чтобы сужение правила не выглядело «пока не нужно»."""
        caught = 0
        for tag, edges, a, b in STARS:
            if not StaticKnowledge(world(edges)).identity_path(a, b).path:
                caught += 1
        self.assertEqual(caught, len(STARS),
                         f"правило поймало {caught} из {len(STARS)} — оно "
                         f"зашито под частный случай")


class TestHonestChainsSurvive(unittest.TestCase):
    """ЖИВОСТЬ: запретив звёзды, обход обязан оставить настоящие цепочки."""

    def test_honest_chain_builds(self):
        for tag, edges, a, b in HONEST:
            with self.subTest(case=tag):
                res = StaticKnowledge(world(edges)).identity_path(a, b)
                self.assertTrue(
                    res.path,
                    f"{tag}: правило шага убило честный путь {edges}. "
                    f"Отказ от лжи не имеет права стоить рассуждения")
                self.assertEqual(res.blocked, [],
                                 f"{tag}: честному пути нечего запрещать, "
                                 f"отказ протёк туда, где его быть не может")

    def test_transitive_is_a_survives(self):
        """Отдельно и явно: транзитивность IS_A заперта test_schemes_contract
        («дождь IS_A вода» обязана сходиться). Любая починка звезды обязана
        оставить её живой."""
        res = StaticKnowledge(world([("дождь", "IS_A", "вода"),
                                     ("вода", "IS_A", "вещество")
                                     ])).identity_path("дождь", "вещество")
        self.assertTrue(res.path, "транзитивный IS_A умер вместе со звездой")

    def test_shared_tail_still_visible_as_relation(self):
        """ОТКАЗ НЕ ЕСТЬ АМНЕЗИЯ: запретив тождество, система обязана
        по-прежнему ВИДЕТЬ, что хвост и лапа мокрый. Иначе починка =
        удалению знания (законы проекта: «не хватило» и «нельзя спросить»
        — разные нули)."""
        sk = StaticKnowledge(world([("хвост", "HAS_PROPERTY", "мокрый"),
                                    ("лапа", "HAS_PROPERTY", "мокрый")]))
        self.assertEqual(sorted(sk.related("хвост", "HAS_PROPERTY")),
                         ["мокрый"])
        self.assertEqual(sorted(sk.related("лапа", "HAS_PROPERTY")),
                         ["мокрый"])
        self.assertIn(("лапа", "HAS_PROPERTY"), sk.referenced_by("мокрый"),
                      "общий хвост перестал быть видимым: тождество "
                      "отказали, а знание выкинули")

    def test_event_refusal_is_untouched(self):
        """СМЕЖНЫЙ ЗАМОК НЕ ПОДВИНУТ (test_step_rule_contract): ложь через
        событие обязана остаться ОТЛИЧИМОЙ по коду от лжи через общий
        хвост — иначе две разные починки склеятся в один ноль."""
        false_story = world(
            [("недолюбливать", "OBJECT_OF", "невестка"),
             ("зинаида", "CAN_DO", "недолюбливать"),
             ("зинаида", "CAN_DO", "гостить"),
             ("гостить", "OBJECT_OF", "егор")],
            kinds={"недолюбливать": "action", "гостить": "action"})
        ev = StaticKnowledge(false_story).identity_path("невестка", "егор")
        star = StaticKnowledge(world(
            [("хвост", "HAS_PROPERTY", "мокрый"),
             ("лапа", "HAS_PROPERTY", "мокрый")])
        ).identity_path("хвост", "лапа")
        self.assertEqual(ev.code, "BLOCKED_BY_EVENT",
                         "запрет события задет этой починкой")
        self.assertFalse(star.path)
        self.assertNotEqual(star.code, ev.code,
                            "звезда и событие склеились в один код: вывод "
                            "не различит «прошли сквозь событие» и «оба "
                            "висят на одном хвосте»")


if __name__ == "__main__":
    unittest.main()
