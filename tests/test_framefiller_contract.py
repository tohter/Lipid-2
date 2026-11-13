"""
КОНТРАКТ: КИРПИЧ 5 «ДУМАЛКА» — ЗАПОЛНЕНИЕ КАРКАСА ИЗ НАБОРА ПОНЯТИЙ.

ЗАМЕРЕННАЯ ДЫРА (11.03)
-----------------------
Понималка и говорилка (кирпичи 3/4) умеют разбирать готовое предложение в
ролевую структуру и собирать структуру обратно. Но если в НАБОРЕ понятий
нет главного предиката (пример хозяина 3: {кожа, мурашки, слушать, музыка} ->
«бежать») или нет деятеля (пример 4: {дом, труба, шатать} -> «ветер»), их не
откуда взять: структура обязана наполниться, а предлога-сказуемого в наборе
нет. Старая программа отвечала молчанием или догадкой.

ЧТО ЗАКРЕПЛЕНО (консультация босса и Жозефины, 11.03, слепой протокол)
----------------------------------------------------------------------
Думалка — СЕЛЕКТОР-ОБХОДЧИК ПО БАЗЕ, а не генератор. Заполнение пропуска —
это способность БАЗЫ, которую думалка только обнаруживает. У неё нет и не
должно быть источника, из которого она «придумывает» слово. Отсюда:

1. БАЗА — ЕДИНСТВЕННЫЙ ИСТОЧНИК ЗАПОЛНИТЕЛЕЙ. Каждый элемент выходного
   каркаса обязан нести основание: EXPLICIT (дано в наборе) или DERIVED
   (выведено по ребру базы). Элемент без основания не существует.
2. БАЗА — ТОЛЬКО ЧТЕНИЕ. Думалка ничего в неё не пишет.
3. ТРИ ИСХОДА, взаимоисключающих:
   * SUCCESS  — путь в базе найден, каркас наполнен, всё заземлено;
   * NO_PATH  — пути нет: думалка НЕ выдумывает пропуск и НАЗЫВАЕТ, какого
                ребра не хватает (питает измеритель покрытия);
   * ASK      — кандидаты есть, но они равноправны или слабо заземлены:
                думалка СПРАШИВАЕТ, а не выбирает произвольно.
4. ПРОБА НА ГАЛЛЮЦИНАЦИЮ: если пропуск не поддержан базой, думалка обязана
   его НЕ породить. Пример 3 («бежать») и пример 4 («ветер») без нужных
   рёбер в базе дают исход NO_PATH/ASK, и слова «бежать»/«ветер» НЕТ в
   выходном каркасе.
5. ГЛАГОЛ ПОСТАНОВКИ УЧАСТВУЕТ. Глагол, данный в наборе, не выбрасывается
   молча: он либо становится предикатом, либо остаётся в кандидатах/причине
   — но не вытесняется чужим глаголом «ради красивого предложения».

МУТАЦИИ, на которых контракт обязан покраснеть:
  * выдумать предикат, когда ребра в базе нет — падает проба на галлюцинацию;
  * записать вывод в базу (не read-only) — падает проверка чтения;
  * вернуть SUCCESS при незаземлённом (HYPOTHETICAL) элементе — падает
    правило «гипотеза не бывает уверенным исходом»;
  * при равных кандидатах выбрать одного — падает ASK;
  * не назвать недостающее ребро в NO_PATH — падает проверка нехватки.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from brain.framefiller import FrameFill, fill_frame          # noqa: E402
from brain.simulator import StaticKnowledge                  # noqa: E402


def obs(*triples) -> dict:
    out = {}
    for head, rel, tail in triples:
        out[f"{head}|{rel}|{tail}"] = {"head": head, "relation": rel,
                                       "tail": tail, "support": 3}
    return out


# МИР ПРИМЕРА 1: «фонарь освещал аптеку на ночной улице».
# Все роли (предикат «освещать») ДОСТУПНЫ в базе, но предиката в наборе нет.
WORLD1 = {
    "entities": {
        "фонарь": {"concept": "фонарь"},
        "аптека": {"concept": "аптека"},
        "улица": {"concept": "улица"},
        "ночь": {"concept": "ночь"},
    },
    "observations": obs(
        ("фонарь", "CAN_DO", "освещать"),
        ("освещать", "OBJECT_OF", "аптека"),
        ("аптека", "LOCATED_IN", "улица"),
        ("освещать", "CIRCUMSTANCE", "ночь"),
    ),
}


def world_without(*triples):
    """Мир, из которого убраны заданные рёбра (по набору (head, rel, tail))."""
    gone = {(h, r, t) for h, r, t in triples}
    w = {"entities": {k: dict(v) for k, v in WORLD1["entities"].items()},
         "observations": {}}
    for key, o in WORLD1["observations"].items():
        if (o["head"], o["relation"], o["tail"]) in gone:
            continue
        w["observations"][key] = dict(o)
    return w


def world_with(*triples):
    w = world_without()
    for head, rel, tail in triples:
        w["observations"][f"{head}|{rel}|{tail}"] = {
            "head": head, "relation": rel, "tail": tail, "support": 3}
    return w


class TestFillsExample1(unittest.TestCase):
    """Предикат «освещать» выводится из базы, а не из набора."""

    def test_success_fills_full_frame(self):
        f = fill_frame(["ночь", "улица", "фонарь", "аптека"],
                       StaticKnowledge(WORLD1))
        self.assertEqual(f.outcome, "SUCCESS", f.gaps)
        self.assertEqual(f.frame["predicate"], "освещать")
        self.assertEqual(f.frame["subject"], "фонарь")
        self.assertEqual(f.frame["object"], "аптека")
        self.assertEqual(f.frame["place"], "улица")
        self.assertEqual(f.frame["time"], "ночь")

    def test_predicate_basis_is_derived_not_explicit(self):
        f = fill_frame(["фонарь", "аптека"], StaticKnowledge(WORLD1))
        self.assertEqual(f.outcome, "SUCCESS", f.gaps)
        self.assertEqual(f.basis["predicate"], "DERIVED",
                         "предиката в наборе не было — он выведен по ребру")
        self.assertEqual(f.basis["subject"], "EXPLICIT")
        self.assertEqual(f.basis["object"], "EXPLICIT")

    def test_reads_base_only(self):
        """Думалка обязана читать базу, а не менять её."""
        k = StaticKnowledge(WORLD1)
        before = sum(len(v) for v in k.world.get("observations", {}))
        fill_frame(["фонарь", "аптека"], k)
        after = sum(len(v) for v in k.world.get("observations", {}))
        self.assertEqual(after, before,
                         "думалка не пишет в базу: вывод — чтение, обучение — "
                         "дело подсистемы обучения")


class TestNoPathIsHonest(unittest.TestCase):
    """Убрали ребро — думалка НЕ выдумывает предикат."""

    def test_missing_predicate_link_gives_no_path(self):
        w = world_without(("фонарь", "CAN_DO", "освещать"))
        f = fill_frame(["ночь", "улица", "фонарь", "аптека"],
                       StaticKnowledge(w))
        self.assertEqual(f.outcome, "NO_PATH")
        self.assertNotIn("освещать", f.frame.values(),
                         "предиката в базе нет — его не должно быть в каркасе")
        self.assertTrue(f.gaps,
                        "отказ обязан назвать, какого ребра не хватает")

    def test_missing_object_link_gives_no_path(self):
        w = world_without(("освещать", "OBJECT_OF", "аптека"))
        f = fill_frame(["фонарь", "аптека"], StaticKnowledge(w))
        self.assertEqual(f.outcome, "NO_PATH")

    def test_gap_names_the_link(self):
        w = world_without(("фонарь", "CAN_DO", "освещать"))
        f = fill_frame(["фонарь", "аптека"], StaticKnowledge(w))
        pairs = {tuple(sorted(g)) for g in f.gaps}
        self.assertIn(tuple(sorted(("фонарь", "освещать"))), pairs,
                      "недостача ровно эта: субъект не связан с действием")

    def test_no_hallucination_of_verb(self):
        """Провокация: набор пустой по предикату, база без ребра — тишина."""
        w = world_without(("фонарь", "CAN_DO", "освещать"))
        f = fill_frame(["фонарь", "аптека"], StaticKnowledge(w))
        self.assertEqual(f.outcome, "NO_PATH")
        self.assertNotIn("освещать", f.frame.values())

    def test_unknown_concept_is_a_gap(self):
        f = fill_frame(["фонарь", "витрина"], StaticKnowledge(WORLD1))
        self.assertEqual(f.outcome, "NO_PATH")
        self.assertTrue(f.unknown,
                        "незнакомое понятие — пробел базы, а не ответ")


class TestExample3And4NoHallucination(unittest.TestCase):
    """Примеры 3/4 без нужных рёбер: «бежать»/«ветер» не выдумываются."""

    def test_example3_no_bezhat(self):
        """{кожа, мурашки, музыка} без ребра «мурашки -> бежать»."""
        w = obs(
            ("музыка", "CAUSES", "мурашки"),
            ("мурашки", "LOCATED_ON", "кожа"),
        )
        world = {"entities": {"кожа": {"concept": "кожа"},
                              "мурашки": {"concept": "мурашки"},
                              "музыка": {"concept": "музыка"}},
                 "observations": w}
        f = fill_frame(["кожа", "мурашки", "музыка"], StaticKnowledge(world))
        self.assertEqual(f.outcome, "NO_PATH",
                         "связи дают причинно-пространственную структуру, но "
                         "НЕ доказывают «бежать»")
        self.assertNotIn("бежать", f.frame.values(),
                         "предикат «бежать» — деривация, без понятия движения "
                         "в базе её выдумывать нельзя")

    def test_example4_no_veter(self):
        """{дом, труба, шатать} без деятеля: «ветер» не выдумывается."""
        w = obs(
            ("шатать", "OBJECT_OF", "труба"),
            ("труба", "PART_OF", "дом"),
        )
        world = {"entities": {"дом": {"concept": "дом"},
                              "труба": {"concept": "труба"},
                              "шатать": {"concept": "шатать"}},
                 "observations": w}
        f = fill_frame(["дом", "труба", "шатать"], StaticKnowledge(world))
        self.assertEqual(f.outcome, "NO_PATH",
                         "объект и принадлежность известны, деятель — нет")
        self.assertNotIn("ветер", f.frame.values(),
                         "деятель без ребра в базе не угадывается")

    def test_example4_possessor_is_derived_but_no_subject(self):
        """Принадлежность (труба PART_OF дом) выводима, деятель — нет."""
        w = obs(
            ("шатать", "OBJECT_OF", "труба"),
            ("труба", "PART_OF", "дом"),
        )
        world = {"entities": {"дом": {"concept": "дом"},
                              "труба": {"concept": "труба"},
                              "шатать": {"concept": "шатать"}},
                 "observations": w}
        f = fill_frame(["дом", "труба", "шатать"], StaticKnowledge(world))
        obj = f.frame.get("object")
        self.assertEqual(obj.get("lemma"), "труба")
        self.assertEqual(obj.get("possessor", {}).get("lemma"), "дом",
                         "принадлежность выведена по PART_OF")
        self.assertEqual(f.basis.get("possessor"), "DERIVED",
                         "принадлежность — вывод по ребру, а не данность набора")
        self.assertEqual(f.basis.get("object"), "EXPLICIT",
                         "объект дан в наборе, выводом является только принадлежность")


class TestExplicitSubjectAndAmbiguity(unittest.TestCase):
    """Данный в наборе деятель — EXPLICIT; равные кандидаты — ASK."""

    def test_explicit_subject_is_accepted(self):
        """«ветер» ДАН в наборе — это не выдумка, а вход."""
        w = obs(
            ("ветер", "CAN_DO", "шатать"),
            ("шатать", "OBJECT_OF", "труба"),
        )
        world = {"entities": {"ветер": {"concept": "ветер"},
                              "дом": {"concept": "дом"},
                              "труба": {"concept": "труба"},
                              "шатать": {"concept": "шатать"}},
                 "observations": w}
        f = fill_frame(["ветер", "дом", "труба", "шатать"],
                       StaticKnowledge(world))
        self.assertEqual(f.outcome, "SUCCESS", f.gaps)
        self.assertEqual(f.frame["subject"], "ветер")
        self.assertEqual(f.basis["subject"], "EXPLICIT")

    def test_tied_candidates_give_ask(self):
        """Два равноправных предиката — думалка спрашивает, а не гадает."""
        w = obs(
            ("фонарь", "CAN_DO", "освещать"),
            ("фонарь", "CAN_DO", "показывать"),
            ("освещать", "OBJECT_OF", "аптека"),
            ("показывать", "OBJECT_OF", "аптека"),
        )
        world = {"entities": {"фонарь": {"concept": "фонарь"},
                              "аптека": {"concept": "аптека"}},
                 "observations": w}
        f = fill_frame(["фонарь", "аптека"], StaticKnowledge(world))
        self.assertEqual(f.outcome, "ASK",
                         "равноправные кандидаты не дают права выбрать одного")
        self.assertTrue(f.candidates and len(f.candidates) >= 2)
        verbs = {c.get("predicate") for c in f.candidates}
        self.assertEqual(verbs, {"освещать", "показывать"})

    def test_hypothetical_never_success(self):
        """Любой HYPOTHETICAL элемент блокирует SUCCESS."""
        w = obs(
            ("фонарь", "CAN_DO", "освещать"),
            ("освещать", "OBJECT_OF", "аптека"),
        )
        world = {"entities": {"фонарь": {"concept": "фонарь"},
                              "аптека": {"concept": "аптека"}},
                 "observations": w}
        f = fill_frame(["фонарь", "аптека"], StaticKnowledge(world))
        self.assertEqual(f.outcome, "SUCCESS")
        self.assertNotIn("HYPOTHETICAL", set(f.basis.values()),
                         "на успехе не бывает гипотез")


if __name__ == "__main__":
    unittest.main()