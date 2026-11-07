"""
КОНТРАКТ: «МОЖЕТ ЛИ X СДЕЛАТЬ Y» — ПУТЬ, А НЕ УГАДКА.

ЗАМЕРЕННАЯ ДЫРА (07.11)
-----------------------
Первая версия думалки (brain/thinker.py) перебирала ВСЕ глаголы, которые
происходят над целью, и НЕ СРАВНИВАЛА их с глаголом вопроса. Следствие
замерено руками:

    «может ли дождь потушить огонь»  -> да (гасить OBJECT_OF огонь)
    «может ли дождь съесть огонь»    -> да (то же самое ребро)

Вопрос про ЕДУ превращался в ответ про ВОДУ, потому что глагол вопроса
выбрасывался сразу после разбора. Это ровно тот класс ошибки, который проект
запрещает законом «ложное ребро хуже отсутствующего»: программа уверенно
отвечает на вопрос, которого не слышала.

ЧТО ЗАКРЕПЛЕНО
--------------
1. ГЛАГОЛ ВОПРОСА ОБЯЗАН УЧАСТВОВАТЬ В ПУТИ. Ребро «вода гасит огонь»
   доказывает «дождь может потушить» только если «потушить» и «гасить» —
   одно действие (синонимия) или найденное действие есть вид запрошенного
   (IS_A вниз: умеет частное => умеет общее). Никакой другой перенос не
   разрешён: «может мыть» не даёт «может чистить».
2. ОТКАЗ ОБЯЗАН НАЗЫВАТЬ НЕХВАТКУ. Если глагол вопроса не связан ни с одним
   действием над целью, код MISSING_ACTION, а в gaps — пара «что связать».
   Это вход для любопытства: не «не знаю», а «не хватает этой конкретной
   связи».
3. ДАННЫЕ ИМЕЮТ ПРАВО ЗАКРЫТЬ ПРОБЕЛ, КОД — НЕТ. В файле мира/в fixture нет
   ни строчки про пожар: есть «вода CAN_DO гасить», «гасить OBJECT_OF огонь»,
   «дождь IS_A вода». Успех появляется ТОЛЬКО после того, как в мир входит
   синоксия «потушить = гасить». Тест доказывает это двумя прогонами одного
   и того же кода: до входа связи — честный отказ, после — да.
4. ЯЗЫК ЖИВЁТ В lang/ru. Мозг получает роли (subject/action/object), но ни
   один русский маркер вопроса не объявлен в brain/ (иначе brain/ тянет язык
   мимо lang/ — категория S измерителя границы, потолок 6 швов).

МУТАЦИИ, на которых контракт обязан покраснеть:
  * вернуть прежнее сравнение «любой глагол над целью» — падает «съесть
    огонь» и «мыть»;
  * разрешить перенос способности по IS_A между СУЩЕСТВИТЕЛЬНЫМИ целей —
    падает «ведро тушит» (ведро СОДЕРЖИТ воду, но не тушит);
  * сделать MISSING_ACTION без gaps — падает проверка любопытства.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from brain.thinker import Thinker                       # noqa: E402
from brain.simulator import StaticKnowledge             # noqa: E402
from lang.ru.ability import parse_ability, render_verdict   # noqa: E402


def obs(*triples) -> dict:
    out = {}
    for head, rel, tail in triples:
        out[f"{head}|{rel}|{tail}"] = {"head": head, "relation": rel,
                                       "tail": tail, "modality": 1.0}
    return out


# МИР БЕЗ ОТВЕТА. Ни одной пары «вопрос -> ответ», ни одного имени вопроса в
# правилах: только факты о воде, огне и дожде.
BASE_WORLD = {
    "entities": {
        "дождь": {"concept": "дождь", "hierarchy": ["вода"]},
        "вода": {"concept": "вода", "hierarchy": ["вещество"],
                 "actions": ["гасить"]},
        "огонь": {"concept": "огонь"},
        "пожар": {"concept": "пожар", "parts": ["огонь"]},
        "ведро": {"concept": "ведро"},
    },
    "observations": obs(
        ("дождь", "IS_A", "вода"),
        ("вода", "CAN_DO", "гасить"),
        ("гасить", "OBJECT_OF", "огонь"),
        ("пожар", "HAS_PART", "огонь"),
        ("ведро", "HAS_PART", "вода"),
    ),
}

# То же знание, но ПРОБЕЛ закрыт данными: лексический факт о русском языке
# (видовая пара), а не правило программы.
BRIDGE_WORLD = {
    "entities": BASE_WORLD["entities"],
    "observations": dict(BASE_WORLD["observations"], **obs(
        ("потушить", "SYNONYM_OF", "гасить"),
    )),
}


def world_with(extra_triples=()):
    w = {"entities": {k: dict(v) for k, v in BASE_WORLD["entities"].items()},
         "observations": dict(BASE_WORLD["observations"])}
    for head, rel, tail in extra_triples:
        w["observations"][f"{head}|{rel}|{tail}"] = {
            "head": head, "relation": rel, "tail": tail, "modality": 1.0}
    return w


class TestQuestionIsParsedAsRoles(unittest.TestCase):
    """Вопрос превращается в роли, а роли не знают русских фраз."""

    def test_roles_from_ability_question(self):
        r = parse_ability("Может ли дождь потушить сильный пожар?")
        self.assertEqual(r["subject"], "дождь")
        self.assertEqual(r["action"], "потушить")
        self.assertEqual(r["object"], "пожар")
        self.assertFalse(r["negative"])

    def test_negation_is_recorded_not_swallowed(self):
        r = parse_ability("Не может ли дождь потушить пожар?")
        self.assertTrue(r["negative"],
                        "отрицание — часть вопроса, а не мусор")

    def test_no_russian_word_is_known_to_brain(self):
        """Ни одного русского слова в КОДЕ мозга (докстринги — не код).

        Это проверка границы, а не стиля. Правило взято у tools/langborder.py:
        строки документации из меры исключены, считается только исполняемый
        код. Импорт морфологии из brain/ считается швом (категория S, потолок
        6): мозг обязан получать роли из lang/ru, а не разбирать русский сам.
        """
        import ast

        path = os.path.join(ROOT, "brain", "thinker.py")
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src)
        docs = {id(n.value) for n in ast.walk(tree)
                if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, str)}
        cyr = [n.s for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.s, str)
               and id(n) not in docs and any("а" <= c <= "я" for c in n.s)]
        self.assertEqual(cyr, [],
                         f"русские строки в коде мозга: {cyr[:3]} — языку место "
                         f"в lang/ru")
        self.assertNotIn("core.morph", src,
                         "мозг обязан получать роли из lang/ru, а не разбирать"
                         " русский сам")


class TestVerbMustLieOnThePath(unittest.TestCase):
    """Глагол вопроса обязан участвовать в пути — иначе ответа нет."""

    def setUp(self):
        self.t = Thinker(StaticKnowledge(BASE_WORLD))

    def test_unrelated_verb_is_not_an_answer(self):
        """«Может ли дождь съесть огонь» — НЕ «да».

        Прежняя версия отвечала да, потому что смотрела только на цель.
        """
        v = self.t.think_link("дождь", "съесть", "огонь")
        self.assertFalse(v.ok,
                         "найденное ребро про ДРУГОЙ глагол — не ответ")
        self.assertEqual(v.code, "MISSING_ACTION")

    def test_gap_names_what_is_missing(self):
        v = self.t.think_link("дождь", "потушить", "огонь")
        self.assertFalse(v.ok)
        self.assertEqual(v.code, "MISSING_ACTION")
        self.assertTrue(v.gaps,
                        "отказ обязан назвать связку, которой не хватает")
        pair = {tuple(sorted(g)) for g in v.gaps}
        self.assertIn(("гасить", "потушить"), pair,
                      "недостача ровно эта: «потушить» не связано с «гасить»")

    def test_synonymy_from_data_turns_refusal_into_answer(self):
        """Один и тот же код: без синоксии — отказ, с ней — «да».

        Знание вошло МИРОМ, а не правкой условия в думалке.
        """
        self.assertFalse(self.t.think_link("дождь", "потушить", "огонь").ok)

        t2 = Thinker(StaticKnowledge(BRIDGE_WORLD))
        v = t2.think_link("дождь", "потушить", "огонь")
        self.assertTrue(v.ok, render_verdict(v))
        self.assertEqual(v.code, "LINK_FOUND")
        self.assertEqual(v.links[0].source, "вода",
                         "ответ обязан назвать, ЧЕМ дождь может: он — вода")
        self.assertEqual(v.links[0].action, "гасить")

    def test_part_of_does_not_transfer_ability(self):
        """Ведро СОДЕРЖИТ воду, но «ведро тушит огонь» — не ответ.

        Иначе перенос способности по любой связи плодит уверенную чушь:
        «orestoran тушит», потому что в ресторане есть вода.
        """
        t2 = Thinker(StaticKnowledge(BRIDGE_WORLD))
        v = t2.think_link("ведро", "потушить", "огонь")
        self.assertFalse(v.ok, render_verdict(v))
        self.assertEqual(v.code, "MISSING_CARRIER",
                         "действие подошло, но носить его некому: это ДРУГАЯ "
                         "дыра, чем «нет действия», и ищется в другом месте")
        self.assertIn(("ведро", "гасить"), [tuple(g) for g in v.gaps])

    def test_subtype_verb_is_enough(self):
        """Если найденное действие — ВИД запрошенного, этого достаточно.

        Умеет частное => умеет общее. Обратное (умеет общее => умеет частное)
        запрещено: вода гасит вообще не доказывает, что она потушит лесной
        пожар, — для этого нужна сила, а её в мире нет.
        """
        w = world_with((("гасить", "IS_A", "потушить"),))
        v = Thinker(StaticKnowledge(w)).think_link("дождь", "потушить", "огонь")
        self.assertTrue(v.ok, render_verdict(v))

        # Обратная стрелка НЕ даёт права: «потушить IS_A гасить» — это про
        # частность потушить, а не про способность воды.
        w2 = world_with((("потушить", "IS_A", "гасить"),))
        v2 = Thinker(StaticKnowledge(w2)).think_link("дождь", "потушить",
                                                     "огонь")
        self.assertFalse(v2.ok, "перенос вверх по IS_A запрещён")

    def test_unknown_node_is_a_gap_not_a_ban(self):
        v = self.t.think_link("морось", "потушить", "огонь")
        self.assertFalse(v.ok)
        self.assertEqual(v.code, "UNKNOWN_WORDS")
        self.assertIn("морось", v.unknown_words,
                      "незнакомое слово — пробел в базе, его надо учить")

    def test_goal_parts_are_searched(self):
        """Запрошен пожар, а действует вода над ОГНЕМ — часть пожара."""
        t2 = Thinker(StaticKnowledge(BRIDGE_WORLD))
        v = t2.think_link("дождь", "потушить", "пожар")
        self.assertTrue(v.ok, render_verdict(v))
        self.assertEqual(v.links[0].target, "огонь",
                         "путь обязан идти через часть цели, а не выдумывать "
                         "ребро «гасить пожар»")


class TestAnswerIsRenderedOutsideBrain(unittest.TestCase):
    """Русскую фразу даёт lang/ru, думалка даёт только структуру."""

    def test_render_names_the_path(self):
        v = Thinker(StaticKnowledge(BRIDGE_WORLD)).think_link(
            "дождь", "потушить", "огонь")
        text = render_verdict(v)
        self.assertIn("дождь", text)
        self.assertIn("вода", text)
        self.assertIn("гасить", text)
        self.assertNotIn("None", text)

    def test_refusal_says_what_is_missing(self):
        v = Thinker(StaticKnowledge(BASE_WORLD)).think_link(
            "дождь", "потушить", "огонь")
        text = render_verdict(v)
        self.assertIn("потушить", text)
        self.assertIn("гасить", text)
        self.assertIn("не", text, "отказ обязан быть отказом")
        self.assertNotIn("потушит ", text,
                         "отказ не имеет права утверждать способность")


if __name__ == "__main__":
    unittest.main()
