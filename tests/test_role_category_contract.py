"""
РОЛЬ НЕ ЕСТ КАК КАТЕГОРИЯ. Замок на ложь, которую ничто другое не мерит.

ЗАЧЕМ ЭТОТ ФАЙЛ. Фраза «Вася — инженер» даёт отношение IS_ROLE (роль),
«Инженер — это профессия» — тоже IS_ROLE. Раньше brain/learner.py по пути в
мир переписывал ИМЯ на IS_A (REL_ALIAS = {"IS_ROLE": "IS_A"}), а IS_A
транзитивен. Замер (01.11), сквозной прогон на живом Learner:

    мир:  вася|IS_A|инженер , инженер|IS_A|профессия
    вывод (brain/simulator.py::_agents_of, оно же core/schemes.py::_matches
           через _neighbors(..., "IS_A", "transitive")):
          вася -> инженер -> профессия

то есть система выводила «Вася ЯВЛЯЕТСЯ профессией». Роль уехала в
категориальное наследование.

ПОЧЕМУ ЭТОГО НЕ ВИДНО НИ ОДНИМ СУЩЕСТВУЮЩИМ ПРИБОРОМ (замерено, а не
предположено): золотой корпус на всех четырёх конфигурациях (с REL_ALIAS и
без, с expect_world и без) даёт forbid_violations = 0. Ложь корпусу НЕВИДИМА
совсем: он сверяет ТРОЙКИ «голова-связь-хвост», а тут три тройки по отдельности
верны и лишь ИХ ЦЕПОЧКА лжёт. Именно поэтому дефект пережил 812 тестов.

ЗАМОК ДВУСТОРОННИЙ НАМЕРЕННО. Дешёвый способ сделать его зелёным — вырубить
транзитивность IS_A совсем. Тогда сломается законное: «дождь IS_A вода» и
схема «гасить огонь» должны сходиться (это заперто в test_schemes_contract и
разделе «почему» витрины). Поэтому здесь стоят ДВА теста: ложь обязана
исчезнуть, а настоящий IS_A обязан остаться транзитивным.

ОБРАТНОГО ПУТИ У СЛИЯНИЯ НЕТ — вот почему это чинят, а не запирают как
измеренный дефект (в отличие от «кот, или собака» в test_alias_contract):
из наблюдения «вася|IS_A|инженер» уже нельзя восстановить, была ли это роль
или порода. Потеря необратима в самих данных, а не в выводе.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from brain.learner import Learner                     # noqa: E402
from brain.simulator import Simulator, StaticKnowledge  # noqa: E402

PHRASES = ("Вася — инженер.", "Инженер — это профессия.")


def learn_world(texts=PHRASES):
    """Мир после урока. Учитель выключен, ядро не подключено, память в памяти
    процесса: тест не имеет права ни писать в живую Вселенную, ни звать LLM."""
    world = {"entities": {}, "observations": {}, "unknown_words": {}}
    lrn = Learner(load_world=lambda: dict(world),
                  save_world=lambda w: world.update(w),
                  teacher=lambda w: None, core=None, max_teacher_calls=0)
    for t in texts:
        lrn.learn_from_text(t, source="role_category_lock", allow_teacher=False)
    return world


def chain_of(world, start):
    """Цепочка «во что входит» ровно тем способом, каким её строит вывод:
    brain/simulator.py::_agents_of (снизу вверх по IS_A)."""
    return Simulator(world)._agents_of(start)


class TestRoleIsNotCategory(unittest.TestCase):
    """Сама ложь: «Вася является профессией»."""

    @classmethod
    def setUpClass(cls):
        cls.world = learn_world()

    def test_world_keeps_role_name_not_category(self):
        """ИМЯ в мире обязано остаться IS_ROLE. Это верхний замок: пока имя
        не переименовано, транзитивность к роли не применяется."""
        rels = sorted({v["relation"] for v in self.world["observations"].values()})
        self.assertIn("IS_ROLE", rels,
                      f"роль не дошла до мира под своим именем: {rels}")
        self.assertNotIn("IS_A", rels,
                         f"в мир снова лёг IS_A (REL_ALIAS вернулся?): {rels}")

    def test_role_is_not_a_category_parent(self):
        """is_a() обязана не знать про роль: «инженер» — не родитель «вася»."""
        self.assertEqual(StaticKnowledge(self.world).is_a("вася"), [],
                         "роль поселилась в is_a(): транзитивное наследование "
                         "начнёт считать инженера категорией Васи")

    def test_false_chain_absent(self):
        """ГЛАВНЫЙ ТЕСТ. Цепочка вася -> профессия обязана отсутствовать."""
        chain = chain_of(self.world, "вася")
        self.assertNotIn("профессия", chain,
                         f"вывод утверждает «Вася является профессией»: {chain}")

    def test_role_still_reachable_as_role(self):
        """Ложь уходит, ЗНАНИЕ остаётся. Роль обязана быть доступна — иначе
        «починка» выродилась бы в выбрасывание фразы, а это не улучшение."""
        roles = {(v["head"], v["relation"], v["tail"])
                 for v in self.world["observations"].values()}
        self.assertIn(("вася", "IS_ROLE", "инженер"), roles,
                      f"роль потеряна при уроке: {sorted(roles)}")
        self.assertIn(("инженер", "IS_ROLE", "профессия"), roles,
                      f"вторая роль потеряна при уроке: {sorted(roles)}")


class TestRealIsAStaysTransitive(unittest.TestCase):
    """
    Вторая сторона замка: нельзя сделать его зелёным, вырубив транзитивность.

    Синтетический мир, а не выученный: здесь проверяется свойство вывода,
    независимое от экстрактора. Если эту проверку убрать, самый дешёвый способ
    «починить» ложь — это сломать законное сведение «дождь -> вода», и витрина
    с разделом «почему» тихо умрёт.
    """

    def test_genuine_is_a_still_transitive(self):
        world = {"entities": {}, "observations": {
            "дождь|IS_A|вода": {"head": "дождь", "relation": "IS_A",
                                "tail": "вода", "negated": False},
            "вода|IS_A|вещество": {"head": "вода", "relation": "IS_A",
                                   "tail": "вещество", "negated": False},
        }, "unknown_words": {}}
        chain = chain_of(world, "дождь")
        self.assertIn("вода", chain, f"прямой IS_A пропал: {chain}")
        self.assertIn("вещество", chain,
                      f"транзитивность IS_A вырублена — цепочка {chain}. "
                      "Ложь «вася -> профессия» лечится именем отношения, "
                      "а не кастрацией наследования")


if __name__ == "__main__":
    unittest.main()
