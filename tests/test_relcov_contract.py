"""
Замок на ИЗМЕРИТЕЛЬ tools/relcov.py. До 01.11 у этой меры тестов не было
вовсе (grep -rln relcov tests/ — пусто), и она успела: (а) называться
«доезжают до мира», глядя только в sf.facts; (б) врать в арифметике вывода
(«ИТОГО 17, ПОРОЖДАЕТСЯ 15» при одном имени в списке).

Мера без замка — это число, которое может тихо сменить смысл, а витрина
(tools/presnum.py) продаст новый смысл под старым словом. Здесь заперто ровно
то, что отличает ДВА числа от одного: что это разные множества, а не один
список с другим фильтром.
"""
import hashlib
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import tools.relcov as relcov                       # noqa: E402

LIVE_WORLD = os.path.join(ROOT, "data", "memory", "lipid_world.json")


def _digest(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class TestTwoNumbersAreReallyDifferent(unittest.TestCase):
    """Два числа обязаны мерить РАЗНОЕ. Иначе второе — декорация."""

    @classmethod
    def setUpClass(cls):
        cls.total, cls.ext, cls.world = relcov.coverage()

    def test_arithmetic_of_lists_is_honest(self):
        """Число и список за ним сходятся. Это не паранойя: ровно на этой мере
        счётчик уже расходился со списком на единицу."""
        reg = {e.name for e in relcov.RelationType}
        self.assertEqual(self.total, len(reg))
        self.assertEqual(len(reg - self.ext), self.total - len(self.ext),
                         "расхождение: имена вне реестра")
        for got, name in ((self.ext, "экстрактор"), (self.world, "мир")):
            self.assertTrue(got <= reg,
                            f"{name}: имена вне реестра {sorted(got - reg)}")

    def test_neither_set_contains_the_other(self):
        """ГЛАВНОЕ. extracted ⊄ world и world ⊄ extracted.

        Если кто-то «починит» меру, сводя оба числа к одному источнику (или
        начнёт переименовывать отношения на пути в мир, как было с REL_ALIAS),
        одно из включений станет полным и тест покраснеет. Именно за это
        включение, а не за сравнение чисел: 16 >= 15 верно и при полностью
        вложенных множествах, а смысл — в составе разности.
        """
        self.assertTrue(self.ext - self.world,
                        "у экстрактора и в мире одно и то же: второе число "
                        "перестало мерять цену конвейера")
        self.assertTrue(self.world - self.ext,
                        "в мире нет ничего помимо экстрактора: потеряно "
                        "происхождение связей (learner порождает IS_A сам)")

    def test_specific_losses_and_gains_are_named(self):
        """Конкретные имена, а не только «множества различны»: иначе мера
        могла бы разойтись чем угодно и тест остался бы зелёным."""
        self.assertEqual(self.ext - self.world, {"INSTRUMENT", "RECIPIENT"},
                         "изменился список падежных ролей, не доходящих до "
                         "мира: это не регрессия, а новое знание — надо "
                         "перечитать _apply_fact и обновить витрину")
        self.assertEqual(self.world - self.ext, {"IS_A"},
                         "изменилось то, что learner порождает без экстрактора"
                         " (иерархия глаголов) или экстрактор научился сам "
                         "порождать IS_A — и то и другое меняет витрину")

    def test_role_reaches_world_under_its_own_name(self):
        """IS_ROLE в мире, IS_A у экстрактора нет. Разведение имён (снятие
        REL_ALIAS) — причина, по которой у этих чисел такой состав; без этого
        теста переименование можно вернуть, и мера этого не заметит."""
        self.assertIn("IS_ROLE", self.world,
                      "роль снова не доходит до мира под своим именем")
        self.assertNotIn("IS_A", self.ext,
                         "экстрактор начал порождать IS_A: число 16 и подпись "
                         "к нему разошлись")


class TestMeasureDoesNotTouchLiveMemory(unittest.TestCase):
    """Измеритель обязан мерить, а не жить в памяти проекта.

    Отдельно от test_memory_isolation_contract.py: там заперты самопроверки
    core/, а здесь — что сам вызов coverage() (он поднимает Learner и пишет
    мир во временный файл) не трогает живую Вселенную.
    """

    def test_coverage_leaves_live_world_alone(self):
        before = _digest(LIVE_WORLD)
        relcov.coverage()
        self.assertEqual(before, _digest(LIVE_WORLD),
                         "измеритель отношений изменил живую Вселенную")


if __name__ == "__main__":
    unittest.main()
