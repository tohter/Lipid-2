"""
КОНТРАКТ: кортизол — РАБОЧИЙ ФЛАГ ПОИСКА, дофамин — РАЗРЕШЕНИЕ ПИСАТЬ.

ЗАМЕРЕННАЯ ДЫРА (07.11, до правки)
----------------------------------
Гормоны в программе пишутся, но НИГДЕ НЕ ЧИТАЮТСЯ. Проверено перебором всех
обращений к `hormones.` в core/, brain/, ui/: два места, и оба — свойства-
геттеры для красивой полоски в окне (`noradrenaline`, `dopamine`). Ни один
`if`, ни один шаг обучения, ни одна запись в базу от значения гормона НЕ
ЗАВИСЯТ. То есть «нейроморфность» была КОСМЕТИКОЙ: кортизол горел, а
поведение не менялось.

ВТОРАЯ ПОЛОВИНА ДЫРЫ ХУЖЕ ПЕРВОЙ. Порядок в `_trigger_curiosity_and_learn`:
учитель ВЫЗЫВАЕТСЯ И САМ ПИШЕТ в lipid_world.json (адрес Вселенной прошит у
него в конструкторе), и только ПОСЛЕ этого мозг ставит дофамин 255. Значит
дофамин — не разрешение, а ОТМЕТКА О СОСТОЯВШЕМСЯ: запись уже на диске, когда
«разрешение» загорелось, снять её нельзя и проверить нечем.

ЧТО ТРЕБУЕТ ХОЗЯИН (словами)
----------------------------
  * пока кортизол высокий — искать: связывать имеющееся, добывать недостающее
    (оба учителя разрешены), до тех пор, пока не появится ответ;
  * ответ найден — кортизол падает, и только ДОФАМИН даёт право записать;
  * «не знаю» права писать не даёт и кортизол не снимает.

ЧЕМ ОТЛИЧАЕТСЯ ОТ «просто поставить флажки»
-------------------------------------------
Знания в базе в момент проверки НЕТ. В файле мира нет узла «роса» (замер:
knows('роса')=False), правильного ответа нет ни в коде, ни в данных. Един-
ственный способ довести кортизол до нуля — ЧЕСТНО закрыть пробел. Тест
проверяет ПОРЯДОК, а не слово в ответе:

    1. вопрос с неизвестным словом -> высокий кортизол, открытая цель,
       запись в базе НЕ появилась (хэш файла мира тот же);
    2. цель закрыта (знание вошло)  -> кортизол низкий, дофамин высокий,
       цель снята, кандидат переехал из карантина;
    3. попытка получить право записи БЕЗ ответа -> отказ.

МУТАЦИИ, на которых контракт обязан покраснеть (проверяется руками):
  * dopamine_ok возвращает True всегда — падает отказ от записи;
  * провал поиска сбрасывает кортизол — падает «флаг поиска»;
  * цель снимается при первом отказе — падает «цель жива»;
  * promote не смотрит на разрешение — падает карантин.
"""
import hashlib
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from brain.curiosity import Curiosity, HormoneGate    # noqa: E402
from brain.simulator import StaticKnowledge           # noqa: E402

WORLD = {
    "entities": {
        "дождь": {"concept": "дождь", "hierarchy": ["вода"], "actions": ["лить"]},
        "вода": {"concept": "вода", "hierarchy": ["вещество"],
                 "actions": ["гасить"]},
        "огонь": {"concept": "огонь", "hierarchy": ["процесс"]},
        "пожар": {"concept": "пожар", "hierarchy": ["бедствие"]},
    },
    "observations": {
        "дождь|IS_A|вода": {"head": "дождь", "relation": "IS_A",
                            "tail": "вода", "modality": 1.0},
        "вода|CAN_DO|гасить": {"head": "вода", "relation": "CAN_DO",
                               "tail": "гасить", "modality": 1.0},
        "гасить|OBJECT_OF|огонь": {"head": "гасить", "relation": "OBJECT_OF",
                                   "tail": "огонь", "modality": 1.0},
        "пожар|HAS_PART|огонь": {"head": "пожар", "relation": "HAS_PART",
                                 "tail": "огонь", "modality": 1.0},
    },
}


def knowledge(world=None) -> StaticKnowledge:
    return StaticKnowledge(world if world is not None else WORLD)


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class TestCortisolIsSearchFlag(unittest.TestCase):
    """Кортизол — флаг, а не украшение: он обязан ТРЕБОВАТЬ поиска."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.world_path = os.path.join(self.dir, "lipid_world.json")
        with open(self.world_path, "w", encoding="utf-8") as fh:
            json.dump(WORLD, fh, ensure_ascii=False)
        self.cur = Curiosity(self.dir)

    def test_unknown_word_opens_goal_and_raises_cortisol(self):
        """Незнакомое слово в вопросе: цель открыта, кортизол ВЫСОКИЙ."""
        gate = HormoneGate()
        goal = self.cur.note_unknown("роса", why="нужно для вопроса")
        gate.stress(200)

        self.assertTrue(goal.open,
                        "цель обязана открыться: «роса» — пробел, а не «нет»")
        self.assertGreaterEqual(gate.cortisol, gate.search_threshold,
                                "высокий кортизол обязан означать «ИЩИ»")
        self.assertTrue(gate.must_search(),
                        "пока тревога высокая, поиск обязан требоваться")

    def test_failed_search_keeps_goal_and_cortisol(self):
        """Нашли «ничего» — цель НЕ закрывается, тревога НЕ снимается.

        Иначе «поискали и не нашли» успокаивает систему, и вопрос перестаёт
        существовать: программа, сказавшая «не знаю» с пустой головой,
        неотличима от той, что вообще не искала.
        """
        gate = HormoneGate()
        goal = self.cur.note_unknown("роса", why="нужно для вопроса")
        gate.stress(200)
        self.cur.failed(goal, reason="оба учителя молчат")

        self.assertTrue(goal.open,
                        "не найденное знание ОБЯЗАНО остаться целью")
        self.assertGreaterEqual(
            gate.cortisol, gate.search_threshold,
            "провал поиска не имеет права снимать тревогу: искать не кончили")

    def test_same_question_does_not_duplicate_goal(self):
        g1 = self.cur.note_unknown("роса", why="первый вопрос")
        g2 = self.cur.note_unknown("роса", why="второй вопрос")
        self.assertIs(g1, g2, "повтор вопроса удваивает ТРЕБОВАНИЕ, а не цель")
        self.assertEqual(len(self.cur.open_goals()), 1)

    def test_world_untouched_while_searching(self):
        """Пока цель не закрыта, файл мира не должен измениться ни на байт.

        Это проверка запрета «дофамин после записи»: если запись случилась до
        разрешения, смысл флага потерян, и поймать это можно только хэшем.
        """
        before = _sha(self.world_path)
        gate = HormoneGate()
        goal = self.cur.note_unknown("роса", why="вопрос")
        gate.stress(200)
        self.cur.failed(goal, reason="не нашли")
        self.assertEqual(before, _sha(self.world_path),
                         "база изменилась без дофамина — запись опередила "
                         "разрешение")


class TestDopamineIsWritePermission(unittest.TestCase):
    """Дофамин — РАЗРЕШЕНИЕ на запись, а не отчёт о состоявшемся."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.world_path = os.path.join(self.dir, "lipid_world.json")
        with open(self.world_path, "w", encoding="utf-8") as fh:
            json.dump(WORLD, fh, ensure_ascii=False)
        self.cur = Curiosity(self.dir, writer=self._writer)

    def _writer(self, entities: dict) -> int:
        """Единственный писатель во Вселенную: ровно то, что пропустил шлюз."""
        with open(self.world_path, encoding="utf-8") as fh:
            world = json.load(fh)
        ents = world.setdefault("entities", {})
        moved = 0
        for name, rec in entities.items():
            if name not in ents:
                moved += 1
            ents[name] = rec
        with open(self.world_path, "w", encoding="utf-8") as fh:
            json.dump(world, fh, ensure_ascii=False)
        return moved

    def test_no_write_permission_without_answer(self):
        gate = HormoneGate()
        gate.stress(255)
        self.assertFalse(gate.dopamine_ok(),
                         "на пике тревоги права писать быть не должно")
        self.assertFalse(gate.dopamine_ok(answer_found=False),
                         "без найденного ответа права писать нет никогда")
        # Второй замок проверяется ОТДЕЛЬНО от первого: спокойствие само по
        # себе не есть ответ. Иначе «ничего не случилось» и «мы разобрались»
        # дают одинаковое право писать.
        gate.relief()
        self.assertTrue(gate.dopamine_ok(answer_found=True),
                        "спокойно + ответ = писать можно")
        self.assertFalse(gate.dopamine_ok(answer_found=False),
                         "спокойно + БЕЗ ответа = писать нельзя: покой — это "
                         "не закрытый вопрос")

    def test_answer_lowers_cortisol_and_grants_write(self):
        gate = HormoneGate()
        goal = self.cur.note_unknown("роса", why="вопрос")
        gate.stress(255)
        self.assertTrue(gate.must_search())

        gate.relief()                       # связь нашлась, вопрос понятен
        self.cur.closed(goal)
        self.assertTrue(gate.dopamine_ok(answer_found=True),
                        "найденный ответ обязан дать право записи")
        self.assertLess(gate.cortisol, gate.search_threshold,
                        "разобралась — тревога обязана упасть ниже порога")
        self.assertFalse(gate.must_search())
        self.assertFalse(goal.open)

    def test_promote_moves_candidate_only_with_permission(self):
        """Кандидат из карантина переезжает во Вселенную ТОЛЬКО с дофамином."""
        gate = HormoneGate()
        candidate = {"роса": {"concept": "роса", "hierarchy": ["вода"]}}
        self.cur.stage(candidate, source="учитель-кандидат")

        moved = self.cur.promote(knowledge, gate, answer_found=False)
        self.assertEqual(moved, 0, "без ответа писать нельзя")
        self.assertEqual(len(self.cur.staged()), 1,
                         "отказ в праве не сжигает кандидата: он ещё нужен")
        with open(self.world_path, encoding="utf-8") as fh:
            self.assertNotIn("роса", json.load(fh)["entities"],
                             "росы в базе не стало без дофамина")

        gate.relief()
        moved = self.cur.promote(knowledge, gate, answer_found=True)
        self.assertGreaterEqual(moved, 1, "с дофамином кандидат обязан войти")
        self.assertEqual(len(self.cur.staged()), 0,
                         "переехавший кандидат обязан покинуть карантин")
        with open(self.world_path, encoding="utf-8") as fh:
            self.assertIn("роса", json.load(fh)["entities"])

    def test_candidate_without_hook_is_not_written(self):
        """Кандидат, не зацепившийся за известное, в базу НЕ входит.

        Иначе учитель может забросить в Вселенную сколь угодно странных узлов
        одним лишь удачно найденным ответом на другой вопрос.
        """
        gate = HormoneGate()
        self.cur.stage({"квазля": {"concept": "квазля",
                                   "hierarchy": ["несуществующее"]}},
                       source="выдумка")
        gate.relief()
        self.assertEqual(self.cur.promote(knowledge, gate, answer_found=True), 0,
                         "узел без единой связи с известным — не знание")
        self.assertEqual(len(self.cur.staged()), 1,
                         "непрошедший кандидат остаётся в карантине")

    def test_gate_persists_between_sessions(self):
        """Кортизол — состояние ПРОЦЕССА: перезапуск не обязан его обнулять.

        Иначе «недосказанность» забывается вместе с окном, а просили искать
        «до такой степени, чтобы уметь выдать ответ».
        """
        gate = HormoneGate()
        gate.stress(200)
        gate.save(self.dir)
        again = HormoneGate().load(self.dir)
        self.assertGreaterEqual(again.cortisol, gate.search_threshold,
                                "снятый при перезапуске кортизол — это "
                                "забытая недосказанность")


if __name__ == "__main__":
    unittest.main()
