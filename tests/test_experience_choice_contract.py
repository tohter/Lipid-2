"""
КОНТРАКТ (п.26): ВЕС ОПЫТА НЕ ВЫБИРАЕТ ОПРЕДЕЛЕНИЕ, а выбор сейчас — это
порядок записи в файле.

Откуда файл. Очередь спрашивала: влияет ли опыт (weight, сила перехода, сон) на
то, КАКОЕ определение выдаст мозг? Замер (experiments/p26_measure.py,
experiments/p26_risk.py, док experiments/EXPERIENCE_CHOICE_TASK.md) ответил:
НЕ ВЛИЯЕТ НИ ОДНИМ ЧИСЛОМ. Единственное чтение веса ребра во всём проекте —
core/graph.py::print_node_associations, то есть ПЕЧАТЬ. Ответ собирается в
_definition_answer по жёсткому приоритету definition > hierarchy[0] > properties.

Замеренные числа, которые здесь замурованы:

  родитель 'обувь' вес 80 (пол), 'кустарный предмет' вес 250 (потолок сна)
      ответ при нулевом опыте:  Валенок — это обувь (категории: обувь -> ...)
      ответ при весе 250:       Валенок — это обувь (категории: обувь -> ...)
      ОТВЕТ НЕ ИЗМЕНИЛСЯ
  вес 255 на категории против definition — проигрывает определение всегда.

НАЙДЕННЫЙ ДЕФЕКТ (он же — тест TestOrderOfRecordingChoosesTheAnswer; он ЗЕЛЁНЫЙ
СЕЙЧАС, потому что фиксирует текущее ДЕФЕКТНОЕ поведение, и ОБЯЗАН
покраснеть в момент починки, чтобы та не прошла молча):

  один и тот же набор фактов, записанный в разном порядке, даёт РАЗНЫЙ ответ:
      ['вещь', 'обувь']  ->  «Валенок — это вещь»
      ['обувь', 'вещь']  ->  «Валенок — это обувь»
  Порядок записи — побочный эффект того, кто ответил раньше (Вики или
  пользователь), и к миру отношения не имеет.

Ранжирование по весу опыта ЗДЕСЬ НЕ РАЗРЕШАЮТ, а запрещают числами: грубая
категория подтверждается опытом чаще именно потому, что она грубая (замер 9:
у 'вещь' подтверждений всегда больше, чем у 'обувь'). Правка обязана идти от
нового ЗАМЕРА по содержанию, а не от частоты.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CONCEPT = "валенок"
NEAR, FAR = "обувь", "кустарный предмет"
COARSE, PRECISE = "вещь", "обувь"


def write_world(directory, hierarchy, weights=None):
    """Мир с одной сущностью и (опционально) наблюдениями заданного веса."""
    obs = {}
    for cat, w in (weights or {}).items():
        obs[f"{CONCEPT}|IS_A|{cat}"] = {
            "head": CONCEPT, "relation": "IS_A", "tail": cat,
            "confidence": 0.99, "negated": False, "support": 40,
            "weight": w, "evidence": "замер п.26", "source": "experience",
            "first_seen": 1,
        }
    world = {
        "entities": {CONCEPT: {
            "concept": CONCEPT, "kind": "thing", "definition": "",
            "hierarchy": list(hierarchy), "properties": [], "actions": [],
            "provenance": [{"source": "user", "note": "замер п.26"}]}},
        "relations": [], "observations": obs, "unknown_words": {},
        "transitions": {},
    }
    with open(os.path.join(directory, "lipid_world.json"), "w",
              encoding="utf-8") as fh:
        json.dump(world, fh, ensure_ascii=False, indent=1)


class P26Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="l2p26_")
        os.environ["LIPID2_MEMORY_DIR"] = self.dir
        os.environ["LIPID2_STATUS"] = "0"

    def tearDown(self):
        os.environ.pop("LIPID2_MEMORY_DIR", None)
        shutil.rmtree(self.dir, ignore_errors=True)

    def answer(self, text=None):
        from brain.lipid2_brain import Lipid2Brain
        b = Lipid2Brain(memory_dir=self.dir)
        # Учитель выключен ВЕСЬ: ответ обязан зависеть от памяти, а не от сети.
        b.use_llm_teacher = b.use_wiki_teacher = b.ask_user_enabled = False
        b.learn_experience = False
        return b.process_input(text or f"что такое {CONCEPT}")


class TestExperienceWeightDoesNotChoose(P26Base):
    """Ни пол, ни потолок сна не меняют ответ на «что такое X»."""

    def test_maximum_experience_weight_changes_nothing(self):
        write_world(self.dir, [NEAR, FAR])
        without = self.answer()
        write_world(self.dir, [NEAR, FAR], {NEAR: 80, FAR: 250})
        with_heavy = self.answer()
        self.assertEqual(
            without, with_heavy,
            "вес опыта 250 против 80 изменил ответ — значит выбор по весу "
            "ПОЯВИЛСЯ. Он не разрешён (см. док, замер 9: частота подтверждений "
            "коррелирует с грубостью). Правка обязана идти от замера по "
            "СОДЕРЖАНИЮ и обязана обновить этот тест осознанно.")

    def test_definition_beats_max_weight_category(self):
        """definition веса не имеет вовсе и выигрывает у веса 255 всегда."""
        world = {
            "entities": {CONCEPT: {
                "concept": CONCEPT, "kind": "thing",
                "definition": "старая русская обувь из войлока",
                "hierarchy": [NEAR], "properties": [], "actions": [],
                "provenance": [{"source": "user", "note": "замер п.26"}]}},
            "relations": [],
            "observations": {f"{CONCEPT}|IS_A|{NEAR}": {
                "head": CONCEPT, "relation": "IS_A", "tail": NEAR,
                "confidence": 0.99, "negated": False, "support": 99,
                "weight": 255, "evidence": "замер", "source": "experience",
                "first_seen": 1}},
            "unknown_words": {}, "transitions": {},
        }
        with open(os.path.join(self.dir, "lipid_world.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(world, fh, ensure_ascii=False, indent=1)
        text = self.answer()
        self.assertIn("старая русская обувь", text)
        self.assertNotIn(f"это {NEAR} ", text,
                         f"категория с весом 255 вытеснила определение: {text}")

    def test_every_category_is_still_visible(self):
        """
        Ни один вариант НЕ теряется — это хорошая часть замера, она обязана
        пережить любую правку порядка: менять порядок можно, терять мнение —
        нельзя.
        """
        write_world(self.dir, [NEAR, FAR], {NEAR: 80, FAR: 250})
        text = self.answer()
        self.assertIn(NEAR, text)
        self.assertIn(FAR, text,
                      f"родитель с весом 250 пропал из ответа: {text}")


class TestOrderOfRecordingChoosesTheAnswer(P26Base):
    """
    Замок на НАЙДЕННЫЙ дефект, а не на желаемое поведение.

    Тест зелёный СЕЙЧАС — он описывает факт: ответ выбирается порядком записи.
    Он обязан покраснеть ровно в один момент — когда порядок перестанет быть
    порядком записи. Тогда этот файл обязан быть ПЕРЕПИСАН, а не удалён:
    новые ожидания обязаны идти с новым замером (док, п.10).
    """

    def test_answer_follows_record_order_not_content(self):
        write_world(self.dir, [COARSE, PRECISE])
        first = self.answer()
        write_world(self.dir, [PRECISE, COARSE])
        second = self.answer()
        self.assertNotEqual(
            first, second,
            "ответ перестал зависеть от ПОРЯДКА ЗАПИСИ. Отлично — дефект "
            "закрыт; перепиши этот тест на новое правило выбора и приложи "
            "замер, почему оно правильное (experiments/EXPERIENCE_CHOICE_TASK.md, "
            "п.10). Полученное: %r / %r" % (first, second))
        self.assertIn(f"это {COARSE}", first)
        self.assertIn(f"это {PRECISE}", second)


class TestNoSilentWeightReader(unittest.TestCase):
    """
    Вес ребра во всём проекте обязан читаться ТОЛЬКО для печати и для диагноза
    расхождений. Если появится четвёртый читатель, выбирающий ответ, — этот
    тест обязан покраснеть первым.
    """

    def test_weight_is_read_only_in_known_places(self):
        import re
        readers = []
        for folder in ("core", "brain"):
            for fname in sorted(os.listdir(os.path.join(ROOT, folder))):
                if not fname.endswith(".py"):
                    continue
                path = os.path.join(ROOT, folder, fname)
                with open(path, encoding="utf-8") as fh:
                    for no, line in enumerate(fh, 1):
                        if line.lstrip().startswith("#"):
                            continue
                        if re.search(r'syn\[.weight.\]|syn_info\[.weight.\]|'
                                      r'\[.weight.\]\s*[\)\]\n,<]', line):
                            readers.append(f"{folder}/{fname}:{no}")
        # core/graph.py:215 — печать; core/storage.py:108 — упаковка в бинарный
        # формат (запись, не выбор). Дальше вес читается только через
        # observation_weight (диагноз, replay, сон), а не из ребра графа.
        allowed_prefixes = ("core/graph.py", "core/storage.py")
        stray = [r for r in readers if not r.startswith(allowed_prefixes)]
        self.assertEqual([], stray,
                         "появилось новое чтение веса РЕБРА — именно так вес "
                         "незаметно превращается в выбор ответа (п.26): "
                         + ", ".join(stray))


if __name__ == "__main__":
    unittest.main(verbosity=2)
