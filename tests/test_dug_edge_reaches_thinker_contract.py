# -*- coding: utf-8 -*-
"""
КОНТРАКТ: «ДОБЫТОЕ УЧИТЕЛЕМ РЕБРО ОБЯЗАНО ДОЙТИ ДО ДУМАЛКИ».

ЗАМЕРЕННАЯ ДЫРА (09.11)
-----------------------
Живой прогон questions/вопроса «Может ли дождь потушить лесной пожар?»
через _handle_ability с фейковым учителем (который УМЕЕТ добыть недостающие
факты) показал:

    учитель выучил «потушить»  -> 1 кандидат в карантине (источник staging)
    _reasoning_view (мир для 2-й думы) -> ни одной синонимии (SYNONYM: [])

Связь «дождь может потушить пожар» МОДЕЛЬ УМЕЕТ собрать (проверено
напрямую think_link: при полном пути, включая «потушить SYNONYM_OF гасить»,
даёт LINK_FOUND). Но в живом потоке добытое учителем РЕБРО теряется: 
синонимия записывается учителем как observation, а collect_from_staging
забирает только сущности (entities), observations выбрасывает. _reasoning_view
строит мир только из сущностей кандидатов, observations не читает вовсе.
Итог: модель отвечает «не нашёл, чем дождь может потушить пожар», хотя
данных достаточно.

ЧТО ЗАКРЕПЛЕНО
--------------
1. РЕБРО ИМЕЕТ ПРАВО ДОЙТИ. Учитель, узнавший слово с синонимией, обязан
   получить шанс, что думалка увидит это ребро во втором проходе. Иначе
   «модель сама догадалась» недостижимо: канал приносит данные, а мозг их
   молча роняет.
2. ЖИВАЯ БАЗА ПО-ПРЕЖНЕМУ НЕ ТРОГАЕТСЯ ДО ОТВЕТА. Правка переноса не
   ослабляет карантин: observation из staging становится КАНДИДАТОМ, а не
   фактом. promote пишет в базу только с найденным ответом и через шлюз.
3. ГИПОТЕЗА ОСТАЁТСЯ ГИПОТЕЗОЙ. Ребро кандидата участвует в РАССУЖДЕНИИ
   (_reasoning_view), но не лежит в lipid_world.json, пока дофамин не
   разрешил. Тест проверяет это двумя прогонами файла мира.

МУТАЦИЯ, на которой контракт обязан покраснеть:
  * collect_from_staging / _reasoning_view, продолжающие выбрасывать
    observations, — вопрос «дождь потушить пожар» остаётся «не нашёл».

Это НЕ подсказка на конкретный вопрос: ни в коде, ни в тесте нет ребра
«дождь -> пожар». Учитель честно узнаёт слово «потушить» и его синонимию
(лексический факт о русском языке), а связь собирает сама думалка.
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# Мир, который мозг УЖЕ знает до вопроса. Ни одного «ответа на вопрос»,
# ни одного имени вопроса в правилах: только факты о дожде, воде, огне,
# пожаре. Связь «дождь потушить пожар» НЕ зашита — она должна собраться.
BASE_WORLD = {
    "entities": {
        "дождь": {"concept": "дождь", "hierarchy": ["вода"],
                  "properties": [], "actions": [], "definition": ""},
        "вода": {"concept": "вода", "hierarchy": ["вещество"],
                 "actions": ["гасить"], "properties": [],
                 "definition": "прозрачная жидкость"},
        "огонь": {"concept": "огонь", "hierarchy": [],
                  "properties": [], "actions": [], "definition": ""},
        "пожар": {"concept": "пожар", "hierarchy": [],
                  "properties": [], "actions": [], "parts": ["огонь"],
                  "definition": ""},
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
    "unknown_words": {},
}

# Что «знает» учитель (локальная модель/Википедия) про слово. Это НЕ факты
# о вопросе: учитель отвечает на запрос про СЛОВО. Потушить он описывает
# как глагол-синоним «гасить» (лексический факт о русском языке).
# Синонимия кладётся как РЕБРО (observation), а не как поле entity-записи, —
# ровно так, как это делает реальный учитель в staging-файл.
FAKE_LEARN = {
    "потушить": {"concept": "потушить", "hierarchy": [], "properties": [],
                 "actions": [], "definition": "погасить"},
}
FAKE_OBS = [("потушить", "SYNONYM_OF", "гасить")]


class FakeTeacher:
    """Эмуляция канала-учителя. Пишет добытое в карантинную копию.

    Знает FAKE_LEARN (сущность) и FAKE_OBS (ребро-синонимию). Всё пишется
    в staging-файл, на который мозг перенаправил учителя (world_filepath).
    """

    def __init__(self):
        self.world_filepath = None
        self.last_failure = None
        self.calls = []

    def run_agent(self, base_concept, derived_concept, filename):
        self.calls.append(base_concept)
        self.last_failure = None
        rec = FAKE_LEARN.get(base_concept)
        if rec is None:
            return None
        path = self.world_filepath
        if not path:
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                w = json.load(fh)
        except Exception:
            w = {"entities": {}, "observations": {}, "unknown_words": {}}
        w.setdefault("entities", {})
        w.setdefault("observations", {})
        w["entities"][base_concept] = dict(rec)
        for h, r, x in FAKE_OBS:
            if h == base_concept:
                w["observations"][f"{h}|{r}|{x}"] = {
                    "head": h, "relation": r, "tail": x, "modality": 1.0}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(w, fh, ensure_ascii=False)
        return dict(rec)


class FakeWiki(FakeTeacher):
    def fetch_concept(self, concept):
        return None

    def write_to_world(self, world):
        return False


def _make_brain(dirpath):
    os.environ["LIPID2_STATUS"] = "0"
    from brain.lipid2_brain import Lipid2Brain
    with open(os.path.join(dirpath, "lipid_world.json"), "w",
              encoding="utf-8") as fh:
        json.dump(BASE_WORLD, fh, ensure_ascii=False)
    b = Lipid2Brain(memory_dir=dirpath)
    b.teacher_llm = FakeTeacher()
    b.teacher_wiki = FakeWiki()
    b.use_llm_teacher = True
    b.use_wiki_teacher = True
    b.ask_user_enabled = False
    return b


class TestDugUpEdgeReachesThinker(unittest.TestCase):
    """Ребро, добытое учителем, обязано дойти до думалки во 2-й проход."""

    def test_model_finds_rain_can_extinguish_fire(self):
        """Вопрос хозяина. Учитель узнаёт «потушить» + синонимию — и модель
        сама собирает связь «дождь может потушить пожар»."""
        with tempfile.TemporaryDirectory() as d:
            b = _make_brain(d)
            answer = b.process_input("Может ли дождь потушить лесной пожар?")
            self.assertTrue(b.teacher_llm.calls,
                            "учитель обязан был спросить про неизвестное слово")
            self.assertNotIn("не нашёл", answer,
                             f"модель обязана сама найти связь, а ответ: "
                             f"{answer!r}")
            self.assertIn("потушить", answer.lower(), answer)
            self.assertIn("дождь", answer.lower(), answer)

    def test_dug_edge_lives_in_reasoning_view(self):
        """Добытая синонимия обязана попасть в _reasoning_view (мир для
        2-й думы), иначе думалка её физически не увидит."""
        with tempfile.TemporaryDirectory() as d:
            b = _make_brain(d)
            # имитируем перенос из staging в кандидата ровно так, как его
            # делает _handle_ability после учителей
            staging = b.curiosity.begin_search()
            t = FakeTeacher()
            t.world_filepath = staging
            t.run_agent("потушить", "потушить", "lipid_world.json")
            b.curiosity.collect_from_staging()
            rv = b._reasoning_view()
            obs = rv.get("observations") or {}
            syn = [k for k in obs if "SYNONYM" in k]
            self.assertTrue(syn,
                            "добытая синонимия потеряна между карантином и "
                            "рассуждением: _reasoning_view не видит observations")

    def test_live_world_still_untouched_before_answer(self):
        """Правка переноса не должна ослабить карантин: добытое ребро —
        КАНДИДАТ, а не факт. До promote оно не имеет права лечь в базу."""
        with tempfile.TemporaryDirectory() as d:
            b = _make_brain(d)
            with open(os.path.join(d, "lipid_world.json"),
                      encoding="utf-8") as fh:
                before = json.dumps(json.load(fh), sort_keys=True,
                                    ensure_ascii=False)
            staging = b.curiosity.begin_search()
            t = FakeTeacher()
            t.world_filepath = staging
            t.run_agent("потушить", "потушить", "lipid_world.json")
            b.curiosity.collect_from_staging()
            with open(os.path.join(d, "lipid_world.json"),
                      encoding="utf-8") as fh:
                after = json.dumps(json.load(fh), sort_keys=True,
                                   ensure_ascii=False)
            self.assertEqual(before, after,
                             "добытое ребро вошло в живую базу ДО ответа/промоута")

    def test_dug_edge_persists_in_world(self):
        """После найденного ответа добытая синонимия ЛЕЖИТ в базе.

        Иначе на следующий раз модель снова добывала бы её у учителя, а
        выученное пропадало бы молча. Замок promote пишет сущность, а ребро
        обязано лечь рядом с ней."""
        with tempfile.TemporaryDirectory() as d:
            b = _make_brain(d)
            b.process_input("Может ли дождь потушить лесной пожар?")
            with open(os.path.join(d, "lipid_world.json"),
                      encoding="utf-8") as fh:
                w = json.load(fh)
            obs = w.get("observations") or {}
            syn = [k for k in obs if "SYNONYM" in k]
            self.assertTrue(syn,
                            "добытая синонимия не легла в базу после ответа: "
                            f"{list(obs.keys())}")
            self.assertIn("потушить|SYNONYM_OF|гасить", obs,
                          "синонимия именно та, что выучил учитель")

    def test_hypothesis_is_hypothesis(self):
        """Ребро кандидата участвует в рассуждении, но не лежит в базе."""
        with tempfile.TemporaryDirectory() as d:
            b = _make_brain(d)
            staging = b.curiosity.begin_search()
            t = FakeTeacher()
            t.world_filepath = staging
            t.run_agent("потушить", "потушить", "lipid_world.json")
            b.curiosity.collect_from_staging()
            self.assertEqual(len(b.curiosity.staged()), 1,
                             "кандидат обязан жить в карантине до промоута")
            # в живом файле сущности «потушить» ещё нет
            with open(os.path.join(d, "lipid_world.json"),
                      encoding="utf-8") as fh:
                w = json.load(fh)
            self.assertNotIn("потушить", w["entities"],
                             "сущность-гипотеза не имеет права лежать в базе")


if __name__ == "__main__":
    unittest.main(verbosity=2)