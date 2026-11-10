# -*- coding: utf-8 -*-
"""Живой замер: найдёт ли модель связь «дождь может погасить пожар».

Не меняет проект. Подменяет учителей фейковыми, которые УМЕЮТ добыть
недостающие факты, и смотрит, догадается ли цепочка
think_link -> любопытство -> учителя -> think_link по кандидатам.
"""
import os
import sys
import tempfile
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["LIPID2_STATUS"] = "0"


# --- базовая база: то, что мозг УЖЕ знает до вопроса ---
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
                  "properties": [], "actions": [],
                  "parts": ["огонь"], "definition": ""},
    },
    "observations": {
        "гасить|OBJECT_OF|огонь": {"head": "гасить",
                                    "relation": "OBJECT_OF",
                                    "tail": "огонь", "modality": 1.0},
        "пожар|HAS_PART|огонь": {"head": "пожар",
                                  "relation": "HAS_PART",
                                  "tail": "огонь", "modality": 1.0},
    },
    "unknown_words": {},
}


# --- фейковый учитель: "умеет" узнать слово, дописав недостающие факты ---
# Это НЕ подсказка на конкретный вопрос. Это эмуляция: локальная модель
# / Википедия отвечают на запрос про слово фактами о нём.
# Важно: учитель даёт ЛЕКСИЧЕСКИЙ факт «потушить = гасить» как синонимию
# (ребро), а не как иерархию (потушить вид гасить). Только так think_link
# увидит связь: глагол вопроса обязан участвовать в пути.
FAKE_WORLD = {
    "дождь": {"concept": "дождь", "hierarchy": ["вода"],
              "properties": ["падает с неба", "капли воды"],
              "actions": [], "definition": "капли воды с неба"},
    "потушить": {"concept": "потушить", "hierarchy": [],
                 "properties": [], "actions": [], "definition": "погасить"},
    "пожар": {"concept": "пожар", "hierarchy": [],
              "properties": [], "actions": [],
              "parts": ["огонь"], "definition": "большой огонь"},
}

# Ребра-синонимии, которые учитель/модель ДОБАВЛЯЕТ как observations
# (не как встроенные подсказки в вопрос).
FAKE_OBS = [
    ("потушить", "SYNONYM_OF", "гасить"),
    ("дождь", "CAN_DO", "гасить"),
]


class FakeTeacher:
    """Эмуляция канала: знает FAKE_WORLD, пишет в карантинную копию."""

    def __init__(self):
        self.world_filepath = None
        self.last_failure = None
        self.calls = []

    def run_agent(self, base_concept, derived_concept, filename):
        self.calls.append(base_concept)
        self.last_failure = None
        rec = FAKE_WORLD.get(base_concept)
        if rec is None:
            return None
        path = self.world_filepath
        if not path:
            return None
        try:
            with open(path, encoding="utf-8") as f:
                w = json.load(f)
        except Exception:
            w = {"entities": {}, "observations": {}, "unknown_words": {}}
        w.setdefault("entities", {})
        w.setdefault("observations", {})
        w["entities"][base_concept] = dict(rec)
        # добавляем синонимию/способность как observation
        for h, r, x in FAKE_OBS:
            if h == base_concept or x == base_concept:
                w["observations"][f"{h}|{r}|{x}"] = {
                    "head": h, "relation": r, "tail": x, "modality": 1.0}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(w, f, ensure_ascii=False)
        return dict(rec)


class FakeWiki(FakeTeacher):
    def fetch_concept(self, concept):
        return None

    def write_to_world(self, world):
        return False


def main():
    from brain.lipid2_brain import Lipid2Brain

    with tempfile.TemporaryDirectory() as d:
        # кладём базовую базу
        with open(os.path.join(d, "lipid_world.json"), "w",
                  encoding="utf-8") as f:
            json.dump(BASE_WORLD, f, ensure_ascii=False)

        b = Lipid2Brain(memory_dir=d)
        llm = FakeTeacher()
        wiki = FakeWiki()
        b.teacher_llm = llm
        b.teacher_wiki = wiki
        b.use_llm_teacher = True
        b.use_wiki_teacher = True
        b.ask_user_enabled = False

        q = "Может ли дождь потушить лесной пожар?"
        print("=== ВОПРОС:", q)
        print("=== что знает мозг ДО:", sorted(BASE_WORLD["entities"].keys()))
        answer = b.process_input(q)
        print("=== ОТВЕТ:", answer)
        print("=== какие слова спрашивал у учителей:", llm.calls)
        # что осталось в карантине (staged)
        staged = b.curiosity.staged()
        print("=== кандидатов в карантине ПОСЛЕ:", len(staged))
        for item in staged:
            print("    кандидат:", item.get("name"), "source=", item.get("source"))
        # что видит reasoning_view (живое + кандидаты)
        rv = b._reasoning_view()
        obs = rv.get("observations") or {}
        syns = [k for k in obs if "SYNONYM" in k]
        print("=== observations в reasoning_view (синонимии):", syns)


if __name__ == "__main__":
    main()