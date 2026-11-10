# -*- coding: utf-8 -*-
"""Живой замер: как мозг понимает типы вопросов (сравнение, почему, общее)."""
import os
import sys
import tempfile
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["LIPID2_STATUS"] = "0"

from lang.ru.questions import IRParser
from brain.lipid2_brain import Lipid2Brain

WORLD = {
    "entities": {
        "птица": {"concept": "птица", "hierarchy": ["живое существо"],
                  "properties": ["умеет летать", "есть крылья", "есть перья"],
                  "actions": ["летать"], "definition": "животное с перьями"},
        "самолёт": {"concept": "самолёт", "hierarchy": ["летательный аппарат"],
                    "properties": ["сделан из металла", "есть крылья", "есть двигатель"],
                    "actions": ["летать"], "definition": "летательная машина"},
    },
    "observations": {},
    "unknown_words": {},
}


def make_brain(d):
    with open(os.path.join(d, "lipid_world.json"), "w", encoding="utf-8") as f:
        json.dump(WORLD, f, ensure_ascii=False)
    b = Lipid2Brain(memory_dir=d)
    class _Noop:
        world_filepath = None
        last_failure = None
    b.teacher_llm = _Noop()
    b.teacher_wiki = _Noop()
    b.use_llm_teacher = False
    b.use_wiki_teacher = False
    b.ask_user_enabled = False
    return b


if __name__ == "__main__":
    p = IRParser()
    questions = [
        "Чем отличается птица от самолёта?",
        "Что общего у птицы и самолёта?",
        "Почему птица летает?",
        "Зачем птице крылья?",
        "Откуда берутся птицы?",
        "Что может быть, если птица врежется в самолёт?",
        "Что такое птица?",
        "Какие свойства у птицы?",
        "Что делает птица?",
    ]
    with tempfile.TemporaryDirectory() as d:
        b = make_brain(d)
        print("=" * 74)
        print("РАЗБОР ВОПРОСА (IRParser) + ОТВЕТ МОЗГА (process_input)")
        print("=" * 74)
        for q in questions:
            ir = p.parse(q)
            ans = b.process_input(q)
            print(f"\nQ: «{q}»")
            print(f"   intent={ir.intent_type} subjects={ir.subjects}")
            print(f"   ANSWER: {ans}")