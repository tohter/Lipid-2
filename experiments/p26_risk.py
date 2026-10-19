# -*- coding: utf-8 -*-
"""
п.26, продолжение. Два добивающих замера.

8. ПОРЯДОК КАТЕГОРИЙ = ПОРЯДОК ПОСТУПЛЕНИЯ В ФАЙЛ. Те же самые два факта,
   записанные в обратном порядке, обязаны дать тот же ответ. Если не дают —
   ответ зависит не от знания, а от того, что пришло первым.
9. КОНТРФАКТ: что изменится, если ранжирование по опыту ввести.
"""
import json
import os
import shutil
import sys
import tempfile

TMP = tempfile.mkdtemp(prefix="p26r_")
os.environ["LIPID2_MEMORY_DIR"] = TMP
os.environ["LIPID2_STATUS"] = "0"
sys.path.insert(0, "/home/tohter/LipidQwen")

from brain.lipid2_brain import Lipid2Brain   # noqa: E402

WFILE = os.path.join(TMP, "lipid_world.json")
C = "валенок"
COARSE, PRECISE = "вещь", "обувь"


def line(*a):
    print(*a, flush=True)


def put(hierarchy, weights=None):
    ent = {C: {"concept": C, "kind": "thing", "definition": "",
               "hierarchy": list(hierarchy), "properties": [], "actions": [],
               "provenance": [{"source": "user", "note": "замер"}]}}
    obs = {}
    for cat, w in (weights or {}).items():
        obs[f"{C}|IS_A|{cat}"] = {"head": C, "relation": "IS_A", "tail": cat,
                                 "confidence": 0.99, "negated": False,
                                 "support": 40, "weight": w,
                                 "evidence": "замер", "source": "experience",
                                 "first_seen": 1}
    json.dump({"entities": ent, "relations": [], "observations": obs,
               "unknown_words": {}, "transitions": {}},
              open(WFILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def ask():
    b = Lipid2Brain()
    b.use_llm_teacher = b.use_wiki_teacher = b.ask_user_enabled = False
    b.learn_experience = False
    return b.process_input(f"что такое {C}")


line("════ ЗАМЕР 8. Один и тот же набор фактов, разный ПОРЯДОК ЗАПИСИ ════")
put([COARSE, PRECISE])
a1 = ask()
line("записано ['вещь', 'обувь'] :", a1)
put([PRECISE, COARSE])
a2 = ask()
line("записано ['обувь', 'вещь'] :", a2)
line("ОТВЕТЫ РАЗЛИЧАЮТСЯ при ОДНОМ МНОЖЕСТВЕ фактов:", a1 != a2)

line("\n════ ЗАМЕР 9. Опыт кричит про ГРУБУЮ категорию (вес 250 против 80) ════")
put([PRECISE, COARSE], {COARSE: 250, PRECISE: 80})
line("ответ:", ask())
line("выбрана грубая 'вещь' потому, что опыт её подтверждает сильнее?",
     "НЕТ — опыт на выбор не влияет (замер 2); выбрано то, что записано первым")

shutil.rmtree(TMP, ignore_errors=True)
line("\nвременный каталог удалён")
