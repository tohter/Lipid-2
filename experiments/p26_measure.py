# -*- coding: utf-8 -*-
"""
п.26 — ВЛИЯЕТ ЛИ ОПЫТ (weight / strength / сон) НА ВЫБОР ОПРЕДЕЛЕНИЯ.

Изолированно: LIPID2_MEMORY_DIR = временный каталог, живая память не трогается
(её sha256 сверяется ВНЕ скрипта, до и после). Учитель выключен весь — LLM,
Wiki, вопрос пользователю: иначе ответ зависел бы от сети, а не от весов.
"""
import json
import os
import shutil
import sys
import tempfile

TMP = tempfile.mkdtemp(prefix="p26_")
os.environ["LIPID2_MEMORY_DIR"] = TMP
os.environ["LIPID2_STATUS"] = "0"
sys.path.insert(0, "/home/tohter/LipidQwen")

from brain.lipid2_brain import Lipid2Brain        # noqa: E402
from core.records import fact_weight, observation_weight  # noqa: E402

WFILE = os.path.join(TMP, "lipid_world.json")
C = "валенок"
P1, P2 = "обувь", "кустарный предмет"          # два конкурирующих родителя
R1, R2 = "сшит из войлока", "держит тепло"     # два конкурирующих свойства


def line(*a):
    print(*a, flush=True)


def put_world(entities, observations=None):
    world = {"entities": entities, "relations": [],
             "observations": observations or {}, "unknown_words": {},
             "transitions": {}}
    with open(WFILE, "w", encoding="utf-8") as f:
        json.dump(world, f, ensure_ascii=False, indent=1)


def entity(hierarchy, properties, definition=""):
    return {C: {"concept": C, "kind": "thing", "definition": definition,
                "hierarchy": list(hierarchy), "properties": list(properties),
                "actions": [],
                "provenance": [{"source": "user", "note": "замер п.26"}]}}


def brain_of():
    b = Lipid2Brain()
    b.use_llm_teacher = False
    b.use_wiki_teacher = False
    b.ask_user_enabled = False
    b.learn_experience = False
    return b


def obs(rel, tail, weight):
    return (f"{C}|{rel}|{tail}",
            {"head": C, "relation": rel, "tail": tail, "confidence": 0.99,
             "negated": False, "support": 40, "weight": weight,
             "evidence": "замер", "source": "experience",
             "first_seen": 1})


line("каталог памяти замера:", TMP)

# ══════════════════════════════════════════════════════════════════
line("\n════ ЗАМЕР 1. Вес НЕ тронут ничем (нулевой опыт) ════")
put_world(entity([P1, P2], [R1, R2]))
b = brain_of()
base = b.process_input(f"что такое {C}")
line("ответ:", base)

# ══════════════════════════════════════════════════════════════════
line("\n════ ЗАМЕР 2. ВТОРОМУ родителю вес 250, первому 80 ════")
# 250 — потолок, какой даёт сон; 80 — пол. Разница максимальная из возможных.
put_world(entity([P1, P2], [R1, R2]),
          dict([obs("IS_A", P1, 80), obs("IS_A", P2, 250)]))
b = brain_of()
w_obs = json.load(open(WFILE, encoding="utf-8"))["observations"]
line("веса наблюдений (прочитаны observation_weight, не заданы):",
     {k: observation_weight(v) for k, v in w_obs.items()})
heavy = b.process_input(f"что такое {C}")
line("ответ:", heavy)
line("ОТВЕТ ИЗМЕНИЛСЯ:", heavy != base)

# ══════════════════════════════════════════════════════════════════
line("\n════ ЗАМЕР 3. Тот же вес переставлен на СВОЙСТВА ════")
put_world(entity([P1, P2], [R1, R2]),
          dict([obs("HAS_PROPERTY", R1, 80), obs("HAS_PROPERTY", R2, 250),
                obs("IS_A", P1, 80), obs("IS_A", P2, 80)]))
b = brain_of()
props = b.process_input(f"какие свойства у {C}")
line("ответ:", props)

# ══════════════════════════════════════════════════════════════════
line("\n════ ЗАМЕР 4. Порядок вариантов и его происхождение ════")
data = b.find_in_memory(C)
h = (data or {}).get("base_entity", {}).get("hierarchy", [])
line("hierarchy, которую видит мозг:", h)
line("совпадает с порядком ЗАПИСИ в JSON (а не с весами):",
     h == [P1, P2])
line("weights у родителей при этом:",
     {t: observation_weight(json.load(open(WFILE, encoding='utf-8'))
                            ["observations"].get(f"{C}|IS_A|{t}", {}))
      for t in (P1, P2)})

# ══════════════════════════════════════════════════════════════════
line("\n════ ЗАМЕР 5. Определение против категории при весе 255 ════")
put_world(entity([P1], [R1], definition="старая русская обувь из войлока"),
          dict([obs("IS_A", P1, 255)]))
b = brain_of()
line("ответ:", b.process_input(f"что такое {C}"))
line("(вес категории 255 — максимум; definition веса не имеет вовсе)")

# ══════════════════════════════════════════════════════════════════
line("\n════ ЗАМЕР 6. Двойной смысл: потеряна ли связь в 250 ════")
put_world(entity([P1, P2], []),
          dict([obs("IS_A", P1, 80), obs("IS_A", P2, 250)]))
b = brain_of()
ans6 = b.process_input(f"что такое {C}")
line("ответ:", ans6)
line("упомянут ли родитель с весом 250 (%r):" % P2, P2 in ans6)
line("упомянут ли родитель с весом 80  (%r):" % P1, P1 in ans6)

# ══════════════════════════════════════════════════════════════════
line("\n════ ЗАМЕР 7. На что вес ВЛИЯЕТ (для контраста) ════")
line("fact_weight(CAN_DO, conf=0.99, support=40) =",
     fact_weight(0.99, False, "CAN_DO", 40))
line("fact_weight(CAN_DO, conf=0.5,  support=1)  =",
     fact_weight(0.5, False, "CAN_DO", 1))
b = brain_of()
put_world(entity([P1, P2], [R1, R2]),
          dict([obs("CAN_DO", "валять", 250)]))
b = brain_of()
conf = b.experience_conflicts()
line("experience_conflicts():", json.dumps(conf, ensure_ascii=False)[:300])

shutil.rmtree(TMP, ignore_errors=True)
line("\nвременный каталог удалён")
