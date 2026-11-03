#!/usr/bin/env python3
"""РОДИТЕЛЬНЫЙ ПРИ СУЩЕСТВИТЕЛЬНОМ — доходит ли дело до выбора отношения.

Зачем. Босс поставила между правилом шага (П4) и звеном 4 извлечение
родственных рёбер ИЗ РАССКАЗА. Прежде чем выбирать отношение и сигнал
различения (одушевлённость по уроку 79), надо измерить две вещи, о которых я
до сих пор рассуждала, а не знала:

  1. НАСТУПАЕТ ЛИ ВООБЩЕ ВЫБОР ОТНОШЕНИЯ. PART_OF рождается только в
     бесглагольном члене («хвост собаки мокрый» — строка 10 core/extract.py:
     «Род.п. без предлога -> OBJECT_OF / PART_OF в бесглагольном»). В
     рассказе у КАЖДОГО предложения есть сказуемое. Если родительный при
     подлежащем уходит глаголу, а не существительному, то выбирать не из
     чего: пары (жена, егор) не возникает ни под каким именем.

  2. ЕСТЬ ЛИ ГДЕ В ПРИРОДЕ РАССКАЗА конструкция, в которой выбор НАСТУПАЕТ.
     Если нет — правка отношения (PART_OF vs HAS_PROPERTY) мертва до написания.

Разбор НЕ трактуется: печатается сырой список токенов с падежами и сырой
список рёбер, которые экстрактор положил в мир.

Изоляция: каждый пример учится в СВОЙ /tmp-мир; живая память не трогается.

    LIPID2_STATUS=0 .venv/bin/python tools/genitive_writer.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STORY = [
    "Зинаида гостит в семье своего сына Егора.",
    "Она недолюбливает невестку и постоянно к ней придирается.",
    "Жена Егора пожаловалась мужу и попросила его поговорить с матерью, "
    "чтобы та не портила ей настроение.",
    "Егор любит жену, но боится мамы.",
    "Он пообещал жене, что поговорит с мамой, но обещание своё не выполнил.",
    "Жена затаила обиду.",
]
# Контроль: единственная конструкция, в которой PART_OF рождается по строке
# 10 core/extract.py — выбор отношения там НАСТУПАЕТ.
CONTROL = [
    "Хвост собаки мокрый.",
    "Маша — жена сына Зинаиды.",
]


def tok_table(sentence):
    from core.morph import analyze
    rows = []
    for t in analyze(sentence):
        vs = t.get("variants") or []
        cs = {v.get("case") for v in vs if v.get("case")}
        rows.append((t.get("word"), t.get("role"),
                     ",".join(sorted(cs)) or "-",
                     (vs[0].get("animacy") or "-") if vs else "-"))
    return rows


def edges_of(sentence):
    os.environ["LIPID2_MEMORY_DIR"] = tempfile.mkdtemp(prefix="gen_")
    from brain.lipid2_brain import Lipid2Brain
    from core.worldfile import read_world
    b = Lipid2Brain(memory_dir=os.environ["LIPID2_MEMORY_DIR"])
    b.use_llm_teacher = False
    b.use_wiki_teacher = False
    b.ask_user_enabled = False
    b.learn_from_text(sentence, source="замер", allow_teacher=False)
    w = read_world(b.world_filepath)
    obs = w.get("observations") or {}
    return sorted((o["head"], o["relation"], o["tail"])
                  for o in obs.values()
                  if isinstance(o, dict) and o.get("head"))


def main():
    print("═" * 74)
    print("ЗАМЕР: родительный ПРИ СУЩЕСТВИТЕЛЬНОМ в рассказе со сказуемым")
    print("═" * 74)
    n_gen_at_noun = 0
    n_edge_person_person = 0
    for sentence in STORY + ["-"] + CONTROL:
        if sentence == "-":
            print("\n" + "=" * 74)
            print("КОНТРОЛЬ: здесь сказуемого-действия нет, выбор отношения "
                  "НАСТУПАЕТ")
            print("=" * 74)
        print("\n" + sentence)
        for word, role, case, an in tok_table(sentence):
            flag = "  <== РОДИТЕЛЬНЫЙ" if "gent" in case or "gen" in case.split(
                ",") else ""
            print(f"    {word:16} роль={role:10} падеж={case:8} "
                  f"одуш={an:5}{flag}")
        edges = edges_of(sentence)
        print("    рёбра:")
        for h, r, t in edges:
            print(f"       {h:16} -{r}-> {t}")
        gen = [w for w, _, c, _ in tok_table(sentence)
               if "gent" in c or "gen" in c.split(",")]
        n_gen_at_noun += len(gen)
        ppl = [e for e in edges if e[2] in (
            "зинаида", "егор", "жена", "муж", "мама", "мать", "сын",
            "невестка", "собака", "шина")]
        n_edge_person_person += len(
            [e for e in ppl if e[1] in ("PART_OF", "IS_ROLE", "HAS_PROPERTY",
                                        "HAS_PART")])
    print("\n" + "═" * 74)
    print(f"родительных токенов всего: {n_gen_at_noun}")
    print(f"рёбер РОДА/ЧАСТИ (PART_OF/IS_ROLE/HAS_PROPERTY), заведённых этим "
          f"разбором: {n_edge_person_person}")
    print("═" * 74)


if __name__ == "__main__":
    main()
