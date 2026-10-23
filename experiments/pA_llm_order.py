"""
ЗАМЕР п.A, шаг 5: ДЕРЖИТ ЛИ ЛОКАЛЬНАЯ МОДЕЛЬ ОБЕЩАННЫЙ ПОРЯДОК.

В промпте Учителя (teachers/teacher1.py:430) записан контракт:
    hierarchy[0] — ВСЕГДА ближайшее семейство;
    hierarchy[-1] — ВСЕГДА верхний абстрактный корень.
Контракт НИГДЕ НЕ ПРОВЕРЯЕТСЯ (замер: grep по репозиторию — ни одного теста
на порядок иерархии УЧИТЕЛЯ, краснеют только тесты порядка ЗАПИСИ). Если
модель его держит, то порядок источника есть данные, и правило порядка
выводится из него; если нет — порядок, обещанный промптом, ничем не лучше
алфавита Вики, и строить на нём ответ нельзя.

Золото — род из первой строки ру-Вики (wiki_lead_snapshot.json), тот же, что
в pA_rank.py. Модель здесь РАЗМЕЧАЕТ, ответ мозга не строит: подменой
причинного ответа она стать не может (запрет дока, п.2).
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
LEADS = os.path.join(ROOT, "experiments", "wiki_lead_snapshot.json")
OUT = os.path.join(ROOT, "experiments", "llm_hierarchy_snapshot.json")

from teachers.teacher1 import AgentTeacherV2                       # noqa: E402
from experiments.pA_rank import gold, lems                          # noqa: E402

WORDS = ["кошка", "собака", "лес", "река", "вода", "огонь", "дом",
         "машина", "книга", "стол", "город", "дерево", "флюгер",
         "солнце", "луна", "земля", "обувь", "валенок", "работа", "экосистема"]


def main():
    leads = json.load(open(LEADS, encoding="utf-8"))
    t = AgentTeacherV2(memory_dir=os.environ.get('LIPID2_MEMORY_DIR'))
    out = {}
    if os.path.exists(OUT):
        out = json.load(open(OUT, encoding="utf-8"))
    for w in WORDS:
        if w in out:
            continue
        got = t.run_agent(base_concept=w, derived_concept=w,
                          filename="lipid_world.json")
        ent = (got or {}).get("base_entity", {}) if got else {}
        out[w] = {"hierarchy": ent.get("hierarchy") or [],
                  "definition": ent.get("definition") or "",
                  "failure": getattr(t, "last_failure", None)}
        print(f"◆ {w:12s} {out[w]['hierarchy']}")
        json.dump(out, open(OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    print("\n──────── ИСПОЛНЕНИЕ КОНТРАКТА hierarchy[0] ────────")
    top_hit = root_hit = both = tot = 0
    for w, rec in out.items():
        h = [x.lower() for x in rec["hierarchy"] if isinstance(x, str)]
        g = gold(w, leads.get(w, {}).get("lead", ""))
        if not h or not g:
            continue
        tot += 1
        gl = lems(g)
        first = bool(lems(h[0]) & gl)
        # «верхний корень» строже: самое общее слово золота должно попасть в
        # последний элемент; если золота мало, считаем только первую часть
        root = bool(lems(h[-1]) & gl) if len(h) > 1 else None
        top_hit += first
        both += (first and bool(root))
        mark = "✓" if first else "✗"
        print(f"{mark} {w:12s} [0]={h[0][:28]:28s} [-1]={h[-1][:24]:24s} "
              f"| золото: {g[:40]}")
    print(f"\nПервый элемент = род из золота: {top_hit} из {tot}"
          f" (доля: {top_hit / max(1, tot):.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
