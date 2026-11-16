#!/usr/bin/env python3
"""Замер 0.3: умеет ли морфология различать одушевлённое/неодушевлённое.

Критерий хозяина (15.11): если морфология справляется с одушевлённостью,
отдельное поле в сущности не нужно — «лишь бы работало». Пробные пары:
  медвежонок (одуш.) vs плюшевый мишка / кукла (неодуш.?)
  ребёнок (одуш.) vs кукла (неодуш.)
  жена/сын/егор/зинаида/кошка (одуш.) vs хвост/стол/дождь/куст (неодуш.)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LIPID2_STATUS", "0")

from core.morph import MorphAnalyzerFacade  # noqa: E402


def main():
    m = MorphAnalyzerFacade()
    print("методы:", [x for x in dir(m) if not x.startswith("_")][:40])
    words = ["медвежонок", "мишка", "плюшевый", "ребёнок", "кукла",
             "жена", "сын", "егор", "зинаида", "кошка",
             "хвост", "стол", "дождь", "куст", "голова"]
    print("=" * 60)
    for w in words:
        d = m.parse(w)
        if not d:
            print(f"{w:12} -> (нет словаря)")
            continue
        print(f"{w:12} animacy={d.get('animacy','?')} case={d.get('case','?')} "
              f"gender={d.get('gender','?')} pos={d.get('pos','?')}")


if __name__ == "__main__":
    main()