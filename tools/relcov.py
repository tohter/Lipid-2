#!/usr/bin/env python3
"""Покрытие реестра отношений экстрактором (замер, а не утверждение в тексте).

README_ЭКСПЕРТУ.md и memory.txt говорят «из 17 отношений реестра экстрактор
порождает 14». Это проверяется здесь, а не верой на слово.

    LIPID2_STATUS=0 .venv/bin/python tools/relcov.py

CASES намеренно бедный и открытый: это НЕ тест на качество, это граница —
какие отношения в принципе недостижимы из русской фразы. Поэтому список
предложений можно дополнять, но «НЕ ПОРОЖДАЕТСЯ» имеет смысл только как
«не порождается НИ ОДНИМ из известных нам способов».
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.extract import SimpleExtractor          # noqa: E402
from core.storage import RelationType             # noqa: E402

# По одному образцу на отношение, плюс два заведомо дырявых.
CASES = [
    "Кошка пушистая.",                                  # HAS_PROPERTY
    "Кошка ловит мышь.",                                # CAN_DO, OBJECT_OF
    "Вася — инженер.",                                  # IS_ROLE
    "Пожар погас вследствие дождя.",                    # CAUSES
    "Пёс спит на подоконнике.",                         # LOCATED_IN
    "Витя дал кошке корм.",                             # RECIPIENT
    "Кошка открыла мышь лапой.",                        # INSTRUMENT
    "Кошка спрятала мышь, чтобы не есть её.",           # PURPOSE
    "Он думает о погоде.",                              # ABOUT  (дыра)
    "Солнце светит днём.",                              # CIRCUMSTANCE
    "Вода кипит при ста градусах.",                     # CONDITION
    "Кошка весит три килограмма.",                      # VALUE
    "Хвост собаки мокрый.",                             # PART_OF
    "У кошки есть хвост.",                              # HAS_PART
    "Гиппопотам, или бегемот, живёт в реке.",           # SYNONYM_OF (дыра)
    "Кошка — млекопитающее.",                           # IS_A (дыра)
]


def main() -> int:
    ex = SimpleExtractor()
    got = set()
    for s in CASES:
        for f in ex.extract(s, 1).facts:
            got.add(f.relation)
    allrel = [e.name for e in RelationType]
    missing = sorted(set(allrel) - got)
    print(f"ИТОГО ОТНОШЕНИЙ={len(allrel)} ПОРОЖДАЕТСЯ={len(got)}")
    print("ПОРОЖДАЮТСЯ   :", " ".join(sorted(got)))
    print("НЕ ПОРОЖДАЕТСЯ:", " ".join(missing) if missing else "(нет)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
