#!/usr/bin/env python3
"""support и bases обязаны врать ОДИНАКОВО, иначе вранье остаётся.

Замер (до правки): «Кошка боится воды.» без документа, затем та же фраза
с документом — learner даёт support=2 при independent_bases=1. Число
«видела 2 раза» стоит на ОДНОМ названном основании: ровно то враньё,
ради которого писался кирпич 6, но отчётливое в другом поле.

Закон после правки: support — ЭТО independent_bases, а не счётчик
свидетелей. Поддержка растёт только когда появляется ОТДЕЛЬНОЕ named-
основание; пересказ, refinement и безымянный свидетель support не греют.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lang.ru.lemma import install  # noqa: E402
install()

from brain.learner import Learner            # noqa: E402
from core.records import independent_bases   # noqa: E402


def learner_on(path):
    def read():
        if os.path.exists(path):
            return json.load(open(path, encoding="utf-8"))
        return {"entities": {}, "observations": {}, "unknown_words": {}}

    def write(w):
        json.dump(w, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return Learner(load_world=read, save_world=write)


def show(tag, rec):
    print(f"{tag:44} support={rec.get('support')} "
          f"bases={independent_bases(rec)} "
          f"weight={rec.get('weight')} law={rec.get('weight_law')}")
    return rec


def main():
    print("== A. одна фраза, второй проход с названием документа ==")
    d = tempfile.mkdtemp()
    lr = learner_on(os.path.join(d, "w.json"))
    lr.learn_from_text("Кошка боится воды.", source="t")
    key = "кошка|CAN_DO|бояться"
    r1 = show("после безымянного", lr.load_world()["observations"][key])
    lr.learn_from_text("Кошка боится воды.", source="t", document="lektsiya-2")
    r2 = show("после называния (refinement)",
              lr.load_world()["observations"][key])
    ok_a = (r1["support"] == independent_bases(r1)
            or independent_bases(r1) == 0)
    ok_b = r2["support"] == independent_bases(r2)
    print(f"   согласованы: первое={ok_a} второе={ok_b}")

    print("== B. пересказ той же статьи не двигает НИ ОДНОГО числа ==")
    d = tempfile.mkdtemp()
    lr = learner_on(os.path.join(d, "w.json"))
    lr.learn_from_text("Дождь гасит пожар.", source="wiki", document="Дождь")
    key = "дождь|CAN_DO|гасить"
    a = show("первая строка статьи", lr.load_world()["observations"][key])
    for i in range(3):
        lr.learn_from_text("Дождь гасит пожар.", source="wiki",
                           document="Дождь")
    b = show("три пересказа той же статьи", lr.load_world()["observations"][key])
    ok_c = a["support"] == b["support"] == independent_bases(b) == 1

    print("== C. вторая названная книга поднимает ОБА числа ==")
    d = tempfile.mkdtemp()
    lr = learner_on(os.path.join(d, "w.json"))
    lr.learn_from_text("Дождь гасит пожар.", source="wiki", document="Дождь")
    lr.learn_from_text("Дождь гасит пожар.", source="text",
                       document="kniga-o-vode")
    c = show("две разные книги", lr.load_world()["observations"][key])
    ok_d = c["support"] == independent_bases(c) == 2

    print("== D. вес monotone: нечем(0) < 1 < 2 < 3 < owner ==")
    from core.records import weight_from_ground as wf
    vals = [wf("testimony", 0), wf("testimony", 1), wf("testimony", 2),
            wf("testimony", 3), wf("owner", 1)]
    print("   ", vals)
    ok_e = vals == sorted(vals) and len(set(vals)) == len(vals)

    tudo = all([ok_a, ok_b, ok_c, ok_d, ok_e])
    print("\nИТОГ:", "ЗАКОН ДЕРЖИТСЯ" if tudo else "НАРУШЕНИЕ")
    return 0 if tudo else 1


if __name__ == "__main__":
    sys.exit(main())
