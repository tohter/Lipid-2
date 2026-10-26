#!/usr/bin/env python3
"""
tools/golden_corpus.py — метрика качества экстрактора (очередь, пункт A).

ЗАЧЕМ. 628 тестов отвечают «не сломалось ли запертое» и НЕ отвечают «стало ли
лучше». Этот прогон отвечает на второй вопрос числом: precision/recall по
золотому корпусу, отдельно DEV (конструкции, по которым писались правила) и
HELD-OUT (свежая лексика, правила по ней НЕ писались).

КОРПУС РАЗМЕЧЕН ДО ПЕРВОГО ПРОГОНА (26.10.26). Эталон, подогнанный под вывод,
перестаёт быть эталоном — это тот же класс лжи, что «зелёный тест на мутанте».

ЧЕСТНОСТЬ МЕРЫ (правила проекта: «НЕ СПРАШИВАЛИ» != «НЕ НАШЛИ»):
  * recall    = найденные эталоны / все эталоны (known_gap-кейсы В отдельном
                знаменателе: это известные дыры, их отсутствие — факт, а не
                регресс);
  * precision = эталоны, которые что-то объяснили, / ВСЁ, что экстрактор
                выдал. Всё лишнее идёт в UNLABELED: это НЕ «ложь», а «не
                размечено» — каждая лишняя тройка обязана быть разобрана
                человеком и либо добавлена в корпус, либо признана мусором.
                Записать её в ложь без разбора — сочинить цифру;
  * forbid    = перевёрнутый смысл. Его нарушение — ПАДЕНИЕ, а не процент:
                перевёрнутая связь опаснее отсутствующей (урок CAUSES).

ЛОГИКУ ЭКСТРАКТОРА ЭТОТ ФАЙЛ НЕ ТРОГАЕТ. Только читает данные и считает.

Замер:  .venv/bin/python tools/golden_corpus.py
        .venv/bin/python tools/golden_corpus.py --json experiments/CORPUS_METRIC.json
"""
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEV_PATH = os.path.join(ROOT, "experiments", "corpus_dev.json")
HELD_PATH = os.path.join(ROOT, "experiments", "corpus_heldout.json")

Triple = Tuple[str, str, str]

#
# Роли, которые ОБРЕЗАЮТСЯ при записи события. Замерено, а не предполагается:
# секция events в core/worldfile.py хранит только subject/action/obj, поэтому
# у фразы в прошедшем времени место, орудие, получатель, цель и условие
# доходят до мира как OBJECT_OF. Падежная роль стёрта, связь верна — это
# отдельная мера (role_loss), а не «ложь» и не «верно».
#
PRODUCTION_ROLES = frozenset({"LOCATED_IN", "INSTRUMENT", "RECIPIENT",
                              "PURPOSE", "CONDITION", "CIRCUMSTANCE", "ABOUT"})


def load_cases(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["cases"]


def _norm(tr: list) -> Triple:
    """Эталон из JSON -> тройка. 4-й элемент (комментарий) отбрасывается."""
    return (tr[0], tr[1], tr[2])


def produce_extract(sf) -> List[Tuple[Triple, bool]]:
    """
    Что экстрактор выдал по фразе: (тройка, отрицана_ли).

    События (basket=event) учитываются и как Fact(is_event), и как EventRecord:
    смысл один, корзина разная, судим по смыслу (запертое поведение
    test_events_contract: прошедшее время = событие).
    """
    out: List[Tuple[Triple, bool]] = []
    for f in sf.facts:
        out.append(((f.head, f.relation, f.tail), f.negated))
    for ev in getattr(sf, "events", []):
        if ev.action:
            out.append(((ev.subject, "CAN_DO", ev.action), bool(getattr(ev, "negated", False))))
            if ev.obj:
                out.append(((ev.action, "OBJECT_OF", ev.obj), bool(getattr(ev, "negated", False))))
    return out


def score_case(case: dict, produced: List[Tuple[Triple, bool]],
               level: str = "extract") -> dict:
    """Арифметика одного кейса. Одна функция — один замер, всё проверяемо.

    УРОВЕНЬ важен: мозг переписывает имя отношения, не меняя смысла
    (REL_ALIAS: IS_ROLE -> IS_A). Судить уровень мира эталоном уровня
    экстрактора — значит записать дырой то, чего дырой нет. Поэтому в кейсе
    могут лежать expect_world / neg_world: они обязательны, если на этом
    уровне отношение зовётся иначе.
    """
    suffix = "_world" if level == "world" else ""

    def getl(key: str) -> list:
        """Список кейса на этом уровне: базовый ПЛЮС уровень-специфичный.

        expect_world ЗАМЕЩАЕТ expect (мозг переименовал отношение — прежний
        эталон на этом уровне неправ), а also_ok_world/junk_world ДОПОЛНЯЮТ
        (связь, рождённая learner'ем, не отменяет верных связей экстрактора).
        Заменять вместо дополнения = потерять покрытие и назвать его 100%.
        """
        base = case.get(key, [])
        if not suffix:
            return base
        return list(base) + list(case.get(key + suffix, []))

    def getrepl(key: str) -> list:
        return case.get(key + suffix) or case.get(key, []) if suffix else case.get(key, [])

    expect = [_norm(t) for t in getrepl("expect")]
    neg = [_norm(t) for t in getrepl("neg")]
    forbid = [_norm(t) for t in case.get("forbid", [])]
    # also_ok/junk тоже бывают уровнем выше: алиас имени и REL_ALIAS работают в
    # learner'е, поэтому на уровне мира появляется связь, которой нет на уровне
    # экстрактора (замерено: SYNONYMOF гиппопотам/бегемот рождается именно там).
    also_ok = [_norm(t) for t in getl("also_ok")]
    junk = [_norm(t) for t in getl("junk")]

    pos = {t for t, n in produced if not n}
    negd = {t for t, n in produced if n}
    every = pos | negd

    # ПЕРЕКЛИКА ЭТАЛОНОВ. Одна и та же тройка, записанная в два кармана,
    # засчитывалась ДВАЖДЫ: в числитель recall (как neg) и в precision (как
    # also_ok). Замерено на h28. Приоритет карманов фиксирован, а переклика
    # видна отдельным полем — иначе разметка может тихо надувать метрику,
    # и отличить её от честной стало бы нельзя.
    conflict = sorted(
        (set(expect) & set(neg)) | (set(expect) & set(also_ok))
        | (set(neg) & set(also_ok)) | (set(junk) & (set(expect) | set(neg)
                                                   | set(also_ok)))
        | (set(forbid) & (set(expect) | set(neg) | set(also_ok))))
    neg = [t for t in neg if t not in set(expect)]
    also_ok = [t for t in also_ok
               if t not in set(expect) | set(neg) | set(forbid) | set(junk)]
    junk = [t for t in junk
            if t not in set(expect) | set(neg) | set(also_ok) | set(forbid)]

    hits = [t for t in expect if t in pos]
    misses = [t for t in expect if t not in pos]
    neg_hits = [t for t in neg if t in negd]
    neg_misses = [t for t in neg if t not in negd]
    forbid_hits = [t for t in forbid if t in pos or t in negd]

    labeled = set(expect) | set(neg) | set(forbid) | set(also_ok) | set(junk)
    extra = sorted({t for t, _ in produced} - labeled)

    # Корзина (fact/event) судится в score_run: там есть sf. Здесь её нет, и
    # делать вид, что судим, — значит написать зелёный тест на пустом месте.
    return {
        "id": case["id"], "text": case["text"],
        "markup_conflict": conflict,
        "expect_n": len(expect), "hits": hits, "misses": misses,
        "neg_n": len(neg), "neg_hits": neg_hits, "neg_misses": neg_misses,
        "forbid_hits": forbid_hits, "extra": extra,
        # Три судьбы каждой ВЫДАННОЙ тройки (precision без этого — одно число
        # по трём разным вещам): верная и размеченная, верная но не размеченная
        # (also_ok), ЛОЖНАЯ (junk) и совсем неизвестная (extra).
        "also_ok": [t for t in also_ok if t in every],
        "junk_hits": [t for t in junk if t in every],
        "known_gap": bool(case.get("known_gap")),
        "produced_n": len({t for t, _ in produced}),
    }


def produce_world(text: str, workdir: str, case_id: str) -> List[Tuple[Triple, bool]]:
    """
    Что экстрактор ВЫДАЛ ПОСЛЕ того, как урок прошёл через мозг.

    Зачем отдельный уровень (урок 1: «снимок данных != поведение кода»): по
    экстрактору видно разбор, но в мир связь проходит через brain/learner.py —
    там срез уверенности, алиас отношения (REL_ALIAS: IS_ROLE -> IS_A) и
    отбраковка частей. Метрика по одному уровню называет дырой то, чего дырой
    нет, и наоборот.

    Мир на каждый кейс СВОЙ: в общем мире одна фраза зачла бы связь другой,
    и recall измерял бы порядок кейсов в файле.
    """
    from brain.learner import Learner
    from core.worldfile import read_world, write_world
    import os

    path = os.path.join(workdir, f"{case_id}.json")
    if os.path.exists(path):
        os.remove(path)
    lr = Learner(load_world=lambda: read_world(path),
                 save_world=lambda w: write_world(path, w),
                 core=None, teacher=None)
    lr.learn_from_text(text, source="text", allow_teacher=False)
    world = read_world(path)
    out: List[Tuple[Triple, bool]] = []
    for key, obs in (world.get("observations") or {}).items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        out.append(((parts[0], parts[1], parts[2]), bool(obs.get("negated"))))
    for ev in world.get("events") or []:
        act = ev.get("action")
        if act:
            neg = bool(ev.get("negated"))
            out.append(((ev.get("subject", ""), "CAN_DO", act), neg))
            if ev.get("obj"):
                out.append(((act, "OBJECT_OF", ev["obj"]), neg))
            if ev.get("cause"):
                out.append(((ev["cause"], "CAUSES", act), neg))
    return out


def score_run(extractor, cases: List[dict], producer=produce_extract,
              workdir: str = "") -> dict:
    per, rel_stats = [], defaultdict(lambda: [0, 0])  # rel -> [hit, total]
    for case in cases:
        if workdir:
            produced = producer(case["text"], workdir, case["id"])
            sf = extractor.extract(case["text"])
        else:
            sf = extractor.extract(case["text"])
            produced = producer(sf)
        r = score_case(case, produced, "world" if workdir else "extract")
        # ПОТЕРЯ РОЛИ (замерено, а не предполагается): событие рассказа хранит
        # только subject/action/obj (core/worldfile, секция events). Значит у
        # фразы в прошедшем времени МЕСТО, ОРУДИЕ, ПОЛУЧАТЕЛЬ, ЦЕЛЬ и УСЛОВИЕ
        # доходят до мира как OBJECT_OF — падежная роль стёрта, хотя связка
        # верна. Это НЕ ложь и НЕ верное знание: отдельная мера, иначе
        # precision смешает «нашло криво» и «нашло криво и сохранило криво».
        if workdir:
            roles = PRODUCTION_ROLES
            flat = {t for t, _ in produced}
            lost = []
            for f in sf.facts:
                if f.relation in roles:
                    collapsed = (f.head, "OBJECT_OF", f.tail)
                    if collapsed in flat and (f.head, f.relation, f.tail) not in flat:
                        lost.append([f.head, f.relation, f.tail])
            r["role_loss"] = lost
        # корзина: для event совпадение обязано быть событием (факт is_event или EventRecord)
        if case.get("basket") == "event":
            ev_triples = set()
            for f in sf.facts:
                if f.is_event:
                    ev_triples.add((f.head, f.relation, f.tail))
            for ev in getattr(sf, "events", []):
                ev_triples.add((ev.subject, "CAN_DO", ev.action))
                if ev.obj:
                    ev_triples.add((ev.action, "OBJECT_OF", ev.obj))
            lost = [t for t in r["hits"] if t not in ev_triples]
            if lost:
                r["basket_bad"] = lost
                r["hits"] = [t for t in r["hits"] if t in ev_triples]
                r["misses"].extend(r.pop("basket_bad"))
        for t in r["hits"]:
            rel_stats[t[1]][0] += 1
            rel_stats[t[1]][1] += 1
        for t in r["misses"]:
            rel_stats[t[1]][1] += 1
        per.append(r)

    n_exp = sum(r["expect_n"] + r["neg_n"] for r in per)
    n_hit = sum(len(r["hits"]) + len(r["neg_hits"]) for r in per)
    n_also = sum(len(r["also_ok"]) for r in per)
    n_junk = sum(len(r["junk_hits"]) for r in per)
    gap_exp = sum(r["expect_n"] + r["neg_n"] for r in per if r["known_gap"])
    gap_hit = sum(len(r["hits"]) + len(r["neg_hits"]) for r in per if r["known_gap"])
    produced_total = sum(r["produced_n"] for r in per)
    extra_total = sum(len(r["extra"]) for r in per)
    forbid_total = sum(len(r["forbid_hits"]) for r in per)
    plain_exp, plain_hit = n_exp - gap_exp, n_hit - gap_hit

    return {
        "cases": len(per), "per_case": per,
        "recall": round(n_hit / n_exp, 4) if n_exp else 0.0,
        "recall_plain": round(plain_hit / plain_exp, 4) if plain_exp else 0.0,
        "known_gap_found": f"{gap_hit}/{gap_exp}",
        "precision_strict": round(n_hit / produced_total, 4) if produced_total else 0.0,
        # precision с учётом триажа: ALSO_OK — второстепенные ВЕРНЫЕ связи,
        # JUNK — ЛОЖНЫЕ. Не Размеченные (extra) в точный числитель НЕ идут:
        # неизвестное не бывает зачтено как верное.
        "precision_triaj": round((n_hit + n_also) / produced_total, 4) if produced_total else 0.0,
        "also_ok": n_also, "junk": n_junk,
        "role_loss": sum(len(r.get("role_loss", [])) for r in per),
        "junk_rate": round(n_junk / produced_total, 4) if produced_total else 0.0,
        "unlabeled": extra_total,
        # Переклика эталонов = метрика надута самим корпусом. Не «0 для красоты»,
        # а предупреждение: пока переклики нет, ни одна precision не trustworthy.
        "markup_conflicts": sum(len(r.get("markup_conflict", [])) for r in per),
        "forbid_violations": forbid_total,
        "produced_total": produced_total,
        "expect_total": n_exp, "hit_total": n_hit,
        "by_rel": {k: f"{v[0]}/{v[1]}" for k, v in sorted(rel_stats.items())},
    }


def main(argv: List[str]) -> int:
    if "--help" in argv:
        print(__doc__)
        return 0
    from core.extract import SimpleExtractor
    extractor = SimpleExtractor()
    level = "extract"
    if "--level" in argv:
        level = argv[argv.index("--level") + 1]
    workdir = ""
    if level == "world":
        import tempfile
        workdir = tempfile.mkdtemp(prefix="lipid_corpus_")
    out = {}
    for name, path in (("dev", DEV_PATH), ("heldout", HELD_PATH)):
        if level == "world":
            out[name] = score_run(extractor, load_cases(path),
                                  produce_world, workdir)
        else:
            out[name] = score_run(extractor, load_cases(path))

    if "--json" in argv:
        dst = argv[argv.index("--json") + 1]
        with open(dst, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print(f"записан {dst}")

    for name, r in out.items():
        print(f"\n=== БЛОК {name.upper()} (уровень {level}): кейсов {r['cases']} ===")
        print(f"recall        {r['recall']}   (обычные {r['recall_plain']}, "
              f"известные дыры найдено {r['known_gap_found']})")
        print(f"precision     {r['precision_strict']} строгая / {r['precision_triaj']} с триажем  "
              f"(выдано {r['produced_total']}: верно {r['hit_total']}/{r['expect_total']}, "
              f"верно но второстепенно {r['also_ok']}, ЛОЖЬЮ {r['junk']} ({r['junk_rate']}), "
              f"не размечено {r['unlabeled']})")
        print(f"PEREBOROV     {r['forbid_violations']}  (перевёрнутый смысл; обязан быть 0)")
        print("по отношениям (попало/ждало):")
        for rel, v in r["by_rel"].items():
            print(f"   {rel:14} {v}")
        for c in r["per_case"]:
            if c["misses"] or c["neg_misses"] or c["forbid_hits"]:
                print(f"  ✗ {c['id']} {c['text']}")
                for t in c["misses"]:
                    print(f"      ПРОПУЩЕНО     {t[0]} --{t[1]}--> {t[2]}")
                for t in c["neg_misses"]:
                    print(f"      НЕ ОТРИЦАЕТ  {t[0]} --{t[1]}--> {t[2]}")
                for t in c["forbid_hits"]:
                    print(f"      ⚠ ПЕРЕВОРОТ  {t[0]} --{t[1]}--> {t[2]}")
        if r.get("role_loss") or any(c.get("role_loss") for c in r["per_case"]):
            rl = sum(len(c.get("role_loss", [])) for c in r["per_case"])
            print(f"потеря РОЛИ в событии (место/орудие/цель стали OBJECT_OF): {rl}")
            for c in r["per_case"]:
                for t in c.get("role_loss", []):
                    print(f"      роль → OBJECT_OF: {t[0]} --{t[1]}--> {t[2]}")
    tot = sum(len(c["extra"]) for r in out.values() for c in r["per_case"])
    conf = sum(r.get("markup_conflicts", 0) for r in out.values())
    if conf:
        print(f"ВНИМАНИЕ: переклика эталонов {conf} — одна связь засчитана дважды, "
              f"метрика надута разметкой (см. поле markup_conflict)")
    print(f"\nИТОГО не размечено троек (каждую обязан разобрать человек): {tot}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
