"""
tools/sleep_cycle.py — измерить сон на ЖИВОЙ памяти и ничего не испортить.

Очередь memory.txt, п.23-доработка (P1). Контракт чисел — brain/sleep.py,
основание — experiments/PLASTICITY_CONTRACT.md.

ЗАЧЕМ ОТДЕЛЬНЫЙ ИНСТРУМЕНТ, если сон уже вызывается из мозга.
Потому что проверка «сон работает» на живой памяти невозможна в самом сне:
чтобы УВИДЕТЬ, что сон сделал бы с настоящим миром, его надо запустить, а
запустить на живом мире «посмотреть» — значит испортить. Поэтому инструмент
работает на КОПИИ: файл мира читается, считается, пишется в /tmp, живой файл
остаётся байт в байт.

Отдельно и жёстко измеряется то, что иначе проверить нечем:

  ЗАТУХАНИЕ НЕ ОТМЕНЯЕТСЯ СОБСТВЕННОЙ ЗАПИСЬЮ.
  Мозг сохраняет опыт через merge_json, а тот берёт максимум. Для накопления
  это правильно (сосед не сотрёт чужой опыт), для decay — убийственно: сон
  уменьшил силу, запись взяла максимум из файла,сила вернулась. Замерено в
  этой сессии: память 5 -> файл 10. Единственный способ это поймать — реальный
  цикл «записал -> прочитал файл», поэтому он здесь и делается (в песочнице).

ВЫХОД: 0 — сон безопасен и измерен; 1 — measurement failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from brain.sleep import (CONSOLIDATION_THRESHOLD, DECAY_RATE, SOURCE,
                        WEIGHT_CAP, WEIGHT_FLOOR, consolidate,
                        decay_transitions, reconcile_saved, sleep,
                        table_of_section)
from core.transitions import merge_json, table_from_world
from core.worldfile import read_world, world_transaction


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


def _copy_world(path: str, dst_dir: str) -> str:
    """Копия мира + рядом секция transitions в «мировом» виде для песочницы."""
    dst = os.path.join(dst_dir, os.path.basename(path))
    shutil.copy2(path, dst)
    return dst


# Текст, на котором измерялся сон (experiments/SLEEP_TASK.md, раздел 4 и 5).
# Встроен deliberately: числа в протоколе обязаны ПЕРЕВОДИМЫМ образом
# получаться заново, а не зависеть от того, какой файл лежал под рукой у
# автора замера. Рассказ короткий, но в нём есть повторяющиеся пары «глагол ->
# объект» — именно их цикл превращает в опыт силы.
SEED_STORY = ("Кошка крадётся к мыши. Кошка ловит мышь. Кошка ест мышь. "
              "Пёс лает на кошку. Пёс гоняет кошку. Кошка видит мышь.")


def seed_world(dst_dir: str, story: str, passes: int, verbose: bool) -> str:
    """
    НАКОПИТЬ опыт в песочнице и вернуть путь к миру для измерения.

    ЗАЧЕМ. Живая память проекта сейчас пуста (переходов 0), и измерить на ней
    продвижение нельзя — инструмент честно рапортует «пустой прогон», но
    числа, записанные в experiments/SLEEP_TASK.md, должны ПЕРЕВОДИМЫМ образом
    получаться заново, а не жить в памяти автора. Поэтому опыт намывается هنا,
    в отдельном каталоге, а живой файл не участвует.

    АВТОСОН ПРИ НАКОПЛЕНИИ ВЫКЛЮЧЕН: иначе измерение сна запускало бы сон по
    ходу накопления и «замер до» перестал бы быть «до».
    """
    from brain.lipid2_brain import Lipid2Brain
    os.makedirs(dst_dir, exist_ok=True)
    brain = Lipid2Brain(memory_dir=dst_dir)
    brain.sleep_enabled = False
    for i in range(max(1, passes)):
        brain.experience_from_text(story)
        if verbose and (i + 1) % 5 == 0:
            print(f"    накопление: прогон {i + 1}/{passes}")
    return brain.world_filepath


def measure(path: str, *, cycles: int, quiet: bool, label: str = "живой файл") -> int:
    """Прогон сон-циклов на КОПИИ. Возвращает код возврата."""
    say = (lambda *a: None) if quiet else print
    real_sha_before = _sha(path) if os.path.exists(path) else "(нет файла)"

    world = read_world(path)
    obs = world.get("observations") or {}
    can_do = [v for v in obs.values() if str(v.get("relation", "")).upper() == "CAN_DO"]
    table = table_from_world(world)
    pairs = [(h, t) for h in table.heads() for t in table.known_tails(h)]

    say(f"мир: {path}")
    say(f"  наблюдений {len(obs)}, из них CAN_DO {len(can_do)}, "
        f"переходов {len(pairs)} (head'ов {table.size()})")
    if not pairs:
        say("  ПЕРЕХОДОВ НЕТ. Сон на пустом опыте создать ничего не может —")
        say("  это НЕ проверка, а пустой прогон. Сначала нужен опыт: цикл")
        say("  предсказания копит силу на повторяющемся тексте.")

    rc = 0
    with tempfile.TemporaryDirectory(prefix="sleep_measure_") as tmp:
        sandbox = _copy_world(path, tmp)

        # ── 1. ПРОБНЫЙ сон: обязан не тронуть копию ────────────────────────
        sha_before = _sha(sandbox)
        dry = sleep(read_world(sandbox), None, table_from_world(read_world(sandbox)),
                    dry_run=True)
        sha_after = _sha(sandbox)
        ok = (sha_before == sha_after
              and json.load(open(sandbox, encoding="utf-8"))
              == json.load(open(path, encoding="utf-8")))
        say(f"\n1. ПРОБНЫЙ сон (dry_run): {dry.as_text()}")
        say(f"   файл не изменился: {'да' if ok else 'НЕТ'}")
        if not ok:
            say("   ⚠ пробный сон тронул файл — это дыра в правах, а неособенность")
            rc = 1

        # ── 2. ЗАТУХАНИЕ НЕ ОТМЕНЯЕТСЯ ЗАПИСЬЮ (главный замер) ────────────
        #
        # Цикл «записал -> прочитал». Разрешает спор двумя способами:
        #   а) naive save (merge_json максимумом) -> decay пропадает;
        #   б) save как делает сон (reconcile)    -> decay остаётся.
        # Если «б» тоже возвращает старую силу, сон — фикция.
        probe = table_from_world(read_world(sandbox))
        sample = probe.to_json()
        rows0 = {r["head"]: list(zip(r["tails"], r["strengths"]))
                 for r in sample.get("rows", [])}
        before_sum = sum(s for v in rows0.values() for _, s in v)
        t_probe = table_from_world(read_world(sandbox))
        d = decay_transitions(t_probe, DECAY_RATE)
        after_sum = sum(s for r in t_probe.to_json().get("rows", [])
                        for s in r["strengths"])
        say(f"\n2. ЗАТУХАНИЕ: затронуто {d.decayed}, удалено {d.compacted}, "
            f"сумма сил {before_sum} -> {after_sum}")
        if pairs and after_sum >= before_sum:
            say("   ⚠ сумма сил не упала: decay на живом мире ничего не делает")
            rc = 1

        # (а) как было бы при обычном сохранении
        naive = merge_json(t_probe.to_json(), sample)
        naive_sum = sum(s for r in naive.get("rows", []) for s in r["strengths"])
        say(f"   обычным merge_json (максимумом): {naive_sum} "
            f"<-- вот почему сон не мог пользоваться обычным сохранением")

        # (б) как делает сон: настоящий save_transitions + перечитывание
        from brain.sleep import save_transitions
        live = table_from_world(read_world(sandbox))
        baseline = live.to_json()
        decay_transitions(live, DECAY_RATE)
        save_transitions(sandbox, live, baseline=baseline)
        reread = table_of_section(read_world(sandbox).get("transitions") or {})
        reread_sum = sum(reread.get(h, t) for h in reread.heads()
                         for t in reread.known_tails(h))
        say(f"   записью сна + перечитывание файла: {reread_sum}")
        if pairs and reread_sum >= before_sum:
            say("   ⚠ decay не дожил до файла и обратно: сон не сохраняет эффект")
            rc = 1

        # ── 3. ПРОДВИЖЕНИЕ ЗНАНИЯ: сколько, чего и ДОЖИВАЕТ ЛИ ДО ФАЙЛА ───
        say(f"\n3. ПРОДВИЖЕНИЕ (порог {CONSOLIDATION_THRESHOLD}, "
            f"вес 80+8*s, потолок {WEIGHT_CAP}, пол {WEIGHT_FLOOR})")
        sandbox2 = _copy_world(path, tmp)
        created_paths, raised_paths, lowered = [], [], 0
        for i in range(max(1, cycles)):
            # ОДНА транзакция на весь сон — как в brain.sleep_now: правки
            # наблюдений и переходов обязаны уходить одним снимком. Отдельная
            # запись transitions (первый вариант инструмента) оставляла
            # знание в памяти процесса: файл его не видел, и каждый следующий
            # сон рапортовал «создано 4» ЗАНОВО. Отчёт, который повторяется,
            # неотличим от работающего — поэтому цикл пишется целиком.
            with world_transaction(sandbox2) as w:
                tbl = table_from_world(w)
                base = tbl.to_json()
                rep = consolidate(w, tbl)
                d2 = decay_transitions(tbl, DECAY_RATE)
                rep.decayed, rep.compacted = d2.decayed, d2.compacted
                sec, lost = reconcile_saved(base, tbl.to_json(),
                                            w.get("transitions") or {})
                w["transitions"] = sec
                rep.file_conflict = lost
            for c in rep.changes:
                if c["kind"] == "created":
                    created_paths.append((i + 1, c["head"], c["tail"],
                                          c["strength"], c["new_weight"]))
                elif c["kind"] == "raised":
                    raised_paths.append((i + 1, c["head"], c["tail"],
                                         c["old_weight"], c["new_weight"]))
                elif c["kind"] == "lowered":
                    lowered += 1
            say(f"   сон #{i + 1}: {rep.as_text()}")

        created_now = [k for k, v in (read_world(sandbox2).get("observations")
                                      or {}).items()
                       if v.get("source") == SOURCE]
        say(f"   создано и УВИДЕНО В ФАЙЛЕ: {len(created_now)} записей опыта")
        for cyc, h, t, s_, w_ in created_paths[:20]:
            say(f"     сон #{cyc}: {h} -> {t}  опыт {s_}, вес {w_}")
        for cyc, h, t, o_, n_ in raised_paths[:20]:
            say(f"     сон #{cyc}: поднято {h} -> {t}  {o_} -> {n_}")
        if lowered:
            say(f"     понижено конфликтом опыта: {lowered}")
        if not created_now:
            say("   создано 0. Причина механическая: ни одна пара опыта не "
                f"набрала силу {CONSOLIDATION_THRESHOLD}. Это НЕ поломка сна.")

        # ПОВТОР = БАГОПОИСК. Знание, созданное сном #1, не имеет права
        # создаваться заново во сне #2: если создаётся, значит продвижение не
        # доживает до файла и отчёт «создано N» — иллюзия прогресса.
        re_created = [c for c in created_paths if c[0] > 1]
        say(f"   идемпотентность: связей, созданных ПОВТОРНО во снах #2..#{cycles},"
            f" — {len(re_created)} (0 = продвижение сохраняется между снами)")
        if re_created:
            say(f"   ⚠ продвижение НЕ доживает до файла: {len(re_created)} "
                f"связей создаются каждый сон заново")
            rc = 1

        # пороги и пол на живом снимке
        w2 = read_world(sandbox2)
        bad = [k for k, v in (w2.get("observations") or {}).items()
               if v.get("source") == SOURCE
               and not (WEIGHT_FLOOR <= int(v.get("weight", 0)) <= WEIGHT_CAP)]
        say(f"   вне коридора [{WEIGHT_FLOOR};{WEIGHT_CAP}] записей опыта: "
            f"{len(bad)}")
        if bad:
            say(f"   ⚠ {bad[:5]}")
            rc = 1

        # отрицания не тронуты НИ ОДНИМ сном
        orig = {k: v.get("weight") for k, v in obs.items() if v.get("negated")}
        now = {k: (read_world(sandbox2).get("observations") or {})
               .get(k, {}).get("weight") for k in orig}
        touched = {k: (a, b) for k, (a, b) in orig.items()
                   if now.get(k) is not None and now[k] != a}
        say(f"   отрицанных наблюдений: {len(orig)}, из них изменено сном: "
            f"{len(touched)}")
        if touched:
            say(f"   ⚠ сон снял отрицание: {list(touched.items())[:3]}")
            rc = 1

    real_sha_after = _sha(path) if os.path.exists(path) else "(нет файла)"
    say(f"\n{label}: {real_sha_before} -> {real_sha_after} "
        f"{'(не тронут)' if real_sha_before == real_sha_after else '⚠ ИЗМЕНЁН'}")
    if real_sha_before != real_sha_after:
        rc = 1
    say("копии: в /tmp (удалены). LLM не вызывался ни разу.")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="измерить сон на копии мира")
    ap.add_argument("--world", default=os.path.join(ROOT, "data", "memory",
                                                    "lipid_world.json"))
    ap.add_argument("--cycles", type=int, default=3,
                    help="сколько сон-циклов прогнать подряд (по умолчанию 3)")
    ap.add_argument("--story", default="",
                    help="файл с текстом для --passes (по умолчанию встроенный "
                         "SEED_STORY, на котором считался experiments/SLEEP_TASK.md)")
    ap.add_argument("--text", default="",
                    help="текст для накопления прямо в командной строке")
    ap.add_argument("--passes", type=int, default=0,
                    help="накопить опыт в ПЕСОЧНИЦЕ (N прогонов текста) и "
                         "измерить сон на нём; 0 = брать --world как есть. "
                         "Нужно потому, что пустой мир измерять нечем")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.passes:
        text = args.text.strip()
        if not text and args.story:
            if not os.path.exists(args.story):
                print(f"файла текста нет: {args.story}", file=sys.stderr)
                return 1
            text = open(args.story, encoding="utf-8").read().strip()
        if not text:
            text = SEED_STORY
            if not args.quiet:
                print("текст не задан — берётся SEED_STORY (тот же, что в замере)")
        tmp = tempfile.mkdtemp(prefix="sleep_seed_")
        if not args.quiet:
            print(f"накопление опыта в песочнице {tmp} "
                  f"({args.passes} прогонов), автосон выключен")
        path = seed_world(tmp, text, args.passes, not args.quiet)
        print(f"мир для измерения (песочница): {path}")
        rc = measure(path, cycles=args.cycles, quiet=args.quiet,
                     label="исходник песочницы")
        shutil.rmtree(tmp, ignore_errors=True)
        return rc
    if not os.path.exists(args.world):
        print(f"мира нет: {args.world}", file=sys.stderr)
        return 1
    return measure(args.world, cycles=args.cycles, quiet=args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
