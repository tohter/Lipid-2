"""
experiments/measure_nonsense_footprint.py — ЗАМЕР (код НЕ изменён): где
физически живёт «чушь» и кому она нужна.

Повод 09.11. Хозяин решил: заведомо ложные связи хранить не надо — «ботинок
умеет читать» нужно лишь симулятору как сказка, заводить ради каждой чуши
живую базу не стоит. Чтобы решение можно было исполнить, а не обсуждать,
замеряются ЧЕТЫРЕ факта:

  [1] СЛЕД. Что именно появляется в файле мира, когда обучающий читает чушь.
      (Раньше я мерил wrong-место: след лежит в observations, а НЕ в
      entities — первая версия мерила чистила не там и молчала.)
  [2] ОТЛИЧИМОСТЬ. Чем след чуши отличается от следа годного факта ВНУТРИ
      записи. Если поля-различия нет, запрет «не хранить чушь» сегодня
      нечем применить: запрещать придётся не чушь, а что-то другое.
  [3] КОМУ НУЖНО. Отвечает ли СИМУЛЯТОР иначе, если записи чуши удалить.
  [4] КТО ИХ ЧИТАЕТ. Отвечает ли ДУМАЛКА по этим записям как по знанию.

Живой мир НЕ трогается — копии в /tmp. LLM не зовётся. Python обязан быть
из .venv: без pymorphy3 разбор деградирует в эвристику и не извлекает фактов
вовсе, и тогда мерило мерит пустоту, а не мир (урок этого замера).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.worldfile import read_world, write_world          # noqa: E402
from brain.learner import Learner                            # noqa: E402
from brain.simulator import Simulator, StaticKnowledge       # noqa: E402

EXP = os.path.join(_ROOT, "experiments")
WORLD = os.path.join(EXP, "fire_rain_world.json")
STORY = os.path.join(EXP, "fire_rain_story.json")

NONSENSE = ("Ботинок читает стихи. Ветер раздувает пламя. "
            "Глаз не может оторваться от тебя.")
SENSE = "Дождь гасит огонь."


def _edges(world: dict) -> dict:
    return world.get("observations") or {}


def _learner(path: str) -> Learner:
    lr = Learner(load_world=lambda: read_world(path),
                 save_world=lambda w: write_world(path, w))
    lr._lr_path = path
    return lr


def _world_with(text: str, tmp: str, name: str) -> tuple:
    """Копия учебного мира + текст через обучающего; (мир, новые ключи)."""
    path = os.path.join(tmp, name + ".json")
    write_world(path, read_world(WORLD))
    before = set(_edges(read_world(path)))
    rep = _learner(path).learn_from_text(text, source="text")
    world = read_world(path)
    fresh = {k: v for k, v in _edges(world).items() if k not in before}
    return world, fresh, rep


def show(title: str) -> None:
    print(f"\n{'─' * 74}\n{title}\n{'─' * 74}")


# ── [1] след чуши ─────────────────────────────────────────────────────────
def footprint() -> None:
    show("[1] СЛЕД: ЧТО ОСТАВЛЯЕТ В СТАТИКЕ РАССКАЗ-ЧУШЬ")
    tmp = tempfile.mkdtemp(prefix="nons_")
    base = read_world(WORLD)
    print(f"  мир ДО: записей-связей {len(_edges(base))}, "
          f"событий {len(base.get('events') or [])}, "
          f"сущностей {len(base.get('entities') or {})}")
    world, fresh, rep = _world_with(NONSENSE, tmp, "n")
    print(f"  прочитано чуши: {NONSENSE!r}")
    print(f"  НОВЫХ СВЯЗЕЙ В СТАТИКЕ: {len(fresh)}  "
          f"(фактов извлечено {rep.facts}, событий рассказа добавлено {rep.events_new})")
    for k, v in fresh.items():
        print(f"    • {k:28} основание={v.get('ground')} вес={v.get('weight')} "
              f"support={v.get('support')} отрицание={v.get('negated')}")
    print(f"  всего записей в базе стало: {len(_edges(world))}")
    ents = world.get("entities") or {}
    orphans = sorted({v["head"] for v in _edges(world).values()} - set(ents))
    print(f"  сущностей создано: {rep.entities_created} — "
          f"«ботинок» в узлах: {'ботинок' in ents}")
    print(f"  головы связей БЕЗ узла (ребро висит в пустоте): {orphans}")
    unk = world.get("unknown_words") or {}
    print(f"  неизвестных слов: {len(unk)} — {sorted(unk) if isinstance(unk, dict) else unk}")
    shutil.rmtree(tmp, ignore_errors=True)


# ── [2] отличима ли чушь от годного ───────────────────────────────────────
def distinguishability() -> None:
    show("[2] ОТЛИЧИМОСТЬ: ЧЕМ СЛЕД ЧУШИ ОТЛИЧАЕТСЯ ОТ СЛЕДА ГОДНОГО ФАКТА")
    tmp = tempfile.mkdtemp(prefix="nons_")
    rows = []
    for label, text in (("чушь   ", NONSENSE), ("годное ", SENSE)):
        _, fresh, _ = _world_with(text, tmp, "d_" + label.strip())
        for k, v in fresh.items():
            rows.append((label.strip(), k, v))
    print(f"  {'источник-текст':12} {'связь':28} осн.  вес  sup  law")
    for label, k, v in rows:
        print(f"  {label:12} {k:28} {str(v.get('ground'))[:11]:11} "
              f"{str(v.get('weight')):4} {str(v.get('support')):4} {v.get('weight_law')}")
    sig = set()
    for _, _, v in rows:
        sig |= set(v.keys())
    print(f"\n  поля записи: {sorted(sig)}")
    fiction = [f for f in sorted(sig) if any(
        t in f.lower() for t in ("fiction", "genre", "made", "unreal",
                                 "believ", "modal", "assert", "world"))]
    print(f"  из них признаком «это сказка / это вымысел»: "
          f"{fiction or 'НЕТ НИ ОДНОГО'}")
    print("\n  СВЕРКА РУКАМИ: 'ветер раздувать' (правда) и 'ботинок читать' "
          "(чушь) имеют одинаковые")
    same = all(rows[0][2].get(f) == rows[1][2].get(f)
               for f in ("ground", "weight", "support", "weight_law", "source"))
    print(f"  основание, вес, поддержку, закон и имя канала: ОДИНАКОВЫ = {same}")
    shutil.rmtree(tmp, ignore_errors=True)


# ── [3] нужен ли симулятору ботинок в статике ─────────────────────────────
def simulator_need() -> None:
    show("[3] КОМУ НУЖНО: ОТВЕЧАЕТ ЛИ СИМУЛЯТОР ИНАЧЕ БЕЗ ЗАПИСЕЙ ЧУШИ")
    tmp = tempfile.mkdtemp(prefix="nons_")
    world, fresh, _ = _world_with(NONSENSE, tmp, "s")
    keys = sorted(fresh)

    def run(w: dict) -> str:
        sim = Simulator(w)
        sim.load_story(STORY)
        r = sim.why_stopped("пожар")
        return f"ok={r.ok} ответ={r.answer!r} путь={len(r.chain)} " \
            f"не хватило={r.missing}"

    print(f"  записи чуши, которые пробую удалить: {keys}")
    with_ans = run(world)
    print(f"  С чушью в статике:    {with_ans}")

    trimmed = json.loads(json.dumps(world, ensure_ascii=False))
    for k in keys:
        trimmed["observations"].pop(k, None)
    without_ans = run(trimmed)
    print(f"  БЕЗ чуши в статике:  {without_ans}")
    print(f"\n  ОТВЕТ СИМУЛЯТОРА ИЗМЕНИЛСЯ: {with_ans != without_ans} "
          f"(связей удалено {len(keys)}, событий рассказа "
          f"{len(world.get('events') or [])} тронуто: НЕТ)")

    sk_full = StaticKnowledge(world)
    sk_cut = StaticKnowledge(trimmed)
    print(f"  related('дождь','CAN_DO')   с чушью {sk_full.related('дождь','CAN_DO')} "
          f"/ без {sk_cut.related('дождь','CAN_DO')}")
    shutil.rmtree(tmp, ignore_errors=True)


# ── [4] кто их читает ─────────────────────────────────────────────────────
def readers() -> None:
    show("[4] КТО ИХ ЧИТАЕТ: ДУМАЛКА ОТВЕЧАЕТ ПО ЧУШИ КАК ПО ЗНАНИЮ")
    tmp = tempfile.mkdtemp(prefix="nons_")
    for label, mutate in (("со связью 'ботинок CAN_DO читать'", lambda w, f: None),
                          ("без неё",
                           lambda w, f: [w["observations"].pop(k, None) for k in f])):
        world, fresh, _ = _world_with(NONSENSE, tmp, "t_" + label[:6])
        mutate(world, sorted(fresh))
        sk = StaticKnowledge(world)
        links = sk.related("ботинок", "CAN_DO")
        print(f"  {label}: related('ботинок','CAN_DO') = {links}")
        try:
            from brain.thinker import Thinker
            th = Thinker(sk)
            v = th.think("ботинок может читать стихи?")
            ok = getattr(v, "ok", "?")
            conf = getattr(v, "certainty", getattr(v, "confidence", "?"))
            txt = getattr(v, "text", None) or getattr(v, "answer", "") or v.code
            print(f"      вердикт думалки: ok={ok} код={getattr(v,'code','?')} "
                  f"достоверность={conf} фраза={str(txt)[:90]!r}")
        except Exception as exc:                              # noqa: BLE001
            print(f"      думалка не позвана: {type(exc).__name__}: {exc}")
    shutil.rmtree(tmp, ignore_errors=True)


# ── [5] один ли рычаг с «Маша была жирная» ───────────────────────────
def one_lever() -> None:
    show("[5] ОДИН ЛИ РЫЧАГ: ОДНО ПРЕДЛОЖЕНИЕ -> СВЯЗЬ ИЛИ СОБЫТИЕ")
    tmp = tempfile.mkdtemp(prefix="nons_")
    for t in ("Ботинок читает стихи.",      # чушь, настоящее
              "Ботинок прочитал стих.",     # та же чушь, прошедшее
              "Ветер раздувает пламя.",     # правда, настоящее
              "Ветер раздуть пламя.",
              "Ветер раздул пламя.",        # правда, прошедшее
              "Маша была жирная.",          # пункт 1 очереди
              "Маша толстеет."):
        world, fresh, rep = _world_with(t, tmp, "e")
        ev = [f"{e['subject']} {e['action']}" for e in (world.get("events") or [])]
        where = "СОБЫТИЕ" if (rep.events_new and not fresh) else (
            "СВЯЗЬ в статике" if fresh else "НИЧЕГО")
        print(f"  {t!r:28} -> {where:15} связей +{len(fresh)}, событий +{rep.events_new}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    footprint()
    distinguishability()
    simulator_need()
    readers()
    one_lever()
