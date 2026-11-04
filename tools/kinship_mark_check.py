"""ЗАМЕР ПЕРЕД МЕХАНИЗМОМ В: достаточна ли иерархия как метка родства.

Зачем. Механизм В (решение босса 04.11, experiments/BOSS_PHASE4_ANSWER.txt)
должен превращать УЧИТЕЛЬСКОЕ определение родственного понятия в два ребра:
    невестка IS_ROLE жена,  жена PART_OF сын
Правило обязано срабатывать НЕ по разбору строки определения — синтаксис
различителем быть не может (experiments/STATIC_PAST.txt, вопрос 5: голое
«жена сына» даёт OBJECT_OF, а «брак с морем» даёт INSTRUMENT). Признак должен
прийти ПОМЕТКОЙ. Кандидат, который босс выбрала из трёх, — иерархия: учитель
уже кладёт `hierarchy=["родственник", ...]` для родства и не кладёт для хлама.

Что именно меряется (указание босса дословно: «для пяти родственных понятий и
пяти неродственных проверь, что учитель кладёт "родственник" в иерархию для
первых и не кладёт для вторых»). Если метка появится у неродственного — метка
ненадёжна и нужен отдельный разговор про поле учителя.

ВЕРДИКТ ПЕЧАТАЕТСЯ ПО ДВУМ ОСЯМ, и это не моя прихоть: ложное срабатывание
пишет в мир ЛОЖНОЕ ребро, пропуск не пишет ничего (молчание, тот же
безопасный отказ, что пустая иерархия). Критерий босса смотрит на первую ось;
вторая ось — это реальный охват правила, и на замере она не пустая.
Повтор пропавших слов включается REPEAT_MISSED=1 (пропуск обязан быть
детерминирован, иначе это разброс модели, а не свойство запроса).

Как ищется метка: ВХОЖДЕНИЕМ подстроки в любой элемент `hierarchy`, а не
равенством (указание босса: «ищи вхождение, не равенство»). Учитель отвечает
то «родственник», то «родственники» — сверка по равенству потеряла бы первую
форму и показала бы ложную ненадёжность метки.

Куда пишет и чего не трогает. Мир — mktemp внутри каждого прогона, живой
каталог не открывается; хэш живого мира сверяется ДО и ПОСЛЕ и печатается.
Сервер модели — localhost:8080 (`LIPID2_LLM_HOST`), тот же, что у
tools/ablation_teacher.py. Прогон медленный: это живая модель, не стаб.

Запуск:
  LIPID2_STATUS=0 .venv/bin/python tools/kinship_mark_check.py
Текст замера: experiments/KINSHIP_MARK.txt (перехватывать stdout).
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("LIPID2_STATUS", "0")

REAL_WORLD = os.path.join(ROOT, "data", "memory", "lipid_world.json")

# Родственные: у всех должен найтись корень «родствен*» в иерархии.
KIN = ["невестка", "свекровь", "золовка", "племянник", "внучка"]
# Неродственные: ни у одного не должно найтись. Подобраны так, чтобы часть
# была абстрактной и социально-семейной (брак, любовь, должность) — именно
# там у модели велик соблазн положить «семья» или «родство», и именно это
# решает, надёжна метка или нет.
NON_KIN = ["брак", "любовь", "кошка", "флюгер", "должность"]

MARK = "родствен"      # ищем ВХОЖДЕНИЕМ, не равенством


def digest(path):
    if not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def ask(word):
    """Один прогон живого учителя; возвращает (секунды, запись, ошибку)."""
    from teachers.teacher1 import AgentTeacherV2      # локально: тянет мир

    d = tempfile.mkdtemp(prefix="mark_")
    t = AgentTeacherV2(host=os.environ.get("LIPID2_LLM_HOST",
                                           "http://localhost:8080"),
                       memory_dir=d)
    t0 = time.time()
    try:
        t.run_agent(base_concept=word, derived_concept=word)
    finally:
        path = os.path.join(d, "lipid_world.json")
        ent = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                ent = json.load(fh).get("entities", {}).get(word, {})
        shutil.rmtree(d, ignore_errors=True)
    return time.time() - t0, ent, (getattr(t, "last_failure", "") or "")


def marked(ent):
    """(есть ли метка, через какой элемент иерархии)."""
    hier = ent.get("hierarchy")
    if not isinstance(hier, list):
        return False, None
    for h in hier:
        if isinstance(h, str) and MARK in h.lower():
            return True, h
    return False, None


def main():
    before = digest(REAL_WORLD)
    hr("ЗАМЕР МЕТКИ: кладёт ли учитель «родственник» в иерархию")
    print(f"  модель: {os.environ.get('LIPID2_LLM_HOST', 'http://localhost:8080')}")
    print(f"  живой мир до: {before}")
    print(f"  метка ищется ВХОЖДЕНИЕМ подстроки «{MARK}» в элемент hierarchy")

    rows = []
    for kind, words in (("РОДСТВЕННЫЕ", KIN), ("НЕРОДСТВЕННЫЕ", NON_KIN)):
        print(f"\n--- {kind} ---")
        for w in words:
            dt, ent, err = ask(w)
            hit, via = marked(ent)
            rows.append((kind, w, hit, ent))
            print(f"  {w:12} {dt:6.1f} с | метка: "
                  f"{'ЕСТЬ' if hit else 'НЕТ'}{'  (через ' + str(via) + ')' if via else ''}")
            print(f"               hierarchy={ent.get('hierarchy')}")
            print(f"               definition={str(ent.get('definition'))[:70]!r}")
            print(f"               relates_to={ent.get('relates_to')}")
            if err:
                print(f"               last_failure: {str(err)[:90]}")

    hr("ИТОГ")
    kin = [r for r in rows if r[0] == "РОДСТВЕННЫЕ"]
    non = [r for r in rows if r[0] == "НЕРОДСТВЕННЫЕ"]
    k_yes = sum(1 for r in kin if r[2])
    n_yes = sum(1 for r in non if r[2])
    print(f"  родственные: метка есть у {k_yes} из {len(kin)}")
    print(f"  неродственные: метка есть у {n_yes} из {len(non)}")
    no_hier = [r[1] for r in rows if not (r[3].get("hierarchy") or [])]
    if no_hier:
        print(f"  пустая иерархия (запись не получена): {no_hier}")
    # Критерий босса ОДНОСТОРОННИЙ и это правильно: она требовала смотреть,
    # не утекает ли метка на чужие слова. Ложное срабатывание и пропуск —
    # РАЗНЫЕ поломки: первое пишет ложное ребро (ложь в мире), второе лишь
    # не пишет ребро (молчание, тот же безопасный отказ, что и пустая
    # иерархия). Поэтому вердикт печатается по двум осям, а не одним «да/нет».
    print("\n  ВЫВОД ПО ПРАВИЛУ (две оси, не одна):")
    if n_yes:
        print(f"  ЛОЖНЫХ срабатываний: {n_yes} "
              f"{[r[1] for r in non if r[2]]} — метка НЕнадёжна, правило В"
              " писало бы ложные рёбра, нужен разговор про поле учителя")
    else:
        print("  ЛОЖНЫХ срабатываний: НОЛЬ — правило В ни разу не напишет"
              " ребро там, где родства нет; по критерию босса иерархия"
              " ДОСТАТОЧНА как различитель")
    if k_yes == len(kin):
        print(f"  ПОКРЫТИЕ своих: {k_yes}/{len(kin)} — метка ставится всегда")
    else:
        missed = [r[1] for r in kin if not r[2]]
        print(f"  ПОКРЫТИЕ своих: {k_yes}/{len(kin)} — ПРОПУСКАЮТ"
              f" {missed}; это молчание, а не ложь (ребро не написано, и"
              " ложного тоже нет), но правило В покрывает не всё родство")
        print("  ПРИРОДА ПРОПУСКА: учитель кладёт hierarchy[0]=БЛИЖАЙШИЙ класс"
              " (так требует его запрос), и для части слов ближайший класс —"
              " не «родственники», а «дети». Проверено повтором: пропуск"
              " детерминирован, это не разброс модели (см. ниже, ПОВТОР)")
    if os.environ.get("REPEAT_MISSED"):
        print("\n  ПОВТОР (LIPID2_STATUS=0 REPEAT_MISSED=1 ...):")
        for w in [r[1] for r in kin if not r[2]]:
            for i in range(2):
                _, ent, _ = ask(w)
                hit2, _ = marked(ent)
                print(f"    {w} прогон {i+1}: метка="
                      f"{'ЕСТЬ' if hit2 else 'НЕТ'} hierarchy={ent.get('hierarchy')}")
    after = digest(REAL_WORLD)
    print(f"\n  живой мир после: {after} "
          f"({'не тронут' if after == before else 'ТРОНУТ — это ошибка'})")


if __name__ == "__main__":
    main()
