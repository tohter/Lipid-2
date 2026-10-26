#!/usr/bin/env python3
"""
experiments/pG_causes_measure.py — ЗАМЕР очереди G: что экстрактор делает
с причиной.

G утверждает: «экстрактор НЕ порождает CAUSES». Проверка обязана быть
ПОВЕДЕНЧЕСКОЙ (что выдаёт extract() на причинных предложениях), а не grep'ом:
grep — снимок строк, а не производство отношений.

ЧТО НАШЁЛ ЗАМЕР (хуже, чем «не порождает»).

1. ПРИЧИНА ЗАПИСЫВАЕТСЯ ЛОЖЬЮ ДРУГОГО СМЫСЛА, а не теряется:
       Пожар гаснет из-за дождя.      -> гаснуть --LOCATED_IN--> дождь
       Металл плавится от жары.       -> плавиться --LOCATED_IN--> жара
  оба predicate=fact, т.е. идут В ОНТОЛОГИЮ (brain/learner.py отсекает только
is_event). Подтверждено СКВОЗНЫМ прогоном через мозг: в наблюдения мира
легло «гаснуть|LOCATED_IN|дождь» с confidence 0.85.

2. ДЫРА ШИРЕ ПРЕДЛОГОВ. «вследствие», «ввиду», «благодаря» — по pymorphy
   предлоги с уверенностью 1.0, но их НЕТ в PREP_CASES, а ALL_PREPS = set(PREP_CASES).
   Значит _tag_prepositions их вообще не помечает: слово идёт в разбор КАК
   САМОСТОЯТЕЛЬНАЯ лексику, а отношение получается наугад:
       Пожар погас вследствие дождя.  -> погаснуть --OBJECT_OF--> дождь
       Успех пришёл благодаря тренеру. -> прийти    --RECIPIENT--> тренер

3. ДВУСМЫСЛЕННЫЕ «из-за» И «от» неразличимы НИКАК: и для места, и для
   причины экстрактор выдаёт ОДНО И ТО ЖЕ отношение (замер корпуса ниже:
   МЕСТО 7/7 LOCATED_IN, ПРИЧИНА 6/6 LOCATED_IN). Морфология тоже молчит:
   «угла», «дождя», «куста» — все неодушевлённые существительные. Значит
   любой дефолт здесь ОБЯЗАН кому-то соврать: это выбор хозяина, а не баг.

4. ОРИЕНТАЦИЯ. В данных мира CAUSES направлен «причина -> следствие»
   (experiments/cat_canary_world.json: предательство -> обида -> гнев ->
   агрессия -> нападение). Наивная правка («тот же хвост, другое имя») дала бы
   «погаснуть -> дождь» = «гаснуть ВЫЗЫВАЕТ дождь», то есть причину шиворот-
   навыворот. Причина обязана стать ГОЛОВОЙ.

5. ПОТРЕБИТЕЛЬ ЕСТЬ, данных нет. SchemeInterpreter._direct и
   StaticKnowledge.causes() читают CAUSES; миров, где он вписан РУКАМИ, —
   ровно один. Никакого канала, который бы его заполнял из текста, нет.

6. ПРОШЕДШЕЕ ВРЕМЯ прячет причину ещё глубже (замер сквозным прогоном).
   «Пожар погас вследствие дождя» разводится как СОБЫТИЕ, а формат события
   хранит только subject/action/obj: в онтологию событие не идёт (верно —
   закон из одного случая не выводится), и причина исчезала бесследно:
   0 из 3 предложений сохранили её в мире. Сейчас событие несёт поле
   "cause" — см. строку past_kept в ИТОГЕ и запертый тестом
   test_past_tense_cause_survives. В онтологию однократный случай не
   попадает — это принцип, а не упущение (заперт отдельным тестом).

Замер ПЕРЕЧИТЫВАЕМЫЙ: строку ИТОГ сверяет тест
tests/test_cause_extraction_contract.py командой, а не на глаз.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.extract import SimpleExtractor                    # noqa: E402

EX = SimpleExtractor()

# Двусмысленные предлоги: ОДИНАКОВЫЙ ответ у места и у причины.
PLACE = ["Мальчик выглянул из-за угла.", "Пёс спрятался из-за куста.",
         "Ребёнок вышел из-за двери.", "Кот прыгнул из-за стола.",
         "Птица улетела из-за крыши.", "Мальчик отошёл от окна.",
         "Пёс отошёл от миски."]
CAUSE_POLY = ["Пожар погас из-за дождя.", "Металл плавится от жары.",
              "Растения гибнут из-за засухи.", "Ребёнок плачет от боли.",
              "Посевы погибли от жары.", "Металл темнеет от времени."]
# Однозначные причинные предлоги — их в таблице нет вообще.
CAUSE_UNAMBIG = ["Пожар погас вследствие дождя.", "Урожай сгорел ввиду жары.",
                 "Успех пришёл благодаря тренеру."]


def rels(sentence):
    return [f.relation for f in EX.extract(sentence, order=1).facts]


def count_rel(sentences, rel):
    return sum(1 for s in sentences for f in EX.extract(s, order=1).facts
               if f.relation == rel)


print("G1. причина как ложь (плоские конструкции, predicate=fact)")
for s in CAUSE_POLY[:2]:
    for f in EX.extract(s, order=1).facts:
        print(f"  {s}\n      {f!r}")

print()
print("G2. предлогов-причин нет в таблице -> отношение наугад")
for s in CAUSE_UNAMBIG:
    print(f"  {s}")
    for f in EX.extract(s, order=1).facts:
        print(f"      {f!r}")

print()
print("G3. неразличимость «из-за»/«от»: МЕСТО против ПРИЧИНЫ")
print(f"  МЕСТО   : LOCATED_IN={count_rel(PLACE, 'LOCATED_IN')} из {len(PLACE)},"
      f" CAUSES={count_rel(PLACE, 'CAUSES')}")
print(f"  ПРИЧИНА  : LOCATED_IN={count_rel(CAUSE_POLY, 'LOCATED_IN')} из "
      f"{len(CAUSE_POLY)}, CAUSES={count_rel(CAUSE_POLY, 'CAUSES')}")

print()
print("G4. ориентация CAUSES в данных и её потребитель")
import json                                          # noqa: E402
import os as _os                                     # noqa: E402


def manual_cause_worlds(exp_dir):
    """
    Миры, где CAUSES вписан РУКАМИ.

    ПРЕЖНЕЕ ОПРЕДЕЛЕНИЕ БЫЛО ЛОЖНЫМ: «любой .json в experiments, где
    встречается строка CAUSES». Значит счётчик зависел от того, что лежит в
    каталоге по СОСЕДСТВУ, а не от формата мира. ЗАМЕРЕНО: добавление метрики
    золотого корпуса (experiments/CORPUS_METRIC*.json, где CAUSES — имя
    отношения в отчёте) подняло manual_worlds с 1 до 5, и запертый тест
    test_manual_world_count упал. Тот же класс хрупкости, что grep вместо
    поведения: ищем подстроку вместо структуры.

    Теперь мир — это мир: секция observations с записью relation == CAUSES.
    Слово в тексте фразы или в отчёте миром не считается.
    """
    found = []
    for fn in sorted(_os.listdir(exp_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(_os.path.join(exp_dir, fn), encoding="utf-8") as fh:
                data = json.load(fh)
        except (ValueError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        obs = data.get("observations")
        if not isinstance(obs, dict):
            continue
        if any(isinstance(v, dict) and v.get("relation") == "CAUSES"
               for v in obs.values()):
            found.append(fn)
    return found


worlds = manual_cause_worlds(_os.path.join(ROOT, "experiments"))
print(f"  миров с CAUSES, вписанным руками: {len(worlds)} {worlds}")
if worlds:
    d = json.load(open(_os.path.join(ROOT, "experiments", worlds[0]),
                       encoding="utf-8"))
    for v in (d.get("observations") or {}).values():
        if isinstance(v, dict) and v.get("relation") == "CAUSES":
            print(f"      {v.get('head')} --CAUSES--> {v.get('tail')}")
from brain.simulator import StaticKnowledge          # noqa: E402
sk = StaticKnowledge(d if worlds else {})
print(f"  читает ли симулятор: causes('обида') = {sk.causes('обида')}")

print()
print("G5. причина в ПРОШЕДШЕМ времени: событие её переживает?")
# «погас» -> predicate='event'; в онтологию событие не идёт (верно: закон из
# одного случая не выводится). ПроверЯЕМ, что причина не исчезла бесследно.
import shutil                                          # noqa: E402
from brain.lipid2_brain import Lipid2Brain            # noqa: E402

PAST = ["Пожар погас вследствие дождя.", "Урожай сгорел ввиду жары.",
        "Успех пришёл благодаря тренеру."]
TMP = "/tmp/pG_measure_mem"
survived = 0
for s in PAST:
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    os.environ["LIPID2_MEMORY_DIR"] = TMP
    brain = Lipid2Brain()
    brain.learner.learn_from_text(s, source="text", allow_teacher=False)
    w = brain.learner.load_world()
    ev = [e for e in (w.get("events") or []) if isinstance(e, dict)]
    kept = bool(ev) and bool(ev[0].get("cause"))
    survived += kept
    print(f"  {'СОХРАНЕНА' if kept else 'ПОТЕРЯНА '} {s} -> "
          f"событий {len(ev)}, причина {ev[0].get('cause') if ev else None!r}")
shutil.rmtree(TMP, ignore_errors=True)
print(f"  причина переживает урок: {survived} из {len(PAST)}")

print()
print("ИТОГ " + " ".join([
    f"past_kept={survived}",
    f"past_total={len(PAST)}",
    f"place_located={count_rel(PLACE, 'LOCATED_IN')}",
    f"cause_poly_located={count_rel(CAUSE_POLY, 'LOCATED_IN')}",
    f"poly_causes={count_rel(PLACE + CAUSE_POLY, 'CAUSES')}",
    f"unambig_causes={count_rel(CAUSE_UNAMBIG, 'CAUSES')}",
    f"vseledstvie={rels('Пожар погас вследствие дождя.')[1]}",
    f"vvidu={rels('Урожай сгорел ввиду жары.')[1]}",
    f"blagodarya={rels('Успех пришёл благодаря тренеру.')[1]}",
    f"izza={rels('Пожар гаснет из-за дождя.')[1]}",
    f"ugol={rels('Мальчик выглянул из-за угла.')[1]}",
    f"manual_worlds={len(worlds)}",
]))
