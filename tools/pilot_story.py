"""
ПИЛОТ 0: короткий сюжет с семантической загадкой -> обучение -> вопросы.

ЗАЧЕМ. Хозяин дал текст, в котором таится загадка: «невестка Зинаиды» и
«жена Егора» — одно лицо, а в тексте это ни сказано, ни записано. Прежде чем
чинить механизм, надо ИЗМЕРИТЬ, что из этого текста доходит до мира и что мозг
умеет ответить. Пилот НИЧЕГО не чинит и НИЧЕГО не подсказывает: никаких правил
под этот текст не заводилось (требование хозяина).

ЦЕНЫ, которых здесь нет ни в одном другом инструменте:
  * цена ОТВЕТА ЖИВОГО Учителя (один концепт, замером). yesterday's
    tools/measure_long_session.py мерил с ВЫКЛЮЧЕННЫМ каналом и давал нижнюю
    границу; цена живого канала в проекте не была измерена ВООБЩЕ.
  * сколько фраз сюжета дошло до мира и сколько слов осталось неизвестными
    при ЗАВОДСКОМ бюджете Учителя (max_teacher_calls = 3).

КОНТРОЛЬ ЖИВОСТИ (урок 54). Пилот обязан показать известное ДО того, как ему
верить: заведомо простая фраза проверяется на «факты есть» и на «ответ на
«что делает X» берётся из памяти». Не прошёл контроль — вывод ниже не замер.

КОНТРОЛЬ ИЗОЛЯЦИИ (урок 52). Живой каталог проверяется по хэшу ДО и ПОСЛЕ;
при расхождении — красный выход, а не предупреждение.

Запуск:
  LIPID2_MEMORY_DIR=/tmp/pilot0 .venv/bin/python tools/pilot_story.py
БЕЗ LIPID2_MEMORY_DIR не запускается: рабочая память Вселенной не двигается.
"""
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MEM = os.environ.get("LIPID2_MEMORY_DIR")
if not MEM:
    print("нужен LIPID2_MEMORY_DIR: пилот не должен трогать рабочую память")
    raise SystemExit(2)

REAL = os.path.join(ROOT, "data", "memory", "lipid_world.json")


def digest(path):
    if not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


real_before = digest(REAL)

# ── МАТЕРИАЛ ХОЗЯИНА (чистовик, правлены только ошибки хозяина: «поговорит»,
#    «боится мамы»; двойные пробелы склеены — см. замеры в памяти 02.11) ──
STORY = (
    "Зинаида гостит в семье своего сына Егора. Она недолюбливает невестку и "
    "постоянно к ней придирается. Жена Егора пожаловалась мужу и попросила "
    "его поговорить с матерью, чтобы та не портила ей настроение. Егор любит "
    "жену, но боится мамы. Он пообещал жене, что поговорит с мамой, но "
    "обещание своё не выполнил. Жена затаила обиду."
)

# ВОПРОСЫ. Ни один не содержит ответа и ни один не назван в тексте как
# правило: это то, что должен СОБРАТЬ мозг.
QUESTIONS = [
    "Кто такая невестка?",
    "Кто такая жена?",
    "Что делает Зинаида?",
    "Что делает Егор?",
    "Кто такая мама?",
]

from brain.lipid2_brain import Lipid2Brain          # noqa: E402
from core.logchannel import ask_handler             # noqa: E402

fail = []


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# Канал 3 в headless отключён: спрашивать некого, и обработчик не ставится.
ask_handler(None)

# ══════════════════════════════════════════════════════════════════════
hr("КОНТРОЛЬ ЖИВОСТИ ПРИБОРА (доверять можно только после зелёного контроля)")
# ══════════════════════════════════════════════════════════════════════
CTL_DIR = "/tmp/pilot0_ctl"
os.makedirs(CTL_DIR, exist_ok=True)
ctl = Lipid2Brain(memory_dir=CTL_DIR)
ctl.use_wiki_teacher = False        # контроль не должен зависеть от интернета
ctl.ask_user_enabled = False
t0 = time.time()
crep = ctl.learn_from_text("Собака охраняет дом.", source="control",
                           allow_teacher=True)
ctl_t = time.time() - t0
print(f"  факты: {crep.facts}  сущностей создано: {crep.entities_created}  "
      f"наблюдений: {crep.observations_new}  ({ctl_t:.1f} с)")
if crep.facts < 1 or crep.observations_new < 1:
    fail.append(f"КОНТРОЛЬ: контрольная фраза не дала фактов ({crep.facts})")
ctl.use_llm_teacher = False         # ответ обязан читаться из памяти
# СНИМОК ДО ВОПРОСА. channel_calls обнуляются на НАЧАЛЕ каждого сеанса
# обучения, поэтому проверять их абсолютным числом нельзя: после урока там
# лежат вызовы ОБУЧЕНИЯ, и контроль краснеет на том, чего не проверял. Меряется
# ДЕЛЬТА вокруг самого вопроса. (Своя ошибка первого прогона: контроль был
# красный при работающем read-путь — «прибор молчит» принято за «прибор сломан".)
before = dict(ctl.channel_calls)
ans = ctl._handle_actions("собака")
delta = {k: ctl.channel_calls.get(k, 0) - before.get(k, 0)
         for k in ctl.channel_calls}
print(f"  ответ «что делает собака»: {ans[:90]}")
print(f"  вызовы каналов ВОКРУГ вопроса: {delta}")
if "не знаю" in ans.lower():
    fail.append("КОНТРОЛЬ: путь чтения не видит выученное знание")
if any(v for v in delta.values()):
    fail.append(f"КОНТРОЛЬ: ответ на вопрос дёрнул канал при выключенном LLM: {delta}")

# ══════════════════════════════════════════════════════════════════════
hr("ЦЕНА ЖИВОГО УЧИТЕЛЯ (не измерена нигде; концепт ВНЕ сюжета, чтобы не "
   "выучить слово загадки до урока)")
# ══════════════════════════════════════════════════════════════════════
PRICE_DIR = "/tmp/pilot0_price"
os.makedirs(PRICE_DIR, exist_ok=True)
pb = Lipid2Brain(memory_dir=PRICE_DIR)
pb.use_wiki_teacher = False
pb.ask_user_enabled = False
t0 = time.time()
try:
    got = pb.teacher_llm.run_agent(base_concept="кенгуру",
                                   derived_concept="кенгуру",
                                   filename="lipid_world.json")
    price = time.time() - t0
    print(f"  ОДИН ответ локальной модели про «кенгуру»: {price:.1f} с; "
          f"знания получено: {'ДА' if got else 'НЕТ'}")
    if not got:
        f = getattr(pb.teacher_llm, "last_failure", None)
        fail.append(f"ЦЕНА: Учитель не ответил ({f}) — цену считать не по чему")
except Exception as exc:
    price = None
    fail.append(f"ЦЕНА: канал 1 недоступен ({type(exc).__name__}: {exc})")
    print(f"  канал 1 НЕДОСТУПЕН: {type(exc).__name__}: {exc}")

# ══════════════════════════════════════════════════════════════════════
hr("УРОК: сюжет хозяина (полный мозг, каналы по умолчанию, правил не добавлял)")
# ══════════════════════════════════════════════════════════════════════
MAIN_DIR = MEM
os.makedirs(MAIN_DIR, exist_ok=True)
brain = Lipid2Brain(memory_dir=MAIN_DIR)
t0 = time.time()
rep = brain.learn_from_text(STORY, source="owner_story")
lesson_t = time.time() - t0

hr(f"ОТЧЁТ ({lesson_t:.1f} с)")
for k in ("sentences", "content_words", "known_words", "unknown_words",
          "facts", "entities_created", "entities_updated", "observations_new",
          "events_new", "expectations_new", "aliases_new", "aliases_written",
          "skipped", "low_confidence", "sentences_silent", "teacher_calls",
          "teacher_success", "teacher_budget", "teacher_skipped_by_budget"):
    if hasattr(rep, k):
        print(f"  {k:26} = {getattr(rep, k)}")
print(f"  каналы: вызовы {getattr(rep,'channel_calls',{})} "
      f"найдено {getattr(rep,'channel_found',{})}")
print(f"  новые концепты: {getattr(rep,'new_concepts',[])}")
print(f"  выученные действия: {getattr(rep,'learned_actions',[])}")
print(f"  НЕ.unknown: {getattr(rep,'unknown',[])}")
if getattr(rep, "parse_notes", None):
    print("  заметки разбора:")
    for n in rep.parse_notes:
        print(f"     - {n}")
if getattr(rep, "notes", None):
    print("  notes:")
    for n in rep.notes:
        print(f"     - {n}")

hr("ЧТО ОСЕЛО В МИРЕ (read-only)")
world = json.load(open(os.path.join(MAIN_DIR, "lipid_world.json")))
ents = world.get("entities", {})
obs = world.get("observations", {})
unk = world.get("unknown_words", {})
print(f"  сущностей {len(ents)}: {sorted(ents)}")
# В observations ДВА РАЗНЫХ сорта записей: связи (head/relation/tail) и
# записи любопытства («егор|curiosity», у них этих полей НЕТ). Первый прогон
# печатал все одним циклом и выдал три строки «None --None--> None» — мёртвый
# вывод, в котором настоящая связь тонула (тот же класс, что урок 54).
links = {k: v for k, v in obs.items() if isinstance(v, dict) and v.get("head")}
curios = {k: v for k, v in obs.items() if k not in links}
print(f"  связей-наблюдений {len(links)} (плюс {len(curios)} записей любопытства):")
for k in sorted(links, key=lambda x: links[x].get("head", "")):
    v = links[k]
    print(f"     {v.get('head')} --{v.get('relation')}--> {v.get('tail')}"
          f"   [{v.get('source')}]")
print(f"  любопытство (без связи): {sorted(k.split('|')[0] for k in curios)}")

# ПРОВЕРКА СЛОВА ИЗ ТЕКСТА. Слово, которое было в уроке, обязано остаться
# видимым: либо узлом, либо в unknown_words, либо в связи. Молча исчезнувшее
# слово = схлопывание с другим узлом (замер 02.11: «семья» пропала, связь
# уехала на свойство «семейная»). Проверка НЕ знает ответа заранее: она
# перебирает существительные сюжета и сверяет их с миром.
import pymorphy3                                       # noqa: E402
_m = pymorphy3.MorphAnalyzer()
nouns = []
for w in STORY.replace(",", " ").split():
    w = w.strip(".?!«»()").lower()
    if len(w) < 4 or w in nouns:
        continue
    t = _m.parse(w)[0].tag
    if str(t.POS) == "NOUN":
        nouns.append(_m.parse(w)[0].normal_form)
visible = set(ents) | set(unk)
for k, v in links.items():
    visible |= {links[k].get("head"), links[k].get("tail")}
missing = [n for n in nouns if n not in visible]
print(f"\n  существительных в сюжете: {len(set(nouns))}; "
      f"НЕВИДИМЫХ в мире: {missing or 'нет'}")
if missing:
    for n in missing:
        # чем именно заменилось: тот же резолвер, что в бою
        hit = brain.lemmas_world.resolve_name(n)
        print(f"     ✗ «{n}» исчезло; резолвер Вселенной ведёт его в "
              f"{hit!r}")
    fail.append(f"СЛОВО ИЗ ТЕКСТА ПРОПАЛО МОЛЧА: {missing}")
print(f"  неизвестных слов {len(unk)}:")
for w, v in sorted(unk.items()):
    st = {k: val for k, val in (v.items() if isinstance(v, dict) else [])
          if k in ("status", "asked_empty", "reason", "provenance")}
    print(f"     {w:14} {st if st else v}")

hr("ВОПРОСЫ К МОЗГУ (только чтение; LLM-канал на время вопросов ВЫКЛЮЧЕН, "
   "чтобы ответ нельзя было получить из модели)")
brain.use_llm_teacher = False
brain.ask_user_enabled = False
for q in QUESTIONS:
    print(f"\n  В: {q}")
    name = q.replace("Кто такая ", "").replace("Что делает ", "").replace("?", "").strip()
    try:
        if q.startswith("Что делает"):
            print(f"  О: {brain._handle_actions(name)[:200]}")
        else:
            data = brain.find_in_memory(name)
            print(f"  О: {'НЕТ ЗАПИСИ' if not data else json.dumps(data, ensure_ascii=False)[:220]}")
    except Exception as exc:
        print(f"  О: ИСКЛЮЧЕНИЕ {type(exc).__name__}: {exc}")

# ══════════════════════════════════════════════════════════════════════
hr("ИТОГ")
# ══════════════════════════════════════════════════════════════════════
real_after = digest(REAL)
print(f"  живой мир: {real_before} -> {real_after}  "
      f"{'НЕ ТРОНУТ' if real_before == real_after else 'ИЗМЕНЁН!!!'}")
if real_before != real_after:
    fail.append("ИЗОЛЯЦИЯ: пилот изменил живой lipid_world.json")
if price:
    print(f"  цена одного ответа живого Учителя: {price:.1f} с")
print(f"  урок целиком: {lesson_t:.1f} с при "
      f"teacher_calls={rep.teacher_calls}/{rep.teacher_budget}")
if fail:
    print("\nКРАСНЫЙ:")
    for f in fail:
        print("  ✗ " + f)
    raise SystemExit(1)
print("\nЗЕЛЁНЫЙ: контроль живости пройден, изоляция цела")
