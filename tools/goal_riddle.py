"""
ЦЕЛЬ ХОЗЯИНА, ЗАФИКСИРОВАННАЯ КАК ПРОГОН (красный тест вместо рассуждений).

Хозяин сформулировал конечный критерий так:

  «надо сделать так, чтобы наше чудище внесло в базу данных новые сущности,
   если они неизвестны — муж, невестка, жена, быть в гостях, портить другому
   человеку нервы, жаловаться, настроение и так далее. Потом установить связи,
   синапсы. И чтобы симулятор на основании этой базы данных мог ответить на
   вопрос: кем друг другу приходятся невестка Зинаиды и жена Егора.
   Правильный ответ — это одно лицо».

Значит проверяются ЧЕТЫРЕ звена, и прогон обязан сказать, КОГДЕ обрыв, а не
просто «не вышло»:

  ЗВЕНО 1  Сущности.  Все значимые слова рассказа стали узлами мира.
  ЗВЕНО 2  Синапсы.   Между ними есть связи, в т.ч. реляционные (невестка —
                      жена сына), а не только «невестка IS_A родственники».
  ЗВЕНО 3  Вывод.     Из связок выводится, что «невестка Зинаиды» и «жена
                      Егора» — один участник рассказа.
  ЗВЕНО 4  Ответ.     Симулятор/мозг отвечает на вопрос текстом, в котором
                      есть «одно лицо / один и тот же человек».

ЗВЕНО 3 и 4 сегодня нечем мерить — ни COREF, ни вопроса «кем приходятся» в
коде нет. Прогон НЕ делает вид, что они есть: он печатает «ПРИБОРА НЕТ» и
красный итог. Это и есть заказ на починку, выраженный числом, а не мнением.

УЧИТЕЛЬ ЖИВОЙ (localhost:8080). Бюджет вызовов здесь задаёт ПРОГОН, а не
молча меняет дефолт мозга: в рассказе больше десяти неизвестных слов, а
заводской бюджет = 3, из-за чего он кончался на именах «егор, жена, зинаида»
и до «невестка» не доходил НИ РАЗУ (замер пилота). Число — параметр хозяина;
здесь оно названо явно и видно в выводе.

ИЗОЛЯЦИЯ (урок 52): своя папка, живой lipid_world.json сверяется по хэшу до
и после; расхождение = красный выход.

Запуск:
  LIPID2_MEMORY_DIR=/tmp/goal .venv/bin/python tools/goal_riddle.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MEM = os.environ.get("LIPID2_MEMORY_DIR")
if not MEM:
    print("нужен LIPID2_MEMORY_DIR: прогон не должен трогать рабочую память")
    raise SystemExit(2)

REAL = os.path.join(ROOT, "data", "memory", "lipid_world.json")


def digest(path):
    if not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


real_before = digest(REAL)
os.environ["LIPID2_STATUS"] = "0"

from brain.lipid2_brain import Lipid2Brain        # noqa: E402
from core.logchannel import ask_handler           # noqa: E402

ask_handler(None)

# ЧИСТОВИК хозяина (29.01): без «поговорить/поговорит»-ошибки, иначе CAN_DO
# ссылка не строится и ЗВЕНО 2 краснеет по вине текста, а не системы.
STORY = (
    "Зинаида гостит в семье своего сына Егора. Она недолюбливает невестку и "
    "постоянно к ней придирается. Жена Егора пожаловалась мужу и попросила "
    "его поговорить с матерью, чтобы та не портила ей настроение. Егор любит "
    "жену, но боится мамы. Он пообещал жене, что поговорит с мамой, но "
    "обещание своё не выполнил. Жена затаила обиду."
)

# Что хозяин назвал сам. Скрипт ищет НЕ подстроку в каталоге, а узел мира:
# «быть в гостях» — это глагол «гостить», «портить нервы» — «портить».
WANTED = {
    "жена": ["жена"],
    "муж": ["муж"],
    "сын": ["сын"],
    "невестка": ["невестка"],
    "мать/мама": ["мать", "мама"],
    "быть в гостях": ["гостить"],
    "портить настроение": ["портить"],
    "настроение": ["настроение"],
    "жаловаться": ["пожаловаться"],
    "придираться": ["придираться"],
    "недолюбливать": ["недолюбливать"],
    "обещание": ["обещание", "пообещать"],
    "обида": ["обида", "затаить"],
    "любить": ["любить"],
    "бояться": ["бояться"],
}

TEACHER_BUDGET = int(os.environ.get("BUDGET", "12"))

fail = []


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ══════════════════════════════════════════════════════════════════════
hr("ЗВЕНО 0: КОНТРОЛЬ ЖИВОСТИ (урок 54) — известное обязано прочитаться")
# ══════════════════════════════════════════════════════════════════════
CTL = os.path.join(MEM, "_ctl")
os.makedirs(CTL, exist_ok=True)
ctl = Lipid2Brain(memory_dir=CTL)
ctl.learn_from_text("Собака гасит пламя. Кошка спит.", source="ctl",
                    allow_teacher=False)
got = sorted(ctl._load_world().get("entities", {}))
print(f"  контрольный урок дал узлов: {got}")
if not got:
    fail.append("КОНТРОЛЬ: контрольный урок не создал НИ ОДНОГО узла — "
                "дальнейшие нули значат «прибор сломан», а не «система глупая»")

# ══════════════════════════════════════════════════════════════════════
hr(f"ЗВЕНО 1: СУЩНОСТИ — обучение рассказу (бюджет учителя {TEACHER_BUDGET})")
# ══════════════════════════════════════════════════════════════════════
RUN = os.path.join(MEM, "run")
if os.path.isdir(RUN):
    shutil.rmtree(RUN)
os.makedirs(RUN)
b = Lipid2Brain(memory_dir=RUN)
b.use_llm_teacher = True
b.use_wiki_teacher = True          # канал 2 включён; ПОРЯДОК остаётся заводской
b.ask_user_enabled = False
# Бюджет — параметр хозяина. learn_from_text его НЕ пробрасывает (замер:
# сигнатура = text/source/allow_teacher), поэтому он ставится в ТОМ месте,
# где он живёт — Learner.max_teacher_calls. Прежняя версия этого прогона
# делала вид, что передаёт бюджет, и молча меряла заводскую тройку: прогон
# за 4 минуты вместо получаса выглядел ПОДОЗРИТЕЛЬНО быстро, и это был
# единственный признак подмены. Теперь число сверяется ДО обучения и после.
b.learner.max_teacher_calls = TEACHER_BUDGET
print(f"  бюджет учителя: выставлен {b.learner.max_teacher_calls} "
      f"(заводской 3), каналы: llm={b.use_llm_teacher} "
      f"wiki={b.use_wiki_teacher} user={b.ask_user_enabled}")
rep = b.learn_from_text(STORY, source="owner_goal", allow_teacher=True)
if rep.teacher_calls > TEACHER_BUDGET:
    fail.append(f"бюджет нарушен: вызовов {rep.teacher_calls} при лимите "
                f"{TEACHER_BUDGET}")
for k in ("sentences", "content_words", "known_words", "unknown_words", "facts",
          "entities_created", "observations_new", "teacher_calls",
          "teacher_success"):
    print(f"    {k:18} = {getattr(rep, k, '?')}")
print(f"    каналы: {getattr(rep,'channel_calls',{})} "
      f"найдено {getattr(rep,'channel_found',{})}")

w = json.load(open(os.path.join(RUN, "lipid_world.json")))
ents = w.get("entities", {})
obs = [(v.get("head"), v.get("relation"), v.get("tail"))
       for v in w.get("observations", {}).values() if isinstance(v, dict) and v.get("head")]
print(f"\n  узлов мира: {len(ents)}\n    {sorted(ents)}")

hr("ЗВЕНО 1: ПРОВЕРКА ПО СПИСКУ ХОЗЯИНА")
miss1 = []
for label, forms in WANTED.items():
    hit = [f for f in forms if f in ents]
    print(f"  {label:20} {'ЕСТЬ ' + str(hit) if hit else 'НЕТ'}")
    if not hit:
        miss1.append(label)
print(f"\n  нет в мире: {len(miss1)} из {len(WANTED)} — {miss1}")
if miss1:
    fail.append(f"ЗВЕНО 1 (сущности): в мире нет {len(miss1)} слов хозяина: {miss1}")

# ══════════════════════════════════════════════════════════════════════
hr("ЗВЕНО 2: СИНАПСЫ — есть ли СВЯЗЬ, а не только узел")
# ══════════════════════════════════════════════════════════════════════
print(f"  наблюдений (рёбер): {len(obs)}")
for h, r, t in obs:
    print(f"     {h} --{r}--> {t}")
n_edges = 0
need_edges = [("жена", "IS_ROLE", "муж"), ("жена", "IS_ROLE", "сын"),
              ("невестка", "IS_ROLE", "сын"), ("невестка", "IS_ROLE", "мать"),
              ("жена", "HAS_PROPERTY", "настроение")]
print("\n  реляционные связи, которых ждём (узел + отношение к кому):")
for h, r, t in need_edges:
    ok = (h, r, t) in obs or any(
        h in str(ents.get(h, {}).get("relations", "")) for _ in [0])
    have_rel = any(hh == h and tt == t for hh, _rr, tt in obs)
    print(f"     {h:10} --{r}--> {t:12} "
          f"{'ЕСТЬ' if ok else ('есть другое отношение' if have_rel else 'НЕТ')}")
    if not ok:
        n_edges += 1
print(f"\n  не хватает {n_edges} реляционных связей из {len(need_edges)}")
if n_edges:
    fail.append(f"ЗВЕНО 2 (синапсы): нет {n_edges} реляционных связей "
                "(отношение «жена сына» в мир не записано)")

# ══════════════════════════════════════════════════════════════════════
hr("ЗВЕНО 3: ВЫВОД «одно лицо» — есть ли вообще такой прибор")
# ══════════════════════════════════════════════════════════════════════
probe = None
for name in ("resolve_mentions", "resolve_coreference", "same_person",
             "who_is_to_whom", "relation_between"):
    if hasattr(b, name):
        probe = name
        break
from brain import simulator as sim_mod                    # noqa: E402
from brain.social_sim import SocialSimulator             # noqa: E402
sim_names = [n for n in dir(SocialSimulator) + dir(sim_mod.Simulator)
             if any(k in n.lower() for k in ("relativ", "coref", "kem", "who"))]
print(f"  метод мозга для «кем приходятся»: {probe or 'НЕТ'}")
print(f"  методы симулятора по теме связей: {sim_names or 'НЕТ'}")
if not probe:
    fail.append("ЗВЕНО 3 (вывод): прибора нет — мозг не умеет ни связывать "
                "упоминания, ни спрашивать «кем X приходится Y»")

# ══════════════════════════════════════════════════════════════════════
hr("ЗВЕНО 4: ОТВЕТ НА ВОПРОС ХОЗЯИНА")
# ══════════════════════════════════════════════════════════════════════
Q = "Кем друг другу приходятся невестка Зинаиды и жена Егора?"
ans = b.process_input(Q)
print(f"  Q: {Q}")
print(f"  A: {ans[:220]}")
low = ans.lower()
good = any(k in low for k in ("одно лицо", "один и тот же", "это она",
                              "одно и то же", "одна и та же"))
print(f"\n  в ответе есть «одно лицо» и т.п.: {'ДА' if good else 'НЕТ'}")
if not good:
    fail.append(f"ЗВЕНО 4 (ответ): мозг ответил не про одно лицо: {ans[:110]!r}")

# ══════════════════════════════════════════════════════════════════════
hr("ЗНАЮЩИЙ СВИДЕТЕЛЬ (замерка, в мир НИЧЕГО не пишет)")
# ══════════════════════════════════════════════════════════════════════
# Локалка в свободном чате загадку РЕШАЛА (замер 30.01, 16.1 с: «это один и тот
# же человек»). Она здесь — эталон верхней границы: если свидетель прав, а
# чудище молчит, виноват конвейер, а не модель.
try:
    import urllib.request
    p = {"messages": [{"role": "user", "content":
                       "Зинаида гостит в семье своего сына Егора. Жена Егора "
                       "пожаловалась мужу. Кем приходится невестка Зинаиде и "
                       "это же человек, что жена Егора, или разные? Кратко."}],
           "temperature": 0.1, "max_tokens": 900, "reasoning_effort": "low"}
    t0 = time.time()
    d = json.load(urllib.request.urlopen(
        os.environ.get("LIPID2_LLM_HOST", "http://localhost:8080")
        + "/v1/chat/completions", data=json.dumps(p).encode(), timeout=240))
    wit = (d["choices"][0]["message"].get("content") or "").replace("\n", " ")
    print(f"  свидетель ({time.time()-t0:.1f} с): {wit[:200]}")
    print(f"  свидетель говорит «одно лицо»: "
          f"{'ДА' if any(k in wit.lower() for k in ('один тот же','тот же','одно лицо','один и тот же')) else 'НЕТ'}")
except Exception as exc:
    print(f"  свидетель недоступен: {type(exc).__name__}: {exc}")

hr("ИТОГ")
real_after = digest(REAL)
print(f"  живой мир: {real_before} -> {real_after} "
      f"{'НЕ ТРОНУТ' if real_before == real_after else 'ИЗМЕНЁН!!!'}")
if real_before != real_after:
    fail.append("ИЗОЛЯЦИЯ: прогон изменил живой lipid_world.json")
if fail:
    print("\nКРАСНЫЙ — чинить по звеньям:")
    for f in fail:
        print("  ✗ " + f)
    raise SystemExit(1)
print("\nЗЕЛЁНЫЙ: цель хозяина достигнута")
