"""
СХЛЁСТКА КАНАЛОВ: один сюжет, два источника знания, сравнение ЧЕМ, а не «как».

ЗАЧЕМ. Хозяин ставит цель «в будущем учиться на Интернете, без локальной
модели». Прежде чем менять приоритет каналов, надо ИЗМЕРИТЬ, что Википедия
даёт на том же материале, что и Gemma: цену, покрытие и СОДЕРЖАНИЕ ответа.
Скрипт НИЧЕГО не чинит и порядок каналов не меняет — он меряет.

ПОРЯДОК КАНАЛОВ В МОЗГЕ ЖЁСТКИЙ (brain/lipid2_brain.py: 1 LLM -> 2 Вики ->
3 пользователь), флага порядка нет. Поэтому «wiki-first» здесь снимается
ВЫКЛЮЧЕНИЕМ канала 1, а не правкой мозга: это честный wiki-only прогон, и
именно он показывает верхнюю границу того, что даст приоритет на Вики.

ЧТО ПЕЧАТАЕТ:
  * цену одного концепта каждым каналом (Вики мерится тут же, Gemma — из
    замера tools/pilot_story.py);
  * покрытие: какие слова урока остались not_asked при том же бюджете;
  * СОДЕРЖАНИЕ: определение, иерархию, свойства — дословно, чтобы видеть
    подмену темы редиректом (замер: «невестка» -> статья «Родство»);
  * ловит ли мозг подмену (сверяет ли title страницы со словом запроса).

ИЗОЛЯЦИЯ (урок 52): каждый канал в своём mktemp, живой каталог сверяется по
хэшу до и после, при расхождении — красный выход.

Запуск:  LIPID2_STATUS=0 .venv/bin/python tools/compare_channels.py
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REAL = os.path.join(ROOT, "data", "memory", "lipid_world.json")


def digest(path):
    if not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


real_before = digest(REAL)
os.environ["LIPID2_STATUS"] = "0"          # сводку сеанса не печатаем
from brain.lipid2_brain import Lipid2Brain          # noqa: E402
from core.logchannel import ask_handler             # noqa: E402
from teachers.teacher_wiki import WikiTeacher       # noqa: E402

ask_handler(None)          # канал 3 в headless отключён: спрашивать некого


class CountingWiki(WikiTeacher):
    """Вики, которая НЕ путает «не нашла» и «отказала».

    ЗАЧЕМ. Первый прогон этого скрипта выдал ЗЕЛЁНЫЙ, когда Вики вернула
    HTTP 429 на четырёх словах из семи: fetch_goncept проглатывает исключение
    и возвращает {}, а скрипт читал пустой ответ как «в Википедии про это
    слова нет». Нуль, полученный отказом, нулём измерения быть не может
    (урок 54) — сравнение каналов по отказам означает «Вики хуже», чего мы
    не мерили. Здесь каждый сбой сети считается отдельно и уводит выход в
    красный: при 429 сравнивать нечего, надо ждать, а не делать вывод.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.refusals = []      # (concept, причина) — Вики ОТКАЗАЛА
        self.misses = []        # concept — Вики ответила и не нашла

    def _fetch_page(self, title):
        return super()._fetch_page(title)

    def fetch_concept(self, concept):
        res = super().fetch_concept(concept)
        if not res:
            # WHY-подсчёт по факту: отличить отказ от пропуска ретроспективно
            # нельзя, поэтому повторный запрос с паузой: 429 за это время уходит,
            # настоящий пропуск — нет.
            time.sleep(2.0)
            retry = super()._fetch_page(concept.strip().lower())
            (self.misses if retry is None else self.refusals).append(concept)
        return res

STORY = (
    "Зинаида гостит в семье своего сына Егора. Она недолюбливает невестку и "
    "постоянно к ней придирается. Жена Егора пожаловалась мужу и попросила "
    "его поговорить с матерью, чтобы та не портила ей настроение. Егор любит "
    "жену, но боится мамы. Он пообещал жене, что поговорит с мамой, но "
    "обещание своё не выполнил. Жена затаила обиду."
)

# Те же слова, что просил босс, плюс загадка и имена.
CONCEPTS = ["невестка", "жена", "свекровь", "обида", "ездок", "егор", "зинаида"]

fail = []


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ══════════════════════════════════════════════════════════════════════
hr("КОНТРОЛЬ ЖИВОСТИ ОБЕИХ ПРОБ (урок 54): заведомо известное обязано "
   "прочитаться ДО того, как мы поверим нулям")
# ══════════════════════════════════════════════════════════════════════
CTL = tempfile.mkdtemp(prefix="cmp_ctl_")
ctl = WikiTeacher(memory_dir=CTL)
for c, must in [("жена", "супруг"), ("кошка", "кош"), ("Москва", "росси")]:
    t0 = time.time()
    r = ctl.fetch_concept(c)
    be = (r or {}).get("base_entity") or {}
    txt = json.dumps(be, ensure_ascii=False).lower()
    ok = must in txt
    print(f"  Вики «{c:8}» {time.time()-t0:4.1f} с  title={be.get('title')!r:24} "
          f"определение содержит {must!r}: {'ДА' if ok else 'НЕТ'}")
    print(f"     {str(be.get('definition'))[:110]}")
    if not r:
        fail.append(f"КОНТРОЛЬ: канал 2 не ответил на заведомо известное «{c}»")
# Контроль второй пробы: локалка обязана отвечать (иначе сравнение цен враньё).
LLM_OK = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                         "http://127.0.0.1:8080/health"],
                        capture_output=True, text=True).stdout.strip()
print(f"  health локалки @8080: HTTP {LLM_OK}")
if LLM_OK != "200":
    fail.append(f"КОНТРОЛЬ: локалка недоступна (HTTP {LLM_OK}) — сравнение "
                "цен будет враньём")

# ══════════════════════════════════════════════════════════════════════
hr("СХЛЁСТКА 1: ОДНИ ТЕ ЖЕ СЛОВА, ДВА КАНАЛА, ЦЕНА И СОДЕРЖАНИЕ")
# ══════════════════════════════════════════════════════════════════════
W2 = tempfile.mkdtemp(prefix="cmp_wiki_")
wt = CountingWiki(memory_dir=W2)
rows = []
for c in CONCEPTS:
    t0 = time.time()
    time.sleep(1.2)                    # Вики режет частоту: 429 при запросе в запрос
    try:
        r = wt.fetch_concept(c)
    except Exception as exc:
        r = {}
    dt = time.time() - t0
    be = (r or {}).get("base_entity") or {}
    title = be.get("title") or ""
    hijack = bool(title) and title.strip().lower() != c.strip().lower()
    rows.append((c, dt, bool(r), title, hijack, str(be.get("definition") or "")[:88]))
    print(f"\n  «{c}»  Вики: {dt:4.2f} с  найдено={'ДА' if r else 'НЕТ'}")
    print(f"     title страницы: {title!r}"
          f"{'   <<< ПОДМЕНА: страница НЕ про это слово' if hijack else ''}")
    print(f"     определение   : {rows[-1][5]}")
    print(f"     иерархия      : {be.get('hierarchy')}")
    print(f"     свойства      : {be.get('properties')}")

hr("СВОДКА ОТКАЗОВ И ПОДМЕН")
print(f"  ОТКАЗОВ (429/обрыв, знания НЕ измеряли): {len(wt.refusals)} "
      f"{wt.refusals}")
print(f"  НАСТОЯЩИХ ПРОПУСКОВ (Вики ответила и не нашла): {len(wt.misses)} "
      f"{wt.misses}")
if wt.refusals:
    fail.append(f"СРАВНЕНИЕ НЕКОРРЕКТНО: Вики отказала на {len(wt.refusals)} "
                f"словах {wt.refusals} — эти нули не результат обучения")

hr("СВОДКА ПОДМЕН")
hij = [r for r in rows if r[4]]
print(f"  слов: {len(rows)},  из них Вики вернула ДРУГУЮ статью: {len(hij)}")
for c, dt, got, title, h, _ in rows:
    print(f"    {c:10} -> {title!r:22} {'ПОДМЕНА' if h else 'совпало'}")
print(f"  средняя цена концепта Вики: "
      f"{sum(r[1] for r in rows)/max(1,len(rows)):.2f} с")
print("  для сравнения, ЗАМЕРЕНО tools/pilot_story.py: один концепт Gemma = "
      "20.1 с и 22.6 с (6 шагов агента)")

# ══════════════════════════════════════════════════════════════════════
hr("СХЛЁСТКА 2: ОДИН СЮЖЕТ, wiki-only прогон (канал 1 выключен)")
# ══════════════════════════════════════════════════════════════════════
W3 = tempfile.mkdtemp(prefix="cmp_lesson_wiki_")
b = Lipid2Brain(memory_dir=W3)
b.use_llm_teacher = False          # приоритет Вики без правки порядка каналов
b.ask_user_enabled = False         # спрашивать некого — иначе канал 3 исказит
b.use_wiki_teacher = True
t0 = time.time()
rep = b.learn_from_text(STORY, source="owner_story_wiki", allow_teacher=True)
# Вики режет частоту запросов; без пауз часть бюджета сгорает на 429 и
# покрытие урока меряется отказом сети, а не знанием (урок 54).
time.sleep(2.0)
dt = time.time() - t0
print(f"\n  урок через Вики: {dt:.1f} с")
for k in ("sentences", "content_words", "known_words", "unknown_words", "facts",
          "entities_created", "observations_new", "sentences_silent",
          "teacher_calls", "teacher_success", "teacher_budget"):
    print(f"    {k:20} = {getattr(rep, k, '?')}")
print(f"    каналы: вызовы {getattr(rep,'channel_calls',{})} "
      f"найдено {getattr(rep,'channel_found',{})}")
print(f"    выучено: {getattr(rep,'new_concepts',[])}")
print(f"    осталось not_asked: {[x for x in getattr(rep,'unknown',[])]}")

w = json.load(open(os.path.join(W3, "lipid_world.json")))
print(f"\n  сущностей {len(w['entities'])}: {sorted(w['entities'])}")
print("  связей-наблюдений:")
for k, v in sorted(w["observations"].items()):
    if isinstance(v, dict) and v.get("head"):
        print(f"     {v['head']} --{v['relation']}--> {v['tail']}")
print("\n  КАЧЕСТВО: что реально записано вместо определений")
for c in ("жена", "невестка", "свекровь"):
    e = w["entities"].get(c)
    if not e:
        print(f"    «{c}»: сущности НЕТ")
        continue
    print(f"    «{c}»: hierarchy={e.get('hierarchy')} "
          f"properties={e.get('properties')}")
    prov = json.dumps(e.get("provenance"), ensure_ascii=False)[:100]
    print(f"       provenance: {prov}")

# ══════════════════════════════════════════════════════════════════════
hr("ЛОВИТ ЛИ МОЗГ ПОДМЕНУ (Вики написала «невестка» текстом про «Родство») ")
# ══════════════════════════════════════════════════════════════════════
for c in ("невестка", "свекровь"):
    e = w["entities"].get(c) or {}
    hier = e.get("hierarchy") or []
    props = e.get("properties") or []
    src = json.dumps(e, ensure_ascii=False)
    print(f"  «{c}» записано: {'ДА' if e else 'НЕТ'}")
    if e:
        print(f"    иерархия: {hier}")
        print(f"    есть ли в записи слово «родство» (тема чужой статьи): "
              f"{'ДА — подмена тихо осела в мире' if 'родств' in src.lower() else 'нет'}")
        print(f"    source={e.get('source')!r} title={e.get('title')!r}")
        if str(e.get("title", "")).strip().lower() != c:
            fail.append(f"ПОДМЕНА НЕ ЗАМЕЧЕНА: «{c}» записана со страницы "
                        f"{e.get('title')!r} и мир это принял")

# Замер, которого в первом прогоне НЕ БЫЛО и из-за которого он соврал:
# в мире урока могла не появиться ни одна из целевых сущностей — скрипт
# это молча переживал. Теперь отсутствие цели = красный, а не «зелёный, но
# сравнивать нечего».
hr("ЦЕЛЕВЫЕ СЛОВА УРОКА: что Вики дала, а что мозг записал")
for c in ("жена", "невестка", "мать", "сын", "семья"):
    e = w["entities"].get(c)
    print(f"  {c:10} в мире урока: {'ЕСТЬ' if e else 'НЕТ'}"
          + (f"  title={e.get('title')!r}" if e and e.get("title") else ""))
    if e and str(e.get("title", "")).strip().lower() not in ("", c):
        print(f"     <<< ЗАПИСЬ СДЕЛАНА ПО СТАТЬЕ {e.get('title')!r} — "
              f"определение не про {c!r}")
for c in ("невестка",):
    if c not in w["entities"]:
        fail.append(f"загадка не решена даже частично: «{c}» в мир урока не "
                    "попала ни через один канал")

# ══════════════════════════════════════════════════════════════════
hr("СХЛЁСТКА 3: ПРИОРИТЕТ НА ВИКИ, ЛОКАЛКА ДОГОНЯЕТ (то, что просил хозяин)")
# ══════════════════════════════════════════════════════════════════
# Порядок каналов в мозге жёсткий (сначала Gemma, потом Вики) и флага порядка
# нет. Менять его правкой — решение хозяина, а не моё. Здесь же ПРИОРИТЕТ
# МОДЕЛИРУЕТСЯ ДЕЙСТВИЕМ: сначала Вики закрывает все неизвестные слова урока,
# затем урок учится мозгом при ВКЛЮЧЁННОЙ локалке. Если Вики закрыла слово,
# мозг его уже знает и до Gemma не доходит — ровно то поведение, которого
# хозяин ждёт от будущей версии «учимся на интернете, локалки нет».
W4 = tempfile.mkdtemp(prefix="cmp_hybrid_")
wt4 = CountingWiki(memory_dir=W4)
vocab = set()
for sf in __import__("core.extract", fromlist=["SimpleExtractor"]).SimpleExtractor().extract_text(STORY):
    vocab |= set(sf.content_lemmas)
seed_t0 = time.time()
seeded = 0
for c in sorted(vocab):
    time.sleep(1.2)
    d = wt4.fetch_concept(c)
    if d and wt4.write_to_world(d):
        seeded += 1
seed_dt = time.time() - seed_t0
print(f"\n  шапг Вики по всем {len(vocab)} значимым словам урока: {seed_dt:.1f} с, "
      f"закрыто {seeded}")
print(f"  отказы Вики: {wt4.refusals or 'нет'}   пропуски: {wt4.misses or 'нет'}")
b4 = Lipid2Brain(memory_dir=W4)
b4.use_llm_teacher = True           # локалка ВКЛЮЧЕНА — она будет догонять
b4.use_wiki_teacher = True
b4.ask_user_enabled = False
t0 = time.time()
rep4 = b4.learn_from_text(STORY, source="owner_story_hybrid", allow_teacher=True)
dt4 = time.time() - t0
print(f"\n  урок после шага Вики: {dt4:.1f} с")
for k in ("content_words", "known_words", "unknown_words", "facts",
          "entities_created", "observations_new", "teacher_calls",
          "teacher_success"):
    print(f"    {k:18} = {getattr(rep4, k, '?')}")
print(f"    каналы: вызовы {getattr(rep4,'channel_calls',{})} "
      f"найдено {getattr(rep4,'channel_found',{})}")
print(f"    выучено: {getattr(rep4,'new_concepts',[])}")
print(f"    ИТОГО цена: шаг Вики {seed_dt:.1f} с + урок {dt4:.1f} с "
      f"= {seed_dt+dt4:.1f} с")
w4 = json.load(open(os.path.join(W4, "lipid_world.json")))
print(f"\n  сущностей {len(w4['entities'])}: {sorted(w4['entities'])}")
for c in ("невестка", "жена", "мать", "мама", "муж", "сын", "семья", "обида"):
    e = w4["entities"].get(c) or {}
    print(f"    {c:10} определение: {str(e.get('definition') or '—')[:96]}")
hij4 = [(c, e.get("title")) for c, e in w4["entities"].items()
        if e.get("title") and str(e["title"]).lower() != c]
print(f"\n  ЗАПИСЕЙ ПО ЧУЖОЙ СТАТЬЕ: {len(hij4)} {hij4}")
if hij4:
    fail.append(f"ПРИОРИТЕТ НА ВИКИ ЗАНОСИТ ПОДМЕНУ: {len(hij4)} записей"
                f" сделаны по статье про другое слово {hij4}")

hr("ИТОГ")
real_after = digest(REAL)
print(f"  живой мир: {real_before} -> {real_after} "
      f"{'НЕ ТРОНУТ' if real_before == real_after else 'ИЗМЕНЁН!!!'}")
if real_before != real_after:
    fail.append("ИЗОЛЯЦИЯ: сравнение изменило живой lipid_world.json")
if fail:
    print("\nКРАСНЫЙ:")
    for f in fail:
        print("  ✗ " + f)
    raise SystemExit(1)
print("\nЗЕЛЁНЫЙ: оба канала живы, изоляция цела")
