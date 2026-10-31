"""
Сквозной прогон п.6 «выучил -> вспомнил» через окно.

Два условия, без которых прогон ничего не доказывает:

* память задаётся ТОЛЬКО переменной LIPID2_MEMORY_DIR (без явного аргумента):
  так проверка доказывает, что переменную слушает сам мозг, а не только тест;
* Учитель выключен: ответ Gemma на незнакомое слово стоит ~70 с, а здесь
  проверяется цикл «прочитал предложение -> записал -> вспомнил по вопросу».

ЧЕГО ЭТОТ ПРОГОН НЕ ПРОВЕРЯЕТ — прочитать раньше, чем гордиться зелёным:
учителя ВЫКЛЮЧЕНЫ, всё знание даёт 3-й канал (ответ пользователя), и строка
отчёта «выучено через Учителя» здесь означает «через канал знаний вообще».
К ЛЛМ этот прогон не обращается НИ РАЗУ — значит «Gemma работает» по нему
утверждать нельзя. Для этого существует tools/e2e_live_teacher.py: он считает
ОБРАЩЕНИЯ ПО КАЖДОМУ КАНАЛУ и требует channel_calls.llm > 0 с контролем ВВЕРХ
(со снятой галочкой — ровно ноль).

ТРИ ИСХОДА, а не два (очередь, пункт F1). Прежний стенд умел только «выучил и
вспомнил» / «НЕ СВЯЗАНО» — и на поломке СТАНДА кричал на мозг:

  ВЫУЧИЛ И ВСПОМНИЛ (rc=0) — цикл прошёл.
  НЕ СВЯЗАНО       (rc=1) — прогон прошёл весь, но знание не прошло путь;
                          за это отвечает код проекта.
  ОТКАЗ СТАНДА     (rc=2) — прогон НЕ ИМЕЛ ПРАВА судить: каталог был грязный,
                          окно не взяло работу, pump не досидел до конца. Про
                          цикл этот запуск НЕ ГОВОРИТ НИЧЕГО. ЗАМЕРЕНО 31.10:
                          повтор в том же каталоге давал rc=1 «НЕ СВЯЗАНО»,
                          хотя знание в файле было ПОЛНОЕ, — во второй раз
                          спрашивать «что такое пчела» уже не о чем, вопрос не
                          задаётся, а стенд требовал asked_of.

Решающая логика (маркеры отказа, вердикт, признаки) живёт в
 tools/e2e_verdict.py — отдельно от Tk, чтобы её можно было звать из теста.
 Сам скрипт НЕИМПОРТИРУЕМ
(окно поднимается на уровне модуля), поэтому запереть его целиком подтестом
нельзя; замок есть у мер.

Запуск: LIPID2_MEMORY_DIR=/tmp/e2e_mem xvfb-run -a .venv/bin/python tools/e2e_gui.py
"""
import json
import os
import sys
import tkinter as tk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.e2e_verdict import (                                # noqa: E402
    is_refusal, knowledge_in_world, rejected_in,
    reports_after_answer, start_problems, verdict)

MEM = os.environ.get("LIPID2_MEMORY_DIR")

# Каталог проверяется ДО tk.Tk(): поднимать окно и грузить мозг ради того,
# чтобы потом сказать «я не начинал», нельзя — это ровно тот случай, где
# прогон краснел не про цикл (см. докстрингу).
_problems = start_problems(MEM)
if _problems:
    print("ОТКАЗ СТАНДА (прогон не начат):")
    for _p in _problems:
        print("   -", _p)
    raise SystemExit(2)

from ui.app_gui import Lipid2GUI                                    # noqa: E402

REAL = os.path.join(ROOT, "data", "memory", "lipid_world.json")
real_mtime = os.path.getmtime(REAL) if os.path.exists(REAL) else None

root = tk.Tk()
app = Lipid2GUI(root)


def pump(times=400, stop=None, tick=None):
    """
    Прокрутка событий UI-потока.

    tick() вызывается ЗДЕСЬ, в main-потоке: Tk запрещает трогать виджеты из
    чужого потока, а «ответить как пользователь» обязан делать именно тот
    поток, который крутит события окна.
    """
    import time as _t
    for _ in range(times):
        root.update()
        if stop is not None and stop():
            return True
        if tick is not None and tick():
            return True
        _t.sleep(0.04)
    return False


print("→ поднимаю ядро")
pump(600, stop=lambda: app.brain is not None)
if app.brain is None:
    print("ЯДРО НЕ ЗАГРУЗИЛОСЬ"); raise SystemExit(1)
print("память прогона:", app.brain.world_filepath)
assert app.brain.world_filepath.startswith(MEM), \
    f"мозг игнорирует LIPID2_MEMORY_DIR: {app.brain.world_filepath}"

app._show_tab("learn")
app.var_llm.set(False)                   # см. докстрингу: LLM здесь не нужен
app._sync_flags("llm_enabled", app.var_llm)
app.var_wiki.set(False)                  # и сеть в этом прогоне не нужна
app._sync_flags("wiki_enabled", app.var_wiki)
# ОЖИДАНИЕ ЗАВЕРШЕНИЯ РАБОТЫ, а не «не занято в этот миг».
# Прежний стенд останавливался на первом `not app.jobs.busy`, и это ловит не
# всё: jobs.start возвращает False (и пишет warn «уже работаю»), если окно ещё
# крутит предыдущую работу, — а стенд после такого молча шёл дальше и судил по
# пустому журналу. Здесь ждём ПРИРОСТА счётчика finished: он растёт под тем же
# замком, что и busy=False, поэтому «finished вырос на 1» значит «работа
# ЗАКОНЧЕНА», а «не busy» может значить «работа ещё не начиналась».
# ЗАМЕРЕНО при первой же проверке: отсчёт от НУЛЯ неверен, загрузка ядра сама
# есть работа (app.jobs.start в _boot), поэтому finished до обучения уже не
# ноль. Отсчёт ведётся ОТ БАЗЫ, снятой перед шагом — заодно пропадают хрупкие
# константы вида «к третьему вопросу finished == 3».
def wait_job(base, what, times=900, tick=None):
    """Дождаться ОДНУ завершённую работу сверх базы. Итог — новая база.

    Не дождалась — ОТКАЗ СТАНДА (rc=2): прогон не досидел и про цикл не
    говорит НИЧЕГО. Причина называется словами, в т.ч. когда окно ОТКЛОНИЛО
    работу — иначе «уже работаю» снова превратилось бы в «НЕ СВЯЗАНО».
    """
    if pump(times, stop=lambda: app.jobs.finished >= base + 1, tick=tick):
        return app.jobs.finished
    rows = app.journal.rows()
    rej = rejected_in(rows)
    print(f"ОТКАЗ СТАНДА: {what} не завершилась за {times} перекачек"
          + (f" — окно ОТКЛОНИЛО работу: {rej[0][:70]}" if rej else ""))
    print("   прогон не досидел, про цикл «выучил -> вспомнил» он НЕ ГОВОРИТ")
    raise SystemExit(2)


app.learn_entry.insert("1.0", "Пчела опыляет цветок на лугу.")
print("→ учу предложение")
_base = app.jobs.finished
app.do_learn()
_base = wait_job(_base, "обучение")

path = os.path.join(MEM, "lipid_world.json")
world = json.load(open(path, encoding="utf-8"))
print("сущности:", sorted(world["entities"]))
print("неизвестные:", {k: v.get("status") for k, v in world["unknown_words"].items()})
both = sorted(set(world["entities"]) & set(world["unknown_words"]))
print("противоречий «известно и неизвестно»:", both or "нет")

# ── второй шаг: вспомнить. LLM выключен, поэтому знание даёт 3-й канал —
# мозг задаёт вопрос ЧЕРЕЗ ОКНО, тест отвечает как пользователь. Так проверяется
# весь путь «не знаю -> спросил -> записал -> знаю», а не один лишь поиск.
app.var_ask.set(True)
app._sync_flags("ask_enabled", app.var_ask)
app._show_tab("sim")


ANSWER = "Пчела — летаемое насекомое; опыляет цветы; живёт в улье"
asked_of = []


def answer_when_asked():
    """Ответить на вопрос мозга, не выходя из main-потока.

    Возвращает ВСЕГДА False. Это важно: pump выходит и по True от tick, а
    TRUE тут означал бы «ответ ОТДАН» — и pump бросал бы прокрутку ДО того,
    как работа с вопросом закончена и строка ответа легла в журнал (замерено:
    по stop «не busy» такие прогоны доходили до ответа, по True от submit —
    нет). Свободу окна ждёт stop по счётчику finished, а не этот ответ.
    """
    if not app.bridge.pending:
        return False
    q = app.bridge.current_question()
    print(f"   вопрос окна: {q}")
    asked_of.append(q)
    app.bridge.submit(ANSWER)
    return False


app.sim_entry.insert("1.0", "что такое пчела")
print("→ спрашиваю")
app.do_ask()
_base = wait_job(_base, "первый вопрос", tick=answer_when_asked)
answers = [t for _, kind, t in app.journal.rows() if kind == "answer"]

if asked_of:
    # ТРЕТИЙ шаг: спросить ЕЩЁ РАЗ. Знание обязано найтись в памяти само,
    # без нового вопроса пользователю и без обращения к учителям.
    print("→ спрашиваю второй раз (проверка, что знание закрепилось)")
    app.sim_entry.insert("1.0", "что такое пчела")
    app.do_ask()
    _base = wait_job(_base, "второй вопрос", tick=answer_when_asked)
    answers = [t for _, kind, t in app.journal.rows() if kind == "answer"]
    print("   второй вопрос пользователю не задан:", len(asked_of) == 1)
print("\nОТВЕТ:", answers[-1] if answers else "НИЧЕГО")
tail = [(k, t[:40].replace("\n", " ")) for _, k, t in app.journal.rows()[-4:]]
print("хвост журнала:")
for k, t in tail:
    print(f"   [{k}] {t}")
# Ответ обязан иметь СВОЙ вид и не тонусть среди отчётов обучения. Строка
# «готово за N с» после ответа законна — это статус работы, а не отчёт.
# ОТВЕТ МОЖЕТ НЕ БЫТЬ ВОВСЕ (окно не взяло работу). max() по пустому
# итератору ронял стенд без ИТОГА и без причины; reports_after_answer
# возвращает None, и ниже это отдельный, читаемый отказ СТАНДА.
_rows = app.journal.rows()
after = reports_after_answer([k for _, k, _ in _rows])
rejected = rejected_in(_rows)
print("отчётов обучения ПОСЛЕ ответа:",
      "ОТВЕТА НЕТ" if after is None else (after or "нет"))
print("работ, ОТКЛОНЁННЫХ окном:", len(rejected) or "нет")

app.bridge.close()
root.destroy()

text = answers[-1] if answers else ""
# Проверка строилась на подстроке «пчел» — и ПРОХОДИЛА на отказе: «О 'пчела'
# в памяти есть только само слово — знания нет» содержит это же слово.
# ЗАМЕРЕНО: прогон печатал «выучил и вспомнил», хотя вопрос пользователю не
# задавался НИ РАЗУ и в памяти осталась одна сущность («опылять»), то есть
# не было выучено НИЧЕГО. Поэтому проверяем ФАКТАМИ, а не лексикой ответа.
# Признаки отказа и сам вердикт — из tools/e2e_verdict.py: у теста и стенда
# ОБЩИЙ КОД, а не два списка, способных разойтись (урок 44).
refused = is_refusal(text)
# И главное — знание обязано лежать в ФАЙЛЕ, а не только в тексте ответа.
# Файл перечитывается: прежний `world` прочитан ДО вопроса и не может содержать
# то, что мозг выучил во время вопроса.
world_after = json.load(open(path, encoding="utf-8"))
bee = (world_after.get("entities") or {}).get("пчела") or {}
remembered = knowledge_in_world(bee)
print("сущность «пчела» в файле:", {
    k: bee.get(k) for k in ("hierarchy", "properties", "definition")}
    if bee else "НЕТ")
print("вопросов пользователю задано:", len(asked_of))
print("ответ это отказ:", refused)
both_after = sorted(set(world_after.get("entities") or {})
                    & set(world_after.get("unknown_words") or {}))
print("противоречий «известно и неизвестно» после вопроса:", both_after or "нет")
ok, why_not = verdict(asked_of, text, after, remembered, both_after)

# ОТКАЗ СТАНДА проверяется ПОСЛЕ вердикта и ВАЖНЕЕ него: если окно не взяло
# работу, пустой asked_of означает не «мозг не спросил», а «мозга не спросили».
# Именно этот порядок и превращал поломку прогона в красный результат проверки.
stand = []
if after is None:
    stand.append("ОТВЕТА в журнале нет: прогон не дошёл до ответа")
if rejected:
    stand.append(f"окно ОТКЛОНИЛО работу: {len(rejected)} — "
                 f"{rejected[0][:60]}")

untouched = ((os.path.getmtime(REAL) if os.path.exists(REAL) else None)
             == real_mtime)
if not untouched:
    stand.append("прогон тронул РАБОЧУЮ память — изоляция не сработала")

print()
if stand:
    print("ИТОГ: ОТКАЗ СТАНДА (прогон не имел права судить):")
    for _s in stand:
        print("   -", _s)
    _rc = 2
elif ok:
    print("ИТОГ: выучил и вспомнил")
    _rc = 0
else:
    print("ИТОГ: НЕ СВЯЗАНО:")
    for _w in why_not:
        print("   -", _w)
    _rc = 1
print("рабочая память не тронута:", untouched)
raise SystemExit(_rc)
