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

Запуск: LIPID2_MEMORY_DIR=/tmp/e2e_mem xvfb-run -a .venv/bin/python tools/e2e_gui.py
"""
import json
import os
import sys
import tkinter as tk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ui.app_gui import Lipid2GUI                                    # noqa: E402

MEM = os.environ.get("LIPID2_MEMORY_DIR")
if not MEM:
    print("нужен LIPID2_MEMORY_DIR: прогон не должен трогать рабочую память")
    raise SystemExit(2)
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
app.var_teacher.set(False)                # см. докстрингу: LLM здесь не нужен
app._sync_flags("teacher_enabled", app.var_teacher)
app.learn_entry.insert("1.0", "Пчела опыляет цветок на лугу.")
print("→ учу предложение")
app.do_learn()
pump(600, stop=lambda: not app.jobs.busy)

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
    """Ответить на вопрос мозга, не выходя из main-потока."""
    if not app.bridge.pending:
        return False
    q = app.bridge.current_question()
    print(f"   вопрос окна: {q}")
    asked_of.append(q)
    return app.bridge.submit(ANSWER)


app.sim_entry.insert("1.0", "что такое пчела")
print("→ спрашиваю")
app.do_ask()
pump(900, stop=lambda: not app.jobs.busy, tick=answer_when_asked)
answers = [t for _, kind, t in app.journal.rows() if kind == "answer"]

if asked_of:
    # ТРЕТИЙ шаг: спросить ЕЩЁ РАЗ. Знание обязано найтись в памяти само,
    # без нового вопроса пользователю и без обращения к учителям.
    print("→ спрашиваю второй раз (проверка, что знание закрепилось)")
    app.sim_entry.insert("1.0", "что такое пчела")
    app.do_ask()
    pump(900, stop=lambda: not app.jobs.busy, tick=answer_when_asked)
    answers = [t for _, kind, t in app.journal.rows() if kind == "answer"]
    print("   второй вопрос пользователю не задан:", len(asked_of) == 1)
print("\nОТВЕТ:", answers[-1] if answers else "НИЧЕГО")
tail = [(k, t[:40].replace("\n", " ")) for _, k, t in app.journal.rows()[-4:]]
print("хвост журнала:")
for k, t in tail:
    print(f"   [{k}] {t}")
# Ответ обязан иметь СВОЙ вид и не тонусть среди отчётов обучения. Строка
# «готово за N с» после ответа законна — это статус работы, а не отчёт.
kinds = [k for _, k, _ in app.journal.rows()]
i_ans = max(i for i, k in enumerate(kinds) if k == "answer")
after = [k for k in kinds[i_ans + 1:] if k == "report"]
print("отчётов обучения ПОСЛЕ ответа:", after or "нет")

app.bridge.close()
root.destroy()

text = (answers[-1] if answers else "").lower()
ok = ("пчел" in text or "пчёл" in text) and both == [] and not after
untouched = ((os.path.getmtime(REAL) if os.path.exists(REAL) else None)
             == real_mtime)
print("\nИТОГ:", "выучил и вспомнил" if ok else "НЕ СВЯЗАНО")
print("рабочая память не тронута:", untouched)
raise SystemExit(0 if (ok and untouched) else 1)
