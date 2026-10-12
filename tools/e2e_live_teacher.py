"""
ЧЕСТНЫЙ сквозной прогон с ЖИВЫМ Учителем (llama.cpp @8080).

Чем отличается от tools/e2e_gui.py и почему их два:
e2e_gui.py идёт с ВЫКЛЮЧЕННЫМ LLM (знание даёт 3-й канал) и потому не может
проверить канал 1 НИКАК — он констатирует «ИТОГ: выучил и вспомнил» даже когда
все учителя выключены и слово выучил сам тест. Он проверяет цикл «спросил ->
записал -> вспомнил», а не ЛЛМ-канал.

Здесь учитель ВКЛЮЧЁН, и проверяется именно канал 1 — по ЧИСЛАМ
(app.last_report, brain.channel_calls/channel_found), а не по подстрокам в
тексте журнала:
  A. контроль ВВЕРХ: со снятой галочкой канала LLM не вызывается НИ РАЗУ и
     слова уходят в not_asked (доказательство, что счётчики не рисуются);
  B. урок без механических отказов канала («missing ... filename», ОБРЕЗАН,
     HTTP Error, Traceback);
  C. знание выучено ЧЕРЕЗ канал: channel_calls.llm > 0 и channel_found.llm > 0;
  D. глагол стал ACTION-узлом с профилем ИЗ СПРАВОЧНИКА;
  E. у глагола нет пожизненного asked_empty;
  F. арифметика отчёта сходится по словам;
  G. ответ «что делает X» берётся из памяти и НЕ дёргает LLM.

БЕЗ LIPID2_MEMORY_DIR не запускается: рабочая память Вселенной не двигается.

Запуск:
  LIPID2_MEMORY_DIR=/tmp/live3 xvfb-run -a .venv/bin/python tools/e2e_live_teacher.py
"""
import json
import os
import sys
import time
import tkinter as tk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ui.app_gui import Lipid2GUI                                    # noqa: E402

SENTENCE = "Вода гасит пламя."
MEM = os.environ.get("LIPID2_MEMORY_DIR")
if not MEM:
    print("нужен LIPID2_MEMORY_DIR: прогон не должен трогать рабочую память")
    raise SystemExit(2)
REAL = os.path.join(ROOT, "data", "memory", "lipid_world.json")
real_stat = os.stat(REAL).st_mtime_ns if os.path.exists(REAL) else None

root = tk.Tk()
app = Lipid2GUI(root)
fail = []


def pump(seconds=30.0, stop=None, tick=None):
    """Крутить события UI-потока до stop()/tick() или до истечения seconds.

    tick() вызывается ЗДЕСЬ, в main-потоке: трогать виджеты и «отвечать как
    пользователь» из рабочего потока нельзя.
    """
    end = time.time() + seconds
    while time.time() < end:
        root.update()
        if stop is not None and stop():
            return True
        if tick is not None and tick():
            return True
        time.sleep(0.03)
    return False


def check(ok, label, detail=""):
    print(f"  {'OK ' if ok else 'ПРОВАЛ'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        fail.append(label)


def world():
    return json.load(open(os.path.join(MEM, "lipid_world.json"), encoding="utf-8"))


print("→ поднимаю ядро")
pump(90, stop=lambda: app.brain is not None)
if app.brain is None:
    print("ЯДРО НЕ ЗАГРУЗИЛОСЬ")
    raise SystemExit(1)
print("память прогона:", app.brain.world_filepath)
check(app.brain.world_filepath.startswith(MEM),
      "мозг слушает LIPID2_MEMORY_DIR", app.brain.world_filepath)

# ── A. КОНТРОЛЬ ВВЕРХ: со снятой галочкой канал обязан молчать ──
print("\n=== A. контрольный урок при ВЫКЛЮЧЕННОМ канале ===")
app.var_teacher.set(False)
app._sync_flags("teacher_enabled", app.var_teacher)
app.var_ask.set(False)
app._sync_flags("ask_enabled", app.var_ask)
app._show_tab("learn")
app.learn_entry.insert("1.0", "Рыбак чинит сеть.")
app.do_learn()
pump(90, stop=lambda: not app.jobs.busy)
ctl = app.last_report
check(ctl is not None, "GUI публикует последний отчёт (last_report)")
if ctl is not None:
    check(ctl.channel_calls.get("llm", -1) == 0,
          "со снятой галочкой LLM НЕ вызывался ни разу",
          str(ctl.channel_calls))
    check(ctl.channel_found.get("llm", -1) == 0,
          "и ничего не нашёл", str(ctl.channel_found))
    check(set(ctl.unknown) == set(ctl.not_asked),
          "невыученное без учителей числится НЕисканным",
          f"unknown={ctl.unknown} not_asked={ctl.not_asked}")
    ctl_statuses = {k: v.get("status") for k, v in world()["unknown_words"].items()}
    check("asked_empty" not in ctl_statuses.values(),
          "выключенные каналы не породили пожизненное «искали и не нашли»",
          str(ctl_statuses))

# ── включаем канал 1, канал 3 остаётся снят ──
app.var_teacher.set(True)
app._sync_flags("teacher_enabled", app.var_teacher)
health = ""
try:
    import urllib.request
    with urllib.request.urlopen("http://localhost:8080/health", timeout=5) as r:
        health = r.read().decode("utf-8", "ignore").strip()
except Exception as exc:                                     # noqa: BLE001
    health = f"НЕДОСТУПЕН: {exc}"
print("\nllama.cpp /health:", health)
if "ok" not in health:
    print("сервер Учителя не жив — прогон бессмысленен, честнее умереть")
    raise SystemExit(2)

print(f"\n=== B..F. урок «{SENTENCE}» с включённым каналом ===")
app.learn_entry.insert("1.0", SENTENCE)
t0 = time.time()
app.do_learn()
pump(240, stop=lambda: not app.jobs.busy)
print(f"   урок {time.time() - t0:.0f} с")

rows = app.journal.rows()
journal_text = "\n".join(t for _, _, t in rows)
rep = app.last_report
if rep is None:
    check(False, "отчёт обучения доступен")
    raise SystemExit(1)
print("   " + "\n   ".join(rep.as_text().splitlines()))

print("\n=== B. механические отказы канала ===")
for needle, why in (("missing 1 required positional argument",
                     "ложная ошибка инструмента (filename от модели)"),
                    ("ОБРЕЗАН", "ответ обрезан по лимиту токенов"),
                    ("finish=length", "то же по-английски"),
                    ("HTTP Error", "транспортный отказ"),
                    ("Traceback", "исключение внутри урока")):
    check(needle not in journal_text, f"нет «{needle}»", why)

print("\n=== C. знание выучено ЧЕРЕЗ канал 1 (числа, не строки) ===")
print("   каналы:", rep.channel_calls, "нашли:", rep.channel_found)
check(rep.channel_calls.get("llm", 0) > 0,
      "к локалке обращались", str(rep.channel_calls.get("llm")))
check(rep.channel_found.get("llm", 0) > 0,
      "локалка ДАЛА знание", str(rep.channel_found.get("llm")))
check(rep.channel_calls.get("user", -1) == 0,
      "канал пользователя НЕ трогался (галочка снята)",
      str(rep.channel_calls.get("user")))

ents = sorted(world()["entities"])
print("   сущности:", ents)
print("   unknown:", {k: v.get("status") for k, v in world()["unknown_words"].items()})
check({"вода", "пламя"}.issubset(ents), "вода и пламя выучены", str(ents))

print("\n=== D. глагол стал действием ===")
g = world()["entities"].get("гасить")
check(isinstance(g, dict), "ACTION-узел «гасить» существует")
if isinstance(g, dict):
    check(g.get("kind") == "action", "kind == action", str(g.get("kind")))
    check(g.get("participants", {}).get("agent") == ["вода"],
          "участник agent = вода", str(g.get("participants")))
    check(g.get("action_profile") == [20, 50, 10],
          "профиль ИЗ СПРАВОЧНИКА [20, 50, 10], а не сочинён",
          str(g.get("action_profile")))
obs = list(world()["observations"])
check("вода|CAN_DO|гасить" in obs, "связь CAN_DO записана", str(obs))
check("гасить" in rep.learned_actions and "гасить" not in rep.learned_concepts,
      "действие учтено отдельно от сущностей",
      f"concepts={rep.learned_concepts} actions={rep.learned_actions}")

print("\n=== E. честные статусы ===")
uw = world()["unknown_words"]
check(uw.get("гасить", {}).get("status") != "asked_empty",
      "у глагола НЕТ пожизненного asked_empty", str(uw.get("гасить", {}).get("status")))
check(all(v.get("status") in ("asked_empty", "not_asked") for v in uw.values()),
      "у каждой unknown-записи есть статус",
      str({k: v.get("status") for k, v in uw.items()}))

print("\n=== F. арифметика отчёта ===")
total = len(rep.learned_concepts) + len(rep.learned_actions) + len(rep.unknown)
check(rep.seen_unknown_words == total,
      "незнакомых = сущности + действия + неизвестные",
      f"{rep.seen_unknown_words} != {total}")
check(len(rep.learned_concepts) == len(set(rep.learned_concepts)),
      "в выученных нет дублей", str(rep.learned_concepts))
check("вода" in rep.learned_concepts and "пламя" in rep.learned_concepts,
      "сущности названы своими словами", str(rep.learned_concepts))

print("\n=== G. вспомнил без нового похода к LLM ===")
llm_before = app.brain.channel_calls.get("llm", 0)
app._show_tab("sim")
app.sim_entry.insert("1.0", "Что делает вода?")
app.do_ask()
pump(120, stop=lambda: not app.jobs.busy)
answers = [t for _, k, t in app.journal.rows() if k == "answer"]
ans = answers[-1] if answers else ""
print("   ответ:", ans[:150])
check("гасить" in ans, "«Что делает вода?» отвечает умением", ans[:80])
check("в разработке" not in ans, "ACTION-интент не заглушка", ans[:80])
llm_after = app.brain.channel_calls.get("llm", 0)
check(llm_after == llm_before, "ответ НЕ дёрнул LLM",
      f"{llm_before} -> {llm_after}")

app.bridge.close()
root.destroy()

untouched = ((os.stat(REAL).st_mtime_ns if os.path.exists(REAL) else None)
             == real_stat)
print("\nрабочая память не тронута:", untouched)
check(untouched, "рабочая память Вселенной не изменена")

print("\nИТОГ:", "успех" if not fail else f"ПРОВАЛЕНО: {fail}")
raise SystemExit(0 if not fail else 1)
