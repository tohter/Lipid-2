#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ИНВЕНТАРИЗАЦИЯ ПО ИСПОЛНЕНИЮ: что движок ЗВАЛ по-настоящему.

ЗАЧЕМ. experiments/inventory.py считает ссылки в коде и честен ровно до
динамического вызова: `getattr(self._h, name)` передаёт имя ПЕРЕМЕННОЙ, и
разбор без исполнения не знает, что там читают. На этой дыре инвентарь
объявил гормональную систему мёртвой («пишут 15, читают 3»), хотя шлюз читает
ТЕ ЖЕ числа и по ним решает «пиши/не пиши». Поймано прогоном живого процесса,
не глазами.

Значит достоверный ответ на «это мертво?» даёт только ИСПОЛНЕНИЕ. Здесь
ставится трассировщик вызовов (sys.settrace) и прогоняются СЦЕНАРИИ, которые
пользователь делает руками: запуск, вопрос, вопрос с пробелом, обучение из
текста, рассказ, сон. Учителя выключены: замер не трогает GPU и не тратит
VRAM. В живую память не пишется — всё под копией каталога в /tmp.

Дочерний процесс нужен потому, что трассировку ставят до импорта модулей:
иначе часть вызовов инициализации уходит за кадр (сам файл умеет быть
дочерним: переменная окружения TRACE_CHILD).

КАК ЧИТАТЬ ВЫВОД:
  * ИСПОЛНЯЛОСЬ — имя прозвучало в реальном прогоне, полкой быть не может;
  * НЕ ПРОЗВУЧАЛО — НЕ приговор: сценарий мог просто не пойти по той ветке.
    Это список на ручную проверку, а не разрешение удалять.

Запуск: .venv/bin/python experiments/trace_live_calls.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ЯДРО = ("brain", "core", "teachers", "lang")
ВЫЗОВ = "call"      # имя-константа, чтобы в теле трассировки латиницы не было

if os.environ.get("TRACE_CHILD") == "1":
    # ── дочерний процесс: трассировка ставится ДО импорта движка ──
    sys.path.insert(0, ROOT)
    os.environ["LIPID2_STATUS"] = "0"
    ПАМЯТЬ = os.environ["TRACE_MEMORY_DIR"]

    вызвано = {}
    префиксы = tuple(os.path.join(ROOT, p) + os.sep for p in ЯДРО)

    def трассировка(фрейм, событие, арг):
        if событие == ВЫЗОВ:
            код = фрейм.f_code
            if код.co_filename.startswith(префиксы):
                имя = код.co_name
                вызвано[имя] = вызвано.get(имя, 0) + 1
            return трассировка
        return трассировка

    from brain.lipid2_brain import Lipid2Brain          # noqa: E402

    мозг = None
    сценарии = [
        ("запуск мозга", lambda: _старт()),
        ("вопрос с ответом", lambda: мозг.process_input("вода гасит огонь?")),
        ("вопрос с пробелом", lambda: мозг.process_input("что такое китовал")),
        ("обучение из текста",
         lambda: мозг.learn_from_text("дождь гасит пожар",
                                      source="gui", allow_teacher=False)),
        ("обучение рассказом",
         lambda: мозг.learn_from_text(
             "вася дал книжку петя . петя прочитал книжку .",
             source="gui", allow_teacher=False)),
        ("сон", lambda: мозг.sleep_now()),
    ]

    def _старт():
        global мозг
        мозг = Lipid2Brain(memory_dir=ПАМЯТЬ)
        # Учителей ВЫКЛЮЧАЕМ сразу после сборки: иначе прогон зависит от
        # сети и GPU, а трассировка меряет ДВИЖОК, а не внешние каналы.
        мозг.use_llm_teacher = False
        мозг.use_wiki_teacher = False
        return True

    for название, действие in сценарии:
        try:
            sys.settrace(трассировка)
            действие()
        except Exception as ошибка:
            sys.stderr.write(f"сценарий «{название}» упал: {ошибка!r}\n")
        finally:
            sys.settrace(None)

    выгрузка = os.environ.get("TRACE_DUMP")
    if выгрузка:
        with open(выгрузка, "w", encoding="utf-8") as fh:
            json.dump(вызвано, fh, ensure_ascii=False, indent=1)
    print("ВЫЗВЫВАЛОСЬ::" + json.dumps(вызвано, ensure_ascii=False))
    raise SystemExit(0)


def прогнать():
    tmp = tempfile.mkdtemp(prefix="trace-")
    живой = os.path.join(ROOT, "data", "memory")
    if os.path.isdir(живой):
        for f in os.listdir(живой):
            src = os.path.join(живой, f)
            if os.path.isfile(src):
                shutil.copy2(src, tmp)
    выгрузка = os.environ.get("TRACE_DUMP")
    env = dict(os.environ, TRACE_CHILD="1", TRACE_MEMORY_DIR=tmp,
               LIPID2_STATUS="0")
    if выгрузка:
        env["TRACE_DUMP"] = выгрузка
    proc = subprocess.run([sys.executable, os.path.abspath(__file__)],
                          env=env, capture_output=True, text=True, timeout=900)
    shutil.rmtree(tmp, ignore_errors=True)
    if proc.returncode != 0:
        print("дочерний процесс упал:\n", proc.stderr[-1500:])
        return None, proc.stderr
    строка = [ln for ln in proc.stdout.splitlines()
              if ln.startswith("ВЫЗВЫВАЛОСЬ::")]
    if not строка:
        print("трейс не напечатан, хвост stdout:\n", proc.stdout[-800:])
        return None, proc.stdout
    if proc.stderr.strip():
        print("   (из дочернего процесса:", proc.stderr.strip()[:400], ")")
    return json.loads(строка[-1][len("ВЫЗВЫВАЛОСЬ::"):]), proc.stderr


if __name__ == "__main__":
    вызвано, stderr = прогнать()
    if вызвано is None:
        raise SystemExit(1)

    print("=" * 74)
    print("ТО, ЧТО ДВИЖОК ЗВАЛ В РЕАЛЬНОМ ПРОГОНЕ (трассировка вызовов)")
    print("=" * 74)
    print(f"имён прозвучало: {len(вызвано)}")

    # сверка с корзиной «мертво» из инвентаря
    инв = os.path.join(ROOT, "experiments", "inventory.py")
    proc = subprocess.run([sys.executable, инв, "--json"], capture_output=True,
                          text=True, timeout=900, cwd=ROOT)
    if proc.returncode != 0:
        print("инвентарь не запустился, сверки нет")
        raise SystemExit(0)
    корзины = json.loads(proc.stdout)
    мертво = {x[0] for x in корзины.get("полка_мертво", [])}
    полка = {x[0] for x in корзины.get("полка_только_тесты", [])}

    ложь = sorted(мертво & set(вызвано))
    print(f"\n--- «мёртвые» имена, которые ЗВУЧАЛИ (инвентарь врал): "
          f"{len(ложь)} ---")
    for имя in ложь:
        print(f"      {имя:34} вызовов {вызвано[имя]}")

    честно_мертво = sorted(мертво - set(вызвано))
    print(f"\n--- не прозвучало НИ в одном сценарии: {len(честно_мертво)} ---")
    print("    (это НЕ приговор: ветка могла не исполняться)")
    for имя in честно_мертво:
        print(f"      {имя}")

    живых_на_полке = sorted(полка & set(вызвано))
    print(f"\n--- «полка только тесты», но движок ЗВАЛ: {len(живых_на_полке)} ---")
    for имя in живых_на_полке:
        print(f"      {имя:34} вызовов {вызвано[имя]}")
