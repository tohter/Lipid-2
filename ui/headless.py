"""
ui/headless.py — собрать окно, отработать событиям и закрыть без человека.

Нужно потому, что GUI-баги этого класса (гонки Tk-потоков, «readonly, из
которого нельзя копировать», подсказка вместо ввода) НЕ видны в юнит-тестах с
FakeBrain и не ловляются проверкой «скомпилировалось». Здесь то же окно, что и
в рантайме, но события генерируются кодом.

Запуск:  xvfb-run -a .venv/bin/python ui/headless.py     (exit 0 = всё хорошо)
"""
import os
import sys
import time

os.environ.setdefault("LIPID2_STATUS", "0")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tkinter as tk                                     # noqa: E402

import ui.app_gui as gui_mod                             # noqa: E402
from ui.app_gui import Lipid2GUI                         # noqa: E402

# Баннер кодировщика проверяет НЕ эту проверку, а tests/test_golden_bits.py.
# Здесь важнее не гонять sentence-transformers ради одной строки в шапке:
# это ~10 с и загрузка модели на каждый прогон.
FAST = os.environ.get("LIPID_HEADLESS_FAST") == "1"
if FAST:
    gui_mod.encoder_info = lambda: ("fake", True, "cfg-test")


class FakeReport:
    content_words = 6
    known_words = 4
    unknown_words = 2
    facts = 5
    observations_new = 3
    teacher_calls = 1
    learned_concepts = ["флюгер"]
    unknown = ["джойстик"]
    quality = {"morph": True, "teacher": True}

    def as_text(self):
        return "📚 Отчёт\n  предложений: 1   слов: 6   фактов: 5"


class FakeSleepReport:
    """Отчёт сна той же формы, что у brain.sleep.SleepReport."""

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.examined, self.created, self.raised = 5, 1, 2
        self.lowered, self.decayed, self.compacted = 0, 5, 1
        self.skipped_negated, self.file_conflict = 0, 0
        self.notes = []
        self.changes = [
            {"kind": "created", "head": "кошка", "tail": "мышь",
             "strength": 9, "new_weight": 152},
            {"kind": "raised", "head": "пёс", "tail": "кошка",
             "old_weight": 144, "new_weight": 164, "strength": 9},
            {"kind": "decay", "head": "кошка", "tail": "мышь",
             "old_strength": 9, "new_strength": 8},
            {"kind": "skipped_negated", "head": "рыба",
             "tail": "бегать", "strength": 12},
        ]

    def as_text(self):
        return ("СОН: " + ("(пробный, не записано) " if self.dry_run else "")
                + "просмотрено 5, создано 1, поднято 2, понижено 0, "
                  "затухло 5, стерто 1")


class FakeBrain:
    """Настоящий мозг поднимается ~10 с и требует torch — здесь он не нужен."""

    def __init__(self):
        self.hormones = type("H", (), {"cortisol": 40})()
        self.noradrenaline = 90
        self.dopamine = 200
        self.ask_user_callback = None
        self.calls = []
        # Флаги каналов: окно обязано дописываться до них (проверяется в лоб).
        self.use_llm_teacher = True
        self.use_wiki_teacher = True
        self.ask_user_enabled = True
        self.learn_experience = True

    def learn_from_text(self, text, source="text", allow_teacher=True):
        time.sleep(0.2)
        self.calls.append(("learn", text, allow_teacher))
        return FakeReport()

    def process_input(self, text):
        time.sleep(0.1)
        self.calls.append(("ask", text))
        return "Флюгер — прибор для определения направления ветра."

    # ── СОН ──────────────────────────────────────────────────────
    def sleep_now(self, dry_run=False, decay_rate=None):
        time.sleep(0.1)
        self.calls.append(("sleep", bool(dry_run)))
        rep = FakeSleepReport(dry_run)
        return {"report": rep, "text": rep.as_text(), "saved": not dry_run}

    def sleep_if_due(self, force=False):
        self.calls.append(("sleep_due", bool(force)))
        return self.sleep_now()

    def _load_world(self):
        return {"entities": {"флюгер": {}}, "observations": {"а": 1},
                "unknown_words": {"б": 1},
                "sleep": {"cycles": 2, "meaningful_cycles": 1, "created": 3,
                          "raised": 1, "lowered": 0, "decayed": 9,
                          "compacted": 2, "skipped_negated": 0,
                          "file_conflict": 0}}


def run_checks(_unused=None) -> list:
    """Возвращает список проблем. Пустой список = окно ведёт себя как заявлено."""
    problems = []
    global app, root

    def pump(n):
        for _ in range(n):
            try:
                root.update()
            except tk.TclError:
                return
            time.sleep(0.05)

    root = tk.Tk()
    app = Lipid2GUI(root, brain=FakeBrain())
    pump(6)

    def check(cond, what):
        if not cond:
            problems.append(what)

    # ── вкладки и поле ввода ────────────────────────────────
    check(app.tab == "learn", "стартовая вкладка не «Обучение»")
    check(app.learn_entry.winfo_manager() == "pack",
          "поле обучения не видимо на вкладке обучения")
    check(app.sim_entry.winfo_manager() == "",
          "поле симуляции видимо на вкладке обучения (Enter уйдёт в скрытое поле)")
    app._show_tab("sim")
    pump(3)
    check(app.sim_entry.winfo_manager() == "pack"
          and app.learn_entry.winfo_manager() == "",
          "переключение вкладки не поменяло панель ввода")
    check(app.tab_btns["sim"].cget("bg") == "#094770", "активная вкладка не подсвечена")
    app._show_tab("learn")
    pump(3)

    # ── вкладка «Симулятор рассказа»: выбор, загрузка, вопрос ──
    app._show_tab("story")
    pump(3)
    check(app.story_entry.winfo_manager() == "pack"
          and app.learn_entry.winfo_manager() == "",
          "вкладка рассказа не поменяла панель ввода")
    check(app.story_entry is not app.sim_entry,
          "поле рассказа то же, что у вкладки «Симуляция»")

    # список берётся с диска, а не из списка имён в коде
    n_stories = len(app.stories)
    check(n_stories > 0,
          "на диске не найдено ни одного рассказа "
          "(experiments/*_world.json) — вкладка будет бесполезна")
    if n_stories:
        check(app.story_combo["values"],
              "combobox рассказа пуст при найденных рассказах")
        # Демонстрационный мир обязан собраться: это проверка, что подбор
        # класса симулятора идёт из данных мира, а не из имени файла.
        app.story_combo.current(0)
        app.do_load_story()
        for _ in range(80):
            pump(2)
            if app.story_loaded:
                break
        check(app.story_loaded is not None,
              "рассказ не загрузился: " + app.story_state.cget("text"))
        if app.story_loaded:
            check(app.story_loaded_index == 0,
                  "индекс загруженного рассказа не совпал с выбранным")
            src = app.stories[0]
            q = ("Почему кошка съела канарейку?"
                 if "social" in " ".join(src.scheme_names)
                 else "Почему пожар погас?")
            app.story_entry.insert("1.0", q)
            app.do_ask_story()
            for _ in range(80):
                pump(2)
                if "ты: " in app.story_chat.get("1.0", "end") and \
                        len(app.story_chat.get("1.0", "end").splitlines()) > 2:
                    break
            chat = app.story_chat.get("1.0", "end")
            check(("ты: " + q) in chat, "вопрос рассказа не виден в чате")
            check("потому что" in chat,
                  "симулятор не ответил через окно: " + chat[-300:])

    # отказ без загруженного рассказа обязан быть СЛОВАМИ, а не тишиной
    app.story_loaded = None
    app.story_entry.insert("1.0", "почему пожар погас?")
    app.do_ask_story()
    pump(3)
    check("не загружен" in app.story_chat.get("1.0", "end"),
          "вопрос без загруженного рассказа не получил отказа (молчание = баг)")
    app._show_tab("learn")
    pump(3)

    # ── П.19: ЧЕТЫРЕ галочки каналов — четыре независимых провода ──
    # Проверяется НАСТОЯЩИМ окном (не подставным объектом, как в юнит-тесте):
    # именно так ловится «галочка есть, а провод от неё отрезан».
    app.var_llm.set(False)
    app._sync_flags("llm_enabled", app.var_llm)
    check(app.brain.use_llm_teacher is False
          and app.brain.use_wiki_teacher is True,
          "галочка LLM не дошла до мозга или погасила Вики: "
          f"llm={app.brain.use_llm_teacher} wiki={app.brain.use_wiki_teacher}")
    app.var_wiki.set(False)
    app._sync_flags("wiki_enabled", app.var_wiki)
    check(app.brain.use_wiki_teacher is False
          and app.brain.use_llm_teacher is False,
          "галочка Вики не дошла до мозга")
    app.var_llm.set(True)
    app._sync_flags("llm_enabled", app.var_llm)
    check(app.brain.use_llm_teacher is True
          and app.brain.use_wiki_teacher is False,
          "включение LLM обратно не дошло или включило Вики следом")
    # При снятых обоих Учителях поиск знания обязан оставаться живым, если
    # включён 3-й канал (иначе галочка Учителя глушит чужой канал).
    app.var_ask.set(True)
    app._sync_flags("ask_enabled", app.var_ask)
    check(app._any_knowledge_channel() is True,
          "снятые УЧИТЕЛЯ отменили поиск знания при включённом 3-м канале")
    # Все три канала сняты — поиск обязан быть ОТМЕНЁН (иначе пустой вызов).
    app.var_llm.set(False)
    app._sync_flags("llm_enabled", app.var_llm)
    app.var_ask.set(False)
    app._sync_flags("ask_enabled", app.var_ask)
    check(app._any_knowledge_channel() is False,
          "все каналы сняты, а поиск знания всё ещё обещается")
    for var, attr in ((app.var_llm, "llm_enabled"),
                      (app.var_wiki, "wiki_enabled"),
                      (app.var_ask, "ask_enabled")):
        var.set(True)
        app._sync_flags(attr, var)
    check(app._any_knowledge_channel() is True,
          "возвращённые каналы не вернули поиск знания")

    # ── вкладка «Сон»: кнопки, dry_run, прозрачность состояния ──
    app._show_tab("sleep")
    pump(3)
    check("sleep" in app.panels, "вкладки «Сон» нет вообще")
    # Сводка обязана читаться ИЗ ФАЙЛА ПАМЯТИ, а не из памяти окна.
    check("снов 2" in app.sleep_state.cget("text"),
          "сводка о снах не показана: " + app.sleep_state.cget("text"))
    check("накоплено" in app.sleep_state.cget("text"),
          "не сказано, сколько тиков накоплено до следующего сна")
    def wait_sleep_log(fragment, tries=40):
        """
        Ждать, ПОКА ПОДСТРОКА НЕ ПОЯВИТСЯ в логе вкладки.

        Ждать надо ПО ЦЕЛИ, а не «пока jobs.busy станет False»: флаг снимает
        рабочий поток СРАЗУ по окончании работы, а сообщение в Tk летит через
        очередь, которую перекачивает after(120). Цикл «крутить update, пока
        jobs.busy» выходил раньше, чем ответ долетал до виджета, и тест падал
        на РАБОТАЮЩЕМ коде (замерено: тот же прогон с другим порядком
        перекачок проходил). Гонка в тесте опаснееMissing теста: она ловит
        случайные падения и случайные же зелёные прогоны.
        """
        for _ in range(tries):
            pump(1)
            if fragment in app.sleep_chat.get("1.0", "end"):
                return True
        return False

    def wait_sleep_count(want, tries=40):
        """Ждать, пока отчётов сна в логе станет не меньше want."""
        for _ in range(tries):
            pump(1)
            if app.sleep_chat.get("1.0", "end").count("СОН:") >= want:
                return True
        return False

    before_sleep_calls = [c for c in app.brain.calls if c[0] == "sleep"]
    app.do_sleep_dry()
    # Проверка БЕЗ выжидания: jobs.start ставит busy и label СИНХРОННО, а
    # «поспать 100 мс и посмотреть» ловило бы и успешный случай, и гонку.
    check(app.jobs.busy and app.jobs.label == "пробный сон",
          "пробный сон пошёл мимо jobs — значит второй поток пишет в тот же "
          f"lipid_world.json, что и обучение (label={app.jobs.label!r})")
    ok_dry = wait_sleep_log("ПРОБНЫЙ")
    chat_s = app.sleep_chat.get("1.0", "end")
    check(ok_dry, "пробный сон не отозвался в лог вкладки")
    check("кошка → мышь" in chat_s,
          "сон не назван поимённо — «создано 1» ничего не проверяет: " + chat_s[-200:])
    check("опыт 9/15, вес 152" in chat_s,
          "в отчёте нет чисел формулы 80+8*сила — на чём тогда проверять контракт?")
    check("ПРОТИВОРЕЧИЕ" in chat_s,
          "опыт против отрицания проглочен молча")
    dry_calls = [c for c in app.brain.calls if c[0] == "sleep"]
    check(dry_calls and dry_calls[-1] == ("sleep", True),
          f"кнопка «посмотреть» позвала НЕ dry_run: {dry_calls[-3:]}")
    check(dry_calls != before_sleep_calls, "пробный сон не дошёл до мозга")

    # Кнопки гасятся ОДНИМ вызовом. Проверяется состоянием, а не «успел ли
    # погаснуть за 100 мс»: у одной кнопки две ручки (jobs и вкладка) — это
    # и есть тот класс бага, который ловится не глазами, а порядком вызовов.
    app._set_buttons(True)
    check(app.sleep_now_btn["state"] == "disabled"
          and app.learn_btn["state"] == "disabled",
          "на время работы гасится не всякая кнопка: сон и обучение пишут "
          "в один файл и не имеют права быть доступными одновременно")
    app._set_buttons(False)
    check(app.sleep_now_btn["state"] == "normal"
          and app.sleep_now_btn.cget("text") == "Спать сейчас",
          "кнопка не вернулась в рабочее состояние/подпись")

    app.do_sleep()
    check(wait_sleep_count(2),
          "реальный сон не дошёл до вкладки: "
          + app.sleep_chat.get("1.0", "end")[-200:])
    chat_s = app.sleep_chat.get("1.0", "end")
    real = [c for c in app.brain.calls if c[0] == "sleep_due"]
    check(real and real[-1] == ("sleep_due", True),
          "кнопка «спать сейчас» не зашла через sleep_if_due(force=True) — "
          "значит порог и кнопка живут разными правилами")
    check(app.sleep_now_btn["state"] == "normal",
          "кнопка сна не вернулась в активное состояние")
    # галочка автосна обязана ДЕЙСТВОВАТЬ, а не просто храниться
    app.var_sleep_auto.set(False)
    app._toggle_sleep_auto()
    check(app.brain.sleep_auto is False,
          "галочка автосна не доходит до мозга (окно врёт про режим)")
    check("ВЫКЛЮЧЕН" in app.sleep_state.cget("text"),
          "отключение автосна не названо словами")
    app.var_sleep_auto.set(True)
    app._toggle_sleep_auto()
    check(app.brain.sleep_auto is True, "автосон не включается обратно")
    app._show_tab("learn")
    pump(3)

    # ── ПКМ: состав меню и доступность пунктов ──────────────
    labels = []
    for i in range(app.clip.menu.index("end") + 1):
        if app.clip.menu.type(i) == "separator":
            continue
        labels.append(app.clip.menu.entrycget(i, "label"))
    for want in ("Вырезать", "Копировать", "Вставить", "Выделить всё"):
        check(want in labels, f"в меню ПКМ нет пункта «{want}»")
    check(bool(root.bind_class("Text", "<Button-3>")),
          "на класс Text не привязан ПКМ (новый виджет получит мёртвую кнопку)")
    check(bool(root.bind_class("Entry", "<Button-3>")),
          "на класс Entry не привязан ПКМ")

    # ── копирование из журнала (главный пункт требования) ───
    app._append(app.log, time.time(), "status", "скопируй меня")
    pump(2)
    app.log.tag_add("sel", "1.0", "end")
    app.clip.target = app.log
    root.clipboard_clear()
    app.log.event_generate("<<Copy>>")
    pump(2)
    try:
        clip = root.clipboard_get()
    except tk.TclError:
        clip = ""
    check("скопируй меня" in clip,
          f"Ctrl+C/«Копировать» из журнала не работает (в буфере: {clip!r})")

    # ── журнал защищён от пользовательского ввода ───────────
    before = app.log.get("1.0", "end")
    app.log.focus_set()
    app.log.event_generate("<KeyPress>", keysym="a")
    pump(2)
    check(app.log.get("1.0", "end") == before,
          "в readonly-журнал можно печатать с клавиатуры")
    app.log.event_generate("<BackSpace>")
    pump(2)
    check(app.log.get("1.0", "end") == before, "BackSpace удаляет из журнала")

    # ── вставка в поле ввода ───────────────────────────────
    root.clipboard_clear(); root.clipboard_append("джойстик")
    app.learn_entry.focus_set()
    app.clip.target = app.learn_entry
    app.learn_entry.event_generate("<<Paste>>")
    pump(2)
    check("джойстик" in app.learn_entry.get("1.0", "end"),
          "«Вставить» не работает в поле ввода")

    # ── обучение доходит до мозга, кнопки реагируют ────────
    app.learn_entry.delete("1.0", "end")
    app.learn_entry.insert("1.0", "Флюгер вертится на ветру")
    app.do_learn()
    pump(2)
    check(app.jobs.busy and app.learn_btn["state"] == "disabled",
          "на время работы кнопка осталась активной")
    check("работает" in app.learn_btn.cget("text"),
          "кнопка не объясняет, что занята (серая «Учить» = сломанная?)")
    pump(30)
    check(not app.jobs.busy, "работа не завершилась")
    check(("learn", "Флюгер вертится на ветру", True) in app.brain.calls,
          "обучение не дошло до мозга")
    check(app.learn_btn["state"] == "normal", "кнопка не вернулась в active")
    text = app.log.get("1.0", "end")
    # ВНИМАНИЕ: статус-бар обновляется перекачкой событий (до ~120 мс), поэтому
    # здесь проверяется «данные дошли до окна», а не мгновенность отражения.
    check("обучение" in app.statusbar.cget("text")
          or "фактов" in text or app.journal.last_status(),
          "итог обучения не дошёл ни до статуса, ни до журнала")
    if not FAST:
        check("семантический" in app.lbl_encoder.cget("text"),
              "шапка не показывает реальную SDR-схему")

    # ── диалог ─────────────────────────────────────────────
    app._show_tab("sim"); pump(2)
    app.sim_entry.insert("1.0", "что такое флюгер")
    app.do_ask(); pump(20)
    chat = app.chat.get("1.0", "end")
    check("ты: что такое флюгер" in chat, "вопрос не виден в чате")
    check("прибор" in chat, "ответ мозга не виден в чате")

    # ── обратный вопрос: вопрос -> ответ -> разблокировка ──
    import threading
    app.ask_enabled = True
    got = []
    th = threading.Thread(target=lambda: got.append(
        app._ask_user_via_bridge("джойстик")), daemon=True)
    th.start()
    pump(10)
    check(bool(app.ask_bar.winfo_ismapped()), "панель вопроса не показалась")
    check("джойстик" in app.ask_lbl.cget("text"), "вопрос не прочитывается")
    app.ask_entry.insert(0, "устройство для игр")
    app._submit_answer()
    th.join(3)
    check(not th.is_alive(), "рабочий поток не отпустило после ответа")
    check(got == ["устройство для игр"], f"ответ не дошёл до мозга: {got}")

    # выключенный канал обязан отказаться явно, а не молчать
    app.ask_enabled = False
    check(app._ask_user_via_bridge("х") is None,
          "выключенный 3-й канал вернул ответ вместо отказа")
    check(any("3-й канал выключен" in r[2] for r in app.journal.rows()),
          "отказ выключенного канала не попал в журнал (молчание = баг)")

    # ── закрытие отпускает ожидание ────────────────────────
    again = []
    app.ask_enabled = True
    th2 = threading.Thread(target=lambda: again.append(
        app._ask_user_via_bridge("юла")), daemon=True)
    th2.start(); pump(8)
    app.on_close()
    th2.join(3)
    check(not th2.is_alive(),
          "закрытие окна не отпустило поток, ждущий ответа (процесс не умрёт)")
    return problems


def main():
    """
    События перебираются root.update() в том же потоке, где созданы виджеты:
    mainloop не нужен, но after-колбэки и очереди рабочих потоков обрабатываются.
    """
    try:
        problems = run_checks(None)
    except Exception as exc:
        import traceback
        problems = [f"исключение: {type(exc).__name__}: {exc}\n"
                    + traceback.format_exc()]
    if problems:
        print("⛔ ПРОБЛЕМЫ GUI:")
        for p in problems:
            print("   •", p)
        return 1
    print("✅ GUI: вкладки, ПКМ-копирование, журнал, обратный вопрос — ок")
    return 0


if __name__ == "__main__":
    sys.exit(main())
