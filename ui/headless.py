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


class FakeBrain:
    """Настоящий мозг поднимается ~10 с и требует torch — здесь он не нужен."""

    def __init__(self):
        self.hormones = type("H", (), {"cortisol": 40})()
        self.noradrenaline = 90
        self.dopamine = 200
        self.ask_user_callback = None
        self.calls = []

    def learn_from_text(self, text, source="text", allow_teacher=True):
        time.sleep(0.2)
        self.calls.append(("learn", text, allow_teacher))
        return FakeReport()

    def process_input(self, text):
        time.sleep(0.1)
        self.calls.append(("ask", text))
        return "Флюгер — прибор для определения направления ветра."

    def _load_world(self):
        return {"entities": {"флюгер": {}}, "observations": {"а": 1},
                "unknown_words": {"б": 1}}


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
