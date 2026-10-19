"""
ui/app_gui.py — окно Lipid-2: вкладки «Обучение» и «Симуляция».

ГЛАВНОЕ ПРАВИЛО ЭТОГО ФАЙЛА
---------------------------
Tkinter НЕ потокобезопасен. Каждый виджет здесь меняет только основной поток;
рабочий поток пишет в два потокабезопасных буфера (ui.presenter.Journal и
обычные строки-поля), а root.after(120) забирает их в окно. Раньше worker
трогал Tk напрямую — такие гонки не воспроизводятся стабильно и ловятся как
«иногда падает», поэтому разделение сделано структурно, а не по невнимательности.

ЗАЧЕМ ПЕРЕПИСАНО, А НЕ ПОДКРАШЕНО
---------------------------------
1. ВКЛАДКИ — КНОПКИ СВЕРХУ. ttk.Notebook темой 'default' рисует нечитаемые
   вкладки, а переключать программой всё равно надо (результат обучения уводит
   в чат). Панели складываются в один Frame, показываются pack/pack_forget.
   ПАНЕЛЬ ВВОДА ОДНА И ПЕРЕКЛЮЧАЕТСЯ ВМЕСТЕ С ВКЛАДКОЙ — иначе Enter в видимом
   поле запускает обработку скрытого (так и было в первой версии этого файла).

2. ОБУЧЕНИЕ ПОКАЗЫВАЕТ ПРОЦЕСС. Окно без журнала = мёртвый экран на 30-60 с,
   потому что Учитель (локальная LLM) делает до шести шагов на слово. Каждый
   шаг идёт через core.logchannel и попадает в журнал + индикатор «чем занята».

3. ОБРАТНЫЙ ВОПРОС. Третий канал знания («спросить пользователя») был веткой
   кода без пути наружу. Здесь есть режим «спрашивать меня»: мост
   ui.presenter.AnswerBridge ждёт ответа из рабочего потока, закрытие окна его
   отпускает, а выключенный режим отказывает ЯВНО.

4. ПКМ = вырезать/копировать/вставить/выделить для ЛЮБОГО текстового виджета.
   Журнал и чат — state="readonly", а не "disabled": disabled нельзя даже
   выделить, то есть «копировать правой кнопкой» было бы невозможно.

5. ЛОГИКА СОСТОЯНИЯ — в ui/presenter.py, тестируется без дисплея
   (tests/test_presenter_contract.py). Здесь только Tk.

ЗАПУСКАТЬ ТОЛЬКО ИЗ .venv: системный python3 не упадёт, а молча откатит SDR на
буквенный хэш (смысл не кодируется). Поэтому схема SDR написана в шапке ALWAYS,
а при деградации — красная строка и отдельная запись в журнал (_report_encoder).
"""
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import scrolledtext, ttk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.logchannel import ask_handler, status, subscribe, unsubscribe
from core.sdr import encoder_info
from brain.question import answer as ask_story
from brain import storyload
from ui.presenter import (AnswerBridge, JobController, Journal, KIND_ANSWER,
                           answer_row, format_report,
                          summarize_report)

BG = "#2d2d2d"
BG_PANEL = "#252526"
BG_INPUT = "#1e1e1e"
BG_TAB_ON = "#094770"
BG_TAB_OFF = "#333333"
FG = "#e0e0e0"
FG_DIM = "#8a8a8a"
ACCENT = "#ff4081"

KIND_COLORS = {
    "status": "#d4d4d4",
    "ok": "#4ecdc4",
    "warn": "#e5c07b",
    "error": "#ff6b6b",
    "report": "#9cdcfe",       # отчёты обучения — голубой
    "answer": "#e5c07b",       # ответ Аманды — отдельный цвет, не «отчёт»
    "ask": "#c586c0",
    "user": "#dcdcaa",
}


# Порядок bindtags виджета в Tk по умолчанию: (сам виджет, класс, toplevel, all)
# — то есть привязка НА ВИДЖЕТЕ отрабатывает РАНЬШЕ классовых, и возвращённый
# "break" останавливает обход, не давая классу Text вставить символ. Это и
# используется ниже.
#
# ДВЕ ЛОВУШКИ, обе пойманы ui/headless.py, а не чтением документации:
#   1) Text.tag_bind() — это привязка к ТЕКСТОВОМУ тегу ("sel", "ts"), а не к
#      bindtag: привязка молча не ставится и ничего не блокирует;
#   2) `lambda e: ... key ...` внутри цикла захватывает ПЕРЕМЕННУЮ, а не её
#      значение: все обработчики начинают видеть последнюю клавишу списка и
#      ведут себя «почти правильно». Отсюда key=k в аргументах лямбды.
_READONLY_KEYS_ALLOWED = {
    # навигация и выделение должны работать: журнал читают, а не редактируют
    "Left", "Right", "Up", "Down", "Prior", "Next", "Home", "End",
    "Shift_L", "Shift_R", "Control_L", "Control_R", "Caps_Lock", "Num_Lock",
    "Tab", "ISO_Left_Tab", "Escape",
}
# Ctrl-сочетания, которые обязан принимать read-only журнал
_READONLY_CTRL_ALLOWED = {"c", "C", "Insert", "a", "A"}


def _readonly_key(event) -> str:
    """Клавиши правки — break; навигация, выделение и Ctrl+C — пропустить."""
    if event.state & 0x0004:                       # ControlMask
        return None if event.keysym in _READONLY_CTRL_ALLOWED else "break"
    if event.keysym in _READONLY_KEYS_ALLOWED:
        return None
    return "break"


def make_readonly(widget: tk.Text) -> tk.Text:
    """
    «Только чтение» для Text в Tk 8.6.

    ЗАМЕРЕНО НА ЭТОЙ МАШИНЕ (Tk 8.6.14):
      * state="readonly" для Text НЕ существует (TclError: bad state "readonly");
      * state="disabled" запрещает и ВЫДЕЛЕНИЕ мышью, то есть «скопировать
        правой кнопкой» стало бы невозможно — а это прямое требование.

    Поэтому состояние остаётся normal, а правка отсекается обработчиками на самом
    виджете (его bindtag первый): клавиши, вставка, вырезание и средняя кнопка
    мыши. Вставки самой программы через insert() это не касается; пользователь
    читает, выделяет и копирует.

    Флаг _l2_readonly читает Clip, чтобы не предлагать «Вставить/Вырезать» там,
    где они запрещены.
    """
    widget._l2_readonly = True
    widget.bind("<KeyPress>", _readonly_key)
    widget.bind("<KeyRelease>", lambda e: None)
    for seq in ("<Delete>", "<BackSpace>"):        # <KeyPress> их уже покрывает,
        widget.bind(seq, lambda e: "break")        # но явность дешевле догадок
    for virt in ("<<Cut>>", "<<Paste>>", "<<Clear>>", "<<Insert>>",
                 "<<Undo>>", "<<Redo>>"):
        widget.bind(virt, lambda e: "break")
    widget.bind("<Button-2>", lambda e: "break")   # средняя кнопка = вставка в X
    return widget



class Clip:
    """
    Контекстное меню (ПКМ) для ВСЕХ текстовых виджетов окна.

    Привязка вешается на ТЕГ КЛАССА (bind_class("Text"/"Entry")), поэтому новый
    виджет получает ПКМ автоматически — иначе через полгода появится поле,
    «в котором правая кнопка не работает», и это будет баг, а не фичу.

    Цель меню определяется по последнему кликнутему/сфокусированному виджету, а
    не через focus_get(): у всплывающего Menu фокус свой, focus_get() возвращает
    None, и «копировать» молча не делает ничего.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.target = None
        self.menu = tk.Menu(root, tearoff=0, bg="#3c3c3c", fg=FG,
                            activebackground=ACCENT, activeforeground="#ffffff",
                            relief="flat", bd=0, font=("DejaVu Sans", 9))
        for label, virt in (("Вырезать", "<<Cut>>"), ("Копировать", "<<Copy>>"),
                            ("Вставить", "<<Paste>>")):
            self.menu.add_command(label=label, command=lambda v=virt: self._send(v))
        self.menu.add_separator()
        self.menu.add_command(label="Выделить всё",
                              command=lambda: self._send("<<SelectAll>>"))
        self.menu.add_command(label="Очистить поле", command=self._clear)
        for cls in ("Text", "Entry"):
            self.root.bind_class(cls, "<Button-3>", self.popup)
            self.root.bind_class(cls, "<Control-Button-3>", self.popup)
            self.root.bind_class(cls, "<Button-1>", self.remember)
            self.root.bind_class(cls, "<FocusIn>", self.remember)
            # Ctrl+C/V/X работают из коробки; Shift+Insert — привычная вставка X
            self.root.bind_class(cls, "<Shift-Insert>",
                                 lambda e: self._send("<<Paste>>"))

    def remember(self, event):
        self.target = event.widget
        return None                                   # не глушим обычное поведение

    def _readonly(self, w) -> bool:
        return bool(getattr(w, "_l2_readonly", False))

    def popup(self, event):
        self.target = event.widget
        ro = self._readonly(event.widget)
        for item, enabled in (("Вырезать", not ro), ("Вставить", not ro),
                              ("Очистить поле", not ro), ("Копировать", True),
                              ("Выделить всё", True)):
            self.menu.entryconfig(item, state="normal" if enabled else "disabled")
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def _send(self, virtual: str):
        w = self.target
        if w is None:
            return
        try:
            w.event_generate(virtual)
        except tk.TclError:
            pass

    def _clear(self):
        w = self.target
        if w is None:
            return
        try:
            w.delete("1.0", "end")
        except tk.TclError:
            try:
                w.delete(0, "end")
            except tk.TclError:
                pass


class Lipid2GUI:
    def __init__(self, root: tk.Tk, brain=None):
        self.root = root
        self.root.title("Lipid-2 · нейроморфное ядро")
        self.root.geometry("1040x740")
        self.root.minsize(860, 580)
        self.root.configure(bg=BG)

        # ── состояние (пишут все потоки, читает только _pump) ──
        self.journal = Journal()
        self._ui_q: "queue.Queue" = queue.Queue()   # колбэки строго в UI-поток
        self._shown = 0
        self._pending_status = None
        # Итог работы («фактов +5, слов 4/6 знакомо») обязан продержаться
        # несколько секунд: иначе следующая строка журнала через 120 мс стирает
        # результат, и вместо итога пользователь видит «записано: ...».
        self._status_hold_until = 0.0

        self.clip = Clip(root)
        self.jobs = JobController(self.journal,
                                  on_state=lambda b: self._ui_q.put(
                                      ("busy", b)),
                                  on_done=lambda: self._ui_q.put(("kb", None)))
        self.bridge = AnswerBridge(
            on_question=lambda q: self._ui_q.put(("question", q)))
        self._sub_id = subscribe(self._on_status)

        self.brain = None
        self.var_llm = tk.BooleanVar(value=True)
        self.var_wiki = tk.BooleanVar(value=True)
        self.var_ask = tk.BooleanVar(value=False)
        self.var_experience = tk.BooleanVar(value=True)
        # РЕАЛЬНОЕ состояние каналов для рабочих потоков. Tk-переменную читать
        # из чужого потока НЕЛЬЗЯ: на этой машине это дало
        # RuntimeError: main thread is not in main loop. Поэтому галочка
        # живёт в UI, а поток читает обычный bool (обновляется command'ом).
        # ПОСЛЕДНИЙ ОТЧЁТ И КАНАЛЫ, доступные МАШИНЕ.
        # Зачем: окно показывает отчёт глазами (текст в журнале), и любой
        # сквозной прогон мог только искать подстроки в тексте журнала.
        # ЗАМЕРЕНО: e2e-прогон утверждал «выучил и вспомнил», даже когда все
        # три канала были выключены, потому что проверял цикл, а не источник
        # знания. Эти два поля дают проверке числа, а не строки.
        self.last_report = None            # brain.learner.LearningReport | None
        self.last_channel_calls = {}       # {"llm": n, "wiki": n, "user": n}
        self.last_channel_found = {}
        # Раздельные галочки каналов (п.19). Раньше одна «спрашивать Учителя
        # (LLM / Вики)» гасила ДВА канала разом: снять её, чтобы не ждать
        # Gemma 70 с, означало молча потерять и Википедию, которая отвечает за
        # секунды и бесплатна.
        self.llm_enabled = True
        self.wiki_enabled = True
        # ВЫБРАННЫЙ РАССКАЗ. Живёт отдельно от brain: рассказ отвечает сам
        # из своего файла, ядро (SDR, Учитель) ему не нужно.
        self.stories = []               # [brain.storyload.StorySource]
        self.story_loaded = None        # {ключ: <Simulator>} | None
        self.story_loaded_index = None  # индекс загруженного в self.stories
        self.ask_enabled = False
        self.experience_enabled = True

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._pump()

        # Инициализируем SDR-кодировщик в фоне, чтобы окно появилось сразу
        self.jobs.start(self._init_encoder, label="SDR-кодировщик")

        if brain is not None:
            self._attach_brain(brain)
        else:
            # Ядро поднимается долго (эмбеддинги ~9 с + калибровка). Окно должно
            # появиться СРАЗУ: «окна нет» из-за загрузки неотличимо от упавшего
            # приложения, и пользователь решит, что оно упало.
            self.jobs.start(self._load_brain, label="загрузка ядра")

    # ══════════════════════════════════════════════════════
    #  МОЗГ
    # ══════════════════════════════════════════════════════
    def _load_brain(self):
        status("поднимаю ядро: SDR-кодировщик, граф знаний, три канала")
        from brain.lipid2_brain import Lipid2Brain
        brain = Lipid2Brain()
        self._ui_q.put(("brain", brain))

    def _init_encoder(self):
        """Инициализирует SDR-кодировщик в фоне, не блокируя GUI."""
        from core.encoder import get_encoder
        enc = get_encoder()
        self._ui_q.put(("encoder_ready", enc))

    def _attach_brain(self, brain):
        self.brain = brain
        # Состояние галочек на момент подключения мозга: окно могло быть
        # настроено раньше, чем ядро поднялось.
        self._push_channel_flags()
        brain.ask_user_callback = self._ask_user_via_bridge
        ask_handler(self.bridge.handler)          # канал 3 получает путь наружу
        status("ядро готово — можно учить и спрашивать")
        for w in (self.learn_entry, self.sim_entry):
            w.config(state="normal")
        # Галочка автосна, снятая ДО готовности ядра, обязана дожить до мозга:
        # иначе окно покажет «включено», а мозг поработает по умолчанию.
        if hasattr(self, "var_sleep_auto"):
            self.brain.sleep_auto = bool(self.var_sleep_auto.get())
        self._refresh_sleep_tab()
        self._set_buttons(busy=False)
        self._refresh_kb()

    def _require_brain(self) -> bool:
        if self.brain is not None:
            return True
        self.journal.add("⛔ ядро ещё не готово — подожди окончания загрузки",
                         "error")
        return False

    # ══════════════════════════════════════════════════════
    #  СБОРКА ОКНА
    # ══════════════════════════════════════════════════════
    def _build_ui(self):
        self._build_header()
        self._build_tabbar()
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self.panels = {}
        self._build_learn_tab(body)
        self._build_sim_tab(body)
        self._build_story_tab(body)
        self._build_sleep_tab(body)
        self._build_bottom()
        self._show_tab("learn")

    def _build_header(self):
        head = tk.Frame(self.root, bg=BG)
        head.pack(fill="x", padx=8, pady=(8, 2))

        left = tk.Frame(head, bg=BG)
        left.pack(side="left")
        tk.Label(left, text="Lipid-2", bg=BG, fg=ACCENT,
                 font=("DejaVu Sans", 13, "bold")).pack(anchor="w")
        self.lbl_encoder = tk.Label(left, text="SDR: проверяю…", bg=BG, fg=FG_DIM,
                                    font=("DejaVu Sans", 8), anchor="w")
        self.lbl_encoder.pack(anchor="w")
        self.lbl_kb = tk.Label(left, text="память: —", bg=BG, fg=FG_DIM,
                               font=("DejaVu Sans", 8), anchor="w")
        self.lbl_kb.pack(anchor="w")

        right = tk.Frame(head, bg=BG)
        right.pack(side="right")
        self.horm = {}
        for key, name, color in (("nor", "норадреналин", "#ff6b6b"),
                                 ("dop", "дофамин", "#4ecdc4"),
                                 ("cor", "кортизол", "#e5c07b")):
            row = tk.Frame(right, bg=BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=name, width=14, anchor="w", bg=BG, fg=color,
                     font=("DejaVu Sans", 8)).pack(side="left")
            bar = ttk.Progressbar(row, length=110, maximum=255)
            bar.pack(side="left", padx=4)
            val = tk.Label(row, text="—", width=4, anchor="e", bg=BG, fg=FG_DIM,
                           font=("DejaVu Sans", 8))
            val.pack(side="left")
            self.horm[key] = (bar, val)

    def _build_tabbar(self):
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=8, pady=(6, 0))
        self.tab_btns = {}
        for key, title in (("learn", " Обучение "), ("sim", " Симуляция "),
                           ("story", " Симулятор рассказа "),
                           ("sleep", " Сон ")):
            b = tk.Button(bar, text=title, relief="flat", bd=0, bg=BG_TAB_OFF,
                          fg=FG, cursor="hand2", activebackground=BG_TAB_ON,
                          activeforeground="#ffffff",
                          font=("DejaVu Sans", 10, "bold"), padx=16, pady=6,
                          command=lambda k=key: self._show_tab(k))
            b.pack(side="left")
            self.tab_btns[key] = b
        tk.Frame(bar, bg=BG_TAB_ON, height=2).pack(fill="x", side="bottom")
        self.tab = None

    def _show_tab(self, key):
        """
        Смена вкладки. Обязательно меняет и ПАНЕЛЬ ВВОДА: если этого не делать,
        Enter в видимом поле запускает обработку скрытого — пользователь видит
        одно, а происходит другое.
        """
        if key not in self.panels:
            return
        self.tab = key
        for k, panel in self.panels.items():
            if k == key:
                panel.pack(fill="both", expand=True)
            else:
                panel.pack_forget()
        for k, btn in self.tab_btns.items():
            btn.config(bg=BG_TAB_ON if k == key else BG_TAB_OFF,
                       fg="#ffffff" if k == key else FG)
        for child in self.entry_host.winfo_children():
            child.pack_forget()
        # Поле ОБЯЗАНО соответствовать вкладке: лишний виджет в entry_host
        # означал бы Enter, который отправляет вопрос в скрытую вкладку.
        if key == "sleep":
            # У СНА НЕТ ПОЛЯ ВВОДА, и это не забытость, а смысл: спать нечем
            # «набирать». Кнопки живут в самой вкладке, а общее нижнее поле
            # остаётся пустым — иначе Enter отправил бы вопрос в скрытую
            # вкладку (тот же баг, из-за которого поля уже разделены по вкладкам).
            return
        entry, button = {"sim": (self.sim_entry, self.sim_btn),
                         "story": (self.story_entry, self.story_btn)}.get(
            key, (self.learn_entry, self.learn_btn))
        button.pack(side="right", padx=(6, 0))
        entry.pack(side="left", fill="x", expand=True)
        entry.focus_set()

    # ── вкладка «Обучение» ──────────────────────────────────
    def _build_learn_tab(self, parent):
        panel = tk.Frame(parent, bg=BG)
        self.panels["learn"] = panel

        top = tk.Frame(panel, bg=BG)
        top.pack(fill="x", pady=(6, 4))
        for var, text, attr in ((self.var_llm,
                                 "Учитель: локальная модель (LLM)", "llm_enabled"),
                                (self.var_wiki,
                                 "Учитель: Википедия (интернет)", "wiki_enabled"),
                                (self.var_ask,
                                 "спрашивать меня (3-й канал)", "ask_enabled"),
                                (self.var_experience,
                                 "наблюдать опыт (предсказания)", "experience_enabled")):
            c = tk.Checkbutton(top, text=text, variable=var, bg=BG, fg=FG,
                               selectcolor=BG_PANEL, activebackground=BG,
                               activeforeground=FG, highlightthickness=0, bd=0,
                               font=("DejaVu Sans", 9), anchor="w",
                               command=lambda a=attr, v=var: self._sync_flags(a, v))
            c.pack(side="left", padx=(0, 14))
        tk.Label(top, text="выключенный канал отказывает ЯВНО: «молча не "
                           "спросили» этот проект считает багом", bg=BG, fg=FG_DIM,
                 font=("DejaVu Sans", 8)).pack(side="left")

        self.log = scrolledtext.ScrolledText(
            panel, wrap="word", bg=BG_INPUT, fg=KIND_COLORS["status"],
            font=("DejaVu Sans Mono", 9), insertbackground=FG,
            selectbackground=ACCENT, selectforeground="#ffffff",
            relief="flat", padx=8, pady=6)
        make_readonly(self.log)
        self.log.pack(fill="both", expand=True)
        self._style_text(self.log)

    # ── вкладка «Симуляция» ─────────────────────────────────
    def _build_sim_tab(self, parent):
        panel = tk.Frame(parent, bg=BG)
        self.panels["sim"] = panel

        note = tk.Frame(panel, bg=BG_PANEL)
        note.pack(fill="x", pady=(8, 6), ipadx=8, ipady=4)
        tk.Label(note, text="◉  Предиктивный цикл подключён — это НЕ заглушка",
                 bg=BG_PANEL, fg="#98c379",
                 font=("DejaVu Sans", 10, "bold"), anchor="w").pack(fill="x",
                                                                    padx=8, pady=(6, 2))
        for line in (
                "Что работает: цикл «предсказание -> ошибка -> пластичность» "
                "наблюдает тот же текст, что и обучение. Граница предложения "
                "разрывает опыт; вход лемматизируется; силы переходов живут "
                "секцией transitions в lipid_world.json и переживают перезапуск.",
                "Чего цикл НЕ делает: не пишет знания, не создаёт узлов и не "
                "трогает уверенность фактов (weight онтологии). У него своя "
                "шкала 0..15 — иначе «факт сомнительный» и «следующее слово "
                "редко угадывали» стали бы одним числом.",
                "Числа видны в отчёте обучения: строгая точность (угадано "
                "именно продолжение) и покрытие (доля тиков, где было что "
                "проверять). Без покрытия «0 ошибок» означает «опыта нет».",
                "Что ещё не сделано: IR-интерпретатор умеет только «что такое X "
                "/ свойства X / взаимодействие X и Y», отдельного режима "
                "прогона последовательностей нет — опыт набирается обучением.",
                "Ниже — настоящий диалог с ядром: он работает и отвечает по любой "
                "форме выученного слова."):
            tk.Label(note, text="•  " + line, bg=BG_PANEL, fg=FG_DIM,
                     font=("DejaVu Sans", 9), anchor="w", justify="left",
                     wraplength=900).pack(fill="x", padx=8, pady=1)

        self.chat = scrolledtext.ScrolledText(
            panel, wrap="word", bg=BG_INPUT, fg=KIND_COLORS["status"],
            font=("DejaVu Sans Mono", 9), insertbackground=FG,
            selectbackground=ACCENT, selectforeground="#ffffff",
            relief="flat", padx=8, pady=6)
        make_readonly(self.chat)
        self.chat.pack(fill="both", expand=True)
        self._style_text(self.chat)

    def _style_text(self, widget):
        for kind, color in KIND_COLORS.items():
            widget.tag_config(kind, foreground=color)
        widget.tag_config("ts", foreground="#5a5a5a")
        widget.bind("<MouseWheel>",
                    lambda e: widget.yview_scroll(-1 * (e.delta // 120), "units"))
        widget.bind("<Button-4>", lambda e: widget.yview_scroll(-4, "units"))
        widget.bind("<Button-5>", lambda e: widget.yview_scroll(4, "units"))

    # ── низ окна: панель вопроса + поле ввода + статус-бар ───
    def _sync_flags(self, attr, var):
        """Галочка нажата в UI-потоке — перекладываем значение в обычный bool."""
        setattr(self, attr, bool(var.get()))
        self._push_channel_flags()

    def _push_channel_flags(self):
        """
        Переключатели окна обязаны доходить до МОЗГА, а не только до обучения.

        ЗАМЕРЕНО: флаг «спрашивать Учителя» читался только в do_learn, поэтому
        ответ на незнакомое слово всё равно уходил в Gemma и висел ~70 с
        (потом «ответ без вызова инструмента»). Снятая галочка должна значить
        «канал не трогает НИЧЕГО» — иначе она не переключатель, а украшение.

        П.19: галочек ЧЕТЫРЕ, LLM и Вики разделены. Мозг различал их и раньше
        (use_llm_teacher / use_wiki_teacher), но окно давало ОДИН провод на оба:
        «подождать Gemma 70 с» стоило молча потерянной Википедии, которая
        отвечает за секунды.
        """
        brain = self.brain
        if brain is None:
            return
        try:
            brain.use_llm_teacher = bool(self.llm_enabled)
            brain.use_wiki_teacher = bool(self.wiki_enabled)
            brain.ask_user_enabled = bool(self.ask_enabled)
            brain.learn_experience = bool(self.experience_enabled)
        except Exception as exc:
            self.journal.add(f"⚠ переключатели каналов не дошли до мозга: "
                             f"{type(exc).__name__}: {exc}", "warn")

    # ── вкладка «Симулятор рассказа» ────────────────────────
    def _build_story_tab(self, parent):
        """
        Вкладка, где берут рассказ с диска и спрашивают ПРИЧИНУ.

        Она появилась потому, что вкладка «Симуляция» симулятора не знала:
        она ходила в ядро («что такое кошка»), а «почему кошка набросилась»
        спросить было нечерез что — весь механизм brain/question.py оставался
        за окном.

        Рассказы берутся списком с диска (brain.storyload.discover), а не
        списком имён в коде. Новый рассказ = новый файл в experiments/ или в
        каталоге из переменной окружения LIPID2_STORY_DIRS.
        """
        panel = tk.Frame(parent, bg=BG)
        self.panels["story"] = panel

        bar = tk.Frame(panel, bg=BG)
        bar.pack(fill="x", pady=(6, 2))
        tk.Label(bar, text="рассказ:", bg=BG, fg=FG_DIM,
                 font=("DejaVu Sans", 9)).pack(side="left")
        self.story_combo = ttk.Combobox(bar, state="readonly", width=52)
        self.story_combo.pack(side="left", padx=6)
        self.story_combo.bind("<<ComboboxSelected>>",
                              lambda e: self._story_selected())
        tk.Button(bar, text="Обновить список", command=self._refresh_stories,
                  bg=BG_PANEL, fg=FG, relief="flat", bd=0, cursor="hand2",
                  font=("DejaVu Sans", 9), padx=10, pady=3).pack(side="left")
        self.story_load_btn = tk.Button(bar, text="Загрузить рассказ",
                                        command=self.do_load_story,
                                        bg=BG_TAB_ON, fg="#ffffff",
                                        relief="flat", bd=0, cursor="hand2",
                                        font=("DejaVu Sans", 9, "bold"),
                                        padx=12, pady=3)
        self.story_load_btn.pack(side="left", padx=(6, 0))

        self.story_state = tk.Label(panel, text="рассказ не загружен", bg=BG,
                                    fg=KIND_COLORS["warn"], anchor="w", justify="left",
                                    wraplength=940, font=("DejaVu Sans", 9))
        self.story_state.pack(fill="x", padx=2, pady=(2, 4))

        self.story_chat = scrolledtext.ScrolledText(
            panel, wrap="word", bg=BG_INPUT, fg=KIND_COLORS["status"],
            font=("DejaVu Sans Mono", 9), insertbackground=FG,
            selectbackground=ACCENT, selectforeground="#ffffff",
            relief="flat", padx=8, pady=6)
        make_readonly(self.story_chat)
        self.story_chat.pack(fill="both", expand=True)
        self._style_text(self.story_chat)
        self._refresh_stories()

    def _refresh_stories(self):
        """Перечитать каталоги. Пустой список обязан быть объяснён словами:
        пустой combobox неотличим от сломанной кнопки."""
        try:
            self.stories = storyload.discover()
        except Exception as exc:
            self.stories = []
            self._story_say("рассказы не читаются: %s: %s"
                            % (type(exc).__name__, exc), error=True)
            return
        labels = [s.label() for s in self.stories]
        try:
            self.story_combo.config(values=labels)
        except tk.TclError:
            return
        if not labels:
            self.story_loaded = None
            self.story_loaded_index = None
            self._story_say("ни одного мира не найдено. Мир — это JSON с "
                            "секциями entities и schemes в experiments/ или в "
                            "каталоге из LIPID2_STORY_DIRS", error=True)
            return
        keep = (self.story_loaded_index
                if self.story_loaded_index in range(len(labels)) else 0)
        self.story_combo.current(keep)
        self._story_selected()

    def _story_selected(self):
        idx = self.story_combo.current()
        if not (0 <= idx < len(self.stories)):
            return
        src = self.stories[idx]
        parts = ["схем: %d" % len(src.scheme_names),
                 "сущностей: %d" % src.entities]
        if src.story_path:
            parts.append("события: " + os.path.basename(src.story_path))
        else:
            parts.append("файл событий не найден")
        if self.story_loaded_index == idx:
            parts.append("ЗАГРУЖЕН")
        self._story_say("%s — %s" % (src.title, ", ".join(parts)),
                        error=bool(src.warning), extra=src.warning)

    def _story_say(self, text, error=False, extra=""):
        try:
            self.story_state.config(text=text + ("  " + extra if extra else ""),
                                     fg=KIND_COLORS["warn"] if error else FG_DIM)
        except tk.TclError:
            pass

    def do_load_story(self):
        """Собрать симулятор выбранного рассказа. Ядро для этого НЕ нужно:
        рассказу не нужны ни SDR-кодировщик, ни Учитель, поэтому кнопка
        сознательно не зависит от готовности brain."""
        idx = self.story_combo.current()
        if not (0 <= idx < len(self.stories)):
            self._story_say("выбери рассказ в списке", error=True)
            return
        src = self.stories[idx]
        self._set_story_busy(True)
        self._story_say("собираю: " + src.title)

        def work():
            try:
                sim, err = storyload.build_simulator(src)
            except Exception as exc:
                self._ui_q.put(("story", (idx, None,
                                          "%s: %s" % (type(exc).__name__, exc))))
                return
            self._ui_q.put(("story", (idx, sim, err)))

        threading.Thread(target=work, daemon=True).start()

    def _after_story_load(self, idx, sim, err):
        self._set_story_busy(False)
        if sim is None:
            self.story_loaded = None
            self.story_loaded_index = None
            self._story_say("рассказ не загружен: " + str(err), error=True)
            self.journal.add("рассказ не загружен: " + str(err), "error")
            return
        src = self.stories[idx]
        if err:
            self._story_say("загружено неполно: " + err, error=True)
        self.story_loaded = {src.key: sim}
        self.story_loaded_index = idx
        kinds = "/".join(sorted({type(s).__name__
                                 for s in self.story_loaded.values()}))
        self._story_say("готово: %s, симулятор %s. Спрашивай внизу: «почему…» "
                        "или «что будет, если…»" % (src.title, kinds))
        self.journal.add("симулятор рассказа: %s (%s)" % (src.title, kinds),
                         "status")

    def _set_story_busy(self, busy):
        try:
            self.story_load_btn.config(
                state="disabled" if busy else "normal",
                text="собираю" if busy else "Загрузить рассказ")
        except tk.TclError:
            pass

    def do_ask_story(self):
        """Вопрос ЗАГРУЖЕННОМУ симулятору. Схему выбирает сам мир по своим
        данным, поэтому окно не знает ни названий рассказов, ни числа
        аргументов обработчика."""
        text = self._entry_text(self.story_entry)
        if not text:
            self._story_say("пусто: задай вопрос", error=True)
            return
        if not self.story_loaded:
            self._append(self.story_chat, time.time(), "error",
                         "рассказ не загружен — нажми «Загрузить рассказ»")
            self._story_say("сначала загрузи рассказ", error=True)
            return
        self.story_entry.delete("1.0", "end")
        self._append(self.story_chat, time.time(), "user", "ты: " + text)
        loaded = self.story_loaded

        def work():
            try:
                res = ask_story(text, loaded)
            except Exception as exc:
                self._ui_q.put(("story_chat",
                                "вопрос не обработан: %s: %s"
                                % (type(exc).__name__, exc)))
                return
            self._ui_q.put(("story_chat", res.as_text()))

        threading.Thread(target=work, daemon=True).start()

    # ═══════════════════════════════════════════════════
    #  Вкладка «Сон» — витрина SLEEP CYCLE (очередь п.23-доработка)
    # ═══════════════════════════════════════════════════
    def _build_sleep_tab(self, parent):
        """
        Вкладка, где сон видно. Она нужна НЕ для красоты, а из-за замера:
        brain/sleep.py существовал, работал и самопроверялся, но НИ ОДИН вызов
        сна из кода не делался — значит «система консолидирует опыт во сне»
        было утверждением, которое нельзя ни увидеть, ни опровергнуть. Витрина
        делает это нажатием кнопки, а не чтением кода.

        Три кнопки, и все три нужны:
          «Посмотреть сон» — dry_run, ничего не пишет;
          «Спать сейчас»   — один цикл, принудительно;
          галочка           — спать автоматически по порогу тиков.
        Кнопка «спать» НЕ привязана к готовности SDR-ядра столь же жёстко, как
        обучение: сон читает файл памяти и таблицу переходов. Но мозг всё равно
        нужен — транзакция, живая таблица и пересборка графа живут там.
        """
        panel = tk.Frame(parent, bg=BG)
        self.panels["sleep"] = panel

        bar = tk.Frame(panel, bg=BG)
        bar.pack(fill="x", pady=(6, 2))
        self.sleep_now_btn = tk.Button(bar, text="Спать сейчас", command=self.do_sleep,
                                       bg=BG_TAB_ON, fg="#ffffff", relief="flat",
                                       bd=0, cursor="hand2",
                                       font=("DejaVu Sans", 9, "bold"),
                                       padx=12, pady=3)
        self.sleep_now_btn.pack(side="left")
        tk.Button(bar, text="Посмотреть сон (не записывая)",
                  command=self.do_sleep_dry, bg=BG_PANEL, fg=FG, relief="flat",
                  bd=0, cursor="hand2", font=("DejaVu Sans", 9),
                  padx=10, pady=3).pack(side="left", padx=(6, 0))
        self.var_sleep_auto = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="спать автоматически по порогу тиков",
                       variable=self.var_sleep_auto,
                       command=self._toggle_sleep_auto, bg=BG, fg=FG,
                       activebackground=BG, activeforeground="#ffffff",
                       selectcolor=BG_PANEL, font=("DejaVu Sans", 9)).pack(
            side="left", padx=(10, 0))
        tk.Button(bar, text="Обновить сводку", command=self._refresh_sleep_tab,
                  bg=BG_PANEL, fg=FG, relief="flat", bd=0, cursor="hand2",
                  font=("DejaVu Sans", 9), padx=10, pady=3).pack(side="left",
                                                                 padx=(6, 0))

        self.sleep_state = tk.Label(panel, text="память о снах: —", bg=BG,
                                    fg=FG_DIM, anchor="w", justify="left",
                                    wraplength=940, font=("DejaVu Sans", 9))
        self.sleep_state.pack(fill="x", padx=2, pady=(2, 4))

        self.sleep_chat = scrolledtext.ScrolledText(
            panel, wrap="word", bg=BG_INPUT, fg=KIND_COLORS["status"],
            font=("DejaVu Sans Mono", 9), insertbackground=FG,
            selectbackground=ACCENT, selectforeground="#ffffff",
            relief="flat", padx=8, pady=6)
        make_readonly(self.sleep_chat)
        self.sleep_chat.pack(fill="both", expand=True)
        self._style_text(self.sleep_chat)
        self._refresh_sleep_tab()

    def _sleep_brain_ready(self) -> bool:
        """
        Сон требует МОЗГ, но не требует SDR-кодировщика. Разница не косметика:
        _require_brain() пишет в журнал «подожди окончания загрузки ядра», а
        сон способен работать, пока кодировщик ещё поднимается, — он читает
        JSON и таблицу переходов, а не векторы.
        """
        if self.brain is not None:
            return True
        self._sleep_say("ядро ещё не готово: сон правит файл памяти, а этим"
                        "владеет мозг", error=True)
        return False

    def _sleep_say(self, text, error=False):
        try:
            self.sleep_state.config(text=text,
                                    fg=KIND_COLORS["warn"] if error else FG_DIM)
        except tk.TclError:
            pass

    def _refresh_sleep_tab(self):
        """
        Сводка о снах — из ФАЙЛА ПАМЯТИ, а не из памяти окна.

        Иначе «спала ли система» зависело бы от того, открыто ли окно: окно
        закрыли — счётчики обнулились, хотя знания-то остались. Числа берутся
        из секции sleep, которую пишет brain.sleep.record_in_world.
        """
        if self.brain is None:
            return
        try:
            world = self.brain._load_world()
        except Exception as exc:
            self._sleep_say(f"память не читается: {type(exc).__name__}: {exc}",
                            error=True)
            return
        sec = world.get("sleep") or {}
        from brain.sleep import SLEEP_TRIGGER_TICKS
        ticks = getattr(self.brain, "ticks_since_sleep", 0)
        if not sec:
            self._sleep_say(
                f"снов ещё не было. Накоплено {ticks} тиков цикла из "
                f"{SLEEP_TRIGGER_TICKS} — при достижении порога мозг спит сам")
            return
        self._sleep_say(
            "снов {cycles} (из них со знанием {meaningful_cycles}) · "
            "создано {created} · поднято {raised} · понижено {lowered} · "
            "затухло {decayed} · стёрто {compacted} · "
            "опыта против отрицаний {skipped_negated} · замен чужого опыта "
            "{file_conflict} · сейчас накоплено {t} тиков из {th}".format(
                t=ticks, th=SLEEP_TRIGGER_TICKS, **{
                    k: int(sec.get(k, 0)) for k in
                    ("cycles", "meaningful_cycles", "created", "raised",
                     "lowered", "decayed", "compacted", "skipped_negated",
                     "file_conflict")}))

    def _sleep_details(self, res) -> str:
        """
        Отчёт сном поимённо. Числа «создано 2» мало: «КАКИЕ именно связи и с
        каким весом» — единственное, чем проверяется формула 80 + 8*сила.
        """
        rep = res.get("report")
        lines = [res.get("text", "нет отчёта")]
        if rep is None:
            return "\n".join(lines)
        for ch in rep.changes:
            kind = ch.get("kind")
            pair = f"{ch.get('head')} → {ch.get('tail')}"
            if kind == "created":
                lines.append(f"   создано {pair}: опыт {ch['strength']}/15, "
                             f"вес {ch['new_weight']}")
            elif kind in ("raised", "lowered"):
                verb = "поднят" if kind == "raised" else "понижен"
                lines.append(f"   {pair}: вес {ch['old_weight']} → "
                             f"{ch['new_weight']} ({verb}, опыт "
                             f"{ch['strength']}/15)")
            elif kind == "decay":
                lines.append(f"   затух {pair}: {ch['old_strength']} → "
                             f"{ch['new_strength']}")
            elif kind == "skipped_negated":
                lines.append(f"   ⚠ опыт ждёт «{pair}», но текст сказал "
                             f"«не делает» — ПРОТИВОРЕЧИЕ, оставлено на лицо")
        for note in rep.notes:
            lines.append("   " + note)
        return "\n".join(lines)

    def do_sleep(self):
        if not self._sleep_brain_ready():
            return
        self._sleep_say("сплю…")
        # НЕ сырой поток, а self.jobs. Сон пишет в тот же lipid_world.json, что
        # и обучение, а JobController держит ОДИН фоновый поток именно поэтому
        # (ui/presenter.py: «второй поток пошёл бы писать в тот же файл»).
        # Заодно кнопка «Учить» гасится на время сна: замер этой сессии — сон
        # вне jobs оставлял все кнопки активными, а кнопка вкладки не гасла
        # никогда, потому что jobs.busy о нём не знал.
        self.jobs.start(lambda: self._ui_q.put(
            ("sleep_chat", self._sleep_details(self.brain.sleep_if_due(
                force=True) or self.brain.sleep_now()))), label="сон")

    def do_sleep_dry(self):
        """
        Пробный сон. Нужен отдельной кнопкой, потому что «что сделает сон» —
        единственный вопрос, который задают ДО, а проверять его реальным сном
        значит править память ради любопытства.
        """
        if not self._sleep_brain_ready():
            return
        # Транзакция dry_run тоже берёт _LOCK, поэтому в общий jobs, а не
        # отдельным потоком: «не пишет» не значит «не мешает».
        self.jobs.start(lambda: self._ui_q.put(
            ("sleep_chat", "ПРОБНЫЙ (не записано)\n" + self._sleep_details(
                self.brain.sleep_now(dry_run=True)))),
            label="пробный сон")

    def _toggle_sleep_auto(self):
        if self.brain is None:
            # Галочка снята ДО ядра — состояние обязано дожить до мозга,
            # иначе оно молча потеряется и «auto» включится сам.
            self._sleep_auto_pending = self.var_sleep_auto.get()
            return
        self.brain.sleep_auto = bool(self.var_sleep_auto.get())
        self._sleep_say("автосон " + ("включён" if self.brain.sleep_auto
                                      else "ВЫКЛЮЧЕН: спать будет только кнопка"))

    def _set_sleep_busy(self, busy):
        """
        Кнопка вкладки. Её состояние выводится ИЗ jobs.busy, а не ставится
        вручную: две ручки у одного переключателя — это всегда расхождение
        (кнопка потухла, а работы нет, или наоборот).
        """
        try:
            self.sleep_now_btn.config(
                state="disabled" if busy else "normal",
                text="спит" if busy else "Спать сейчас")
        except (tk.TclError, AttributeError):
            # AttributeError — не мягкость: _set_buttons вызывается из _pump и
            # из _attach_brain, а вкладка сна строится в _build_ui. Окно без
            # собранного интерфейса (тесты, headless) обязано жить.
            pass

    def _build_bottom(self):
        self.ask_bar = tk.Frame(self.root, bg="#3a2b12")
        self.ask_lbl = tk.Label(self.ask_bar, text="", bg="#3a2b12", fg="#ffd479",
                                font=("DejaVu Sans", 9, "bold"), anchor="w",
                                justify="left", wraplength=760)
        self.ask_lbl.pack(side="left", fill="x", expand=True, padx=8, pady=4)
        tk.Button(self.ask_bar, text="Ответить", command=self._submit_answer,
                  bg=ACCENT, fg="#ffffff", relief="flat", bd=0, cursor="hand2",
                  padx=12, pady=2).pack(side="right", padx=(6, 2), pady=4)
        tk.Button(self.ask_bar, text="Не знаю", command=self._decline_answer,
                  bg="#555555", fg="#ffffff", relief="flat", bd=0, cursor="hand2",
                  padx=12, pady=2).pack(side="right", padx=2, pady=4)
        self.ask_entry = tk.Entry(self.ask_bar, bg="#241a08", fg="#ffd479",
                                  insertbackground="#ffd479", relief="flat",
                                  font=("DejaVu Sans", 10))
        self.ask_entry.pack(side="right", fill="x", expand=True, padx=6, ipady=3)
        self.ask_entry.bind("<Return>", lambda e: self._submit_answer())
        self.ask_entry.bind("<Escape>", lambda e: self._decline_answer())

        self.bottom = tk.Frame(self.root, bg=BG)
        self.bottom.pack(fill="x", padx=8, pady=(2, 4))
        self.entry_host = tk.Frame(self.bottom, bg=BG)
        self.entry_host.pack(fill="x")
        # Подсказка ЖИВЁТ ОТДЕЛЬНО ОТ ПОЛЯ. Первая версия прятала её в самом
        # Text (серый текст, стираемый по FocusIn) — и это был баг: текст
        # подсказки оставался в поле, пока событие фокуса не обработано, а
        # Enter в этот момент означал «пусто». Поле теперь всегда настоящее.
        self.hint_lbl = tk.Label(self.bottom, text="", bg=BG, fg=FG_DIM,
                                 font=("DejaVu Sans", 8), anchor="w", justify="left")
        self.hint_lbl.pack(fill="x")

        self.learn_entry = self._make_input(
            "одно слово · словосочетание · предложение · короткий текст."
            "   Enter — учить, Shift+Enter — новая строка")
        self.learn_btn = tk.Button(self.entry_host, text="…загрузка ядра",
                                   command=self.do_learn, bg=BG_TAB_ON, fg=FG,
                                   relief="flat", bd=0, cursor="hand2",
                                   font=("DejaVu Sans", 10, "bold"),
                                   padx=18, pady=6, state="disabled")

        self.sim_entry = self._make_input(
            "спроси о том, что уже в памяти: «что такое флюгер» · "
            "«свойства кошки» · «кошка и собака»")
        self.sim_btn = tk.Button(self.entry_host, text="Спросить",
                                 command=self.do_ask, bg=BG_TAB_ON, fg=FG,
                                 relief="flat", bd=0, cursor="hand2",
                                 font=("DejaVu Sans", 10, "bold"),
                                 padx=18, pady=6, state="disabled")

        self.story_entry = self._make_input(
            "вопрос о ПРИЧИНЕ в загруженном рассказе: «почему пожар погас» · "
            "«что будет, если лить воду» · «кто остановил пожар»")
        self.story_btn = tk.Button(self.entry_host, text="Спросить",
                                   command=self.do_ask_story, bg=BG_TAB_ON,
                                   fg=FG, relief="flat", bd=0, cursor="hand2",
                                   font=("DejaVu Sans", 10, "bold"),
                                   padx=18, pady=6)
        # Поле рассказа НЕ ждёт готовности ядра. _make_input создаёт поле
        # state="disabled", а включают его в _attach_brain — только learn и sim.
        # Рассказ отвечает из своего файла без SDR и без Учителя, поэтому
        # «поле серое, а ядро грузится 40 с» было бы враньём про зависимости.
        self.story_entry.config(state="normal")

        self.statusbar = tk.Label(self.root, text="стартую", anchor="w",

                                 bg=BG_TAB_ON, fg="#dcdcaa",
                                 font=("DejaVu Sans", 9), padx=8, pady=4)
        self.statusbar.pack(fill="x", side="bottom")

    def _make_input(self, hint: str) -> tk.Text:
        """Text, а не Entry: одно слово и абзац — один и тот же ввод, а Entry
        физически однострочный."""
        box = tk.Text(self.entry_host, height=2, wrap="word", bg=BG_INPUT, fg=FG,
                      insertbackground=FG, selectbackground=ACCENT, relief="flat",
                      bd=6, font=("DejaVu Sans", 10), state="disabled")
        box._hint = hint
        box.bind("<FocusIn>", self._entry_focus_in)
        box.bind("<FocusOut>", self._entry_focus_out)
        box.bind("<Return>", self._entry_return)
        box.bind("<Shift-Return>", lambda e: None)          # перевод строки
        return box

    def _entry_focus_in(self, event):
        self.clip.remember(event)
        self.hint_lbl.config(text=event.widget._hint)

    def _entry_focus_out(self, event):
        self.hint_lbl.config(text="")

    def _entry_return(self, event):
        # Развилка по АКТИВНОЙ вкладке, а не по «sim иначе learn»: с третьей
        # вкладкой прежнее if/else угоняло Enter рассказа в обучение.
        if self.tab == "sim":
            self.do_ask()
        elif self.tab == "story":
            self.do_ask_story()
        else:
            self.do_learn()
        return "break"

    def _entry_text(self, box) -> str:
        return box.get("1.0", "end").strip()

    # ══════════════════════════════════════════════════════
    #  ДЕЙСТВИЯ
    # ══════════════════════════════════════════════════════
    def _any_knowledge_channel(self) -> bool:
        """
        Искать знание вообще = ЛЮБОЙ из трёх каналов включён.

        ЗАМЕРЕНО (п.19): раньше сюда шёл общий флаг Учителя, и снятие
        «спрашивать Учителя» глушило и 3-й канал: при включённом
        «спрашивать меня» вопрос пользователю не дошёл НИ РАЗУ
        (вопросов 0, слов не спрошено 4). Конкретный КАНАЛ выбирает мозг по
        своим флагам; окно решает только «идти ли в поиск знания».
        """
        return bool(self.llm_enabled or self.wiki_enabled or self.ask_enabled)

    def do_learn(self):
        text = self._entry_text(self.learn_entry)
        if not text:
            self._pending_status = "пусто: введи слово, фразу или текст"
            return
        if not self._require_brain():
            return
        self.journal.add(f"обучаю: {text[:200]}", "user")
        self.learn_entry.delete("1.0", "end")
        allow_teacher = self._any_knowledge_channel()
        # Кнопка гасится СРАЗУ, не в следующей перекачке событий (до 120 мс):
        # двойной клик за это время — реальный сценарий, а второй поток полез бы
        # писать в тот же lipid_world.json.
        self._set_buttons(busy=True)

        def work():
            rep = self.brain.learn_from_text(text, source="gui",
                                             allow_teacher=allow_teacher)
            # Публикация отчёта и каналов — ДО журнала: прогонам нужны числа,
            # а не разбор текста (см. комментарий полей last_report).
            self.last_report = rep
            self.last_channel_calls = dict(getattr(self.brain, "channel_calls", {}) or {})
            self.last_channel_found = dict(getattr(self.brain, "channel_found", {}) or {})
            # Отчёт — в журнал окна: «выучил» без чисел невозможно проверить.
            self.journal.add(format_report(rep), "report")
            self._pending_status = summarize_report(rep)   # итог держится 8 с
            if not allow_teacher and rep.unknown:
                self.journal.add(
                    f"{len(rep.unknown)} незнакомых слов НЕ выяснялись: все "
                    f"три канала знания выключены (список — в unknown_words)",
                    "warn")
            if rep.learned_concepts:
                self.journal.add("выучено: " + ", ".join(rep.learned_concepts[:12]),
                                 "ok")

        self.jobs.start(work, label=f"обучение на {len(text)} симв.")

    def do_ask(self):
        text = self._entry_text(self.sim_entry)
        if not text:
            self._pending_status = "пусто: задай вопрос"
            return
        if not self._require_brain():
            return
        self._ui_q.put(("chat", ("ты", text, "user")))
        self.sim_entry.delete("1.0", "end")
        self._set_buttons(busy=True)

        def work():
            answer = self.brain.process_input(text)
            # ОТВЕТ — не отчёт. chat-виджет рисует реплику, журнал — строку
            # ответа своим цветом; отчёт обучения выглядит иначе (см. presenter).
            self._ui_q.put(("chat", ("Lipid-2", answer, KIND_ANSWER)))
            self.journal.add(*answer_row(answer))

        self.jobs.start(work, label=f"вопрос: {text[:60]}")

    # ══════════════════════════════════════════════════════
    #  ПОТОК ДАННЫХ: рабочие потоки -> буферы -> Tk
    # ══════════════════════════════════════════════════════
    def _on_status(self, message: str):
        """
        Подписчик канала статуса. Приходит ИЗ РАБОЧЕГО ПОТОКА, поэтому только
        пишет в потокобезопасный Journal — ни одного обращения к Tk.
        """
        kind = ("error" if ("⛔" in message or "ОТВЕРГНУТ" in message
                            or "НЕ ОТВЕТИЛ" in message)
                else "warn" if ("⚠" in message or "НЕ" in message) else "status")
        self.journal.add(message, kind)

    def _pump(self):
        """Единственный, кто пишет в Tk."""
        try:
            self._drain_queue()
        except Exception as exc:
            self.journal.add(f"⛔ обработчик очереди: {type(exc).__name__}: {exc}",
                             "error")

        rows = self.journal.rows()
        if len(rows) > self._shown:
            for row in rows[self._shown:]:
                self._append(self.log, *row)
            self._shown = len(rows)

        text = self.journal.last_status()
        if self._pending_status is not None:
            text = self._pending_status
            self._pending_status = None
            self._status_hold_until = time.time() + 8.0
        elif time.time() < self._status_hold_until and text.startswith(
                ("начал", "закончил", "●")):
            # служебные строки не имеют права затирать итог работы
            text = self.statusbar.cget("text")
        if text and self.statusbar.cget("text") != text:
            self.statusbar.config(text=text)

        # САМОПРОВЕРКА кнопок: если событие «освободился» не дошло (исключение
        # в обработчике очереди), кнопки остались бы серыми навсегда — а это
        # выглядит как «программа сломалась». Дешевле сверить состояние каждый кадр.
        want = "disabled" if self.jobs.busy else "normal"
        if self.learn_btn["state"] != want:
            self._set_buttons(busy=self.jobs.busy)
        self._refresh_hormones()
        self.root.after(120, self._pump)

    def _drain_queue(self):
        while True:
            try:
                kind, payload = self._ui_q.get_nowait()
            except queue.Empty:
                return
            if kind == "busy":
                self._set_buttons(busy=payload)
            elif kind == "brain":
                self._attach_brain(payload)
            elif kind == "encoder_ready":
                self._report_encoder(payload)
            elif kind == "kb":
                self._refresh_kb()
            elif kind == "chat":
                who, text, style = payload
                self._append(self.chat, time.time(), style, f"{who}: {text}")
            elif kind == "question":
                self._show_question(payload)
            elif kind == "story":
                self._after_story_load(*payload)
            elif kind == "story_chat":
                self._append(self.story_chat, time.time(), KIND_ANSWER,
                             payload)
            elif kind == "sleep_chat":
                self._append(self.sleep_chat, time.time(), KIND_ANSWER, payload)
                self._refresh_sleep_tab()
                self._refresh_kb()

    def _set_buttons(self, busy: bool):
        """
        Кнопки обязаны менять подпись: серая кнопка с надписью «Учить» неотличима
        от сломанной, а «…работает» объясняет, почему нажатие ничего не дало.
        """
        for btn, label in ((self.learn_btn, "Учить"),
                           (self.sim_btn, "Спросить"),
                           (self.story_btn, "Спросить")):
            try:
                btn.config(state="disabled" if busy else "normal",
                           text="…работает" if busy else label)
            except tk.TclError:
                pass
        # Кнопка сна гасится ТЕМ ЖЕ вызовом: сон идёт через jobs, и если бы
        # его состояние держалось отдельно, окно показывало бы «можно учить»
        # во время записи памяти.
        self._set_sleep_busy(busy)

    def _append(self, widget, ts, kind, text):
        """
        Журнал и чат остаются state=normal (см. make_readonly), поэтому вставка
        проходит, а пользовательский ввод отсечён привязками — переключать
        состояние туда-сюда не нужно.
        """
        widget.insert("end", time.strftime("%H:%M:%S ", time.localtime(ts)), "ts")
        widget.insert("end", str(text).rstrip() + "\n", kind)
        widget.see("end")

    def _refresh_kb(self):
        if self.brain is None:
            return
        try:
            world = self.brain._load_world()
            self.lbl_kb.config(
                text="память: сущностей {} · связей {} · неизвестных {}".format(
                    len(world.get("entities") or {}),
                    len(world.get("observations") or {}),
                    len(world.get("unknown_words") or {})))
        except Exception as exc:
            self.lbl_kb.config(
                text=f"память: НЕ ЧИТАЕТСЯ ({type(exc).__name__}: {exc})")

    def _refresh_hormones(self):
        brain = self.brain
        if brain is None:
            return
        hormones = getattr(brain, "hormones", None)
        vals = {"nor": getattr(brain, "noradrenaline", 0),
                "dop": getattr(brain, "dopamine", 0),
                "cor": getattr(hormones, "cortisol", 0)}
        for key, val in vals.items():
            bar, lbl = self.horm[key]
            try:
                bar["value"] = int(val)
                lbl.config(text=str(int(val)))
            except (tk.TclError, TypeError, ValueError):
                pass

    def _report_encoder(self, enc=None):
        """
        Какая SDR-схема работает РЕАЛЬНО — постоянно на виду, не в пролетающей
        строке. На буквенном хэше приложение выглядит здоровым и отвечает,
        но overlap несвязанных слов у него ВЫШЕ, чем у синонимов: узнавание и
        пороги 13/17 теряют смысл. Пользователь обязан видеть это сразу.
        """
        if enc is None:
            from core.encoder import get_encoder
            enc = get_encoder()
        try:
            name, semantic, cfg = encoder_info()
        except Exception as exc:
            self.lbl_encoder.config(text=f"SDR: НЕ ОПРЕДЕЛЁН ({exc})",
                                    fg="#ff6b6b")
            self.journal.add(f"⛔ не удалось определить SDR-кодировщик: {exc}",
                             "error")
            return
        if semantic:
            self.lbl_encoder.config(text=f"SDR: {name} · семантический · cfg {cfg}",
                                    fg="#4ecdc4")
            return
        from core.encoder import letterhash_reason
        self.lbl_encoder.config(text="⛔ SDR: БУКВЕННЫЙ ХЭШ — смысл НЕ кодируется",
                                fg="#ff6b6b")
        self.journal.add(
            "⛔ КРИТИЧНО: SDR на буквенном хэше.Overlap несвязанных слов выше, "
            "чем у синонимов — узнавание и пороги (13/17) бессмысленны.\n"
            f"   причина: {letterhash_reason()}\n"
            "   запускать так:  .venv/bin/python run.py", "error")
        self._pending_status = "⛔ SDR без смысла — запускай из .venv"

    # ══════════════════════════════════════════════════════
    #  3-й КАНАЛ: вопрос пользователю
    # ══════════════════════════════════════════════════════
    def _ask_user_via_bridge(self, concept: str):
        """
        Колбэк мозга. Вызывается из рабочего потока. Когда режим «спрашивать
        меня» выключен — отказ ЯВНЫЙ (None + строка в журнале): молчаливый отказ
        превращает канал в несуществующий, а это ровно тот класс багов, который
        проект считает ошибкой.
        """
        if not self.ask_enabled:
            self.journal.add("3-й канал выключен (галочка «спрашивать меня») — "
                             "вопрос не задан", "warn")
            return None
        return self.bridge.handler(
            f"Что такое «{concept}»? коротко: категория; свойства; действия")

    def _show_question(self, question: str):
        self.ask_bar.pack(fill="x", padx=8, pady=(0, 2), before=self.bottom)
        self.ask_lbl.config(text="❓ " + question)
        self.ask_entry.delete(0, "end")
        self.ask_entry.focus_set()
        self.journal.add(question, "ask")

    def _submit_answer(self):
        text = self.ask_entry.get().strip()
        self._hide_ask()
        if not text:
            self._decline_answer()
            return
        if self.bridge.submit(text):
            self.journal.add("ты: " + text, "user")
        else:
            self.journal.add("вопрос уже не актуален — ответ проигнорирован",
                             "warn")

    def _decline_answer(self):
        self._hide_ask()
        self.bridge.submit(None)
        self.journal.add("ответ не дан — 3-й канал по этому слову исчерпан", "warn")

    def _hide_ask(self):
        if self.ask_bar.winfo_ismapped():
            self.ask_bar.pack_forget()

    # ══════════════════════════════════════════════════════
    def on_close(self):
        """
        Закрытие отпускает ожидание. Иначе рабочий поток висит на AnswerBridge
        вечно: окно закрыто, а процесс жив — тихое зависание, которое ловится
        только «почему python не убивается».
        """
        try:
            unsubscribe(self._sub_id)
        except Exception:
            pass
        self.bridge.close()
        try:
            from core.encoder import get_encoder
            enc = get_encoder()
            if enc.semantic:
                enc.flush()          # битовый кэш обязан дожить до следующего старта
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    Lipid2GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
