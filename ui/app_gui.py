import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
from brain.lipid2_brain import Lipid2Brain
from core.sdr import encoder_info
from core.encoder import letterhash_reason


class Lipid2GUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Lipid-2 :: Neuromorphic Control Panel")
        self.root.geometry("850x600")
        self.root.configure(bg="#2d2d2d")

        # Инициализация Мозга
        self.brain = Lipid2Brain()

        self._init_context_menu()
        self._build_ui()
        self._update_hormone_bars()
        self._report_encoder()

    def _init_context_menu(self):
        """Создание универсального контекстного меню (ПКМ) для копипаста."""
        self.context_menu = tk.Menu(self.root, tearoff=0, bg="#333333", fg="#ffffff", activebackground="#ff4081", activeforeground="#ffffff")
        self.context_menu.add_command(label="Копировать", command=lambda: self.root.focus_get().event_generate("<<Copy>>"))
        self.context_menu.add_command(label="Вставить", command=lambda: self.root.focus_get().event_generate("<<Paste>>"))
        self.context_menu.add_command(label="Вырезать", command=lambda: self.root.focus_get().event_generate("<<Cut>>"))
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Выделить всё", command=lambda: self.root.focus_get().event_generate("<<SelectAll>>"))

    def _show_context_menu(self, event):
        """Показ меню по ПКМ."""
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def _build_ui(self):
        # --- СТИЛИ И ЦВЕТОВАЯ СХЕМА ---
        style = ttk.Style()
        style.theme_use('default')

        # Базовые тёмные цвета
        bg_dark = "#2d2d2d"
        fg_white = "#ffffff"
        accent_pink = "#ff4081"

        style.configure("TFrame", background=bg_dark)
        style.configure("TLabelframetext", foreground=fg_white, background=bg_dark, font=("Arial", 9, "bold"))
        style.configure("TLabelframe", background=bg_dark, bordercolor="#444444")
        
        # Настройка чекбоксов: убираем белый ховер и белые фоны
        style.configure("TCheckbutton", 
                        background=bg_dark, 
                        foreground=fg_white, 
                        font=("Arial", 9))
        
        style.map("TCheckbutton",
                  background=[('active', bg_dark), ('selected', bg_dark)],
                  foreground=[('active', accent_pink), ('selected', fg_white)],
                  indicatorbackground=[('selected', accent_pink), ('pressed', accent_pink), ('!selected', '#444444')],
                  indicatorforeground=[('selected', fg_white)])

        # --- ВЕРХНЯЯ ПАНЕЛЬ: Гормональный дашборд ---
        hormone_frame = ttk.LabelFrame(self.root, text=" Гомеостаз / Гормоны ", padding=10)
        hormone_frame.pack(fill="x", padx=10, pady=5)

        # Норадреналин
        ttk.Label(hormone_frame, text="Норадреналин (Тревога/Новизна):", foreground="#ff6b6b", background=bg_dark).grid(row=0, column=0, sticky="w", padx=5)
        self.nor_bar = ttk.Progressbar(hormone_frame, length=200, maximum=255)
        self.nor_bar.grid(row=0, column=1, padx=5, pady=2)
        self.nor_lbl = ttk.Label(hormone_frame, text="0", foreground=fg_white, background=bg_dark)
        self.nor_lbl.grid(row=0, column=2, padx=5)

        # Дофамин
        ttk.Label(hormone_frame, text="Дофамин (Удовлетворение):", foreground="#4ecdc4", background=bg_dark).grid(row=1, column=0, sticky="w", padx=5)
        self.dop_bar = ttk.Progressbar(hormone_frame, length=200, maximum=255)
        self.dop_bar.grid(row=1, column=1, padx=5, pady=2)
        self.dop_lbl = ttk.Label(hormone_frame, text="0", foreground=fg_white, background=bg_dark)
        self.dop_lbl.grid(row=1, column=2, padx=5)

        # --- СРЕДНЯЯ ПАНЕЛЬ: Чат и Чекбоксы источников ---
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Область вывода чата
        self.chat_area = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, bg="#1e1e1e", fg="#e0e0e0", 
            font=("Consolas", 10), insertbackground="white", selectbackground="#ff4081"
        )
        self.chat_area.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.chat_area.config(state="disabled")
        
        # Привязка ПКМ к чату
        self.chat_area.bind("<Button-3>", self._show_context_menu)

        # Панель режимов обучения справа
        control_frame = ttk.LabelFrame(main_frame, text=" Источники обучения ", padding=10)
        control_frame.pack(side="right", fill="y")

        self.mode_user = tk.BooleanVar(value=True)
        self.mode_llm = tk.BooleanVar(value=True)
        self.mode_wiki = tk.BooleanVar(value=False)

        chk_user = ttk.Checkbutton(control_frame, text="Прямой ввод (User)", variable=self.mode_user)
        chk_llm = ttk.Checkbutton(control_frame, text="Локальный Учитель (LLM)", variable=self.mode_llm)
        chk_wiki = ttk.Checkbutton(control_frame, text="Википедия Парсер", variable=self.mode_wiki)

        chk_user.pack(anchor="w", pady=5)
        chk_llm.pack(anchor="w", pady=5)
        chk_wiki.pack(anchor="w", pady=5)

        # --- НИЖНЯЯ ПАНЕЛЬ: Ввод текста ---
        input_frame = ttk.Frame(self.root)
        input_frame.pack(fill="x", padx=10, pady=10)

        self.entry_box = tk.Entry(
            input_frame, bg="#333333", fg="#ffffff", font=("Arial", 11), 
            insertbackground="white", selectbackground="#ff4081", relief="flat", bd=5
        )
        self.entry_box.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.entry_box.bind("<Return>", self._send_message)
        
        # Привязка ПКМ к полю ввода
        self.entry_box.bind("<Button-3>", self._show_context_menu)

        btn_send = tk.Button(
            input_frame, text="Отправить", bg=accent_pink, fg="white", 
            font=("Arial", 10, "bold"), activebackground="#f50057", activeforeground="white",
            relief="flat", cursor="hand2", command=self._send_message
        )
        btn_send.pack(side="right")

    def _append_chat(self, sender: str, text: str):
        self.chat_area.config(state="normal")
        self.chat_area.insert(tk.END, f"{sender}: {text}\n\n")
        self.chat_area.see(tk.END)
        self.chat_area.config(state="disabled")

    def _report_encoder(self):
        """
        какая SDR-схема реально работает — это НЕ косметика.

        На буквенном хэше приложение выглядит здоровым, но смысл не кодируется,
        и пользователь узнаёт об этом только по бессмысленным ответам. Предупреждение
        выводится в то место, куда пользователь смотрит.
        """
        try:
            name, semantic, cfg = encoder_info()
        except Exception as exc:
            self._append_chat("Lipid-2", f"⚠️ Не удалось определить SDR-кодировщик: {exc}")
            return
        if semantic:
            self._append_chat("LIPID-2", f"SDR: {name} (семантический, cfg={cfg}). Готов.")
        else:
            self._append_chat(
                "LIPID-2",
                "⛔ КРИТИЧНО: SDR работает на БУКВЕННОМ ХЭШЕ — смысл НЕ кодируется, "
                "узнавание и пороги бессмысленны.\n"
                f"   Причина: {letterhash_reason()}")

    def _update_hormone_bars(self):
        """Синхронизация полос гормонов с состоянием Мозга."""
        self.nor_bar["value"] = self.brain.noradrenaline
        self.nor_lbl.config(text=str(self.brain.noradrenaline))

        self.dop_bar["value"] = self.brain.dopamine
        self.dop_lbl.config(text=str(self.brain.dopamine))

        self.root.after(200, self._update_hormone_bars)

    def _send_message(self, event=None):
        query = self.entry_box.get().strip()
        if not query:
            return

        self.entry_box.delete(0, tk.END)
        self._append_chat("Вы", query)

        def worker():
            # try/except ОБЯЗАТЕЛЕН: без него любое исключение из мозга (LLM
            # отвалился, файл памяти побит) молча убивает поток, и пользователь
            # остаётся с отправленным запросом и НИКАКИМ ответом — выглядит как
            # зависание, а не как ошибка.
            try:
                response = self.brain.process_input(query)
            except Exception as exc:
                response = (f"⛔ Сбой обработки: {type(exc).__name__}: {exc}\n"
                            "   Подробности в консоли, запущенной из терминала.")
            self.root.after(0, lambda: self._append_chat("Lipid-2", response))

        threading.Thread(target=worker, daemon=True).start()


def main():
    root = tk.Tk()
    app = Lipid2GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()