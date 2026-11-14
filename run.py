import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.app_gui import Lipid2GUI, _DND_Tk


def main():
    # Корень окна — TkinterDnD.Tk, если tkdnd доступен (в .venv он есть).
    # Это позволяет ПЕРЕТАСКИВАТЬ текстовый файл мышью в поле обучения.
    # Если библиотеки нет, используется обычный tk.Tk и кнопка «+» остаётся.
    root = _DND_Tk()
    app = Lipid2GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()