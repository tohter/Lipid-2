import sys
import os
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.app_gui import Lipid2GUI

def main():
    root = tk.Tk()
    app = Lipid2GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()