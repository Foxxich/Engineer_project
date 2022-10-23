import os
import tkinter as tk

from utils.gui.gui_utils import center_window
from utils.gui.main_window import MainWindow


def main():
    root = tk.Tk()
    center_window(root, 400, 350)
    root.iconbitmap(os.getcwd() + '\\images\\app_images\\icon.ico')
    root.title("Face recognition")
    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
