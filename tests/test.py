import definitons
import tkinter as tk

from tests.gui.main_window_test import TestMainWindow
from utils.gui_utils import center_window


def main():
    root = tk.Tk()
    center_window(root, 400, 350)
    root.iconbitmap(definitons.app_images_dir + '\\icon.ico')
    root.title("Algorithms testing")
    TestMainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
