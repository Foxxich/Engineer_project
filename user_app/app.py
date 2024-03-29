import tkinter as tk

import definitons
from user_app.gui.main_window import MainWindow
from utils.gui_utils import center_window


def main():
    root = tk.Tk()
    center_window(root, 400, 350)
    root.iconbitmap(definitons.app_images_dir + '\\icon.ico')
    root.title("Face recognition")
    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
