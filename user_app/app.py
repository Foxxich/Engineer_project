import tkinter as tk

import definitons
from user_app.gui.main_window import MainWindow


def main():
    root = tk.Tk()
    root.iconbitmap(definitons.app_images_dir + '\\icon.ico')
    root.title("Face recognition")
    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
