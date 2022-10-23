import os
import tkinter as tk
from tkinter import *
import definitons
from PIL import Image
from PIL import ImageTk

from utils.gui.gui_utils import center_window


class LoggedWindow:

    def __init__(self, master, main_window, is_successful):
        self.master = master
        self.main_window = main_window
        self.frame = tk.Frame(self.master, width=300, height=300)
        center_window(master, 300, 300)
        master.iconbitmap(os.getcwd() + '\\images\\app_images\\icon.ico')

        if is_successful:
            f = open(os.getcwd() + '\\utils\\username.txt', "r")
            master.title("Welcome back, " + f.read())
            img = Image.open(definitons.app_images_dir + 'logged.png')
        else:
            master.title("Unknown person")
            img = Image.open(definitons.app_images_dir + 'error.png')
        self.tk_image = ImageTk.PhotoImage(img)
        Label(self.master, image=self.tk_image).place(x=0, y=0, relwidth=1, relheight=1)

        master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.master.destroy()
        self.main_window.destroy()


def show_logged(self, main_window):
    self.master.withdraw()
    self.newWindow = tk.Toplevel(self.master)
    self.app = LoggedWindow(self.newWindow, main_window, True)
