import tkinter as tk
from tkinter import *

from PIL import Image
from PIL import ImageTk
from sys import exit

import definitons
from user_app.model.username_reader import read_username
from utils.gui_utils import center_window


class LoggedWindow:

    def __init__(self, master, main_window, is_successful):
        self.master = master
        self.main_window = main_window
        center_window(master, 300, 300)
        self.frame = tk.Frame(self.master, width=300, height=300)
        master.iconbitmap(definitons.app_images_dir + '\\icon.ico')

        if is_successful == 'success':
            master.title("Welcome back, " + read_username())
            img = Image.open(definitons.app_images_dir + 'logged.png')
        elif is_successful == 'fail':
            master.title("Unknown person")
            img = Image.open(definitons.app_images_dir + 'error.png')
        else:
            master.title("You are registered, " + read_username())
            img = Image.open(definitons.app_images_dir + 'registered.jpg')
        self.tk_image = ImageTk.PhotoImage(img.resize((300, 300), Image.ANTIALIAS))
        Label(self.master, image=self.tk_image).place(x=0, y=0, relwidth=1, relheight=1)

        master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.master.destroy()
        self.main_window.destroy()
        exit()


def show_logged(self, main_window):
    self.master.withdraw()
    self.newWindow = tk.Toplevel(self.master)
    self.app = LoggedWindow(self.newWindow, main_window, 'initial')
