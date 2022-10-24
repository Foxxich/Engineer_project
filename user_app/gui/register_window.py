import tkinter as tk
from functools import partial
from tkinter import *

import definitons
from user_app.gui.app_window import App
from user_app.gui.gui_utils import center_window


class RegisterWindow:

    def validate_login(self, username):
        print("username entered :", username.get())
        with open(definitons.root_dir + '\\utils\\username.txt', 'w') as f:
            f.write(username.get())
        self.newWindow = tk.Toplevel(self.main_window)
        self.app = App(self.newWindow, 'Take image to login', self.main_window, False)
        self.master.destroy()
        return

    def __init__(self, master, main_window):
        self.app = None
        self.newWindow = None
        self.master = master
        self.main_window = main_window
        self.frame = tk.Frame(self.master)
        master.title('Create account')
        center_window(master, 250, 250)
        master.iconbitmap(definitons.app_images_dir + '\\icon.ico')

        Label(master, text="User Name").grid(row=1, column=1, padx=10, pady=10)
        username = StringVar()
        Entry(master, textvariable=username).grid(row=1, column=2, padx=10, pady=10)

        validate_login = partial(self.validate_login, username)

        Button(master, text="Login", command=validate_login).grid(row=4, column=1, padx=10, pady=10)

    def close_windows(self):
        self.master.destroy()
