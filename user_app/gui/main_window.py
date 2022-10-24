import tkinter as tk

from user_app.gui.app_window import App
from user_app.gui.register_window import RegisterWindow


class MainWindow:
    def __init__(self, master):
        self.app = None
        self.newWindow = None
        self.master = master
        self.frame = tk.Frame(self.master, width=200, height=300)
        self.frame.size()
        self.button1 = tk.Button(self.frame, text='Login', height="2", width="30", command=self.login_window)
        self.button1.pack(padx=50, pady=50)
        self.button2 = tk.Button(self.frame, text='Register', height="2", width="30", command=self.register_window)
        self.button2.pack(padx=50, pady=50)
        self.frame.pack()

    def login_window(self):
        self.master.withdraw()
        self.newWindow = tk.Toplevel(self.master)
        self.app = App(self.newWindow, 'Take image to login', self.master, True)

    def register_window(self):
        self.master.withdraw()
        self.newWindow = tk.Toplevel(self.master)
        self.app = RegisterWindow(self.newWindow, self.master)
