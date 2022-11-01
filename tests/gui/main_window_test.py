import tkinter as tk
from tkinter import *
from sys import exit
from tkinter import messagebox

from tests.gui.cnn_window import CnnWindow
from tests.gui.pca_window import PcaWindow
from tests.gui.results import Results
from tests.gui.save_window import SaveWindow
from tests.gui.sift_window import SiftWindow
from tests.gui.vgg_window import VggWindow

results = Results()


class TestMainWindow:
    def __init__(self, master):
        self.app = None
        self.newWindow = None
        self.master = master
        self.frame = tk.Frame(self.master, width=200, height=300)
        self.master.protocol("WM_DELETE_WINDOW", self.close_window)
        self.frame.size()
        check_var1 = IntVar()
        check_var2 = IntVar()
        check_var3 = IntVar()
        check_var4 = IntVar()
        cnn_check = Checkbutton(text="CNN", variable=check_var1,
                                onvalue=1, offvalue=0, height=3,
                                width=20, command=self.cnn_checkbox)
        pca_check = Checkbutton(text="PCA", variable=check_var2,
                                onvalue=1, offvalue=0, height=3,
                                width=20, command=self.pca_checkbox)
        vgg_check = Checkbutton(text="VGG", variable=check_var3,
                                onvalue=1, offvalue=0, height=3,
                                width=20, command=self.vgg_checkbox)
        sift_check = Checkbutton(text="SIFT", variable=check_var4,
                                 onvalue=1, offvalue=0, height=3,
                                 width=20, command=self.sift_checkbox)
        cnn_check.pack()
        pca_check.pack()
        vgg_check.pack()
        sift_check.pack()

        self.button = tk.Button(self.frame, text='Run testing', height=5, width=20, command=self.save_location_window)
        self.button.pack(padx=50, pady=50)
        self.frame.pack()

    def close_window(self):
        self.master.destroy()
        exit()

    def cnn_checkbox(self):
        if len(results.get_cnn()) != 5:
            self.master.withdraw()
            self.newWindow = tk.Toplevel(self.master)
            self.app = CnnWindow(self.newWindow, 'CNN parameters', self.master, results)
        else:
            messagebox.showerror(title=None, message='You have chosen parameters for CNN')

    def vgg_checkbox(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = VggWindow(self.newWindow, 'VGG parameters', self.master, True)

    def sift_checkbox(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = SiftWindow(self.newWindow, 'SIFT parameters', self.master, True)

    def pca_checkbox(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = PcaWindow(self.newWindow, 'PCA Parameters', self.master, True)

    def save_location_window(self):
        self.master.withdraw()
        self.newWindow = tk.Toplevel(self.master)
        self.app = SaveWindow(self.newWindow, 'Tests execution', results)
