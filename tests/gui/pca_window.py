from functools import partial
from tkinter import *
from tkinter import ttk

import definitons
from utils.gui_utils import center_window, close_algorithm_window

pca_parameters = []


def checkout(parameters, label):
    for parameter in pca_parameters:
        if label in parameter[0]:
            pca_parameters.remove(parameter)
    pca_parameters.append([label, parameters])


class PcaWindow:
    def __init__(self, window, window_title, main_window, results):
        self.app = None
        self.photo = None
        self.window = window
        window.iconbitmap(definitons.app_images_dir + '\\icon.ico')
        center_window(window, 640, 520)
        self.results = results
        self.var1 = IntVar(value=0)
        self.main_window = main_window
        self.window.title(window_title)
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)

        x = 10
        x_coordinate = 140

        Label(self.window, text="Components number").place(x=x, y=60)
        Radiobutton(self.window, text="60", variable=self.var1, value=1,
                    command=partial(checkout, "60", "components")).place(x=x_coordinate, y=60)
        Radiobutton(self.window, text="70", variable=self.var1, value=2,
                    command=partial(checkout, "70", "components")).place(x=x_coordinate + 85, y=60)
        Radiobutton(self.window, text="80", variable=self.var1, value=3,
                    command=partial(checkout, "80", "components")).place(x=x_coordinate + 85 * 2, y=60)
        Radiobutton(self.window, text="90", variable=self.var1, value=4,
                    command=partial(checkout, "0.6", "components")).place(x=x_coordinate + 85 * 3, y=60)
        Radiobutton(self.window, text="100", variable=self.var1, value=5,
                    command=partial(checkout, "100", "components")).place(x=x_coordinate + 85 * 4, y=60)

        exit_button = ttk.Button(
            self.window,
            text='Save parameters and exit',
            command=lambda: self.close_window()
        )

        exit_button.pack(
            ipadx=5,
            ipady=5,
            expand=True
        )

    def close_window(self):
        close_algorithm_window(pca_parameters, self.main_window, self.window, self.results, 'pca')

