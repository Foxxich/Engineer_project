from functools import partial
from tkinter import *
from tkinter import ttk

import definitons
from utils.gui_utils import center_window, close_algorithm_window

sift_parameters = []


def checkout(parameters, label):
    for parameter in sift_parameters:
        if label in parameter[0]:
            sift_parameters.remove(parameter)
    sift_parameters.append([label, parameters])


class SiftWindow:
    def __init__(self, window, window_title, main_window, results):
        self.app = None
        self.photo = None
        self.window = window
        self.var1 = IntVar(value=0)
        self.var2 = IntVar(value=0)
        window.iconbitmap(definitons.app_images_dir + '\\icon.ico')
        center_window(window, 640, 520)
        self.results = results
        self.main_window = main_window
        self.window.title(window_title)
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)

        x = 10
        x_coordinate = 120

        Label(self.window, text="Cascades for face").place(x=x, y=60)
        Radiobutton(self.window, text="default", variable=self.var1, value=1,
                    command=partial(checkout, "default", "cascades")).place(x=x_coordinate, y=60)
        Radiobutton(self.window, text="alt", variable=self.var1, value=2,
                    command=partial(checkout, "alt", "cascades")).place(x=x_coordinate + 85, y=60)
        Radiobutton(self.window, text="alt_tree", variable=self.var1, value=3,
                    command=partial(checkout, "alt_tree", "cascades")).place(x=x_coordinate + 85 * 2, y=60)
        Radiobutton(self.window, text="alt2", variable=self.var1, value=4,
                    command=partial(checkout, "alt2", "cascades")).place(x=x_coordinate + 85 * 3, y=60)

        Label(self.window, text="Percentage").place(x=x, y=80)
        Radiobutton(self.window, text="1.0", variable=self.var2, value=1,
                    command=partial(checkout, "1.0", "percentage")).place(x=x_coordinate, y=80)
        Radiobutton(self.window, text="2.0", variable=self.var2, value=2,
                    command=partial(checkout, "2.0", "percentage")).place(x=x_coordinate + 85, y=80)
        Radiobutton(self.window, text="3.0", variable=self.var2, value=3,
                    command=partial(checkout, "3.0", "percentage")).place(x=x_coordinate + 85 * 2, y=80)
        Radiobutton(self.window, text="4.0", variable=self.var2, value=4,
                    command=partial(checkout, "4.0", "percentage")).place(x=x_coordinate + 85 * 3, y=80)
        Radiobutton(self.window, text="5.0", variable=self.var2, value=5,
                    command=partial(checkout, "5.0", "percentage")).place(x=x_coordinate + 85 * 2, y=80)
        Radiobutton(self.window, text="10.0", variable=self.var2, value=6,
                    command=partial(checkout, "10.0", "percentage")).place(x=x_coordinate + 85 * 3, y=80)

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
        close_algorithm_window(sift_parameters, self.main_window, self.window, self.results, 'sift')
