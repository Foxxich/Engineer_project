from functools import partial
from tkinter import *
from tkinter import ttk

import definitons
from utils.gui_utils import center_window, close_algorithm_window

vgg_parameters = []


def checkout(parameters, label):
    for parameter in vgg_parameters:
        if label in parameter[0]:
            vgg_parameters.remove(parameter)
    vgg_parameters.append([label, parameters])


class VggWindow:
    def __init__(self, window, window_title, main_window, results):
        self.app = None
        self.photo = None
        self.window = window
        window.iconbitmap(definitons.app_images_dir + '\\icon.ico')
        center_window(window, 640, 520)
        self.var1 = IntVar(value=0)
        self.var2 = IntVar(value=0)
        self.results = results
        self.main_window = main_window
        self.window.title(window_title)
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)

        x_coordinate = 50
        x = 10

        Label(self.window, text="Thresh").place(x=x, y=60)
        Radiobutton(self.window, text="0.3", variable=self.var1, value=1,
                    command=partial(checkout, "0.3", "thresh")).place(x=x_coordinate, y=60)
        Radiobutton(self.window, text="0.4", variable=self.var1, value=2,
                    command=partial(checkout, "0.4", "thresh")).place(x=x_coordinate + 85, y=60)
        Radiobutton(self.window, text="0.5", variable=self.var1, value=3,
                    command=partial(checkout, "0.5", "thresh")).place(x=x_coordinate + 85 * 2, y=60)
        Radiobutton(self.window, text="0.6", variable=self.var1, value=4,
                    command=partial(checkout, "0.6", "thresh")).place(x=x_coordinate + 85 * 3, y=60)
        Radiobutton(self.window, text="0.7", variable=self.var1, value=5,
                    command=partial(checkout, "0.7", "thresh")).place(x=x_coordinate + 85 * 4, y=60)

        Label(self.window, text="Model").place(x=x, y=80)
        Radiobutton(self.window, text="resnet50", variable=self.var2, value=1,
                    command=partial(checkout, "resnet50", "model")).place(x=x_coordinate, y=80)
        Radiobutton(self.window, text="vgg16", variable=self.var2, value=2,
                    command=partial(checkout, "vgg16", "model")).place(x=x_coordinate + 85, y=80)
        Radiobutton(self.window, text="senet50", variable=self.var2, value=3,
                    command=partial(checkout, "senet50", "model")).place(x=x_coordinate + 85 * 2, y=80)

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
        close_algorithm_window(vgg_parameters, self.main_window, self.window, self.results, 'vgg')
