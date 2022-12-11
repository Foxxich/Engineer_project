from functools import partial
from tkinter import *
from tkinter import ttk

import definitons
from utils.gui_utils import center_window, close_algorithm_window

cnn_parameters = []


def checkout(parameters, label):
    for parameter in cnn_parameters:
        if label in parameter[0]:
            cnn_parameters.remove(parameter)
    cnn_parameters.append([label, parameters])


class CnnWindow:
    def __init__(self, window, window_title, main_window, results):
        self.app = None
        self.photo = None
        self.window = window
        window.iconbitmap(definitons.app_images_dir + '\\icon.ico')
        center_window(window, 640, 520)
        self.var1 = IntVar(value=0)
        self.var2 = IntVar(value=0)
        self.var3 = IntVar(value=0)
        self.var4 = IntVar(value=0)
        self.var5 = IntVar(value=0)
        self.results = results
        self.main_window = main_window
        self.window.title(window_title)
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)

        x = 10
        x_coordinate = 120

        Label(self.window, text="Loss (crossentropy)").place(x=x, y=60)
        Radiobutton(self.window, text="binary", variable=self.var1, value=1,
                    command=partial(checkout, "binary", "loss")).place(x=x_coordinate, y=60)
        Radiobutton(self.window, text="categorical", variable=self.var1, value=2,
                    command=partial(checkout, "categorical", "loss")).place(x=x_coordinate + 85, y=60)

        Label(self.window, text="Metrics (accuracy)").place(x=x, y=80)
        Radiobutton(self.window, text="accuracy", variable=self.var2, value=1,
                    command=partial(checkout, "accuracy", "metrics")).place(x=x_coordinate, y=80)
        Radiobutton(self.window, text="binary", variable=self.var2, value=2,
                    command=partial(checkout, "binary", "metrics")).place(x=x_coordinate + 85, y=80)
        Radiobutton(self.window, text="categorical", variable=self.var2, value=3,
                    command=partial(checkout, "categorical", "metrics")).place(x=x_coordinate + 85 * 2, y=80)
        Radiobutton(self.window, text="top_k_categorical", variable=self.var2, value=4,
                    command=partial(checkout, "top_k_categorical", "metrics")).place(x=x_coordinate + 85 * 3, y=80)

        Label(self.window, text="Optimizers").place(x=x, y=100)
        Radiobutton(self.window, text="adam", variable=self.var3, value=1,
                    command=partial(checkout, "adam", "optimizer")).place(x=x_coordinate, y=100)
        Radiobutton(self.window, text="rmsprop", variable=self.var3, value=2,
                    command=partial(checkout, "rmsprop", "optimizer")).place(x=x_coordinate + 85, y=100)
        Radiobutton(self.window, text="Ftrl", variable=self.var3, value=3,
                    command=partial(checkout, "Ftrl", "optimizer")).place(x=x_coordinate + 85 * 2, y=100)
        Radiobutton(self.window, text="Nadam", variable=self.var3, value=4,
                    command=partial(checkout, "Nadam", "optimizer")).place(x=x_coordinate + 85 * 3, y=100)
        Radiobutton(self.window, text="Adamax", variable=self.var3, value=5,
                    command=partial(checkout, "Adamax", "optimizer")).place(x=x_coordinate + 85 * 4, y=100)

        Label(self.window, text="Epochs number").place(x=x, y=120)
        Radiobutton(self.window, text="10", variable=self.var4, value=1,
                    command=partial(checkout, "10", "epochs_number")).place(x=x_coordinate, y=120)
        Radiobutton(self.window, text="15", variable=self.var4, value=2,
                    command=partial(checkout, "15", "epochs_number")).place(x=x_coordinate + 85, y=120)
        Radiobutton(self.window, text="20", variable=self.var4, value=3,
                    command=partial(checkout, "20", "epochs_number")).place(x=x_coordinate + 85 * 2, y=120)
        Radiobutton(self.window, text="45", variable=self.var4, value=4,
                    command=partial(checkout, "45", "epochs_number")).place(x=x_coordinate + 85 * 3, y=120)
        Radiobutton(self.window, text="50", variable=self.var4, value=5,
                    command=partial(checkout, "50", "epochs_number")).place(x=x_coordinate + 85 * 4, y=120)

        Label(self.window, text="Steps for validation").place(x=x, y=140)
        Radiobutton(self.window, text="10", variable=self.var5, value=1,
                    command=partial(checkout, "10", "steps_for_validation")).place(x=x_coordinate, y=140)
        Radiobutton(self.window, text="9", variable=self.var5, value=2,
                    command=partial(checkout, "9", "steps_for_validation")).place(x=x_coordinate + 85, y=140)
        Radiobutton(self.window, text="8", variable=self.var5, value=3,
                    command=partial(checkout, "8", "steps_for_validation")).place(x=x_coordinate + 85 * 2, y=140)
        Radiobutton(self.window, text="7", variable=self.var5, value=4,
                    command=partial(checkout, "7", "steps_for_validation")).place(x=x_coordinate + 85 * 3, y=140)
        Radiobutton(self.window, text="6", variable=self.var5, value=5,
                    command=partial(checkout, "6", "steps_for_validation")).place(x=x_coordinate + 85 * 4, y=140)

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
        close_algorithm_window(cnn_parameters, self.main_window, self.window, self.results, 'cnn')
