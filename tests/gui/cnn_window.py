from functools import partial
from tkinter import *
from tkinter import messagebox
from tkinter import ttk

import definitons
from utils.gui.gui_utils import center_window

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
        self.results = results
        self.main_window = main_window
        self.window.title(window_title)
        self.window.protocol("WM_DELETE_WINDOW", self.disable_event)

        x = 10

        Label(self.window, text="Loss (crossentropy)").place(x=x, y=60)

        labels = {'binary': 'loss', 'categorical': 'loss'}
        x_coordinate = 50
        for option in labels:
            x_coordinate += 85
            self.CheckVar = IntVar(value=0)
            parameter_validator = partial(checkout, option, labels[option])
            Checkbutton(self.window, text=option, command=parameter_validator, variable=self.CheckVar).place(
                x=x_coordinate, y=60)

        Label(self.window, text="Metrics (accuracy)").place(x=x, y=80)
        x_coordinate = 50
        labels = {'accuracy': 'metrics', 'binary': 'metrics', 'categorical': 'metrics', 'top_k_categorical': 'metrics'}
        for option in labels:
            x_coordinate += 85
            self.CheckVar = IntVar(value=0)
            parameter_validator = partial(checkout, option, labels[option])
            Checkbutton(self.window, text=option, command=parameter_validator, variable=self.CheckVar).place(
                x=x_coordinate, y=80)

        Label(self.window, text="Optimizers").place(x=x, y=100)
        x_coordinate = 50
        labels = {'adam': 'optimizer', 'rmsprop': 'optimizer', 'Ftrl': 'optimizer', 'Nadam': 'optimizer',
                  'Adamax': 'optimizer'}
        for option in labels:
            x_coordinate += 85
            self.CheckVar = IntVar(value=0)
            parameter_validator = partial(checkout, option, labels[option])
            Checkbutton(self.window, text=option, command=parameter_validator, variable=self.CheckVar).place(
                x=x_coordinate, y=100)

        Label(self.window, text="Epochs number").place(x=x, y=120)
        x_coordinate = 50
        labels = {'10': 'epochs_number', '15': 'epochs_number', '20': 'epochs_number', '45': 'epochs_number',
                  '50': 'epochs_number'}
        for option in labels:
            x_coordinate += 85
            self.CheckVar = IntVar(value=0)
            parameter_validator = partial(checkout, option, labels[option])
            Checkbutton(self.window, text=option, command=parameter_validator, variable=self.CheckVar).place(
                x=x_coordinate, y=120)

        Label(self.window, text="Steps for validation").place(x=x, y=140)
        x_coordinate = 50
        labels = {'10': 'steps_for_validation', '9': 'steps_for_validation', '8': 'steps_for_validation',
                  '7': 'steps_for_validation', '6': 'steps_for_validation'}
        for option in labels:
            x_coordinate += 85
            self.CheckVar = IntVar(value=0)
            parameter_validator = partial(checkout, option, labels[option])
            Checkbutton(self.window, text=option, command=parameter_validator, variable=self.CheckVar).place(
                x=x_coordinate, y=140)

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
        if len(cnn_parameters) == 5:
            self.results.set_cnn(cnn_parameters)
            self.window.destroy()
            self.main_window.deiconify()
        else:
            messagebox.showerror(title=None, message='Choose all the parameters!')

    def disable_event(self):
        pass
