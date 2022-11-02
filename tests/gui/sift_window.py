from functools import partial
from tkinter import *
from tkinter import messagebox
from tkinter import ttk

import definitons
from utils.gui.gui_utils import center_window

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
        window.iconbitmap(definitons.app_images_dir + '\\icon.ico')
        center_window(window, 640, 520)
        self.results = results
        self.main_window = main_window
        self.window.title(window_title)
        self.window.protocol("WM_DELETE_WINDOW", self.disable_event)

        x = 10

        Label(self.window, text="Cascades for face").place(x=x, y=60)
        labels = {'default': 'cascades', 'alt': 'cascades', 'alt_tree': 'cascades', 'alt2': 'cascades'}
        x_coordinate = 50
        for option in labels:
            x_coordinate += 85
            self.CheckVar = IntVar(value=0)
            parameter_validator = partial(checkout, option, labels[option])
            Checkbutton(self.window, text=option, command=parameter_validator, variable=self.CheckVar).place(
                x=x_coordinate, y=60)

        Label(self.window, text="Percentage").place(x=x, y=80)
        x_coordinate = 50
        labels = {'1.0': 'percentage', '2.0': 'percentage', '3.0': 'percentage', '4.0': 'percentage',
                  '5.0': 'percentage', '10.0': 'percentage'}
        for option in labels:
            x_coordinate += 85
            self.CheckVar = IntVar(value=0)
            parameter_validator = partial(checkout, option, labels[option])
            Checkbutton(self.window, text=option, command=parameter_validator, variable=self.CheckVar).place(
                x=x_coordinate, y=80)

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
        if len(sift_parameters) == 2:
            self.results.set_sift(sift_parameters)
            self.window.destroy()
            self.main_window.deiconify()
        else:
            messagebox.showerror(title=None, message='Choose all the parameters!')

    def disable_event(self):
        pass
