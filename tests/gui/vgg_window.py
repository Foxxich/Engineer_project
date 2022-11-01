from functools import partial
from tkinter import *
from tkinter import messagebox
from tkinter import ttk

import definitons
from utils.gui.gui_utils import center_window

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
        self.results = results
        self.main_window = main_window
        self.window.title(window_title)
        self.window.protocol("WM_DELETE_WINDOW", self.disable_event)

        x = 10

        Label(self.window, text="Thresh").place(x=x, y=60)

        labels = {'0.3': 'thresh', '0.4': 'thresh', '0.5': 'thresh', '0.6': 'thresh', '0.7': 'thresh'}
        x_coordinate = 50
        for option in labels:
            x_coordinate += 85
            self.CheckVar = IntVar(value=0)
            parameter_validator = partial(checkout, option, labels[option])
            Checkbutton(self.window, text=option, command=parameter_validator, variable=self.CheckVar).place(
                x=x_coordinate, y=60)

        Label(self.window, text="Model").place(x=x, y=80)
        x_coordinate = 50
        labels = {'resnet50': 'model', 'vgg16': 'model', 'senet50': 'model'}
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
        if len(vgg_parameters) == 2:
            self.results.set_vgg(vgg_parameters)
            self.window.destroy()
            self.main_window.deiconify()
        else:
            messagebox.showerror(title=None, message='Choose all the parameters!')

    def disable_event(self):
        pass
