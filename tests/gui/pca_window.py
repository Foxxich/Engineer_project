from functools import partial
from tkinter import *
from tkinter import messagebox
from tkinter import ttk

import definitons
from utils.gui.gui_utils import center_window

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
        self.main_window = main_window
        self.window.title(window_title)
        self.window.protocol("WM_DELETE_WINDOW", self.disable_event)

        x = 10

        Label(self.window, text="Components number").place(x=x, y=60)
        labels = {'60': 'components', '70': 'components', '80': 'components', '90': 'components', '100': 'components'}
        x_coordinate = 50
        for option in labels:
            x_coordinate += 85
            self.CheckVar = IntVar(value=0)
            parameter_validator = partial(checkout, option, labels[option])
            Checkbutton(self.window, text=option, command=parameter_validator, variable=self.CheckVar).place(
                x=x_coordinate, y=60)

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
        if len(pca_parameters) == 1:
            self.results.set_pca(pca_parameters)
            self.window.destroy()
            self.main_window.deiconify()
        else:
            messagebox.showerror(title=None, message='Choose all the parameters!')

    def disable_event(self):
        pass
