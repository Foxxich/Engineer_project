from functools import partial
from tkinter import ttk
from tkinter import *
import definitons
from utils.gui.gui_utils import center_window


class CnnWindow:
    def __init__(self, window, window_title, main_window, testing):
        self.app = None
        self.photo = None
        self.window = window
        window.iconbitmap(definitons.app_images_dir + '\\icon.ico')
        center_window(window, 640, 520)
        self.testing = testing
        self.main_window = main_window
        self.window.title(window_title)
        self.window.protocol("WM_DELETE_WINDOW", self.disable_event)

        Label(self.window, text="Optimizers").place(x=40, y=60)

        enable = {'categorical_crossentropy': 0, 'binary_crossentropy': 0}
        x_coordinate = 60
        for machine in enable:
            enable[machine] = Variable()
            x_coordinate += 70
            validate_login = partial(self.checkout, machine)
            Checkbutton(self.window, text=machine, variable=enable[machine], command=validate_login).place(x=x_coordinate, y=60)

        Label(self.window, text="Optimizers").place(x=40, y=60)

        exit_button = ttk.Button(
            self.window,
            text='Exit',
            command=lambda: self.close_window()
        )

        exit_button.pack(
            ipadx=5,
            ipady=5,
            expand=True
        )

    def checkout(self, par):
        print(par)

    def close_window(self):
        self.window.destroy()

    def disable_event(self):
        pass

        # cnn_optimizers = []
        # cnn_metrics = ['accuracy', 'binary_accuracy', 'categorical_accuracy', 'top_k_categorical_accuracy']
        # cnn_epochs_number = [50, 45, 20, 15, 10]
        # cnn_steps_for_validation = [10, 9, 8, 7, 6]
