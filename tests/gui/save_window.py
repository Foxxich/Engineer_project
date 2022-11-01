from tkinter.filedialog import asksaveasfile

import definitons
from tkinter import *
from utils.gui.gui_utils import center_window

def upload(root):
    statusvar = StringVar()
    statusvar.set("Ready")
    sbar = Label(root, textvariable=statusvar, relief=SUNKEN, anchor="w")
    sbar.pack(side=BOTTOM, fill=X)
    statusvar.set("Busy..")
    sbar.update()
    import time
    time.sleep(2)
    statusvar.set("Ready")


class SaveWindow:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        window.iconbitmap(definitons.app_images_dir + '\\icon.ico')
        center_window(window, 640, 520)

        self.save_file()

    def save_file(self):
        f = asksaveasfile(initialfile='results.csv',
                          defaultextension=".txt", filetypes=[("CSV", "*.csv")])
        upload(self.window)




