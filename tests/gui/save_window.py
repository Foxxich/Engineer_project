import time
from tkinter.filedialog import asksaveasfile

import definitons
from tkinter import *
from sys import exit

from tests.run_tests import add_data, run_sift, run_vgg, run_pca, run_cnn
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


def write_results(algorithm_parameters, name, total_time):
    with open(name, 'a', encoding='UTF8', newline='') as f:
        for parameter in algorithm_parameters:
            f.write(str(parameter[0]) + ':' + str(parameter[1]) + '\n')
        f.write('Total time:' + str(total_time))


class SaveWindow:
    def __init__(self, window, window_title, results, master):
        self.window = window
        self.window.title(window_title)
        self.results = results
        self.master = master
        window.iconbitmap(definitons.app_images_dir + '\\icon.ico')
        center_window(window, 640, 520)
        self.save_file()
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)

    def save_file(self):
        f = asksaveasfile(initialfile='results.csv',
                          defaultextension=".txt", filetypes=[("CSV", "*.csv")])
        upload(self.window)
        add_data()
        sift = self.results.get_sift()
        vgg = self.results.get_vgg()
        pca = self.results.get_pca()
        cnn = self.results.get_cnn()
        start_time = time.time()
        data_to_print = []
        if len(sift) != 0:
            data_to_print.append(['SIFT', sift])
            run_sift(f.name, sift)
        if len(vgg) != 0:
            data_to_print.append(['VGG', vgg])
            run_vgg(f.name, vgg)
        if len(pca) != 0:
            run_pca(f.name, pca)
            data_to_print.append(['PCA', pca])
        if len(cnn) != 0:
            data_to_print.append(['CNN', cnn])
            run_cnn(f.name, cnn)
        write_results(data_to_print, f.name, round((time.time() - start_time), 3))

    def close_window(self):
        self.window.destroy()
        self.master.destroy()
        exit()
