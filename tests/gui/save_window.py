import time
from sys import exit
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import asksaveasfile
from tkinter.ttk import Progressbar

import definitons
from tests.model.test_runner import run_sift, run_vgg, run_pca, run_cnn
from utils.gui_utils import center_window


def bar(progress, root, percent):
    progress['value'] = percent
    root.update_idletasks()
    if percent == 100:
        messagebox.showinfo("showinfo", "Testing is finished")


def write_results(algorithm_parameters, name, total_time):
    with open(name, 'a', encoding='UTF8') as f:
        for parameter in algorithm_parameters:
            f.write(str(parameter[0]) + ':' + str(parameter[1]) + '\n')
        f.write('Total time: ' + str(total_time))
        f.close()


class SaveWindow:
    def __init__(self, window, window_title, results, master, set_size):
        self.window = window
        self.window.title(window_title)
        self.results = results
        self.master = master
        self.set_size = set_size
        window.iconbitmap(definitons.app_images_dir + '\\icon.ico')
        center_window(window, 640, 520)
        self.save_file()

    def save_file(self):

        f = asksaveasfile(initialfile='results.txt',
                          defaultextension=".txt", filetypes=[("txt", "*.txt")])
        progress = Progressbar(self.window, orient=HORIZONTAL,
                               length=300, mode='determinate')
        progress.pack(pady=10)
        sift = self.results.get_sift()
        vgg = self.results.get_vgg()
        pca = self.results.get_pca()
        cnn = self.results.get_cnn()
        start_time = time.time()
        data_to_print = []
        bar(progress, self.window, 0)
        if len(sift) != 0:
            data_to_print.append(['SIFT', sift])
            run_sift(f.name, sift, self.set_size)
        bar(progress, self.window, 25)
        if len(vgg) != 0:
            data_to_print.append(['VGG', vgg])
            run_vgg(f.name, vgg, self.set_size)
        bar(progress, self.window, 50)
        if len(pca) != 0:
            run_pca(f.name, pca, self.set_size)
            data_to_print.append(['PCA', pca])
        bar(progress, self.window, 75)
        if len(cnn) != 0:
            data_to_print.append(['CNN', cnn])
            run_cnn(f.name, cnn, self.set_size)
        write_results(data_to_print, f.name, round((time.time() - start_time), 3))
        bar(progress, self.window, 100)
        self.close_window()

    def close_window(self):
        self.window.destroy()
        self.master.destroy()
        exit()
