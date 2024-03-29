import os
import tkinter as tk

import PIL.ImageTk
import cv2
from PIL import Image
from sys import exit

import definitons
from user_app.model.algorithm_utils import run_algorithm
from user_app.gui.logged_window import show_logged
from user_app.gui.video_capture import VideoCapture
from utils.gui_utils import center_window


class App:
    def __init__(self, window, window_title, main_window, is_usual_user):
        self.app = None
        self.newWindow = None
        self.photo = None
        self.window = window
        self.window.attributes('-topmost',True)
        window.iconbitmap(definitons.app_images_dir + '\\icon.ico')
        self.testing = is_usual_user
        self.main_window = main_window
        self.window.title(window_title)
        center_window(window, 640, 520)
        self.main_window.call('wm', 'attributes', '.', '-topmost', True)
        self.main_window.after_idle(self.main_window.call, 'wm', 'attributes', '.', '-topmost', False)
        self.video_source = 0
        self.ok = False

        self.vid = VideoCapture(self.video_source)
        self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()
        if is_usual_user:
            self.btn_snapshot = tk.Button(window, text="SIFT", command=lambda: self.open_files('sift'))
            self.btn_snapshot.pack(side=tk.LEFT, padx=5, pady=5)
            self.btn_cnn = tk.Button(window, text="CNN", command=lambda: self.open_files('cnn'))
            self.btn_cnn.pack(side=tk.LEFT, padx=5, pady=5)
            self.btn_pca = tk.Button(window, text="PCA", command=lambda: self.open_files('pca'))
            self.btn_pca.pack(side=tk.LEFT, padx=5, pady=5)
            self.btn_vgg = tk.Button(window, text="VGG", command=lambda: self.open_files('vgg'))
            self.btn_vgg.pack(side=tk.LEFT, padx=5, pady=5)
        else:
            self.btn_snapshot = tk.Button(window, text="Take photo", command=lambda: self.open_files('initial'))
            self.btn_snapshot.pack(side=tk.LEFT, padx=5, pady=5)
            new_start()
        self.delay = 10
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_frame()

    def on_closing(self):
        self.vid.destroy()
        self.window.destroy()
        self.main_window.destroy()
        cv2.destroyAllWindows()
        exit()

    def open_files(self, algorithm_type):
        ret, frame = self.vid.get_frame()
        if ret:
            if os.path.isfile(definitons.root_dir + '\\images\\user_images\\new_image.jpg'):
                os.rename(definitons.root_dir + '\\images\\user_images\\new_image.jpg',
                          definitons.root_dir + '\\images\\user_images\\previous_image.jpg')
                os.replace(definitons.root_dir + '\\images\\user_images\\previous_image.jpg',
                           definitons.root_dir + '\\images\\user_images\\previous_images\\previous_image.jpg'
                           )
                cv2.imwrite(definitons.root_dir + '\\images\\user_images\\new_image.jpg',
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self.vid.destroy()
                self.window.destroy()
                cv2.destroyAllWindows()
                run_algorithm(self.window, self.main_window, algorithm_type)
            else:
                cv2.imwrite(definitons.root_dir + '\\images\\user_images\\new_image.jpg',
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self.vid.destroy()
                self.window.destroy()
                cv2.destroyAllWindows()
                show_logged(self.window, self.main_window)

    def update_frame(self):
        ret = None
        frame = None
        try:
            ret, frame = self.vid.get_frame()
        except TypeError:
            self.window.destroy()
            self.main_window.destroy()
            exit()
        if self.ok:
            self.vid.out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update_frame)


def new_start():
    try:
        os.remove(definitons.root_dir + '\\images\\user_images\\previous_images\\previous_image.jpg')
    except FileNotFoundError:
        print("File previous_image.jpg was not found")
    try:
        os.remove(definitons.root_dir + '\\images\\user_images\\new_image.jpg')
    except FileNotFoundError:
        print("File new_image.jpg was not found")
