import os
import tkinter as tk

import PIL.ImageTk
import cv2
from PIL import Image

import definitons
from user_app.algorithm_utils import run_algorithm
from user_app.gui.gui_utils import center_window
from user_app.gui.logged_window import show_logged
from user_app.gui.video_capture import VideoCapture


class App:
    def __init__(self, window, window_title, main_window, testing):
        self.app = None
        self.newWindow = None
        self.photo = None
        self.window = window
        window.iconbitmap(definitons.app_images_dir + '\\icon.ico')
        center_window(window, 640, 520)
        self.testing = testing
        self.main_window = main_window
        self.window.title(window_title)
        self.video_source = 0
        self.ok = False

        self.vid = VideoCapture(self.video_source)
        self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()
        if testing:
            self.btn_snapshot = tk.Button(window, text="SIFT", command=lambda: self.open_files('sift'))
            self.btn_snapshot.pack(side=tk.LEFT, padx=5, pady=5)
            self.btn_cnn = tk.Button(window, text="CNN", command=lambda: self.open_files('cnn'))
            self.btn_cnn.pack(side=tk.LEFT, padx=5, pady=5)
            self.btn_vgg = tk.Button(window, text="PCA", command=lambda: self.open_files('pca'))
            self.btn_vgg.pack(side=tk.LEFT, padx=5, pady=5)
            self.btn_vgg = tk.Button(window, text="VGG", command=lambda: self.open_files('vgg'))
            self.btn_vgg.pack(side=tk.LEFT, padx=5, pady=5)
        else:
            self.btn_snapshot = tk.Button(window, text="Make photo", command=lambda: self.open_files('initial'))
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

    def open_files(self, algorithm_type):
        ret, frame = self.vid.get_frame()
        if ret:
            if os.path.isfile(os.getcwd() + '\\images\\user_images\\new_image.jpg'):
                os.rename(os.getcwd() + '\\images\\user_images\\new_image.jpg',
                          os.getcwd() + '\\images\\user_images\\previous_image.jpg')
                os.replace(os.getcwd() + '\\images\\user_images\\previous_image.jpg',
                           os.getcwd() + '\\images\\user_images\\previous_images\\previous_image.jpg'
                           )
                cv2.imwrite(os.getcwd() + '\\images\\user_images\\new_image.jpg',
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self.vid.destroy()
                self.window.destroy()
                cv2.destroyAllWindows()
                run_algorithm(self.window, self.main_window, algorithm_type)
            else:
                cv2.imwrite(os.getcwd() + '\\images\\user_images\\new_image.jpg',
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self.vid.destroy()
                self.window.destroy()
                cv2.destroyAllWindows()
                show_logged(self.window, self.main_window)

    def update_frame(self):
        ret, frame = self.vid.get_frame()
        if self.ok:
            self.vid.out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update_frame)


def new_start():
    try:
        os.remove(os.getcwd() + '\\images\\user_images\\previous_images\\previous_image.jpg')
    except FileNotFoundError:
        print("File previous_image.jpg was not found")
    try:
        os.remove(os.getcwd() + '\\images\\user_images\\new_image.jpg')
    except FileNotFoundError:
        print("File new_image.jpg was not found")
