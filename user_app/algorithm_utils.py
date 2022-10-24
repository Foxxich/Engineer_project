import os
import time
import tkinter as tk

from algorithms import sift, vgg_face, cnn, pca
from user_app.gui.logged_window import LoggedWindow


def run_algorithm(self, main_window, algorithm_type):
    result = False
    img1 = os.getcwd() + '\\images\\user_images\\previous_images\\previous_image.jpg'
    img2 = os.getcwd() + '\\images\\user_images\\new_image.jpg'
    start_time = time.time()

    if algorithm_type == 'sift':
        result = sift.comparison(img1, img2)
    elif algorithm_type == 'vgg':
        result = vgg_face.comparison(img1, img2)
    elif algorithm_type == 'pca':
        path = os.getcwd() + '\\images\\user_images\\'
        start_time = time.time()
        face_name = pca.comparison(img2, path, 'app')
        if face_name == 'previous_images.jpg':
            result = True
    elif algorithm_type == 'cnn':
        folder = os.getcwd() + '\\images\\user_images\\'
        face_name = cnn.comparison(folder, img2)
        if face_name == 'previous_images':
            result = True
    elif algorithm_type == 'initial':
        result = True

    end_time = time.time()
    print("Total time: ", round((end_time - start_time)), ' Seconds')
    if result:
        self.newWindow = tk.Toplevel(self.master)
        self.app = LoggedWindow(self.newWindow, main_window, True)
    else:
        self.newWindow = tk.Toplevel(self.master)
        self.app = LoggedWindow(self.newWindow, main_window, False)