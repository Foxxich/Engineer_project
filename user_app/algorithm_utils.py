import os
import shutil
import time
import tkinter as tk
from random import randrange

import definitons
from algorithms import sift, vgg_face, pca
from algorithms.cnn import CNN
from user_app.gui.logged_window import LoggedWindow


def run_algorithm(self, main_window, algorithm_type):
    result = False
    img1 = definitons.root_dir + '\\images\\user_images\\previous_images\\previous_image.jpg'
    img2 = definitons.root_dir + '\\images\\user_images\\new_image.jpg'
    start_time = time.time()

    if algorithm_type == 'sift':
        result = sift.comparison(img1, img2, percent_delta=15.0)
    elif algorithm_type == 'vgg':
        result = vgg_face.comparison(img1, img2)
    elif algorithm_type == 'pca':
        path = definitons.root_dir + '\\images\\user_images\\'
        start_time = time.time()
        face_name = pca.comparison(img2, path, 'app')
        if face_name == 'previous_images.jpg':
            result = True
    elif algorithm_type == 'cnn':
        n = 20
        folder = definitons.root_dir + '\\images\\user_images\\'
        for i in range(n):
            file = definitons.root_dir + '\\images\\user_images\\previous_images\\previous_image' + str(randrange(100)) + '.jpg'
            shutil.copy(img1, file)
        cnn = CNN(folder, img2, 'user')
        face_name = cnn.comparison()
        filenames = next(os.walk(definitons.root_dir + '\\images\\user_images\\previous_images\\'), (None, None, []))[2]
        for file in filenames:
            if 'previous_image.jpg' not in file:
                os.remove(definitons.root_dir + '\\images\\user_images\\previous_images\\' + file)
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
        os.rename(definitons.root_dir + '\\images\\user_images\\previous_images\\previous_image.jpg',
                  definitons.root_dir + '\\images\\user_images\\previous_images\\new_image.jpg')
        os.replace(definitons.root_dir + '\\images\\user_images\\previous_images\\new_image.jpg',
                   definitons.root_dir + '\\images\\user_images\\new_image.jpg')

        self.newWindow = tk.Toplevel(self.master)
        self.app = LoggedWindow(self.newWindow, main_window, False)