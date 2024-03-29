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
    img1 = definitons.user_images_dir + 'previous_images\\previous_image.jpg'
    img2 = definitons.user_images_dir + 'new_image.jpg'
    start_time = time.time()

    try:
        if algorithm_type == 'sift':
            result = sift.comparison(img1, img2, percent_delta=15.0)
        elif algorithm_type == 'vgg':
            result = vgg_face.comparison(img1, img2)
        elif algorithm_type == 'pca':
            path = definitons.user_images_dir
            start_time = time.time()
            face_name = pca.comparison(img2, path, 'app')
            if face_name == 'previous_images.jpg':
                result = True
        elif algorithm_type == 'cnn':
            n = 20
            folder = definitons.user_images_dir
            for i in range(n):
                file = definitons.user_images_dir + 'previous_images\\previous_image' + str(
                    randrange(100)) + '.jpg'
                shutil.copy(img1, file)
            cnn = CNN(folder, img2, 'user')
            face_name = cnn.comparison()
            filenames = \
            next(os.walk(definitons.user_images_dir + 'previous_images\\'), (None, None, []))[2]
            for file in filenames:
                if 'previous_image.jpg' not in file:
                    os.remove(definitons.user_images_dir + 'previous_images\\' + file)
            if face_name == 'previous_images':
                result = True
        elif algorithm_type == 'initial':
            result = True
    except:
        print('Error')

    end_time = time.time()
    print("Total time: ", round((end_time - start_time)), ' Seconds')
    if result:
        self.newWindow = tk.Toplevel(self.master)
        self.app = LoggedWindow(self.newWindow, main_window, 'success')
    else:
        os.rename(definitons.user_images_dir + 'previous_images\\previous_image.jpg',
                  definitons.user_images_dir + 'previous_images\\new_image.jpg')
        os.replace(definitons.user_images_dir + 'previous_images\\new_image.jpg',
                   definitons.user_images_dir + 'new_image.jpg')

        self.newWindow = tk.Toplevel(self.master)
        self.app = LoggedWindow(self.newWindow, main_window, 'fail')
