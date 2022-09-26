import glob
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

import definitons


def convert_formats():
    path = definitons.root_dir + "\\images\\att_faces_dataset\\"
    image_folders = os.listdir(path)
    for i in range(1, len(image_folders) + 1):
        filepath = path + str(i) + "\\*.pgm"
        files_list = glob.glob(filepath)
        for j in range(1, len(files_list) + 1):
            new_file = "{}.jpg".format(j)
            image_path = path + str(i) + "\\" + str(j) + ".pgm"
            with Image.open(image_path) as im:
                Path(definitons.root_dir + "\\images\\converted_images\\" + str(i)).mkdir(parents=True, exist_ok=True)
                im.save(definitons.root_dir + "\\images\\converted_images\\" + str(i) + "\\" + new_file)
                source = cv2.imread(image_path, 0)
                image = convert_bgr_to_rgb(source)
                cv2.imwrite(os.getcwd() + "\\images\\converted_images\\" + str(i) + "\\" + new_file, image)


def convert_bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_rgb_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def blur(image_path, blur_percent):
    return Image.open(image_path).filter(ImageFilter.GaussianBlur(blur_percent))


def gaussian_noise(image_path):
    img = cv2.imread(image_path)
    gauss = np.random.normal(0, 1, img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
    return cv2.add(img, gauss)


def main():
    blur(definitons.root_dir + "\\images\\user_images\\new_image.jpg", 5)


if __name__ == "__main__":
    main()
