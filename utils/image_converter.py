import glob
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

import definitons


# This function is used to convert .PGM files to .JPG
# with changing images formats from BGR to RGB
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
                Path(definitons.root_dir + "\\images\\datasets\\converted_images\\" + str(i)).mkdir(parents=True, exist_ok=True)
                im.save(definitons.root_dir + "\\images\\datasets\\converted_images\\" + str(i) + "\\" + new_file)
                source = cv2.imread(image_path, 0)
                image = convert_bgr_to_rgb(source)
                cv2.imwrite(os.getcwd() + "\\images\\datasets\\converted_images\\" + str(i) + "\\" + new_file, image)


def convert_bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_rgb_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# This function is used to add blur for image with given path, image_type, percent (default is 5)
# and later save it to `blured\\image_type` folder
def blur(folder_path, image_name, blur_percent=5):
    Image.open(definitons.root_dir + "\\images\\datasets\\" + folder_path + "\\" + image_name + ".jpg") \
        .filter(ImageFilter.GaussianBlur(blur_percent)) \
        .save(definitons.root_dir + "\\images\\tests\\blured\\" + folder_path + "\\" + image_name + ".jpg")
    return definitons.root_dir + "\\images\\tests\\blured\\" + folder_path + "\\" + image_name + ".jpg"


# This function is used to add noise for image with given path, image_type
# and later save it to `noised\\image_type` folder
def gaussian_noise(folder_path, image_name):
    img = cv2.imread(definitons.root_dir + "\\images\\datasets\\" + folder_path + "\\" + image_name + ".jpg")
    gauss = np.random.normal(0, 1, img.size)
    gauss = gauss.reshape((img.shape[0], img.shape[1], img.shape[2])).astype('uint8')
    noise = img + img * gauss
    cv2.imwrite(definitons.root_dir + "\\images\\tests\\noised\\" + folder_path + "\\" + image_name + ".jpg", noise)
    return definitons.root_dir + "\\images\\tests\\noised\\" + folder_path + "\\" + image_name + ".jpg"


# This function is used to choose data needed to be blured/noised
def run_image_selection(folder, images_number, modification_type):
    mode = 0o666
    converted_images_path = definitons.root_dir + "\\images\\datasets\\" + folder + "\\"
    image_folders = os.listdir(converted_images_path)
    for i in range(0, len(image_folders)):
        filepath = converted_images_path + str(image_folders[i]) + "\\*.jpg"
        glob.glob(filepath)
        for j in range(1, images_number):
            try:
                path = os.path.join(
                    definitons.root_dir +
                    "\\images\\tests\\" +
                    modification_type +
                    "\\" + folder +
                    "\\" + str(image_folders[i]))
                os.mkdir(path, mode)
            except FileExistsError:
                print('Error during creating folder, some of them exist')
            if modification_type == 'noised':
                if folder == 'converted_images':
                    gaussian_noise(folder + "\\" + image_folders[i], str(j))
                else:
                    gaussian_noise(folder + "\\" + image_folders[i],
                                   str(j) + str(image_folders[i]))
            else:
                if folder == 'converted_images':
                    blur(folder + "\\" + image_folders[i], str(j))
                else:
                    blur(folder + "\\" + image_folders[i],
                         str(j) + str(image_folders[i]))
    print('Successfully added ' + folder + ' with ' + modification_type)