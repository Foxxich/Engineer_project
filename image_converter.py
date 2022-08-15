import glob
import cv2
from PIL import Image
import os
from pathlib import Path


def convert_formats():
    path = os.getcwd() + "\\att_faces\\"
    image_folders = os.listdir(path)
    for i in range(1, len(image_folders) + 1):
        filepath = path + str(i) + "\\*.pgm"
        files_list = glob.glob(filepath)
        for j in range(1, len(files_list) + 1):
            new_file = "{}.jpg".format(j)
            image_path = path + str(i) + "\\" + str(j) + ".pgm"
            with Image.open(image_path) as im:
                Path(os.getcwd() + "\\test_images\\" + str(i)).mkdir(parents=True, exist_ok=True)
                im.save(os.getcwd() + "\\test_images\\" + str(i) + "\\" + new_file)
                source = cv2.imread(image_path, 0)
                image = convert_bgr_to_rgb(source)
                cv2.imwrite(os.getcwd() + "\\test_images\\" + str(i) + "\\" + new_file, image)


def convert_bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_rgb_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
