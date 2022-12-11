import random
import time

import definitons
from tests.model.testing_data import test_data


def prepare_dataset(image1, image2=None):
    folder = definitons.root_dir + '\\' + image1[3] + '\\'
    if image2:
        face = image2[0].replace("\\", " ").split()[0]
        image_path = definitons.root_dir + '\\' + image2[3] + '\\' + image2[0]
    else:
        face = image1[0].replace("\\", " ").split()[0]
        image_path = definitons.root_dir + '\\' + image1[3] + image1[0]
    res = False
    start_time = time.time()
    return face, folder, image_path, res, start_time


def get_second_image(image1):
    image2 = None
    while image2 is None:
        score = random.choice(test_data)
        if score[2] == image1[2] and score[0] != image1[0]:
            image2 = score
    test_image = definitons.root_dir + '\\' + image1[3] + '\\' + image1[0]
    original_image = definitons.root_dir + '\\' + image2[3] + '\\' + image2[0]
    s1 = image1[0].replace("\\", " ").split()[0].replace(".jpg", " ")
    s2 = image2[0].replace("\\", " ").split()[0].replace(".jpg", " ")
    return image2, original_image, s1, s2, test_image


def save_compared_images_result(correct, data, image1, image2, incorrect, result, image1_name, image2_name, start_time):
    end_time = time.time()
    is_same_person = False
    if image1_name == image2_name:
        is_same_person = True
    data.append([
        str(image1[0]),
        str(image2[0]),
        str(result),
        str(is_same_person),
        str(round((end_time - start_time), 3)),
        image1[2],
        image1[1],
        image2[1],
    ])
    if result == is_same_person:
        correct += 1
    else:
        incorrect += 1
    return correct, incorrect


def save_dataset_images_result(correct, incorrect, res):
    if res:
        correct += 1
    else:
        incorrect += 1
    return correct, incorrect
