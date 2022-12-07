import glob
import random
import time

import definitons
from algorithms import sift, vgg_face, pca
from algorithms.cnn import CNN
from utils.files_utils import write
from utils.image_converter import run_image_selection
from matplotlib import pyplot as plt
import numpy as np

blur_percents = [1, 2, 3, 4, 5]
labels = ['correct', 'incorrect']

datasets = [
    'images\\datasets\\converted_images\\',  # 0
    'images\\datasets\\tt_dataset\\Final Training Images',  # 1
    'images\\datasets\\tt_dataset\\Final Testing Images',  # 2

    'images\\tests\\noised\\converted_images\\',  # 3
    'images\\tests\\noised\\tt_dataset\\Final Training Images',  # 4
    'images\\tests\\noised\\tt_dataset\\Final Testing Images',  # 5

    'images\\tests\\blured\\converted_images\\',  # 6
    'images\\tests\\blured\\tt_dataset\\Final Training Images',  # 7
    'images\\tests\\blured\\tt_dataset\\Final Testing Images',  # 8
]

test_data = [
    # converted_images (30)
    ['1\\1.jpg', 'usual', 'att', datasets[0]],
    ['1\\5.jpg', 'usual', 'att', datasets[0]],
    ['1\\9.jpg', 'usual', 'att', datasets[0]],
    ['1\\1.jpg', 'usual', 'att', datasets[0]],
    ['1\\2.jpg', 'usual', 'att', datasets[0]],
    ['1\\5.jpg', 'usual', 'att', datasets[0]],
    ['1\\6.jpg', 'usual', 'att', datasets[0]],
    ['1\\8.jpg', 'usual', 'att', datasets[0]],
    ['1\\9.jpg', 'usual', 'att', datasets[0]],
    ['1\\10.jpg', 'usual', 'att', datasets[0]],
    ['12\\7.jpg', 'usual', 'att', datasets[0]],
    ['12\\7.jpg', 'usual', 'att', datasets[0]],
    ['12\\5.jpg', 'usual', 'att', datasets[0]],
    ['12\\4.jpg', 'usual', 'att', datasets[0]],
    ['12\\3.jpg', 'usual', 'att', datasets[0]],
    ['12\\2.jpg', 'usual', 'att', datasets[0]],
    ['12\\1.jpg', 'usual', 'att', datasets[0]],
    ['12\\9.jpg', 'usual', 'att', datasets[0]],
    ['40\\8.jpg', 'usual', 'att', datasets[0]],
    ['40\\7.jpg', 'usual', 'att', datasets[0]],
    ['40\\6.jpg', 'usual', 'att', datasets[0]],
    ['40\\5.jpg', 'usual', 'att', datasets[0]],
    ['40\\4.jpg', 'usual', 'att', datasets[0]],
    ['40\\3.jpg', 'usual', 'att', datasets[0]],
    ['40\\2.jpg', 'usual', 'att', datasets[0]],
    ['40\\1.jpg', 'usual', 'att', datasets[0]],
    ['6\\7.jpg', 'usual', 'att', datasets[0]],
    ['6\\8.jpg', 'usual', 'att', datasets[0]],
    ['6\\9.jpg', 'usual', 'att', datasets[0]],
    ['6\\6.jpg', 'usual', 'att', datasets[0]],
    ['6\\5.jpg', 'usual', 'att', datasets[0]],
    ['6\\2.jpg', 'usual', 'att', datasets[0]],

    # tt_dataset\\Final Training Image (15)
    ['face1\\1face1.jpg', 'usual', 'tt', datasets[1]],
    ['face1\\2face1.jpg', 'usual', 'tt', datasets[1]],
    ['face2\\10face2.jpg', 'usual', 'tt', datasets[1]],
    ['face2\\12face2.jpg', 'usual', 'tt', datasets[1]],
    ['face1\\1face1.jpg', 'usual', 'tt', datasets[1]],
    ['face1\\2face1.jpg', 'usual', 'tt', datasets[1]],
    ['face3\\10face3.jpg', 'usual', 'tt', datasets[1]],
    ['face3\\12face3.jpg', 'usual', 'tt', datasets[1]],
    ['face4\\1face4.jpg', 'usual', 'tt', datasets[1]],
    ['face4\\2face4.jpg', 'usual', 'tt', datasets[1]],
    ['face5\\10face5.jpg', 'usual', 'tt', datasets[1]],
    ['face5\\12face5.jpg', 'usual', 'tt', datasets[1]],
    ['face6\\1face6.jpg', 'usual', 'tt', datasets[1]],
    ['face6\\2face6.jpg', 'usual', 'tt', datasets[1]],
    ['face5\\10face5.jpg', 'usual', 'tt', datasets[1]],
    ['face4\\12face4.jpg', 'usual', 'tt', datasets[1]],

    # tt_dataset\\Final Testing Image (15)
    ['face16\\1face16.jpg', 'usual', 'tt', datasets[2]],
    ['face16\\3face16.jpg', 'usual', 'tt', datasets[2]],
    ['face12\\1face12.jpg', 'usual', 'tt', datasets[2]],
    ['face12\\2face12.jpg', 'usual', 'tt', datasets[2]],
    ['face10\\2face10.jpg', 'usual', 'tt', datasets[2]],
    ['face10\\2face10.jpg', 'usual', 'tt', datasets[2]],
    ['face13\\1face13.jpg', 'usual', 'tt', datasets[2]],
    ['face13\\2face13.jpg', 'usual', 'tt', datasets[2]],
    ['face8\\3face8.jpg', 'usual', 'tt', datasets[2]],
    ['face8\\4face8.jpg', 'usual', 'tt', datasets[2]],
    ['face7\\1face7.jpg', 'usual', 'tt', datasets[2]],
    ['face7\\2face7.jpg', 'usual', 'tt', datasets[2]],
    ['face9\\1face9.jpg', 'usual', 'tt', datasets[2]],
    ['face9\\2face9.jpg', 'usual', 'tt', datasets[2]],
    ['face4\\1face4.jpg', 'usual', 'tt', datasets[2]],
    ['face4\\2face4.jpg', 'usual', 'tt', datasets[2]],

    # converted_images MIXED (30)
    ['1\\1.jpg', 'noised', 'att', datasets[3]],
    ['1\\5.jpg', 'noised', 'att', datasets[3]],
    ['1\\9.jpg', 'noised', 'att', datasets[3]],
    ['5\\1.jpg', 'noised', 'att', datasets[3]],
    ['5\\2.jpg', 'noised', 'att', datasets[3]],
    ['6\\5.jpg', 'noised', 'att', datasets[3]],
    ['6\\6.jpg', 'noised', 'att', datasets[3]],
    ['7\\8.jpg', 'noised', 'att', datasets[3]],
    ['8\\9.jpg', 'blured', 'att', datasets[6]],
    ['8\\10.jpg', 'noised', 'att', datasets[3]],
    ['12\\7.jpg', 'blured', 'att', datasets[6]],
    ['12\\7.jpg', 'noised', 'att', datasets[3]],
    ['12\\5.jpg', 'blured', 'att', datasets[6]],
    ['33\\4.jpg', 'noised', 'att', datasets[3]],
    ['33\\3.jpg', 'blured', 'att', datasets[6]],
    ['23\\2.jpg', 'noised', 'att', datasets[3]],
    ['34\\1.jpg', 'blured', 'att', datasets[6]],
    ['39\\9.jpg', 'noised', 'att', datasets[3]],
    ['40\\8.jpg', 'blured', 'att', datasets[6]],
    ['26\\7.jpg', 'noised', 'att', datasets[3]],
    ['26\\6.jpg', 'blured', 'att', datasets[6]],
    ['15\\5.jpg', 'noised', 'att', datasets[3]],
    ['15\\4.jpg', 'blured', 'att', datasets[6]],
    ['19\\3.jpg', 'noised', 'att', datasets[3]],
    ['17\\2.jpg', 'blured', 'att', datasets[6]],
    ['6\\1.jpg', 'noised', 'att', datasets[3]],
    ['6\\7.jpg', 'blured', 'att', datasets[6]],
    ['13\\8.jpg', 'noised', 'att', datasets[3]],
    ['13\\9.jpg', 'blured', 'att', datasets[6]],
    ['23\\6.jpg', 'noised', 'att', datasets[3]],
    ['22\\5.jpg', 'blured', 'att', datasets[6]],
    ['22\\2.jpg', 'noised', 'att', datasets[3]],

    # tt_dataset\\Final Training Image MIXED (15)
    ['face1\\1face1.jpg', 'noised', 'tt', datasets[4]],
    ['face1\\2face1.jpg', 'blured', 'tt', datasets[7]],
    ['face2\\10face2.jpg', 'noised', 'tt', datasets[4]],
    ['face2\\12face2.jpg', 'blured', 'tt', datasets[7]],
    ['face1\\1face1.jpg', 'noised', 'tt', datasets[4]],
    ['face1\\2face1.jpg', 'blured', 'tt', datasets[7]],
    ['face3\\10face3.jpg', 'noised', 'tt', datasets[4]],
    ['face3\\12face3.jpg', 'blured', 'tt', datasets[7]],
    ['face4\\1face4.jpg', 'noised', 'tt', datasets[4]],
    ['face4\\2face4.jpg', 'blured', 'tt', datasets[7]],
    ['face5\\10face5.jpg', 'noised', 'tt', datasets[4]],
    ['face5\\12face5.jpg', 'blured', 'tt', datasets[7]],
    ['face6\\1face6.jpg', 'noised', 'tt', datasets[4]],
    ['face6\\2face6.jpg', 'blured', 'tt', datasets[7]],
    ['face5\\10face5.jpg', 'noised', 'tt', datasets[4]],
    ['face4\\12face4.jpg', 'blured', 'tt', datasets[7]],

    # tt_dataset\\Final Testing Image MIXED (15)
    ['face16\\1face16.jpg', 'noised', 'tt', datasets[5]],
    ['face16\\3face16.jpg', 'blured', 'tt', datasets[8]],
    ['face12\\1face12.jpg', 'noised', 'tt', datasets[5]],
    ['face12\\2face12.jpg', 'blured', 'tt', datasets[8]],
    ['face10\\2face10.jpg', 'noised', 'tt', datasets[5]],
    ['face10\\2face10.jpg', 'blured', 'tt', datasets[8]],
    ['face13\\1face13.jpg', 'noised', 'tt', datasets[5]],
    ['face13\\2face13.jpg', 'blured', 'tt', datasets[8]],
    ['face8\\3face8.jpg', 'noised', 'tt', datasets[5]],
    ['face8\\4face8.jpg', 'blured', 'tt', datasets[8]],
    ['face7\\1face7.jpg', 'noised', 'tt', datasets[5]],
    ['face7\\2face7.jpg', 'blured', 'tt', datasets[8]],
    ['face9\\1face9.jpg', 'noised', 'tt', datasets[8]],
    ['face9\\2face9.jpg', 'blured', 'tt', datasets[8]],
    ['face4\\1face4.jpg', 'noised', 'tt', datasets[5]],
    ['face4\\2face4.jpg', 'blured', 'tt', datasets[8]],
]


def run_sift(path, parameters, set_size):
    correct = 0
    incorrect = 0
    sift_cascades = None
    delta = None
    for parameter in parameters:
        if 'cascades' in parameter:
            sift_cascades = parameter[1]
        else:
            delta = parameter[1]
    data = []
    for image1 in test_data[0:set_size - 1]:
        image2 = None
        while image2 is None:
            score = random.choice(test_data)
            if score[2] == image1[2] and score[0] != image1[0]:
                image2 = score
        test_image = definitons.root_dir + '\\' + image1[3] + '\\' + image1[0]
        original_image = definitons.root_dir + '\\' + image2[3] + '\\' + image2[0]
        start_time = time.time()
        s1 = image1[0].replace("\\", " ").split()[0].replace(".jpg", " ")
        s2 = image2[0].replace("\\", " ").split()[0].replace(".jpg", " ")
        res = sift.comparison(test_image, original_image, sift_cascades, delta, path='haarcascade_frontalface_')
        end_time = time.time()
        is_same_person = False
        if s1 == s2:
            is_same_person = True
        data.append([
            str(image1[0]),
            str(image2[0]),
            str(res),
            str(is_same_person),
            str(round((end_time - start_time), 3)),
            image1[2],
            image1[1],
            image2[1],
        ])
        if res == is_same_person:
            correct += 1
        else:
            incorrect += 1
        print("Total time: ", round((end_time - start_time), 3), ' Seconds')
    final_statistics = [correct, incorrect]
    plt.figure(figsize=(10, 7))
    plt.pie(final_statistics, labels=labels, autopct='%1.1f%%')
    plt.savefig(definitons.root_dir + '\\sift_average.jpg')
    write(data, 'sift', 'usual', path)


def run_vgg(path, parameters, set_size):
    correct = 0
    incorrect = 0
    vgg_thresh = None
    vgg_model = None
    for parameter in parameters:
        if 'thresh' in parameter:
            vgg_thresh = parameter[1]
        else:
            vgg_model = parameter[1]
    data = []
    for image1 in test_data[0:set_size - 1]:
        try:
            image2 = None
            while image2 is None:
                score = random.choice(test_data)
                if score[2] == image1[2] and score[0] != image1[0]:
                    image2 = score
            test_image = definitons.root_dir + '\\' + image1[3] + '\\' + image1[0]
            original_image = definitons.root_dir + '\\' + image2[3] + '\\' + image2[0]
            start_time = time.time()
            s1 = image1[0].replace("\\", " ").split()[0].replace(".jpg", " ")
            s2 = image2[0].replace("\\", " ").split()[0].replace(".jpg", " ")
            res = vgg_face.comparison(test_image, original_image, vgg_model,
                                      vgg_thresh)
            end_time = time.time()
            is_same_person = False
            if s1 == s2:
                is_same_person = True
            data.append([
                str(image1[0]),
                str(image2[0]),
                str(res),
                str(is_same_person),
                str(round((end_time - start_time), 3)),
                image1[2],
                image1[1],
                image2[1],
            ])
            if res == is_same_person:
                correct += 1
            else:
                incorrect += 1
            print("Total time: ", round((end_time - start_time), 3), ' Seconds')
        except IndexError:
            print('Sift do not support these type of images')
    final_statistics = [correct, incorrect]
    plt.figure(figsize=(10, 7))
    plt.pie(final_statistics, labels=labels, autopct='%1.1f%%')
    plt.savefig(definitons.root_dir + '\\vgg_average.jpg')
    write(data, 'vgg_model', 'usual', path)


def run_cnn(path, parameters, set_size):
    correct = 0
    incorrect = 0
    cnn_optimizers = None
    cnn_loss = None
    cnn_metrics = None
    cnn_epochs_number = None
    cnn_steps_for_validation = None
    for parameter in parameters:
        if 'loss' in parameter:
            cnn_loss = parameter[1] + '_crossentropy'
        elif 'metrics' in parameter:
            if 'accuracy' not in parameter[1]:
                cnn_metrics = parameter[1] + '_accuracy'
            else:
                cnn_metrics = parameter[1]
        elif 'optimizer' in parameter:
            cnn_optimizers = parameter[1]
        elif 'epochs_number' in parameter:
            cnn_epochs_number = int(parameter[1])
        else:
            cnn_steps_for_validation = int(parameter[1])
    data = []
    for image1 in test_data[0:set_size - 1]:
        image2 = None
        while image2 is None:
            score = random.choice(test_data)
            if score[2] == image1[2] and score[0] != image1[0]:
                image2 = score
        face = image2[0].replace("\\", " ").split()[0]
        print(face)
        folder = definitons.root_dir + '\\' + image1[3] + '\\'
        print(folder)
        image_path = definitons.root_dir + '\\' + image2[3] + '\\' + image2[0]
        print(image_path)
        start_time = time.time()
        res = False
        print(cnn_loss)
        print(cnn_optimizers)
        print(cnn_metrics)
        cnn = CNN(
            folder,
            image_path,
            'test',
            cnn_epochs_number,
            cnn_steps_for_validation,
            cnn_optimizers,
            cnn_loss,
            cnn_metrics
        )
        if cnn.comparison() == str(face):
            res = True

        end_time = time.time()
        data.append([
            str(folder),
            str(image2[0]),
            str(res),
            str(round((end_time - start_time), 3)),
            image1[2],
            image1[1],
        ])
        if res:
            correct += 1
        else:
            incorrect += 1
        print("Total time: ", round((end_time - start_time), 3), ' Seconds')
    write(data, 'cnn', 'complex', path)
    final_statistics = [correct, incorrect]
    plt.figure(figsize=(10, 7))
    plt.pie(final_statistics, labels=labels, autopct='%1.1f%%')
    plt.savefig(definitons.root_dir + '\\cnn_average.jpg')


def run_pca(path, parameter, set_size):
    correct = 0
    incorrect = 0
    pca_components = int(parameter[0][1])
    data = []
    for image1 in test_data[0:set_size - 1]:
        face = image1[0].replace("\\", " ").split()[0]
        folder = definitons.root_dir + '\\' + image1[3] + '\\'
        image_path = definitons.root_dir + '\\' + image1[3] + '\\' + image1[0]
        start_time = time.time()
        res = False
        comparison_result = pca.comparison(
            image_path,
            folder,
            'test',
            pca_components)
        print(str(face))
        print(comparison_result)
        if image1[2] == 'tt':
            if 'face' + comparison_result == str(face):
                res = True
        else:
            if comparison_result == str(face):
                res = True

        end_time = time.time()
        data.append([
            str(folder),
            str(image1[0]),
            str(res),
            str(round((end_time - start_time), 3)),
            image1[2],
            image1[1],
        ])
        if res:
            correct += 1
        else:
            incorrect += 1
        print("Total time: ", round((end_time - start_time), 3), ' Seconds')
    write(data, 'pca', 'complex', path)
    final_statistics = [correct, incorrect]
    plt.figure(figsize=(10, 7))
    plt.pie(final_statistics, labels=labels, autopct='%1.1f%%')
    plt.savefig(definitons.root_dir + '\\pca_average.jpg')


def generate_blured_images():
    run_image_selection('tt_dataset\\Final Training Images', 14, 'blured')
    run_image_selection('tt_dataset\\Final Testing Images', 5, 'blured')
    run_image_selection('converted_images', 11, 'blured')


def generate_gaussian():
    run_image_selection('tt_dataset\\Final Training Images', 14, 'noised')
    run_image_selection('tt_dataset\\Final Testing Images', 5, 'noised')
    run_image_selection('converted_images', 11, 'noised')


def add_data():
    if len(glob.glob(definitons.root_dir + '\\images\\tests\\*')) == 0:
        generate_blured_images()
        generate_gaussian()
