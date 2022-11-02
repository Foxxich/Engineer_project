import glob
import random
import time

import definitons
from algorithms import sift, vgg_face, pca
from algorithms.cnn import CNN
from utils.files_utils import write
from utils.image_converter import run_image_selection

blur_percents = [1, 2, 3, 4, 5]

datasets = [
    ['images\\datasets\\converted_images\\'],
    ['images\\datasets\\tt_dataset\\Final Training Images'],
    ['images\\datasets\\tt_dataset\\Final Testing Images'],

    ['images\\tests\\noised\\converted_images\\'],
    ['images\\tests\\noised\\tt_dataset\\Final Training Images'],
    ['images\\tests\\noised\\tt_dataset\\Final Testing Images'],

    ['images\\tests\\blured\\converted_images\\'],
    ['images\\tests\\blured\\tt_dataset\\Final Training Images'],
    ['images\\tests\\blured\\tt_dataset\\Final Testing Images'],
]

test_data = [
    # Usual tt
    ['face1\\1face1.jpg', 'usual', 'tt', datasets[1][0]],
    ['face1\\2face1.jpg', 'usual', 'tt', datasets[1][0]],

    ['face2\\10face2.jpg', 'usual', 'tt', datasets[1][0]],
    ['face2\\12face2.jpg', 'usual', 'tt', datasets[1][0]],
]


def run_sift(path, parameters):
    sift_cascades = None
    delta = None
    for parameter in parameters:
        if 'cascades' in parameter:
            sift_cascades = parameter[1]
        else:
            delta = parameter[1]
    data = []
    for image1 in test_data:
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
        print("Total time: ", round((end_time - start_time), 3), ' Seconds')
    write(data, 'sift', 'usual', path)


def run_vgg(path, parameters):
    vgg_thresh = None
    vgg_model = None
    for parameter in parameters:
        if 'thresh' in parameter:
            vgg_thresh = parameter[1]
        else:
            vgg_model = parameter[1]
    data = []
    for image1 in test_data:
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
            print("Total time: ", round((end_time - start_time), 3), ' Seconds')
        except IndexError:
            print('Sift do not support these type of images')
    write(data, 'vgg_model', 'usual', path)


def run_cnn(path, parameters):
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
    for image1 in test_data:
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
        print("Total time: ", round((end_time - start_time), 3), ' Seconds')
    write(data, 'cnn', 'complex', path)


def run_pca(path, parameter):
    pca_components = int(parameter[0][1])
    data = []
    for image1 in test_data:
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
        print("Total time: ", round((end_time - start_time), 3), ' Seconds')
    write(data, 'pca', 'complex', path)


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
