import random
import time

from matplotlib import pyplot as plt

import definitons
from algorithms import sift, vgg_face, pca
from algorithms.cnn import CNN
from tests.model.testing_data import test_data, labels
from tests.model.utils import save_compared_images_result, get_second_image, prepare_dataset, save_dataset_images_result
from utils.files_utils import write


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
        start_time = time.time()
        image2, original_image, s1, s2, test_image = get_second_image(image1)
        result = sift.comparison(test_image, original_image, sift_cascades, delta, path='haarcascade_frontalface_')
        correct, incorrect = save_compared_images_result(correct, data, image1, image2, incorrect, result, s1, s2,
                                                         start_time)
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
            start_time = time.time()
            image2, original_image, s1, s2, test_image = get_second_image(image1)
            result = vgg_face.comparison(test_image, original_image, vgg_model,
                                         vgg_thresh)
            correct, incorrect = save_compared_images_result(correct, data, image1, image2, incorrect, result, s1, s2,
                                                             start_time)
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
        face, folder, image_path, res, start_time = prepare_dataset(image1, image2)
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
        correct, incorrect = save_dataset_images_result(correct, incorrect, res)
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
        face, folder, image_path, res, start_time = prepare_dataset(image1)
        comparison_result = pca.comparison(
            image_path,
            folder,
            'test',
            pca_components)
        if image1[2] == 'second_set':
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
        correct, incorrect = save_dataset_images_result(correct, incorrect, res)
    write(data, 'pca', 'complex', path)
    final_statistics = [correct, incorrect]
    plt.figure(figsize=(10, 7))
    plt.pie(final_statistics, labels=labels, autopct='%1.1f%%')
    plt.savefig(definitons.root_dir + '\\pca_average.jpg')
