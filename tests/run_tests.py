import glob
import random
import time

import definitons
from algorithms import sift, vgg_face, cnn, pca
from utils.files_utils import write
from utils.image_converter import run_image_selection

# First element is set by default for running in every algorithm
cnn_optimizers = ['adam', 'rmsprop', 'Ftrl', 'Nadam', 'Adamax']
cnn_loss = ['categorical_crossentropy', 'binary_crossentropy']
cnn_metrics = ['accuracy', 'binary_accuracy', 'categorical_accuracy', 'top_k_categorical_accuracy']
cnn_epochs_number = [50, 45, 20, 15, 10]
cnn_steps_for_validation = [10, 9, 8, 7, 6]
vgg_thresh = [0.5, 0.6, 0.7, 0.4, 0.3]
vgg_model = ['resnet50', 'vgg16', 'senet50']
pca_components = [100, 90, 80, 70, 60]
sift_cascades = ['haarcascade_frontalface_default',
                 'haarcascade_frontalface_alt',
                 'haarcascade_frontalface_alt_tree',
                 'haarcascade_frontalface_alt2']
sift_percent_delta = [2.0, 2.5, 4.0, 1.5, 1.0]
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
    # Blured and noise with 5% of blur att
    ['1\\1.jpg', 'noised', 'att', datasets[3][0]],
    ['1\\2.jpg', 'noised', 'att', datasets[3][0]],

    ['1\\1.jpg', 'blured', 'att', datasets[6][0]],
    ['1\\2.jpg', 'blured', 'att', datasets[6][0]],

    ['10\\5.jpg', 'noised', 'att', datasets[3][0]],
    ['10\\8.jpg', 'noised', 'att', datasets[3][0]],

    ['10\\5.jpg', 'blured', 'att', datasets[6][0]],
    ['10\\8.jpg', 'blured', 'att', datasets[6][0]],

    ['38\\5.jpg', 'blured', 'att', datasets[6][0]],
    ['38\\8.jpg', 'blured', 'att', datasets[6][0]],

    ['38\\5.jpg', 'noised', 'att', datasets[3][0]],
    ['38\\8.jpg', 'noised', 'att', datasets[3][0]],

    # Usual att
    ['1\\1.jpg', 'usual', 'att', datasets[0][0]],
    ['1\\2.jpg', 'usual', 'att', datasets[0][0]],

    ['2\\5.jpg', 'usual', 'att', datasets[0][0]],
    ['2\\8.jpg', 'usual', 'att', datasets[0][0]],

    ['10\\5.jpg', 'usual', 'att', datasets[0][0]],
    ['10\\8.jpg', 'usual', 'att', datasets[0][0]],

    ['17\\5.jpg', 'usual', 'att', datasets[0][0]],
    ['18\\9.jpg', 'usual', 'att', datasets[0][0]],

    ['19\\2.jpg', 'usual', 'att', datasets[0][0]],
    ['19\\10.jpg', 'usual', 'att', datasets[0][0]],

    ['20\\2.jpg', 'usual', 'att', datasets[0][0]],
    ['20\\10.jpg', 'usual', 'att', datasets[0][0]],

    ['29\\2.jpg', 'usual', 'att', datasets[0][0]],
    ['29\\10.jpg', 'usual', 'att', datasets[0][0]],

    ['38\\5.jpg', 'usual', 'att', datasets[0][0]],
    ['38\\8.jpg', 'usual', 'att', datasets[0][0]],

    # Usual tt
    ['face1\\1face1.jpg', 'usual', 'tt', datasets[1][0]],
    ['face1\\2face1.jpg', 'usual', 'tt', datasets[1][0]],

    ['face2\\10face2.jpg', 'usual', 'tt', datasets[1][0]],
    ['face2\\12face2.jpg', 'usual', 'tt', datasets[1][0]],

    ['face3\\5face3.jpg', 'usual', 'tt', datasets[1][0]],
    ['face3\\8face3.jpg', 'usual', 'tt', datasets[1][0]],

    ['face4\\5face4.jpg', 'usual', 'tt', datasets[1][0]],
    ['face4\\10face4.jpg', 'usual', 'tt', datasets[1][0]],

    ['face5\\10face5.jpg', 'usual', 'tt', datasets[1][0]],
    ['face5\\2face5.jpg', 'usual', 'tt', datasets[1][0]],

    ['face6\\8face6.jpg', 'usual', 'tt', datasets[1][0]],
    ['face6\\12face6.jpg', 'usual', 'tt', datasets[1][0]],

    # Blured and noise with 5% of blur tt
    ['face4\\5face4.jpg', 'blured', 'tt', datasets[4][0]],
    ['face4\\10face4.jpg', 'blured', 'tt', datasets[4][0]],

    ['face5\\10face5.jpg', 'blured', 'tt', datasets[4][0]],
    ['face5\\2face5.jpg', 'blured', 'tt', datasets[4][0]],

    ['face6\\8face6.jpg', 'blured', 'tt', datasets[4][0]],
    ['face6\\12face6.jpg', 'blured', 'tt', datasets[4][0]],

    ['face4\\5face4.jpg', 'noised', 'tt', datasets[7][0]],
    ['face4\\10face4.jpg', 'noised', 'tt', datasets[7][0]],

    ['face5\\10face5.jpg', 'noised', 'tt', datasets[7][0]],
    ['face5\\2face5.jpg', 'noised', 'tt', datasets[7][0]],

    ['face6\\8face6.jpg', 'noised', 'tt', datasets[7][0]],
    ['face6\\12face6.jpg', 'noised', 'tt', datasets[7][0]]
]


def run_sift():
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
        res = sift.comparison(test_image, original_image, sift_cascades[0])
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
    write(data, 'sift', 'usual')


def run_vgg():
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
            res = vgg_face.comparison(test_image, original_image, vgg_model[0],
                                      vgg_thresh[0])
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
    write(data, 'vgg_model', 'usual')


def run_cnn():
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
        if cnn.comparison(folder,
                          image_path,
                          cnn_epochs_number[0],
                          cnn_steps_for_validation[0],
                          cnn_optimizers[0],
                          cnn_loss[0],
                          cnn_metrics[0]) == str(face):
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
    write(data, 'cnn', 'complex')


def run_pca():
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
            pca_components[0])
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
    write(data, 'pca', 'complex')


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


def main():
    add_data()
    run_sift()
    # run_vgg()
    # run_pca()
    # run_cnn()


if __name__ == "__main__":
    main()
