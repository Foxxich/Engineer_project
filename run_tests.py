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
    ['images\\datasets\\tt_dataset\\Final Training Images', 'training_dataset'],
    ['images\\datasets\\tt_dataset\\Final Testing Images', 'testing_dataset'],
]

test_data = [
    ['face1\\1face1.jpg', 'usual', 'tt_dataset', datasets[0][0], 'face1'],
    ['face1\\2face1.jpg', 'usual', 'tt_dataset', datasets[0][0], 'face1'],
]


def run_sift():
    data = []
    for image1 in test_data:
        image2 = None
        for score in test_data:
            if score[2] == image1[2] and score[0] != image1[0]:
                image2 = score
        test_image = definitons.root_dir + '\\' + image1[3] + '\\' + image1[0]
        original_image = definitons.root_dir + '\\' + image2[3] + '\\' + image2[0]
        start_time = time.time()
        print(test_image)
        res = sift.comparison(test_image, original_image, sift_cascades[0])
        end_time = time.time()
        is_same_person = False
        if image1[4] == image2[4]:
            is_same_person = True
        data.append([
            str(image1[0]),
            str(image2[0]),
            str(res),
            str(is_same_person),
            str(round((end_time - start_time))),
            image1[2],
            image1[1],
        ])
        print("Total time: ", round((end_time - start_time)), ' Seconds')
    write(data, 'sift_cascades')


def run_vgg():
    data = []
    for image1 in test_data:
        image2 = None
        for score in test_data:
            if score[2] == image1[2] and score[0] != image1[0]:
                image2 = score
        test_image = definitons.root_dir + '\\' + image1[3] + '\\' + image1[0]
        original_image = definitons.root_dir + '\\' + image2[3] + '\\' + image2[0]
        start_time = time.time()
        print(test_image)
        res = vgg_face.comparison(test_image, original_image, vgg_model[0],
                                  vgg_thresh[0])
        end_time = time.time()
        is_same_person = False
        if image1[4] == image2[4]:
            is_same_person = True
        data.append([
            str(image1[0]),
            str(image2[0]),
            str(res),
            str(is_same_person),
            str(round((end_time - start_time))),
            image1[2],
            image1[1],
        ])
        print("Total time: ", round((end_time - start_time)), ' Seconds')
    write(data, 'vgg_model')


def run_cnn():
    data = []
    for image1 in test_data:
        image2 = None
        for score in test_data:
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
        is_same_person = False
        if image1[4] == image2[4]:
            is_same_person = True
        data.append([
            str(image1[0]),
            str(image2[0]),
            str(res),
            str(is_same_person),
            str(round((end_time - start_time))),
            image1[2],
            image1[1],
        ])
        print("Total time: ", round((end_time - start_time)), ' Seconds')
    write(data, 'cnn')


def run_pca():
    data = []
    for image1 in test_data:
        image2 = None
        for score in test_data:
            if score[2] == image1[2] and score[0] != image1[0]:
                image2 = score
        face = image2[0].replace("\\", " ").split()[0]
        folder = definitons.root_dir + '\\' + image1[3] + '\\'
        image_path = definitons.root_dir + '\\' + image2[3] + '\\' + image2[0]
        start_time = time.time()
        res = False
        comparison_result = pca.comparison(
                image_path,
                folder,
                'test',
                pca_components[0])
        if image1[2] == 'tt_dataset':
            if 'face' + comparison_result == str(face):
                res = True
        else:
            if comparison_result == str(face):
                res = True

        end_time = time.time()
        is_same_person = False
        if image1[4] == image2[4]:
            is_same_person = True
        data.append([
            str(image1[0]),
            str(image2[0]),
            str(res),
            str(is_same_person),
            str(round((end_time - start_time))),
            image1[2],
            image1[1],
        ])
        print("Total time: ", round((end_time - start_time)), ' Seconds')
    write(data, 'pca')


def generate_blured_images():
    run_image_selection('tt_dataset\\Final Training Images', 14, 'blured')
    run_image_selection('tt_dataset\\Final Testing Images', 5, 'blured')
    run_image_selection('converted_images', 11, 'blured')


def generate_gaussian():
    run_image_selection('tt_dataset\\Final Training Images', 14, 'noised')
    run_image_selection('tt_dataset\\Final Testing Images', 5, 'noised')
    run_image_selection('converted_images', 11, 'noised')


def main():
    run_sift()
    run_vgg()
    run_cnn()
    run_pca()


if __name__ == "__main__":
    main()
