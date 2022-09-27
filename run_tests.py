import random
import time

import definitons
from algorithms import sift, vgg_face, cnn, pca


# First element is set by default for running in every algorithm
cnn_optimizers = ['adam', 'rmsprop', 'Ftrl', 'Nadam', 'Adamax']
cnn_loss = ['categorical_crossentropy', 'binary_crossentropy']
cnn_metrics = ['accuracy', 'binary_accuracy', 'categorical_accuracy', 'top_k_categorical_accuracy']
cnn_epochs_number = [50, 45, 40, 35, 30, 25, 20, 15, 10]
cnn_steps_for_validation = [10, 9, 8, 7, 6, 5]
vgg_thresh = [0.5, 0.6, 0.7, 0.8, 0.4, 0.3]
vgg_model = ['resnet50', 'vgg16', 'senet50']
pca_components = [100, 90, 80, 70, 60, 50]
sift_cascades = ['haarcascade_frontalface_default',
                 'haarcascade_frontalface_alt']
sift_percent_delta = [2.0, 2.5, 3.0, 5.0, 1.5, 1.0]


def run_sift():
    test_image = definitons.root_dir + '\\images\\random_images\\1.jpg'
    original_image = definitons.root_dir + '\\images\\random_images\\2.jpg'
    start_time = time.time()
    sift.comparison(test_image, original_image, sift_cascades[0])
    end_time = time.time()
    print("Total time: ", round((end_time - start_time)), ' Seconds')


def run_vgg():
    test_image = definitons.root_dir + '\\images\\random_images\\1.jpg'
    original_image = definitons.root_dir + '\\images\\random_images\\2.jpg'
    start_time = time.time()
    result = vgg_face.comparison(test_image, original_image, vgg_model[0], vgg_thresh[0])
    end_time = time.time()
    print("Total time: ", round((end_time - start_time)), ' Seconds')


def run_cnn():
    face = 2
    random_image_number = random.randrange(1, 4, 1)
    folder = definitons.root_dir + '\\images\\tt_dataset\\Final Training Images\\'
    image_path = definitons.root_dir + '\\images\\tt_dataset\\Final Testing Images\\face' + str(face) + '\\' + str(
        random_image_number) + 'face' + str(face) + '.jpg'
    start_time = time.time()
    if cnn.comparison(folder,
                      image_path,
                      cnn_epochs_number[0],
                      cnn_steps_for_validation[0],
                      cnn_optimizers[0],
                      cnn_loss[0],
                      cnn_metrics[0]) == 'face' + str(face):
        print('Same person on both images')
    else:
        print('Different persons on both images')
    end_time = time.time()
    print("Total time: ", round((end_time - start_time)), ' Seconds')


def run_pca():
    face = 21
    random_image_number = random.randrange(1, 10, 1)
    print(random_image_number)
    test_file = definitons.root_dir + "\\images\\converted_images\\11\\8.jpg"
    path = definitons.root_dir + "\\images\\converted_images\\"
    start_time = time.time()
    if int(pca.comparison(test_file, path, 'test', pca_components[0])) == 11:
        print('Same person on both images')
    else:
        print('Different persons on both images')
    end_time = time.time()
    print("Total time: ", round((end_time - start_time)), ' Seconds')


def main():
    # run_sift()
    # run_vgg()
    # run_cnn()
    run_pca()


if __name__ == "__main__":
    main()
