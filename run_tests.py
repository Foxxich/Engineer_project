import random
import time

import definitons
from algorithms import sift, vgg_face, cnn, pca


def run_sift():
    test_image = definitons.root_dir + '\\images\\random_images\\1.jpg'
    original_image = definitons.root_dir + '\\images\\random_images\\2.jpg'
    start_time = time.time()
    sift.comparison(test_image, original_image)
    end_time = time.time()
    print("Total time: ", round((end_time - start_time)), ' Seconds')


def run_vgg():
    test_image = definitons.root_dir + '\\images\\random_images\\1.jpg'
    original_image = definitons.root_dir + '\\images\\random_images\\2.jpg'
    start_time = time.time()
    result = vgg_face.comparison(test_image, original_image)
    end_time = time.time()
    print("Total time: ", round((end_time - start_time)), ' Seconds')


def run_cnn():
    face = 2
    random_image_number = random.randrange(1, 4, 1)
    folder = definitons.root_dir + '\\images\\tt_dataset\\Final Training Images\\'
    image_path = definitons.root_dir + '\\images\\tt_dataset\\Final Testing Images\\face' + str(face) + '\\' + str(
        random_image_number) + 'face' + str(face) + '.jpg'
    start_time = time.time()
    epochs_number = 50
    steps_for_validation = 10
    if cnn.comparison(folder, image_path, epochs_number, steps_for_validation) == 'face' + str(face):
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
    if int(pca.comparison(test_file, path, 'test')) == 11:
        print('Same person on both images')
    else:
        print('Different persons on both images')
    end_time = time.time()
    print("Total time: ", round((end_time - start_time)), ' Seconds')


def main():
    run_sift()
    run_vgg()
    run_cnn()
    run_pca()


if __name__ == "__main__":
    main()
