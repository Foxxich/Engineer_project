import time
import definitons
from algorithms import sift, vgg_face, cnn, pca


def run_sift():
    test_image = definitons.root_dir + '\\images\\user_images\\1.jpg'
    original_image = definitons.root_dir + '\\images\\user_images\\2.jpg'
    start_time = time.time()
    sift.comparison(test_image, original_image)
    end_time = time.time()
    print("Total time: ", round((end_time - start_time)), ' Seconds')


def run_vgg():
    test_image = definitons.root_dir + '\\images\\user_images\\1.jpg'
    original_image = definitons.root_dir + '\\images\\user_images\\2.jpg'
    start_time = time.time()
    result = vgg_face.comparison(test_image, original_image)
    end_time = time.time()
    print("Total time: ", round((end_time - start_time)), ' Seconds')


def run_cnn():
    print("LOL")
    folder = definitons.root_dir + '\\images\\tt_dataset\\Final Training Images\\'
    image_path = definitons.root_dir + '\\images\\tt_dataset\\Final Testing Images\\face2\\1face2.jpg'
    start_time = time.time()
    epochs_number = 50
    steps_for_validation = 10
    result = cnn.comparison(folder, image_path, epochs_number, steps_for_validation)
    end_time = time.time()
    print("Total time: ", round((end_time - start_time)), ' Seconds')


def run_pca():
    test_file = "21/10.jpg"
    path = definitons.root_dir + "\\images\\converted_images\\"
    start_time = time.time()
    result = pca.comparison(test_file, path)
    end_time = time.time()
    print("Total time: ", round((end_time - start_time)), ' Seconds')


def main():
    run_pca()


if __name__ == "__main__":
    main()
