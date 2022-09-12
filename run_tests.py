import pickle
import time
from algorithms import sift, vgg_face, cnn
import definitons


def run_sift():
    test_image = definitons.ROOT_DIR + '\\1.jpg'
    original_image = definitons.ROOT_DIR + '\\3.jpg'
    start_time = time.time()
    sift.comparison(test_image, original_image)
    end_time = time.time()
    print("Total time: ", round((end_time - start_time)), ' Seconds')


def run_vgg():
    test_image = definitons.ROOT_DIR + '\\1.jpg'
    original_image = definitons.ROOT_DIR + '\\3.jpg'
    start_time = time.time()
    result = vgg_face.comparison(test_image, original_image)
    end_time = time.time()
    print("Total time: ", round((end_time - start_time)), ' Seconds')


def run_cnn():
    training_image_path = definitons.ROOT_DIR + '\\Face Images\\Final Training Images\\'
    image_path = definitons.ROOT_DIR + '\\Face Images\\Final Testing Images\\face2\\1face2.jpg'
    epochs_number = 50
    steps_for_validation = 10
    training_set, test_set = cnn.generate_sets(training_image_path)
    train_classes = training_set.class_indices
    result_map = {}
    for faceValue, faceName in zip(train_classes.values(), train_classes.keys()):
        result_map[faceValue] = faceName

    with open("algorithms/ResultsMap.pkl", 'wb') as fileWriteStream:
        pickle.dump(result_map, fileWriteStream)
    output_neurons = len(result_map)

    print("Mapping of Face and its ID", result_map)
    print('\n The Number of output neurons: ', output_neurons)

    classifier = cnn.prepare_classifier(output_neurons)
    start_time = time.time()
    steps_per_epoch = len(test_set)

    classifier.fit(
        training_set,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs_number,
        validation_data=test_set,
        validation_steps=steps_for_validation)

    end_time = time.time()
    print("Total time: ", round((end_time - start_time)), ' Seconds')
    cnn.final_prediction(image_path, classifier, result_map)


def run_pca():
    pass


def main():
    run_cnn()


if __name__ == "__main__":
    main()
