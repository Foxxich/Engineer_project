import pickle

import numpy as np
import tensorflow
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


# This function is used to prepare both training and final test sets.
# First is prepared with defining pre-processing transformations on raw images of testing data.
# The final test set is generated without transformations.
import definitons


def generate_sets(training_image_path):
    test_datagen = ImageDataGenerator()

    train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

    training_set = train_datagen.flow_from_directory(
        training_image_path,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical', )

    test_set = test_datagen.flow_from_directory(
        training_image_path,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical', )

    return training_set, test_set


# This function consists of a few main functionalities:
# 1. adding the first layer of CNN;
# 2. max pooling and adding of additional layer of convolution for better accuracy;
# 3. flattening;
# 4. connecting of Neural Network;
# This implementation is used with the format (64,64,3) for TensorFlow backend;
# It is done to represent 3 matrix (with each size 64 to 64) for
# representing Red, Green and Blue components of pixels;
def prepare_classifier(output_neurons, optimizer, loss, metrics):
    classifier = Sequential()
    classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64, 64, 3), activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2, 2)))
    classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(64, activation='relu'))
    classifier.add(Dense(output_neurons, activation='softmax'))
    classifier.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
    return classifier


# This function is used to get prediction what type of image is in data set;
# The way it's done is getting the max value between all of them in map of the results
def final_prediction(image_path, classifier, result_map):
    test_image = tensorflow.keras.utils.load_img(image_path, target_size=(64, 64))
    test_image = tensorflow.keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image, verbose=0)
    print('####' * 10)
    print('Prediction is: ', result_map[np.argmax(result)])
    return result_map[np.argmax(result)]


# This function is used to run code both for tests/app, implementing
# the logic of CNN, like getting generated set, prepare classifier and making final prediction
def comparison(training_image_path,
               image_path,
               map_type,
               epochs_number=30,
               steps_for_validation=20,
               optimizer='adam',
               loss='categorical_crossentropy',
               metrics='top_k_categorical_accuracy',
               ):
    training_set, test_set = generate_sets(training_image_path)
    train_classes = training_set.class_indices
    result_map = {}
    for faceValue, faceName in zip(train_classes.values(), train_classes.keys()):
        result_map[faceValue] = faceName
    if map_type == 'user':
        with open(definitons.root_dir + "\\utils\\cnn\\ResultsMapUser.pkl", 'wb') as fileWriteStream:
            pickle.dump(result_map, fileWriteStream)
    else:
        with open(definitons.root_dir + "\\utils\\cnn\\ResultsMapTest.pkl", 'wb') as fileWriteStream:
            pickle.dump(result_map, fileWriteStream)
    output_neurons = len(result_map)

    print("Mapping of Face and its ID", result_map)
    print('\n The Number of output neurons: ', output_neurons)

    classifier = prepare_classifier(output_neurons, optimizer, loss, metrics)
    steps_per_epoch = len(test_set)

    classifier.fit(
        training_set,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs_number,
        validation_data=test_set,
        validation_steps=steps_for_validation)

    return final_prediction(image_path, classifier, result_map)
