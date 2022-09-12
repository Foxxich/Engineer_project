import os

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
def prepare_classifier(output_neurons):
    classifier = Sequential()
    classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64, 64, 3), activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2, 2)))
    classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(64, activation='relu'))
    classifier.add(Dense(output_neurons, activation='softmax'))
    classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
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
