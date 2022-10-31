import pickle

import numpy as np
import tensorflow
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


import definitons


class CNN:
    def __init__(self,
                 training_image_path,
                 image_path,
                 map_type,
                 epochs_number=30,
                 steps_for_validation=20,
                 optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics='top_k_categorical_accuracy'):
        self.training_image_path = training_image_path
        self.image_path = image_path
        self.map_type = map_type
        self.epochs_number = epochs_number
        self.steps_for_validation = steps_for_validation
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.training_set, self.test_set = self.generate_sets()
        self.train_classes = self.training_set.class_indices
        self.result_map = {}
        for faceValue, faceName in zip(self.train_classes.values(), self.train_classes.keys()):
            self.result_map[faceValue] = faceName
        if map_type == 'user':
            with open(definitons.root_dir + "\\utils\\cnn\\ResultsMapUser.pkl", 'wb') as fileWriteStream:
                pickle.dump(self.result_map, fileWriteStream)
        else:
            with open(definitons.root_dir + "\\utils\\cnn\\ResultsMapTest.pkl", 'wb') as fileWriteStream:
                pickle.dump(self.result_map, fileWriteStream)
        self.output_neurons = len(self.result_map)

    # This function is used to prepare both training and final test sets.
    # First is prepared with defining pre-processing transformations on raw images of testing data.
    # The final test set is generated without transformations.
    def generate_sets(self):
        test_datagen = ImageDataGenerator()

        train_datagen = ImageDataGenerator(
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True)

        training_set = train_datagen.flow_from_directory(
            self.training_image_path,
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical', )

        test_set = test_datagen.flow_from_directory(
            self.training_image_path,
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
    def prepare_classifier(self):
        classifier = Sequential()
        classifier.add(
            Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64, 64, 3), activation='relu'))
        classifier.add(MaxPool2D(pool_size=(2, 2)))
        classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        classifier.add(MaxPool2D(pool_size=(2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(64, activation='relu'))
        classifier.add(Dense(self.output_neurons, activation='softmax'))
        classifier.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metrics])
        return classifier

    # This function is used to get prediction what type of image is in data set;
    # The way it's done is getting the max value between all of them in map of the results
    def final_prediction(self, classifier):
        test_image = tensorflow.keras.utils.load_img(self.image_path, target_size=(64, 64))
        test_image = tensorflow.keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = classifier.predict(test_image, verbose=0)
        print('####' * 10)
        print('Prediction is: ', self.result_map[np.argmax(result)])
        return self.result_map[np.argmax(result)]

    # This function is used to run code both for tests/app, implementing
    # the logic of CNN, like getting generated set, prepare classifier and making final prediction
    def comparison(self):
        print("Mapping of Face and its ID", self.result_map)
        print('\n The Number of output neurons: ', self.output_neurons)

        classifier = self.prepare_classifier()
        steps_per_epoch = len(self.test_set)

        classifier.fit(
            self.training_set,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs_number,
            validation_data=self.test_set,
            validation_steps=self.steps_for_validation)

        return self.final_prediction(classifier)
