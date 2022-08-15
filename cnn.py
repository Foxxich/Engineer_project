import pickle
import time
import tensorflow
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.models import Sequential
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


training_image_path = 'C:\\Users\\Vadym\\PycharmProjects\\Engineer_project\\Face Images\\Final Training Images\\'
image_path = 'C:\\Users\\Vadym\\PycharmProjects\\Engineer_project\\Face Images\\Final Testing Images\\face1\\1.jpg'


# This function is used to prepare both training and final test sets.
# First is prepared with defining pre-processing transformations on raw images of testing data.
# The final test set is generated without transformations.
def generate_sets():
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
    '''# Compiling the CNN'''
    # classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return classifier


def main():
    training_set, test_set = generate_sets()
    # class_indices have the numeric tag for each face
    train_classes = training_set.class_indices

    # Storing the face and the numeric tag for future reference
    result_map = {}
    for faceValue, faceName in zip(train_classes.values(), train_classes.keys()):
        result_map[faceValue] = faceName

    with open("ResultsMap.pkl", 'wb') as fileWriteStream:
        pickle.dump(result_map, fileWriteStream)

    # The model will give answer as a numeric tag
    # This mapping will help to get the corresponding face name for it
    print("Mapping of Face and its ID", result_map)

    # The number of neurons for the output layer is equal to the number of faces
    output_neurons = len(result_map)
    print('\n The Number of output neurons: ', output_neurons)

    classifier = prepare_classifier(output_neurons)
    # Measuring the time taken by the model to train
    start_time = time.time()
    steps_per_epoch = len(test_set)

    # Starting the model training
    classifier.fit(
        training_set,
        steps_per_epoch=steps_per_epoch,
        epochs=50,
        validation_data=test_set,
        validation_steps=10)

    end_time = time.time()
    print("###### Total Time Taken: ", round((end_time - start_time)), 'Seconds ######')

    test_image = tensorflow.keras.utils.load_img(image_path, target_size=(64, 64))
    test_image = tensorflow.keras.utils.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis=0)

    result = classifier.predict(test_image, verbose=0)
    # print(training_set.class_indices)

    print('####' * 10)
    print('Prediction is: ', result_map[np.argmax(result)])


if __name__ == "__main__":
    main()
