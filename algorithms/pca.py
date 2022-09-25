import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


# noinspection PyTypeChecker


# This function is used to load images from path of test set
def load_data_set(path):
    faces = {}
    image_folders = os.listdir(path)
    for i in range(1, len(image_folders) + 1):
        filepath = path + str(i) + "\\*.jpg"
        files_list = glob.glob(filepath)
        for j in range(1, len(files_list) + 1):
            image_path = path + str(i) + "\\" + str(j) + ".jpg"
            order = str(i) + "/" + str(j) + ".jpg"
            img = Image.open(image_path)
            img.load()
            data = np.asarray(img, dtype="int32")
            faces[order] = data
    return faces


def load_set(path):
    faces = {}
    final_faces = {}
    image_folders = os.listdir(path)
    for i in range(1, len(image_folders)):
        if '.jpg' not in image_folders[i]:
            filepath = path + str(image_folders[i]) + "\\*.jpg"
            files_list = glob.glob(filepath)
            order = str(image_folders[i]) + ".jpg"
            img = Image.open(files_list[0])
            img.load()
            data = np.asarray(img, dtype="int32")
            faces[order] = data
    return faces


def comparison(test_filename, path, data_type):
    if data_type == 'test':
        faces = load_data_set(path)
    else:
        faces = load_set(path)
    face_shape = list(faces.values())[0].shape
    classes = set(filename.split("/")[0] for filename in faces.keys())

    print("Face image shape:", face_shape)
    print("Number of classes:", len(classes))
    print("Number of images:", len(faces))

    face_matrix = []
    face_label = []
    for key, val in faces.items():
        face_matrix.append(val.flatten())
        face_label.append(key.split("/")[0])

    # Create NxM matrix with N images and M pixels per image
    face_matrix = np.array(face_matrix)
    # Apply PCA to use first K principal components as eigenfaces
    pca = PCA().fit(face_matrix)
    n_components = 100
    eigenfaces = pca.components_[:n_components]
    # Generate weights as a KxN matrix where K is the number of eigenfaces and N the number of samples
    weights = eigenfaces @ (face_matrix - pca.mean_).T
    print("Shape of the weight matrix:", weights.shape)
    # Test on out-of-sample image of existing class

    load_test_file = Image.open(test_filename)
    load_test_file.load()
    data = np.asarray(load_test_file, dtype="int32")
    test = {'test': data}

    query = test['test'].reshape(1, -1)
    query_weight = eigenfaces @ (query - pca.mean_).T
    euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
    best_match = np.argmin(euclidean_distance)
    print("Best match %s with Euclidean distance %f" % (face_label[best_match], euclidean_distance[best_match]))
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    axes[0].imshow(query.reshape(face_shape), cmap="gray")
    axes[0].set_title("Query")
    axes[1].imshow(face_matrix[best_match].reshape(face_shape), cmap="gray")
    axes[1].set_title("Best match")
    plt.show()
    print('Person number', face_label[best_match])
    return face_label[best_match]
