import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

new_width = 326
new_height = 327


# noinspection PyTypeChecker
# This function is used to load images from path of test set,
# where all images are located in special folders
def load_data_set(path):
    faces = {}
    image_folders = os.listdir(path)
    for i in range(0, len(image_folders)):
        filepath = path + str(image_folders[i]) + "\\*.jpg"
        files_list = glob.glob(filepath)
        for j in range(0, len(files_list)):
            image_path = files_list[j]
            order = str(i+1) + "/" + str(j+1) + ".jpg"
            img = Image.open(image_path)
            img.load()
            if 'tt_dataset' in files_list[j]:
                img = img.resize((new_width, new_height), Image.ANTIALIAS)
            data = np.asarray(img, dtype="int32")
            faces[order] = data
    return faces, 'tt_dataset'


# noinspection PyTypeChecker
# This function is used to load images from path of user images
# or any other images, which are located in `common folder`
def load_set(path):
    faces = {}
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


# This function is used to run code both for tests/app
# (has different ways of face_shapes loading),
# implementing the logic of PCA, like:
# 1. Create NxM matrix with N images and M pixels per image
# 2. Apply PCA to use first K principal components as eigenfaces
# 3. Prepare KxN matrix (K is the number of eigenfaces, N is the number of samples)
def comparison(test_filename, path, data_type, n_components=100):
    if data_type == 'test':
        faces, set_type = load_data_set(path)
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

    face_matrix = np.array(face_matrix)
    pca = PCA().fit(face_matrix)
    eigenfaces = pca.components_[:n_components]
    weights = eigenfaces @ (face_matrix - pca.mean_).T
    print("Shape of the weight matrix:", weights.shape)
    load_test_file = Image.open(test_filename)
    load_test_file.load()
    if set_type == 'tt_dataset':
        load_test_file = load_test_file\
            .resize((new_width, new_height), Image.ANTIALIAS)

    # noinspection PyTypeChecker
    data = np.asarray(load_test_file, dtype="int32")
    image = {'image': data}

    query = image['image'].reshape(1, -1)
    query_weight = eigenfaces @ (query - pca.mean_).T
    euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
    best_match = np.argmin(euclidean_distance)
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    axes[0].imshow(query.reshape(face_shape), cmap="gray")
    axes[0].set_title("Query")
    axes[1].imshow(face_matrix[best_match].reshape(face_shape), cmap="gray")
    axes[1].set_title("Best match")
    plt.show()
    return face_label[best_match]
