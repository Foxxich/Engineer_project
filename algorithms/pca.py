import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import definitons
from sklearn.decomposition import PCA
# noinspection PyTypeChecker


def load_data_set():
    faces = {}
    path = definitons.ROOT_DIR + "\\images\\converted_images\\"
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


def main():
    faces = load_data_set()
    face_shape = list(faces.values())[0].shape
    classes = set(filename.split("/")[0] for filename in faces.keys())
    test_filename = "21/10.jpg"

    print("Face image shape:", face_shape)
    print("Number of classes:", len(classes))
    print("Number of images:", len(faces))

    face_matrix = []
    face_label = []
    for key, val in faces.items():
        if key == test_filename:
            continue
        face_matrix.append(val.flatten())
        face_label.append(key.split("/")[0])

    # Create NxM matrix with N images and M pixels per image
    face_matrix = np.array(face_matrix)
    # Apply PCA to use first K principal components as eigenfaces
    pca = PCA().fit(face_matrix)
    n_components = 50
    eigenfaces = pca.components_[:n_components]
    # Generate weights as a KxN matrix where K is the number of eigenfaces and N the number of samples
    weights = eigenfaces @ (face_matrix - pca.mean_).T
    print("Shape of the weight matrix:", weights.shape)
    # Test on out-of-sample image of existing class
    query = faces[test_filename].reshape(1, -1)
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


if __name__ == "__main__":
    main()
