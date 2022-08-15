# Import libraries
from matplotlib import pyplot as plt
from matplotlib.image import imread
from numpy.linalg import inv
import numpy as np
import os

dataset_path = 'C:\\Users\\Vadym\\PycharmProjects\\Engineer_project\\att_faces\\'
dataset_dir  = os.listdir(dataset_path)

width  = 92
height = 112

number_of_classes=10
img_in_class=6

print('Train Images:')

# to store all the training images in an array
training_tensor   = np.ndarray(shape=(number_of_classes*img_in_class, height*width), dtype=np.float64)

for i in range(number_of_classes):
    for j in range(img_in_class):
        img = plt.imread(dataset_path + 'training1\\s'+str(i+1)+'/'+str(j+1)+'.pgm')
        # copying images to the training array
        training_tensor[img_in_class*i+j,:] = np.array(img, dtype='float64').flatten()
        # plotting the training images
        plt.subplot(number_of_classes,img_in_class,1+img_in_class*i+j)
        plt.imshow(img, cmap='gray')
        plt.subplots_adjust(right=1.2, top=2.5)
        plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
plt.show()

print('Test Images:')
testing_tensor = np.ndarray(shape=(44, height*width), dtype=np.float64)

for i in range(44):
    img = imread(dataset_path + 'test\\'+str(i+1)+'.pgm')
    testing_tensor[i,:] = np.array(img, dtype='float64').flatten()
    plt.subplot(11,4,1+i)
    plt.imshow(img, cmap='gray')
    plt.subplots_adjust(right=1.2, top=2.5)
    plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
plt.show()


def PCA(training_tensor, number_chosen_components):
    mean_face = np.zeros((1, height * width))
    for i in training_tensor:
        mean_face = np.add(mean_face, i)
    mean_face = np.divide(mean_face, float(training_tensor.shape[0])).flatten()

    #     plt.title('Mean face')
    #     plt.imshow(mean_face.reshape(height, width), cmap='gray')
    #     plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
    #     plt.show()

    #     print('Normalized faces:')
    normalised_training_tensor = np.ndarray(shape=(training_tensor.shape))
    for i in range(training_tensor.shape[0]):
        normalised_training_tensor[i] = np.subtract(training_tensor[i], mean_face)
    #     for i in range(len(training_tensor)):
    #         img = normalised_training_tensor[i].reshape(height,width)
    #         plt.subplot(10,6,1+i)
    #         plt.imshow(img, cmap='gray')
    #         plt.subplots_adjust(right=1.2, top=2.5)
    #         plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
    #     plt.show()
    cov_matrix = np.cov(normalised_training_tensor)
    cov_matrix = np.divide(cov_matrix, float(training_tensor.shape[0]))

    eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
    eig_pairs = [(eigenvalues[index], eigenvectors[:, index]) for index in range(len(eigenvalues))]

    # Sort the eigen pairs in descending order:
    eig_pairs.sort(reverse=True)
    eigvalues_sort = [eig_pairs[index][0] for index in range(len(eigenvalues))]
    eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]

    reduced_data = np.array(eigvectors_sort[:number_chosen_components]).transpose()

    proj_data = np.dot(training_tensor.transpose(), reduced_data)
    proj_data = proj_data.transpose()

    #     print('Projected Data:')
    #     for i in range(proj_data.shape[0]):
    #         img = proj_data[i].reshape(height,width)
    #         plt.subplot(10,10,1+i)
    #         plt.imshow(img, cmap='jet')
    #         plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
    #         plt.subplots_adjust(right=1.2, top=2.5)
    #     plt.show()

    wx = np.array([np.dot(proj_data, img) for img in normalised_training_tensor])

    return proj_data, wx

number_chosen_components = 30
projected_data, projected_sig = PCA(training_tensor, number_chosen_components)
projected_sig.shape

mew = np.zeros((number_of_classes, number_chosen_components))
M = np.zeros((1,number_chosen_components))

for i in range(number_of_classes):
    xa = projected_sig[img_in_class*i:img_in_class*i+img_in_class,:]
    for j in xa:
        mew[i,:] = np.add(mew[i,:],j)
    mew[i,:] = np.divide(mew[i,:],float(len(xa)))

for i in projected_sig:
    M = np.add(M,i)
M = np.divide(M,float(len(projected_sig)))

M.shape
mew.shape

# normalised within class data
normalised_wc_proj_sig = np.ndarray(shape=(number_of_classes*img_in_class, number_chosen_components), dtype=np.float64)

for i in range(number_of_classes):
    for j in range(img_in_class):
        normalised_wc_proj_sig[i*img_in_class+j,:] = np.subtract(projected_sig[i*img_in_class+j,:],mew[i,:])
normalised_wc_proj_sig.shape

sw = np.zeros((number_chosen_components,number_chosen_components))

for i in range(number_of_classes):
    xa = normalised_wc_proj_sig[img_in_class*i:img_in_class*i+img_in_class,:]
    xa = xa.transpose()
    cov = np.dot(xa,xa.T)
    sw = sw + cov
sw.shape

normalised_proj_sig = np.ndarray(shape=(number_of_classes*img_in_class, number_chosen_components), dtype=np.float64)
for i in range(number_of_classes*img_in_class):
    normalised_proj_sig[i,:] = np.subtract(projected_sig[i,:],M)

sb = np.dot(normalised_proj_sig.T,normalised_proj_sig)
sb = np.multiply(sb,float(img_in_class))
sb.shape

J = np.dot(inv(sw), sb)
J.shape

J = np.dot(inv(sw), sb)
J.shape

eigenvalues, eigenvectors, = np.linalg.eig(J)
# eigenvectors = abs(eigenvectors)
print('Eigenvectors of Cov(X):')
print(eigenvectors)
# eigenvalues = abs(eigenvalues)
print('Eigenvalues of Cov(X):',eigenvalues)

# get corresponding eigenvectors to eigen values
# so as to get the eigenvectors at the same corresponding index to eigen values when sorted
eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]

# Sort the eigen pairs in descending order:
eig_pairs.sort(reverse=True)
eigvalues_sort  = [eig_pairs[index][0] for index in range(len(eigenvalues))]
eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]

# Find cumulative variance of each principle component
var_comp_sum = np.cumsum(eigvalues_sort)/sum(eigvalues_sort)

# Show cumulative proportion of varaince with respect to components
print("Cumulative proportion of variance explained vector:", var_comp_sum)

# x-axis for number of principal components kept
num_comp = range(1,len(eigvalues_sort)+1)
plt.title('Cum. Prop. Variance and Components Kept')
plt.xlabel('Principal Components')
plt.ylabel('Cum. Prop. Variance ')

plt.scatter(num_comp, var_comp_sum)
plt.show()

print('Number of eigen vectors:',len(eigvalues_sort))

# Choosing the necessary number of principle components
number_chosen_components = 15
print("k:",number_chosen_components)
reduced_data = np.array(eigvectors_sort[:number_chosen_components]).transpose()
reduced_data.shape

print('Number of eigen vectors:',len(eigvalues_sort))

# Choosing the necessary number of principle components
number_chosen_components = 15
print("k:",number_chosen_components)
reduced_data = np.array(eigvectors_sort[:number_chosen_components]).transpose()
reduced_data.shape

projected_sig.shape
FP = np.dot(projected_sig, reduced_data)
FP.shape

# get projected data ---> eigen space

proj_data1 = np.dot(training_tensor.transpose(),FP)
proj_data1 = proj_data1.transpose()
proj_data1.shape

# plotting of eigen faces --> the information retained after applying lossing transformation
for i in range(proj_data1.shape[0]):
    img = proj_data1[i].reshape(height,width)
    #print(img)
    plt.subplot(10,3,1+i)
    plt.imshow((img.real*img.real + img.imag*img.imag)**0.5, cmap='gray')
    plt.subplots_adjust(right=1.2, top=2.5)
    plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
plt.show()

mean_face = np.zeros((1,height*width))

for i in training_tensor:
    mean_face = np.add(mean_face,i)

mean_face = np.divide(mean_face,float(len(training_tensor))).flatten()

plt.imshow(mean_face.reshape(height, width), cmap='gray')
plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
plt.show()

# Testing all the images

count = 0
num_images = 0
correct_pred = 0


def recogniser(img_number):
    global count, highest_min, num_images, correct_pred

    num_images += 1
    unknown_face_vector = testing_tensor[img_number, :]
    normalised_uface_vector = np.subtract(unknown_face_vector, mean_face)

    plt.subplot(11, 8, 1 + count)
    plt.imshow(unknown_face_vector.reshape(height, width), cmap='gray')
    plt.title('Input:' + str(img_number + 1))
    plt.subplots_adjust(right=1.2, top=2.5)
    plt.tick_params(labelleft=False, labelbottom=False, bottom=False, top=False, right=False, left=False, which='both')
    count += 1

    PEF = np.dot(projected_data, normalised_uface_vector)
    proj_fisher_test_img = np.dot(reduced_data.T, PEF)
    diff = FP - proj_fisher_test_img
    norms = np.linalg.norm(diff, axis=1)
    #     print(norms.shape)
    index = np.argmin(norms)

    plt.subplot(11, 8, 1 + count)

    set_number = int(img_number / 4)
    #     print(set_number)

    t0 = 7000000

    #     if(img_number>=40):
    #         print(norms[index])

    if norms[index] < t0:
        if (index >= (6 * set_number) and index < (6 * (set_number + 1))):
            plt.title('Matched with:' + str(index + 1), color='g')
            plt.imshow(training_tensor[index, :].reshape(height, width), cmap='gray')
            correct_pred += 1
        else:
            plt.title('Matched with:' + str(index + 1), color='r')
            plt.imshow(training_tensor[index, :].reshape(height, width), cmap='gray')
    else:
        if (img_number >= 40):
            plt.title('Unknown face!', color='g')
            correct_pred += 1
        else:
            plt.title('Unknown face!', color='r')
    plt.subplots_adjust(right=1.2, top=2.5)
    plt.tick_params(labelleft=False, labelbottom=False, bottom=False, top=False, right=False, left=False, which='both')
    count += 1


fig = plt.figure(figsize=(10, 10))
for i in range(len(testing_tensor)):
    recogniser(i)

plt.show()

print('Correct predictions: {}/{} = {}%'.format(correct_pred, num_images, correct_pred / num_images * 100.00))

accuracy = np.zeros(len(eigvalues_sort))


def func(img_number, reduced_data, FP, num_images, correct_pred):
    num_images += 1
    unknown_face_vector = testing_tensor[img_number, :]
    normalised_uface_vector = np.subtract(unknown_face_vector, mean_face)

    PEF = np.dot(projected_data, normalised_uface_vector)
    proj_fisher_test_img = np.dot(reduced_data.T, PEF)
    diff = FP - proj_fisher_test_img
    norms = np.linalg.norm(diff, axis=1)
    index = np.argmin(norms)

    set_number = int(img_number / 4)

    t0 = 7000000

    if norms[index] < t0:
        if (index >= (6 * set_number) and index < (6 * (set_number + 1))):
            correct_pred += 1
    else:
        if (img_number >= 40):
            correct_pred += 1

    return num_images, correct_pred


def calculate(k):
    #     print("k:",k)
    reduced_data = np.array(eigvectors_sort[:k]).transpose()

    FP = np.dot(projected_sig, reduced_data)

    num_images = 0
    correct_pred = 0

    for i in range(len(testing_tensor)):
        num_images, correct_pred = func(i, reduced_data, FP, num_images, correct_pred)
    #     print(FP.shape)
    accuracy[k] = correct_pred / num_images * 100.00


print('Total Number of eigenvectors:', len(eigvalues_sort))
for i in range(1, len(eigvalues_sort)):
    calculate(i)

fig, axi = plt.subplots()
axi.plot(np.arange(len(eigvalues_sort)), accuracy, 'b')
axi.set_xlabel('Number of eigen values')
axi.set_ylabel('Accuracy')
axi.set_title('Accuracy vs. k-value')

