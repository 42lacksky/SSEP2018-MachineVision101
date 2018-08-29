import os
import mnist
import numpy as np

from scipy.misc import imsave, imread, imresize
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from model.model import MV101_KNN


def prepare_mnist_data(image_arrays):
    data = np.zeros((image_arrays.shape[0], 784))

    for i, image_array in enumerate(image_arrays):
        data[i] = image_array.flatten()

    return np.array(data)


def get_mnist_data():
    X_train_mnist = prepare_mnist_data(mnist.train_images())
    y_train_mnist = mnist.train_labels()

    X_test_mnist = prepare_mnist_data(mnist.test_images())
    y_test_mnist = mnist.test_labels()

    return np.concatenate((X_train_mnist, X_test_mnist)), np.concatenate((y_train_mnist, y_test_mnist))


def get_our_data():
    test_data_root = os.path.abspath('./data/')
    abs_path = lambda x: os.path.join(test_data_root, x)

    images = []
    labels = []

    for file_name in os.listdir(test_data_root):
        if (".png" not in file_name) or ("_" not in file_name) or ("None" in file_name):
            continue
        x = imread(abs_path(file_name), mode='L')
        x = np.invert(x)
        x = resize(x, (28, 28))
        x = x.flatten() * 255

        images.append(x)
        labels.append(int(file_name.split("_")[0]))

    return np.array(images), np.array(labels)


def get_all_data():
    X_train, X_text, y_train, y_test = train_test_split(*get_our_data(), test_size=0.3)
    X_mnist, y_mnist = get_mnist_data()

    return np.concatenate((X_mnist, X_train)), np.concatenate((y_mnist, y_train)), X_text, y_test


def train_model():
    X_train, X_test, y_train, y_test = get_all_data()

    knn = MV101_KNN(k=5)
    knn = knn.fit(X_train, y_train)

    knn.save()

    return knn
