import os
import mnist
import numpy as np

from scipy.misc import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from model.model import MV101_KNN


def get_mnist_data():
    def prepare_mnist_data(image_arrays):
        data = np.zeros((image_arrays.shape[0], 784))

        for i, image_array in enumerate(image_arrays):
            data[i] = image_array.flatten()

        return np.array(data)

    X_train_mnist = prepare_mnist_data(mnist.train_images())
    y_train_mnist = mnist.train_labels()

    X_test_mnist = prepare_mnist_data(mnist.test_images())
    y_test_mnist = mnist.test_labels()

    return np.concatenate((X_train_mnist, X_test_mnist)), np.concatenate((y_train_mnist, y_test_mnist))


def get_our_data(data_path):
    test_data_root = os.path.abspath(data_path)
    abs_path = lambda x: os.path.join(test_data_root, x)

    images = []
    labels = []

    for file_name in os.listdir(test_data_root):
        label = file_name.split("_")[0]
        if not label.isnumeric():
            continue

        x = imread(abs_path(file_name), mode='L')
        x = np.invert(x)
        x = resize(x, (28, 28))
        x = x.flatten() * 255

        images.append(x)
        labels.append(int(label))

    return np.array(images), np.array(labels)


def get_all_data(data_path, use_mnist=False):
    X, y = get_our_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    if use_mnist:
        X_mnist, y_mnist = get_mnist_data()

        X_train = np.concatenate((X_mnist, X_train))
        y_train = np.concatenate((y_mnist, y_train))

    return X_train, y_train, X_test, y_test


def train_model(k=5, data_path="data"):
    X_train, y_train, X_test, y_test = get_all_data(data_path)

    knn = MV101_KNN(k)
    knn = knn.fit(np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)))

    knn.save()

    return knn