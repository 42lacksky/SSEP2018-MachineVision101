import time
import numpy as np
import pickle


class MV101_KNN(object):
    def __init__(self, k):
        self.k = k
        self.data = None
        self.labels = None

    def fit(self, data, labels):
        self.data = data
        self.labels = labels

        return self

    def predict(self, one_data):
        distances = self.__dist(one_data)
        sorted_indexes = self.__sort(distances).astype(int)
        first_k_indexes = self.__choose_k_first(sorted_indexes)
        correct_labels = self.__choose_correct_labels(first_k_indexes)
        labels_counts = self.__count_labels(correct_labels)
        max_index = self.__find_max(labels_counts)

        print(max_index)
        return max_index

    def save(self):
        with open("knn.pkl", "wb") as f:
            pickle.dump(self, f)

    def __dist(self, one_data):
        """
        1. Измерение расстояний (Паша)
        """
        start_time = time.time()
        distances = np.zeros((self.data.shape[0],))

        for i, image in enumerate(self.data):
            distances[i] = np.sum(np.abs(one_data - image))

        print(time.time() - start_time)
        return distances

    def __sort(self, distances):
        """
        2. Быстрая сортировка (Даня)
        """

        def sort(a):
            if a.shape[0] <= 1:
                return a

            central = a[a.shape[0] // 2]

            low = a[a < central]
            equal = a[a == central]
            high = a[a > central]

            low = sort(low)
            high = sort(high)

            return np.concatenate((low, equal, high))

        def recreate(array, shuffled):
            result = np.zeros((array.shape[0],))

            for i, el in enumerate(shuffled):
                for similar in np.where(array == el)[0]:
                    if result[result == similar].shape[0] > 0:
                        continue
                    result[i] = int(similar)
            return result

        start_time = time.time()
        result = recreate(distances, sort(distances))
        print(time.time() - start_time)

        return result

    def __choose_k_first(self, sorted_indexes):
        """
        3. Первые k элементов (Ваня)
        """
        start_time = time.time()
        first_k_indexes = []
        for i in range(self.k if sorted_indexes.shape[0] > self.k else sorted_indexes.shape[0]):
            first_k_indexes.append(sorted_indexes[i])

        print(time.time() - start_time)

        return np.array(first_k_indexes)

    def __choose_correct_labels(self, first_k_indexes):
        """
        4. Получить нужные лейблы (Гоша)
        """
        start_time = time.time()
        correct_labels = self.labels[first_k_indexes]
        print(time.time() - start_time)

        return correct_labels

    def __count_labels(self, correct_labels):
        """
        5. Подсчитать количество каждого лейбла (Гоша)
        """
        labels_counts = np.zeros((10,))
        start_time = time.time()
        for label in correct_labels:
            labels_counts[label] += 1

        print(time.time() - start_time)

        return labels_counts

    def __find_max(self, label_counts):
        """
        6. Найти максимальный элемент массива (Маша)
        """
        starttime = time.time()
        i = 0
        x = -1
        y = 0
        for i in range(len(label_counts)):
            if label_counts[i] > x:
                x = label_counts[i]
                y = i
        max_index = y
        nt = time.time() - starttime

        return max_index