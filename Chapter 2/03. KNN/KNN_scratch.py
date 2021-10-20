import numpy as np
from collections import Counter

class KNN:

    def __init__(self, k):
        self.k = k

    def square_diff(self, x1, x2):
        distance = []
        for i, j in zip(x1, x2):
            distance.append((i - j) ** 2)
        return np.array(distance)

    def root_sum_squared(self, v1):
        return np.sqrt(sum([i for i in v1]))

    def euclidean_distances(self, v0, v1):
        return self.root_sum_squared(self.square_diff(v0, v1))

    def evaluate(self, y, y_p):
        return len([i for i, v in zip(y, y_p) if i == v]) / len(y)

    def fit(self, x, y):
        self.x_true = x
        self.y_true = y
        return self

    def predict(self, x_test):
        y_hat = []
        for test in x_test:
            dist_list = [(i, self.euclidean_distances(self.x_true[i], test)) for i in range(0, len(self.x_true))]
            sl = sorted(dist_list, key=lambda x: x[1])
            k_list = [0] * self.k + [1] * (len(dist_list) - self.k)
            final_n = list(zip(sl, k_list))
            indexes = [i[0][0] for i in final_n if i[1] == 0]
            y_hat.append(Counter(self.y_true[indexes]).most_common(1)[0][0])
        return np.array(y_hat)

