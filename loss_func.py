import numpy as np


class MeanSquaredError():
    @staticmethod
    def loss(X, Y, W):
        M = X.shape[0]
        return np.sum((np.dot(X, W).T - Y) ** 2) / (2 * M)

    def derivative(X, Y, W):
        M = X.shape[0]
        return np.dot((np.dot(X, W).T - Y), X).T / M
