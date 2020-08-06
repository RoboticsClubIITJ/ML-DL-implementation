import numpy as np
from .activations import sigmoid


class MeanSquaredError():
    @staticmethod
    def loss(X, Y, W):
        M = X.shape[0]
        return np.sum((np.dot(X, W).T - Y) ** 2) / (2 * M)

    @staticmethod
    def derivative(X, Y, W):
        M = X.shape[0]
        return np.dot((np.dot(X, W).T - Y), X).T / M


class LogarithmicError():
    @staticmethod
    def loss(X, Y, W):
        M = X.shape[0]
        H = sigmoid(np.dot(X, W).T)
        e = (1/M)*(np.sum((-Y)*np.log(H)-(1-Y)*np.log(1-H)))
        return e

    @staticmethod
    def derivative(X, Y, W):
        M = X.shape[0]
        H = sigmoid(np.dot(X, W).T)
        return (1/M)*(np.dot(X.T, (H-Y).T))
