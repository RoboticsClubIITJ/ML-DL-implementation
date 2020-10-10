import numpy as np
from MLlib.activations import sigmoid


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
        return (1/M)*(np.sum((-Y)*np.log(H)-(1-Y)*np.log(1-H)))

    @staticmethod
    def derivative(X, Y, W):
        M = X.shape[0]
        H = sigmoid(np.dot(X, W).T)
        return (1/M)*(np.dot(X.T, (H-Y).T))


class AbsoluteError():
    @staticmethod
    def loss(X, Y, W):
        M = X.shape[0]
        return np.sum(np.absolute(np.dot(X, W).T - Y)) / M

    @staticmethod
    def derivative(X, Y, W):
        M = X.shape[0]
        AbsError = (np.dot(X, W).T-Y)
        return np.dot(
            np.divide(
                AbsError,
                np.absolute(AbsError),
                out=np.zeros_like(AbsError),
                where=(np.absolute(AbsError)) != 0),
            X
        ).T/M
