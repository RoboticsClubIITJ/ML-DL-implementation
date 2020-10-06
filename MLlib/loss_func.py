import numpy as np
from activations import sigmoid
from activations import softmax


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

class SparseCategoricalCrossEntropy():
    @staticmethod
    def loss(X,Y,W,n):
        # n = total number of classes for classification
        # W of dimension (X.shape[1],n)
        Yprime = []
        for ydash in Y:
            for y in ydash:
                y=int(y)
                yprime = list([0]*y+[1]+[0]*(n-y-1))
                Yprime.append(yprime)
        Yprime = np.array(Yprime)
        H=np.dot(X,W)
        for i in range(len(H)):
            H[i]=softmax(H[i])
        loss=0
        for y in range(len(Yprime)):
            for feature_index in range(len(Yprime[y])):
                if Yprime[y][feature_index] ==1:
                    loss-=np.log(H[y][feature_index])
        return loss

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
