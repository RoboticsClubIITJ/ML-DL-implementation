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
        return (1/M)*(np.sum((-Y)*np.log(H)-(1-Y)*np.log(1-H)))

    @staticmethod
    def derivative(X, Y, W):
        M = X.shape[0]
        H = sigmoid(np.dot(X, W).T)
        return (1/M)*(np.dot(X.T, (H-Y).T))

class Huber_Loss_Function():
    @staticmethod
    def loss(m, b, X, Y, delta):
        N = len(X)
        for i in range(N):
            
            if abs(Y[i] - m*X[i] - b) <= delta: #Quadratic Derivative for <=delta
                m += -X[i] * (Y[i] - (m*X[i] + b))
                b += - (Y[i] - (m*X[i] + b))
            else:
                m += delta * X[i] * ((m*X[i] + b) - Y[i]) / abs((m*X[i] + b) - Y[i]) #Linear Derivative for >delta
                b += delta * ((m*X[i] + b) - Y[i]) / abs((m*X[i] + b) - Y[i])
        return m, b