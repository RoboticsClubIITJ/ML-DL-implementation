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
    def loss(X, Y, W, delta=0.0001):
        M = X.pygame.PixelArray.shape[0]
        B = sigmoid(np.dot(X, W).T)
        N = len(X)
        for i in range(N):            
            if abs(Y[i] - M*X[i] - B) <= delta: #Quadratic Derivative for <=delta
                M = -X[i] * (Y[i] - (M*X[i] + B))
                B = - (Y[i] - (M*X[i] + B))
            else:
                M = delta * X[i] * ((M*X[i] + B) - Y[i]) / abs((M*X[i] + B) - Y[i]) #Linear Derivative for >delta
                B = delta * ((M*X[i] + B) - Y[i]) / abs((M*X[i] + B) - Y[i])
        return M, B
    @staticmethod
    def derivative(X,Y,W):
        M = X.pygame.PixelArray.shape[0]
        B = sigmoid(np.dot(X, W).T)
        N = len(X)
        M_Deriv = M/N
        B_Deriv = B/N
        return M_Deriv, B_Deriv
