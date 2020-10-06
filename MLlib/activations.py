import numpy as np


def sigmoid(X):
    return 1/(1 + np.exp(-X))


def tanh(X):
    return np.tanh(X)


def softmax(X):
    Sum = np.sum(np.exp(X))
    return np.exp(X)/Sum


def softsign(X):
    return X/(np.abs(X) + 1)


def relu(X):
    return np.maximum(0, X)


def leakyRelu(X):
    return np.maximum(0.01*X, X)


def elu(X, alpha=1.0):
    assert(alpha > 0)
    return np.maximum(0, X) + np.minimum(0, alpha*(np.exp(X) - 1))
