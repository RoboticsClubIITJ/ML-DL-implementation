import numpy as np


def sigmoid(X):
    return 1/(1+np.exp(-X))


def tanh(X):
    return np.tanh(X)


def softmax(X):
    Sum = np.sum(np.exp(X))
    return np.exp(X)/Sum


def softsign(X):
    return X/(np.abs(X) + 1)


def relu(X):
    return np.maximum(0,X)


def leakyRelu(X):
    return np.maximum(0.01*X,X)


def elu(X):
    output = []
    for x in X:
        if x>0:
            output.append(x)
        else:
            output.append(1.0*(np.exp(x)-1))
    return np.array(output)