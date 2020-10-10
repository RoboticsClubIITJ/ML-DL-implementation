import numpy as np


def activate(fxn, r):
    """
    Method to call other activation function.
    """
    activate_dict = {"linear": linear, "tanh": tanh,
                     "sigmoid": sigmoid,
                     "softmax": softmax, "relu": relu
                     }
    return activate_dict[fxn](r)


def deactivate(fxn, r):
    """
    A method to call derivative of activation function.
    """
    deactivate_dict = {
        "linear": dlinear,
        "tanh": dtanh,
        "sigmoid": dsigmoid,
        "softmax": dsoftmax,
        "relu": drelu}
    return deactivate_dict[fxn](r)


def linear(x):
    return x


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x = x - np.max(x)
    s = np.exp(x)
    return s / np.sum(s)


def relu(x):
    x[x < 0] = 0
    return x


def dlinear(x):
    return np.ones(x.shape)


def dtanh(x):
    return 2 * x / (1 + x) ** 2


def dsigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


def dsoftmax(x):
    soft = softmax(x)
    diag_soft = soft * (1 - soft)
    return diag_soft


def drelu(x):
    x[x < 0] = 0
    return x
