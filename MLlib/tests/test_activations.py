import numpy as np
from MLlib.activations import Sigmoid, Relu
from MLlib.activations import unit_step


def test_Sigmoid():
    X = np.random.random((3, 2))
    if np.array_equal(
        (1 / (1 + np.exp(-X))),
        Sigmoid.activation(X)
            ) is not True:
        raise AssertionError
    if np.array_equal(
        (1 / (1 + np.exp(-X)))*(1-(1 / (1 + np.exp(-X)))),
        Sigmoid.derivative(X)
            ) is not True:
        raise AssertionError


def test_Relu():
    X = np.array([[1, -2, 3], [-1, 2, 1]])
    if np.array_equal(
        np.maximum(0, X),
        Relu.activation(X)
            ) is not True:
        raise AssertionError
    if np.array_equal(
        np.greater(X, 0).astype(int),
        Relu.derivative(X)
            ) is not True:
        raise AssertionError


def test_unit_step():
    X = np.array([[1, -2, 3], [-1, 2, 1]])
    if np.array_equal(
        np.heaviside(X, 1),
        unit_step(X)
            ) is not True:
        raise AssertionError
