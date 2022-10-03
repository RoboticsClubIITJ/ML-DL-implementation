import numpy as np
from MLlib.activations import Sigmoid, Relu, Softsign, Swish
from MLlib.activations import unit_step, TanH, LeakyRelu, Elu, Softmax


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
    print(np.heaviside(X, 1)),
    if np.array_equal(
        np.heaviside(X, 1),
        unit_step(X)
    ) is not True:
        raise AssertionError


def test_TanH():
    X = np.array([[1, -2, 3], [-1, 2, 1], [0, -5, 6]])
    if np.array_equal(
        np.tanh(X),
        TanH.activation(X)
    ) is not True:
        raise AssertionError
    if np.array_equal(
        1.0 - np.tanh(X)**2,
        TanH.derivative(X)
    ) is not True:
        raise AssertionError


def test_LeakyRelu(alpha):
    X = np.array([[1, -2, 3], [-1, 2, 1], [0, -5, 6]])
    if np.array_equal(
        np.maximum(alpha*X, X),
        LeakyRelu.activation(X, alpha)
    ) is not True:
        raise AssertionError
    dx = np.greater(X, 0).astype(float)
    dx[X < 0] = -alpha
    if np.array_equal(
        dx,
        LeakyRelu.derivative(X, alpha)
    ) is not True:
        raise AssertionError


def test_Elu(alpha):
    X = np.array([[1, -2, 3], [-1, 2, 1], [0, -5, 6]])
    if np.array_equal(
        np.maximum(X, 0)+np.minimum(0, alpha * (np.exp(X) - 1)),
        Elu.activation(X, alpha)
    ) is not True:
        raise AssertionError


def test_Softmax():
    X = np.array([1.3, 5.1, 2.2, 0.7, 1.1])
    x_vector = X.reshape(X.shape[0], 1)
    if np.array_equal(
        np.exp(X)/np.sum(np.exp(X)),
        Softmax.activation(X)
    ) is not True:
        raise AssertionError
    x_vector = X.reshape(X.shape[0], 1)
    x_matrix = np.tile(x_vector, X.shape[0])
    x_der = np.diag(X) - (x_matrix * np.transpose(x_matrix))
    if np.array_equal(
        x_der,
        Softmax.derivative(X)
    ) is not True:
        raise AssertionError


def test_Softsign():
    X = np.array([1.3, 5.1, 2.2, 0.7, 1.1])
    if np.array_equal(
        X / (np.abs(X) + 1),
        Softsign.activation(X)
    ) is not True:
        raise AssertionError
    if np.array_equal(
        1 / (np.abs(X) + 1)**2,
        Softsign.derivative(X)
    ) is not True:
        raise AssertionError


def test_Swish(alpha):
    X = np.array([1.3, 5.1, 2.2, 0.7, 1.1])
    if np.array_equal(
        X / (1 + np.exp(-(alpha*X))),
        Swish.activation(X)
    ) is not True:
        raise AssertionError
    s = 1 / (1 + np.exp(-X))
    f = X / (1 + np.exp(-(alpha*X)))
    df = f + (s * (1 - f))
    if np.array_equal(
        df,
        Swish.derivative(X)
    ) is not True:
        raise AssertionError


# This is a Demonstration Code