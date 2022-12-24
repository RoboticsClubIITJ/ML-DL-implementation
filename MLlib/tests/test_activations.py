import numpy as np
from MLlib.activations import Sigmoid, TanH, Softmax


def test_sigmoid():
    X = 0
    assert Sigmoid.activation(X) == 0.5
    assert Sigmoid.derivative(X) == 0.25


def test_tanh():
    X = 0
    assert TanH.activation(X) == 0
    assert TanH.derivative(X) == 1


def test_Softmax():
    X = np.array([0])
    assert Softmax.activation(X) ==  np.array([1, 2, 3, 4]).all()
    assert Softmax.derivative(X) == np.array([0, 1, 2, 3, 4, 5]).all()
