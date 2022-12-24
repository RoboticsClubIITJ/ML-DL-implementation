import numpy as np
from MLlib.activations import Sigmoid, TanH, Softmax, Relu, LeakyRelu


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
    assert Softmax.activation(X) == np.array([1, 2, 3, 4]).all()
    assert Softmax.derivative(X) == np.array([0, 1, 2, 3, 4, 5]).all()


def test_Relu():
    X=np.array([0])
    assert Relu.activation(X)==np.array([0]).all()
    X=np.array([1,2.0,3.0]).all()
    assert Relu.activation(X)==np.array([1,2.0,3.0]).all()
    assert Relu.derivative(X)==np.array([1,2.0,3.0]).all()


def test_LeakyRelu():
    X=np.array([0])
    assert LeakyRelu.activation(X,0.01)==np.array(0)
    X=np.array([2])
    assert LeakyRelu.activation(X,0.01)==np.array(2)

    

