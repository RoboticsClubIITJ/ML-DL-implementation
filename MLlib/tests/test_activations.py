from MLlib.activations import Sigmoid, TanH


def test_sigmoid():
    X = 0
    assert Sigmoid.activation(X) == 0.5
    assert Sigmoid.derivative(X) == 0.25


def test_tanh():
    X = 0
    assert TanH.activation(X) == 0
    assert TanH.derivative(X) == 1
