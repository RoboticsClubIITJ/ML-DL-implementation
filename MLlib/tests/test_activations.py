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
    X = 0
    assert Softmax.activation(X) == 1
    assert Softmax.derivative(X) == 1.5
    