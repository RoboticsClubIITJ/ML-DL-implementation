from MLlib.activations import Sigmoid

def test_sigmoid():
    X = 0
    assert Sigmoid.activation(X) == 0.5
    assert Sigmoid.derivative(X) == 0.25