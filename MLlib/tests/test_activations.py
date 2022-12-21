from MLlib.activations import Sigmoid, Softmax

def test_sigmoid():
    X = 0
    assert Sigmoid.activation(X) == 0.5
    assert Sigmoid.derivative(X) == 0.25

def test_Softmax():
    X = 0
    assert Softmax.activation(X) == 1
    assert Softmax.derivative(X) == 1.5    
