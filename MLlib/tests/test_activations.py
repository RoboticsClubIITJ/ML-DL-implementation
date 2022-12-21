from MLlib.activations import Sigmoid, TanH, Relu
import numpy as np
import pytest

def test_sigmoid():
    X = 0
    assert Sigmoid.activation(X) == 0.5
    assert Sigmoid.derivative(X) == 0.25


def test_tanh():
    X = 0
    assert TanH.activation(X) == 0
    assert TanH.derivative(X) == 1

@pytest.mark.parametrize("X", [np.array([1,2,3,4,5]),np.array([-1,-2,-3,-4,-5]),np.array([0,0,0,0,0])])
def test_Relu(X):
    Y=np.zeros(len(X))
    dY=np.zeros(len(X))
    for i in range(len(X)):
        if X[i]>0:
            Y[i]=X[i]
        else:
            Y[i]=0

    for i in range(len(X)):
        if X[i]>0:
            dY[i]=1
        else:
            dY[i]=0

    assert(np.array_equal(Relu.activation(X),Y))
    assert(np.array_equal(Relu.derivative(X),dY))