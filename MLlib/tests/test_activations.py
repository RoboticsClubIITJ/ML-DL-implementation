import numpy as np
from MLlib.utils.raises_util import raises
from MLlib.activations import Sigmoid

def test_Sigmoid():
    X=np.random.random((3,1))
    assert 1 / (1 + np.exp(-X)) == Sigmoid.activation(X)
    assert (1 / (1 + np.exp(-X)))*(1-(1 / (1 + np.exp(-X))))==Sigmoid.derivative(X)