import numpy as np
from MLlib.utils.raises_util import raises
from MLlib.activations import Sigmoid

def test_Sigmoid():
    X=np.random.random((3,2))
    print(1 / (1 + np.exp(-X)))
    assert np.array_equal( (1 / (1 + np.exp(-X))), Sigmoid.activation(X)) is True
    assert np.array_equal((1 / (1 + np.exp(-X)))*(1-(1 / (1 + np.exp(-X)))),Sigmoid.derivative(X))