
from MLlib.activations import TanH
from MLlib.activations import Softmax
import numpy as np

def test_1():
    assert 2 == 2.0

def test_2():
    assert type(5) is int

def test_tanh_fuction():
    # X= np.array([0])
    X = np.array([ 0 , 0 ,0 ])
    
    assert (TanH.activation(X) == np.array([[ 0 , 0 , 0] ])).all()
    assert (TanH.derivative(X) == np.array([1 , 1 , 1])).all()
    
        
    

    

    
    
