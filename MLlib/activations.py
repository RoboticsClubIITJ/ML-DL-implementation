import numpy as np

def sigmoid(X):
    return 1/(1+np.exp(-X))