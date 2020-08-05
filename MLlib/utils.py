import numpy as np
import pickle


def read_data(file):
    '''
    Read the training data from a file in the specified directory.

    Parameters
    ==========
    file:
        data type : str
        Name of the file to be read with extension.

    Example
    =======

    If the training data is stored in "dataset.txt" use

    >>> read_data('dataset.txt')

    '''
    A = np.genfromtxt(file)

    # Read all the column except the last into X
    # Add an extra column of 1s at the end to act as constant
    # Read the last column into Y
    X = np.hstack((A[:, 1:2], np.ones((A.shape[0], 1))))
    M, N = X.shape

    Y = A[:, -1]
    Y.shape = (1, M)

    return X, Y


def printmat(name, matrix):
    '''
    Prints matrix in a easy to read form with
    dimension and label.

    Parameters
    ==========
    name:
        data type : str
        The name displayed in the output.

    matrix:
        data type : numpy array
        The matrix to be displayed.
    '''
    print('matrix ' + name + ':', matrix.shape)
    print(matrix, '\n')


def generate_weights(rows, cols, zeros=False):
    '''
    Generates a Matrix of weights according to the
    specified rows and columns
    '''
    if zeros:
        return np.zeros((rows, cols))
    else:
        return np.random.rand(rows, cols)


def load_model(name):
    with open(name, 'rb') as robfile:
        model = pickle.load(robfile)

    return model
