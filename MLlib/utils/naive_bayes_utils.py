import numpy as np


def make_frequency_table(x, y, X, Y):
    """
    This function prepares a frequency table
    for every label in respective column.

    PARAMETERS
    ==========

    x: ndarray(dtype=int,ndim=1,axis=1)
        Array of labels.

    y: ndarray(dtype=int,ndim=1,axis=1)
        Array of classes.

    X: ndarray(dtype=int,ndim=1,axis=1)
        Array of unique labels.

    Y: ndarray(dtype=int,ndim=1,axis=1)
        Array of unique classes.


    RETURNS
    =======

    freq: dict
        Dictionary, with key points as
        unique labels, and values as its
        frequency of y.

    """
    freq = {}

    for i in range(len(X)):
        freq[X[i]] = [0, 0]

    # merging the two to get a matrix

    M = np.array([[x[i], y[i]] for i in range(len(x))])

    for i in range(len(M)):
        if M[i][1] == Y[0]:
            freq[M[i][0]][0] += 1
        else:
            freq[M[i][0]][1] += 1

    return freq


def make_likelihood_table(x, y):
    """
    This function prepares a likelihood
    table for each item we divide frequency
    by column sum.

    PARAMETERS
    ==========

    x: ndarray(dtype=int,ndim=1,axis=1)
        Array of labels.

    y: ndarray(dtype=int,ndim=1,axis=1)
        Array of classes.

    X: ndarray(dtype=int,ndim=1,axis=1)
        Array of unique labels.

    Y: ndarray(dtype=int,ndim=1,axis=1)
        Array of unique classes.


    RETURNS
    =======

    likelihood: dict
        Dictionary, with key points as
        unique labels, and values as ratio
        of frequency and cumulative frequency
        for that column(y_class).

    """

    Y = np.unique(y)
    X = np.unique(x)

    likelihood = [[0 for i in range(len(Y))] for j in range(len(X))]

    freq = make_frequency_table(x, y, X, Y)

    for j in range(len(Y)):
        Sum = (y == Y[j]).sum()
        for i in range(len(X)):
            likelihood[i][j] = freq[X[i]][j] / Sum

    return likelihood
