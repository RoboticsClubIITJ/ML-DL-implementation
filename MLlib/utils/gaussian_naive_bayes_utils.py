import numpy as np


def get_mean_var(x, y):
    """
    Calculates the mean and variance for
    each list x and y.

    PARAMETERS
    ==========

    x: ndarray(dtype=int,ndim=1,axis=1)
        Array of labels.

    y: ndarray(dtype=int,ndim=1,axis=1)
        Array of classes.

    M: 2D Matrix
        The first column contains list
        of labels, rest of columns have
        list of y1,y2..yn, classes.

    RETURNS
    =======

    mean: dict
        Dictionary, with key points as
        unique labels, and values as list
        of mean each class(y1, y2.. yn).

    var: dict
        Dictionary, with key points as
        unique labels, and values as list
        of variance each class(y1, y2.. yn).

    """
    M = []

    for i in range(len(x)):
        M.append([x[i], y[i][0], y[i][1], y[i][2]])

    dataset = dict()

    for j in range(len(M)):
        if M[j][0] not in dataset:
            dataset[M[j][0]] = list()
        dataset[M[j][0]].append(M[j][1:])

    mean = dict()

    for key, value in dataset.items():
        v = np.array(value)
        mean[key] = v.mean(axis=0)

    var = dict()

    for key, value in dataset.items():
        v = np.array(value)
        var[key] = v.var(axis=0)

    return mean, var


def p_y_given_x(X, mean_x, variance_x):
    """
    Calculates the probablity of class
    value being y, given label is x.

    PARAMETERS
    ==========

    X: list
        Input of unknown class values
        given by user.

    mean_x: ndarray(dtype=int,ndim=1,axis=1)
        Mean for given label.

    variance_x: ndarray(dtype=int,ndim=1,axis=1)
        Variance for given label.


    RETURNS
    =======

    p: float
        Probability, according to gaussian
        distribution, for given mean and variance.

    """
    p = 1 / (np.sqrt(2 * np.pi * variance_x)) * \
        np.exp((-(X - mean_x)**2) / (2 * variance_x))
    return p
