import numpy as np


def get_mean_var(x, y):

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

    # Input the arguments into a probability density function
    p = 1 / (np.sqrt(2 * np.pi * variance_x)) * \
        np.exp((-(X - mean_x)**2) / (2 * variance_x))
    # return p
    return p
