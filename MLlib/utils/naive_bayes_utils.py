import numpy as np


def make_frequency_table(x, y, X, Y):
    """
    This function prepares a frequency table
    for every label in respective column.
    """
    freq = dict()

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

    # for each item divide by column sum
    # an array for unique values

    Y = np.unique(y)
    X = np.unique(x)

    likelihood = [[0 for i in range(len(Y))] for j in range(len(X))]

    freq = make_frequency_table(x, y, X, Y)

    for j in range(len(Y)):
        Sum = (y == Y[j]).sum()
        for i in range(len(X)):
            likelihood[i][j] = freq[X[i]][j] / Sum

    return likelihood
