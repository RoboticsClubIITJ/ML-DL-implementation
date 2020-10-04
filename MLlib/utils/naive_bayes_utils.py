import numpy as np
import math


def get_data():

    with open('../datasets/naive_bayes_dataset.txt', 'r') as f:
    	l = [[string.strip('\n') for string in line.split(',')] for line in f]

    # for testing default label="outlook" (sunny,rainy or overcast)

    x_label = np.array([l[i][0] for i in range(len(l))])
    y_class = np.array([l[i][-1] for i in range(len(l))])

    # merging the two to get a matrix

    M = np.array([[l[i][0], l[i][-1]] for i in range(len(l))])

    # an array for unique values

    Y = np.unique(y_class)
    X = np.unique(x_label)

    return(x_label, y_class, X, Y)


def make_frequency_table(X, Y):
    """
    This function prepares a frequency table for every label in respective column.
    """
    freq = dict()

    for i in range(len(X)):
        freq[X[i]] = [0, 0]

    for i in range(len(M)):
        if M[i][1] == Y[0]:
            freq[M[i][0]][0] += 1
        else:
            freq[M[i][0]][1] += 1

    return freq


def make_likelihood_table():

    # for each item divide by column sum

    x, y, X, Y = get_data()

    likelihood = [[0 for i in range(len(Y))] for j in range(len(X))]

    freq = make_frequency_table(X, Y)

    for j in range(len(Y)):
        Sum = (y == Y[j]).sum()
        for i in range(len(X)):
            likelihood[i][j] = freq[X[i]][j] / Sum

    return likelihood
