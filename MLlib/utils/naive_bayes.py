import numpy as np
import math

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


def make_likelihood_table(X, Y, x, y):

    # for each item divide by column sum

    likelihood = [[0 for i in range(len(Y))] for j in range(len(X))]

    freq = make_frequency_table(X, Y)

    for j in range(len(Y)):
        Sum = (y == Y[j]).sum()
        for i in range(len(X)):
            likelihood[i][j] = freq[X[i]][j] / Sum

    return likelihood


def naive_bayes(X, Y):
    """
    pyx: P(y/X) is proportional to p(x1/y)*p(x2/y)...*p(y)
    using log and adding as multiplying for smaller numbers can make them very small
    As denominator P(X)=P(x1)*P(x2).. is common we can ignore it
    """

    pyx = []

    likelihood = make_likelihood_table(X, Y, x_label, y_class)

    for j in range(len(Y)):
        Sum = 0
        for i in range(len(X)):
            if(likelihood[i][j] == 0):
                continue

            Sum += math.log(likelihood[i][j])

            y_sum = (y_class == Y[j]).sum()

            if y_sum:
                Sum += math.log(y_sum / len(y_class))
                pyx.append([Sum, X[i], Y[j]])

    return pyx


def most_likely():
    """
    predicts the most likely label,class
    """
    prediction = max(naive_bayes(X, Y))
    return [prediction[1], prediction[2]]
