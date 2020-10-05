from math import sqrt
import numpy as np


def read_KNN_dataFile(file):
    """
    A function to read data for KNN dataset provided.
    """
    A = np.genfromtxt(file)
    return A


def euclidean_distance(p1, p2):
    """
    Returns the Euclidean Distance of a particular
    point from rest of the points in dataset.
    """
    distance = 0
    for i in range(len(p1)-1):
        distance += (p1[i]-p2[i])**(2)
    return sqrt(distance)


def block_distance(p1, p2):
    """
    Returns the Block Distance of a particular
    point from rest of the points in dataset.
    """
    distance = 0
    for i in range(len(p1)-1):
        distance += abs(p1[i]-p2[i])
    return distance


def get_neighbours(train, test_row, num_neighbours, distance_metrics="block"):
    """
    Returns n nearest neighbours of a particular point
    in dataset based on euclidean or block distance.
    """
    distances = []
    for train_row in train:
        if distance_metrics == "block":
            distance = block_distance(test_row, train_row)
        else:
            distance = euclidean_distance(test_row, train_row)
        distances.append((train_row, distance))
    distances.sort(key=lambda tup: tup[1])
    neigbours = []
    for i in range(num_neighbours):
        neigbours.append(distances[i][0])
    return neigbours
