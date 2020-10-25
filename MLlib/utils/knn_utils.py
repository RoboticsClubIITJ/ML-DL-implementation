from math import sqrt
import numpy as np


def read_KNN_dataFile(file):
    """
    A function to read data for KNN dataset provided.

    PARAMETERS
    ==========

    file: str
        Working Directory Path where
        dataset File is located.

    A: ndarray(dtype=int)
        Array representation of
        dataset File.

    RETURNS
    =======

    A: ndarray(dtype=int)
        Array representation of
        dataset File.
    """
    A = np.genfromtxt(file)
    return A


def euclidean_distance(p1, p2):
    """
    Returns the Euclidean Distance of a particular
    point from rest of the points in dataset.

    PARAMETERS
    ==========

    p1: ndarray(dtype=int,ndim=1,axis=1)
        Array representation of Point 1,
        with corresponding x1,x2 and y fe-
        atures.

    p2: ndarray(dtype=int,ndim=1,axis=1)
        Array representation of Point 2,
        with corresponding x1,x2 and y fe-
        atures.

    distance: int
        Squarred Distance between 2 points,i.e
        p1 and p2, by considering sum of
        sqaurred differences in respect
        to x1,x2 and y features.

    RETURNS
    =======

    float
        Euclidean Distance between 2 points,i.e
        p1 and p2.
    """
    distance = 0
    for i in range(len(p1)-1):
        distance += (p1[i]-p2[i])**(2)
    return sqrt(distance)


def block_distance(p1, p2):
    """
    Returns the Block Distance of a particular
    point from rest of the points in dataset.

    PARAMETERS
    ==========

    p1: ndarray(dtype=int,ndim=1,axis=1)
        Array Representation of Point 1,
        with corresponding x1,x2 and y fe-
        atures.

    p2: ndarray(dtype=int,ndim=1,axis=1)
        Array Representation of Point 2,
        with corresponding x1,x2 and y fe-
        atures.

    distance: int
        Absolute differences of p1 and p2
        components in respect to x1,x2
        and y features.

    RETURNS
    =======

    distance: int
        Sum of Absolute differences of p1
        and p2 components in respect to x1,
        x2 and y features.
    """
    distance = 0
    for i in range(len(p1)-1):
        distance += abs(p1[i]-p2[i])
    return distance


def get_neighbours(train, test_row, num_neighbours, distance_metrics="block"):
    """
    Returns n nearest neighbours of a particular point
    in dataset based on euclidean or block distance.

    PARAMETERS
    ==========

    train: ndarray(dtype=int)
        Array Representation of Collection
        of Points with x1,x2 and y feature-
        s.

    test_row: ndarray(dtype=int,ndim=1,axis=1)
        Array Representation of a Test
        Point having its own x1,x2 and y f-
        eatures.

    num_neighbours: int
        Number of nearest neighbours among
        collection of Points Array.

    distance_metrics: str
        Type of Distance the model wants
        to use among Block and Euclidean.

    distances: list
        List of Distances calculated from
        a test point to other points,
        either by using Block or Euclidean
        Metric.

    distance: int/float
        Value of Distance from test point
        to one of the other points in
        Dataset, calculated by using either
        Block or Euclidean Metric, respectively.

    key: function
        Mode to be used in sorting distances
        List.

    tup: tuple
        In this scenario, tuple of Point Array
        and Distance corresponding to
        its associated Point Array.

    neigbours: list
        List of Neaarest Neighbours, close to
        test point, on the basis of either
        Block or Euclidean Distance.

    RETURNS
    =======

    neigbours: list
        List of n Nearest Neighbours, close to
        test point.
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
