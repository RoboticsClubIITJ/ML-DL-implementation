import numpy as np


def compute_distance(samples):
    """
    Creates a matrix of distances between individual samples and clusters
    attained at a particular step
    """
    distance_mat = np.zeros((len(samples), len(samples)))
    for i in range(distance_mat.shape[0]):
        for j in range(distance_mat.shape[0]):
            if i != j:
                distance_mat[i, j] = float(
                    distance_calculate(samples[i], samples[j]))
            else:
                distance_mat[i, j] = 10**4
    return distance_mat


def distance_calculate(sample1, sample2):
    """
    Distance calulated between two samples.
    If both of them are samples/clusters, then
    simple norm is used. In other cases, we refer
    it as an exception case and calculates the
    necessary distance between cluster and a sample
    """
    dist = []
    for i in range(len(sample1)):
        for j in range(len(sample2)):
            try:
                dist.append(np.linalg.norm(
                    np.array(sample1[i])-np.array(sample2[j])))
            except TypeError:
                dist.append(intersampledist(sample1[i], sample2[j]))
    return min(dist)


def intersampledist(s1, s2):
    """
    To be used in case we have one sample and one cluster.
    It takes the help of one method 'interclusterdist'
    to compute the distances between elements of a
    cluster(which are samples) and the actual sample given.
    """
    if str(type(s2[0])) != '<class \'list\'>':
        s2 = [s2]
    if str(type(s1[0])) != '<class \'list\'>':
        s1 = [s1]
    m = len(s1)
    n = len(s2)
    dist = []
    if n >= m:
        for i in range(n):
            for j in range(m):
                if (str(type(s2[i][0]) != '<class \'list\'>')):
                    dist.append(interclusterdist(s2[i], s1[j]))
                else:
                    dist.append(np.linalg.norm(
                        np.array(s2[i])-np.array(s1[j])))
    else:
        for i in range(m):
            for j in range(n):
                if (str(type(s1[i][0]) != '<class \'list\'>')):
                    dist.append(interclusterdist(s1[i], s2[j]))
                else:
                    dist.append(np.linalg.norm(
                        np.array(s1[i])-np.array(s2[j])))
    return min(dist)


def interclusterdist(cluster, sample):
    if sample[0] != '<class \'list\'>':
        sample = [sample]
    dist = []
    for i in range(len(cluster)):
        for j in range(len(sample)):
            dist.append(np.linalg.norm(
                np.array(cluster[i])-np.array(sample[j])))
    return min(dist)
