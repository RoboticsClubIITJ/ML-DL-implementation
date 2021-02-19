import numpy as np


def distcalc(p1, p2):
    """
    Calculates the Euclidean Distance
    between p1 and p2 points.

    PARAMETERS
    ==========

    p1: ndarray(dtype=int,ndim=1,axis=1)
        Point array with its corresponding
        x and y coordinates.

    p2: ndarray(dtype=int,ndim=1,axis=1)
        Point array with its corresponding
        x and y coordinates.

    dist: int
        Sum of sqaurred difference of
        coordinates between p1 and p2
        points.

    distance: float
        Euclidean Distance between p1
        and p2 points.

    RETURNS
    =======

    distance: float
        Euclidean Distance between p1
        and p2 points.

    """
    dist = 0
    for i in range(0, len(p1)-1):
        dist += (p1[i]-p2[i])**2
    distance = dist**0.5
    return distance


def initi_centroid(M, num_cluster):
    """
    Randomly pick centroids, out of
    Dataset Points, to be updated with
    each epoch.

    PARAMETERS
    ==========

    M: ndarray(dtype=int,ndim=2)
        Dataset Matrix of points,
        with their corresponding x
        and y coordinates.

    num_cluster: int
        Number of clusters, to be made
        using points from Dataset Matrix.

    indexes: ndarray(dtype=int,ndim=1,axis=1)
        Array of ramdomly picked indexes of
        points from Dataset Matrix, to become
        centroids.

    replace: Boolean
        Feature, which enables or disables
        the usage of index repetitions,
        depending upon Boolean value used.

    centroid: ndarray(dtype=int,ndim=2)
        Matrix of Centroids, made out of
        Dataset's Randomly picked points,
        without any repetitions.

    RETURNS
    =======

    centroid: ndarray(dtype=int,ndim=2)
        Initialised Matrix of Centroids,
        used as a building block of k-means
        clustering model.

    """
    indexes = np.random.choice(M.shape[0], num_cluster, replace=False)
    indexes.sort()
    centroid = []
    for indice in indexes:
        centroid.append(M[indice])
    return centroid


def new_centroid(lis):
    """
    Update the Matrix of centroid,
    provided with required information
    to process according to nearby neighbour
    points.

    PARAMETERS
    ==========

    lis: ndarray(dtype=int,ndim=2)
        Matrix of Clusters, with their
        corresponding points count,
        coordinates sum.

    new: list
        List of New Centroids, using
        reference list, updating with
        each epoch.

    newx: float
        X-coordinate of New Centroids.

    newy: float
        Y-coordinate of New Centroids

    RETURNS
    =======

    new: list
        Updated List of Centroids, as
        a result of change in nearby
        points of older centroids.

    """
    new = []
    for n in lis:
        newx = n[0]/n[2]
        newy = n[1]/n[2]
        new.append([newx, newy])
    return new


def xy_calc(M, centroid):
    """
    With each epoch, collects the
    required information, in order to
    update the Matrix of Centroids, in
    accordance with nearby neighbouring
    points.

    PARAMETERS
    ==========

    M: ndarray(dtype=int,ndim=2)
        Dataset Matrix of points,
        with their corresponding x
        and y coordinates.

    centroid: ndarray(dtype=int/float,ndim=2)
        Matrix of Centroid, which
        will be going to update, due
        to change in arrangement of
        nearby neighbours.

    lis: ndarray(dtype=int,ndim=2)
        Matrix, containing information
        of each cluster about their nearby
        points count, sum of coordinates of
        those points.

    dis: list
        Comparison list of Distances of
        current point with centroids, in
        order to find the closest centroid
        associated with it.

    indice: int
        Index of cluster to which the test
        point is close to its centroid and
        where it belongs.

    RETURNS
    =======

    lis: ndarray(dtype=int,ndim=2)
        Matrix of each cluster, with
        information required to update
        its corresponding centroid.

    """
    lis = np.zeros((len(centroid), 3))
    for point in M:
        dis = []
        for c in centroid:
            dis.append(distcalc(point, c))
        indice = dis.index(min(dis))
        lis[indice][0] += point[0]
        lis[indice][1] += point[1]
        lis[indice][2] += 1
  
    return lis


def cluster_allot(M, centroid):
    """
    After certain number of epochs,
    the allotment of Dataset points take
    place, in accordance to updated
    Centroids Matrix, clusterwise.

    PARAMETERS
    ==========

    M: ndarray(dtype=int,ndim=2)
        Dataset Matrix of points,
        with their corresponding x
        and y coordinates.

    centroid: ndarray(dtype=float,ndim=2)
        List of Final Centroids, which will
        be used in allotment of every point,
        belonging to Dataset Matrix.

    cluster_array: list
        List of Furnished Clusters, with their
        centroid and points associated with
        them.

    dista: list
        Comparison list of Distances of test point,
        to be alloted to either of the clusters,
        so that the nearest centroid will be picked
        and then point will be alloted to its
        respective cluster.

    indice: int
        Index of the cluster, to which the test point
        will going to be alloted, on the basis of
        lowest distance measure.

    RETURNS
    =======

    cluster_array: list
        List of Resultant Clusters, with their
        centroids and corresponding nearby points,
        having their x and y coordinates.

    """
    cluster_array = []
    for point in centroid:
        cluster_array.append([point])
    for poin in M:
        dista = []
        for c in centroid:
            dista.append(distcalc(poin, c))
            indice = dista.index(min(dista))
        cluster_array[indice].append(poin)
    return cluster_array
