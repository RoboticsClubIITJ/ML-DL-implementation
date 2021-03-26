import numpy as np
import matplotlib.pyplot as plt
import math
# from time import perf_counter
# import logging
# logging.basicConfig(filename='MLlib/tests/divisive_clustering.log',
# level=logging.DEBUG, filemode='w', format='\n%(asctime)s\n%(message)s')


class KMeans:
    def runKMeans(self, X: np.ndarray, n_clusters: int,
                  n_iterations: int) -> (list, np.array):
        '''
        The KMeans clustering algorithm.
        Returns:
        clusters: list of np.ndarrays of clusters.
        centroids: np.array of size = n_clusters
        '''
        self.n_clusters = n_clusters
        # Initialize centroids with random points from dataset
        self.init_centroids(X)
        for i in range(n_iterations):
            self.allocate(X)  # allocate every point to its closest centroid
            self.update_centroids()  # calculate new centroids of the
        # newly-formed clusters
        return self.clusters, self.centroids

    def init_centroids(self, X: np.ndarray):
        '''
        Initialize centroids with random examples (or points) from the dataset.
        '''
        # Number of examples
        datalen = X.shape[0]
        # Initialize centroids array with points from X with random indices
        # chosen from 0 to number of examples
        rng = np.random.default_rng()
        self.centroids = X[rng.choice(datalen,           size=self.n_clusters,
                                      replace=False)]
        # self.centroids = X[np.random.randint(0, l, size=self.n_clusters)]
        self.centroids.astype(np.float32)

    def allocate(self, X: np.ndarray):
        '''
        This function forms new clusters from the centroids updated in the
        previous iterations.
        '''
        # Step 1: Fill new clusters with a single point
        # Calculate the differences in the features between X and centroids
        # using broadcast subtract
        res = X - self.centroids[:, np.newaxis]

        # Find Euclidean distances using the above differences
        euc = np.linalg.norm(res, axis=2)
        # (n_clusters, X.shape[0]); contains distances of each centroid to
        # every example point
        m, n = euc.shape
        # indices of first points to allot to each cluster
        first_indices = np.full((m,), -1, dtype=int)

        # Add the closest point to the corresponding centroid to the
        # cluster array.
        # We do this to avoid formation of empty clusters
        # Loop to allot unique first points to each cluster  #!FIXME: Could it
        # be better?
        while True:
            # indices of minimum values from each row of distance matrix
            res = np.argmin(euc, axis=1)
            res.astype(int)
            resu = np.unique(res)  # one index per row
            lres = len(res)
            lresu = len(resu)
            if lresu == lres:  # already unique
                first_indices = res
                break
            else:
                # list to classify row indices w.r.t. columns
                arr = []
                # initialization
                for i in range(lres):
                    # logging.debug(f'arr entries: {np.where(res==i)}, i:{i}')
                    arr.append(np.where(res == i)[0])
                # logging.debug(f'euc:{euc}\nres:{res}\narr:{arr}')
                # assign exactly one row index as first index
                for i in range(lres):  # here i plays the role of column index
                    if len(arr[i]) == 1:
                        first_indices[arr[i]] = i  # arr[i] has row indices.
                        # logging.debug(f'fi after equal:\nfi:{first_indices},
                        # i:{i}')
                    elif len(arr[i]) > 1:
                        # logging.debug(f'euc entries for arr[i]={arr[i]} and
                        # i={i}\neuc[arr[i], i]:{euc[arr[i], i]}')
                        temp = np.argmin(euc[arr[i], i])
                        first_indices[temp] = i
            # logging.debug(f'argmin:{np.argmin(euc, axis=1)}\neuc:{euc}\
            # \neuc.shape:{euc.shape}\nres(first_indices): {res}\
            # \nres2: {res2}\neuc[res]: {euc[res2]}\n')
            # check if unique indices for all rows are found
            if -1 in first_indices:
                # column indices where min distances were found
                col_ind = np.nonzero(np.isin(np.arange(n), first_indices))
                # to avoid them in future iterations
                euc[:, col_ind[0]] = euc[col_ind[0], :] = np.inf
                m, n = euc.shape
                # logging.debug(f'euc in while:{euc}\nnp.isin(np.arange(n),
                # first_indices):{
                # np.nonzero(np.isin(np.arange(n),first_indices))}')
            else:
                break
        # logging.debug(f'euc:{euc}\nfirst_indices on
        # completion:{first_indices}')
        cluster_array = X[first_indices]  # assign first point to each cluster
        # add another dimension to make containers for clusters
        cluster_array = list(np.expand_dims(cluster_array, axis=1))

        # Step 2: Allocate the remaining points to the closest clusters
        # Calculate the differences in the features between centroids and X
        # using broadcast subtract
        res = self.centroids - X[:, np.newaxis]
        # logging.debug(res.shape)    #(X.shape[0], n_clusters, X.shape[1])

        # Find Euclidean distances of each point with all centroids
        euc = np.linalg.norm(res, axis=2)

        # Find the closest centroid from each point.
        # Find unique indices of the closest points. Using res again for
        # optimization
        # not unique indices
        res = np.where(euc == euc.min(axis=1)[:, np.newaxis])
        # res[0] is used as indices for row-wise indices in res[1]
        min_indices = res[1][np.unique(res[0])]
        # logging.debug(f'len(min_indices)={len(min_indices)}')
        # Set first indices to -1 to avoid adding their data points again
        min_indices[first_indices] = -1
        # logging.debug(f'len(min_indices)={len(min_indices)}')
        for i, c in enumerate(min_indices):
            if not c == -1:
                # add the point to the corresponding cluster
                cluster_array[c] = np.append(cluster_array[c], [X[i]], axis=0)
        # if len(X) == 2 and (cluster_array[0].shape == (2,2) or \
        # cluster_array[1].shape == (2,2)):
        # logging.debug(f'first_indices: {first_indices}\nmin_indices: \
        # {min_indices}\ncentroids: {self.centroids}')
        # update the fair clusters array
        self.clusters = cluster_array

    def update_centroids(self):
        '''
        This function updates the centroids based on the updated clusters.
        '''
        # Make a rough copy
        centroids = self.centroids
        # Find mean for every cluster
        for i in range(self.n_clusters):
            centroids[i] = np.mean(self.clusters[i], axis=0)

        # Update fair copy
        self.centroids = centroids


def visualize_clusters(global_clusters, global_centroids, iteration):
    '''
    Utility to visualize the changes in clusters at every iteration of the
    work algorithm in models.py
    '''
    # centroids = np.array(centroids)
    fig, ax = plt.subplots()
    n_clusters = iteration
    rng = np.random.default_rng()
    # generate random color values for data points in each cluster
    colors = rng.random(size=(n_clusters, 4), dtype=np.float32)
    # opacity set to 0.5 for all points
    colors[:, 3] = 0.5
    for i in range(n_clusters):
        # scatter-plot of all data points belonging to different clusters
        ax.scatter(global_clusters[i][:, 0], global_clusters[i]
                   [:, 1], marker='.', color=tuple(colors[i]))
    # scatter-plot of centroids
    ax.scatter(global_centroids[:, 0], global_centroids[:, 1],
               marker='s', c='#F008', label='Centroids')
    # give labels to centroids
    for i in range(n_clusters):
        ax.annotate(f'c{i}', (global_centroids[i, 0], global_centroids[i, 1]))
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'Iteration {iteration}')
    # ax.set_legend()


def sse(cluster: np.array, centroid: np.array):
    '''
    Sum of squared error calculation
    '''
    # broadcast subtract with all cluster points
    return np.sum((cluster - centroid)**2)


def to_adjacency_matrix(global_centroids: np.ndarray, n_clusters) -> \
        np.ndarray:
    '''
    Creates an adjacency matrix of the distances of the result centroids.
    '''
    res = global_centroids - global_centroids[:, np.newaxis]
    # find distance between centroids and store them into an adjacency matrix
    centroid_dist_mat = np.linalg.norm(res, axis=2)
    np.fill_diagonal(centroid_dist_mat, np.nan)
    return centroid_dist_mat


def update_mat(centroid_dist_mat, n_clusters):
    '''
    Updates the adjacency matrix of distances of centroids for every centroid
    '''
    # Find the index of the smallest value (other than np.nan) from the
    # centroid distance matrix
    ind = np.unravel_index(np.nanargmin(
        centroid_dist_mat), (n_clusters, n_clusters))
    dist = centroid_dist_mat[ind]
    # ind is a tuple of x and y indices
    c = max(ind[0], ind[1])
    o = min(ind[0], ind[1])
    for i in range(n_clusters):
        a = centroid_dist_mat[c, i]
        b = centroid_dist_mat[o, i]
        # Appollonius theorem: used to find median length given the sides of
        # the triangle
        m = math.sqrt(0.5 * (a * a + b * b - dist * dist * 0.5))
        centroid_dist_mat[o, i] = centroid_dist_mat[i, o] = m
    centroid_dist_mat[:, c] = centroid_dist_mat[c, :] = np.nan
    return centroid_dist_mat, ind, dist


def create_label_map(locations, n_clusters):
    '''
    Create a label map for construction of dendrogram.
    '''
    linkedlists = {}  # track all the sequences of x positions
    # track list elements to their head
    ref_arr = np.full((n_clusters,), -1, dtype=int)
    for loc in locations:
        # logging.debug(f'linkedlists: {linkedlists}, ref_arr:{ref_arr}')
        x, y = loc
        # out, out case; min -> max in a new linkedlist indexed by min.
        if ref_arr[x] == -1 and ref_arr[y] == -1:
            mn, mx = (x, y) if min(x, y) == x else (y, x)
            linkedlists[mn] = [mn, mx]
            ref_arr[mn] = ref_arr[mx] = mn
        # out, in case; (out -> in) in the linkedlist where in is present
        elif ref_arr[x] == -1 and not ref_arr[y] == -1:
            r = ref_arr[y]
            linkedlists[r].insert(0, x)
            ref_arr[x] = r
            # if x < r, shift the linkedlist from index r to x
            if x < r:
                temp = linkedlists.pop(r)
                linkedlists[x] = temp
                ref_arr[x] = x
        # in, out case; (out -> in) in the linkedlist where in is present
        elif not ref_arr[x] == -1 and ref_arr[y] == -1:
            r = ref_arr[x]
            linkedlists[r].insert(0, y)
            ref_arr[y] = r
            # if y < r, shift the linkedlist from index r to y
            if y < r:
                temp = linkedlists.pop(r)
                linkedlists[y] = temp
                ref_arr[y] = y
        # in, in case; append linkedlist from max reference to min
        elif not ref_arr[x] == -1 and not ref_arr[y] == -1:
            rx = ref_arr[x]
            ry = ref_arr[y]
            if rx < ry:
                linkedlists[rx] += linkedlists[ry]
                linkedlists.pop(ry)  # remove the appended linkedlist
            if ry < rx:
                linkedlists[ry] += linkedlists[rx]
                linkedlists.pop(rx)
    # the result label map used to construct dendrogram
    # logging.debug(f'linkedlists: {linkedlists}, ref_arr:{ref_arr}')
    label_map = {}
    # function to initialize dicts inside of label map

    def init_label_map(a, b):
        return {'label': a, 'xpos': b, 'ypos': 0}
    # init_label_map = lambda a, b:
    for i, l in enumerate(linkedlists[0]):
        label_map[l] = init_label_map(l, i + 1)
    # logging.debug(f'label_map:{label_map}')
    return label_map, linkedlists[0]


def mk_fork(x0, x1, y0, y1, new_level):
    '''
    Utility function to generate connectors in dendrogram
    '''
    points = [[x0, x0, x1, x1], [y0, new_level, new_level, y1]]
    connector = [(x0 + x1) / 2., new_level]
    return (points), connector


def visualize(global_clusters, global_centroids, n_clusters, datasize):
    rng = np.random.default_rng()
    # generate random color values for data points in each cluster
    colors = rng.random(size=(n_clusters, 4), dtype=np.float32)
    # opacity set to 0.5 for all points
    colors[:, 3] = 0.5
    for i in range(n_clusters):
        # scatter-plot of all data points belonging to different clusters
        plt.scatter(global_clusters[i][:, 0], global_clusters[i]
                    [:, 1], marker='.', color=tuple(colors[i]))
    # scatter-plot of centroids
    plt.scatter(global_centroids[:, 0], global_centroids[:, 1],
                marker='s', c='#F008', label='Centroids')
    # give labels to centroids
    for i in range(n_clusters):
        plt.annotate(f'c{i}', (global_centroids[i, 0], global_centroids[i, 1]))
    plt.xlim((0, datasize))
    plt.ylim((0, datasize))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    # dendrogram using adjacency matrix of the distances of the centroids from
    # each other
    locations = []
    levels = []
    centroid_dist_mat = to_adjacency_matrix(global_centroids, n_clusters)
    # logging.debug('=========================================================
    # =')
    # logging.debug(f'centroid dist matrix initially: \n{centroid_dist_mat}')

    for i in range(n_clusters - 1):
        centroid_dist_mat, tup, dist = update_mat(
            centroid_dist_mat, n_clusters)
        # logging.debug('=====================================================
        # =====')
        # logging.debug(f'updated matrix at {i}th iteration:
        # \n{centroid_dist_mat}')
        locations.append(tup)
        levels.append(dist)
    # logging.debug(f'locations: {locations}')
    # logging.debug(f'levels: {levels}')
    label_map, xticklabels = create_label_map(locations, n_clusters)
    # logging.debug(label_map)

    fig, ax = plt.subplots()

    for i, (new_level, (loc0, loc1)) in enumerate(zip(levels, locations)):
        x0, y0 = label_map[loc0]['xpos'], label_map[loc0]['ypos']
        x1, y1 = label_map[loc1]['xpos'], label_map[loc1]['ypos']
        # logging.debug('\t points are: {0}:({2},{3}) and
        # {1}:({4},{5})'.format(loc0,loc1,x0,y0,x1,y1))

        p, c = mk_fork(x0, x1, y0, y1, new_level)
        ax.plot(*p)  # plot the lines in dendrogram
        label_map[loc0]['xpos'] = c[0]
        label_map[loc0]['ypos'] = c[1]
    # insert one more element to algn the xticklabels
    xticklabels.insert(0, 0)
    ax.set_xticks(np.arange(n_clusters + 1))
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('Cluster Number')
    ax.set_ylim(0, 1.05 * np.max(levels))
    ax.set_ylabel('Distance')


def numConcat(li: list):
    '''
    Utility function to concatenate two float numbers in a list.
    Reference: https://www.geeksforgeeks.org/python-program-to-concatenate-two
    -integer-values-into-one/
    '''
    num1, num2 = li
    # Convert both the numbers to strings
    num1 = str(int(num1))
    num2 = str(int(num2))
    # Concatenate the strings
    num1 += num2
    return int(num1)


'''
Notable References:
Algorithm: https://www.youtube.com/watch?v=Fm01pqWLqzU
Numpy docs: https://numpy.org/doc/1.20/
Bisecting KMeans: https://www.geeksforgeeks.org/bisecting-k-means-algorithm-in
troduction/
    https://cs.fit.edu/~pkc/classes/ml-internet/papers/steinbach00tr.pdf
Dendrogram: https://stackoverflow.com/questions/56123380/how-to-draw-dendrogra
m-in-matplotlib-without-using-scipy
'''
