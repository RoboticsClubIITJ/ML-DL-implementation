from MLlib.models import DivisiveClustering
from MLlib.utils.divisive_clustering_utils import visualize, sse, numConcat
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
# import logging
# logging.basicConfig(filename='MLlib/tests/divisive_example.log', level=logging.DEBUG, filemode='w', format='\n%(asctime)s\n%(message)s')

# Example 1
# X = np.genfromtxt('Examples/datasets/k_means_clustering.txt')
# count = X.shape[0]
# n_clusters = 8

# Example 2: Redundant datapoints not allowed
# Try with different number of clusters
LOAD = False  # set to True to load the previously saved dataset, False for custom config
SAVE = False  # set to True if you want to save below configurations
if LOAD:
    with open('Examples/datasets/divisive_clustering.npy', 'rb') as f:
        X = np.load(f)
else:
    n_clusters = 7
    count = 7
    X = np.empty((0, 2))
    rng = np.random.default_rng()
    i = 0
    while i < count:
        xy = rng.choice(count, 2, replace=False)
        if numConcat(xy) not in list(map(numConcat, X)):
            X = np.append(X, [xy], axis=0)
        else:
            i -= 1
        i += 1
    if SAVE:
        with open('Examples/datasets/divisive_clustering.npy', 'wb') as f:
            np.save(f, X)

# # Example 3: Redundant datapoints allowed
# X = np.empty((0, 2))
# count = 7
# n_clusters = 7
# for i in range(count):
#     x = np.random.randint(count)
#     y = np.random.randint(count)
#     X = np.append(X, [[x, y]], axis=0)

# print(len(X), len(X[0]), type(X), type(X[0]), type(X[0][0]))
start = perf_counter()
# create divisive clustering object
DC = DivisiveClustering()
# find the result clusters and result centroids
result_clusters, result_centroids = DC.work(X, n_clusters, 7)
stop = perf_counter()
# print(f'len(result_centroids): {len(result_centroids)}')
# print(f'result_centroids: {result_centroids}')
# print(f'len(result_clusters): {len(result_clusters)}')
# print(f'result_clusters: {result_clusters}')
print(f'Time taken by this algorithm: {stop-start}')
# scatter plot of data and dendrogram
visualize(result_clusters, result_centroids, n_clusters, count)

# scipy's agglomerative used for benchmarking. Comparable only when n_clusters=count
# start = perf_counter()
Z = linkage(X, 'median')
# stop = perf_counter()
plt.figure()
# print(f'Linkage done in {stop-start} time.')
dn = dendrogram(Z, n_clusters, 'lastp')
plt.show()
