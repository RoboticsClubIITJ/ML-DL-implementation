from MLlib.models import DivisiveClustering
from MLlib.utils.divisive_clustering_utils import visualize, sse
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import logging
logging.basicConfig(filename='MLlib/tests/divisive_Example.log', level=logging.DEBUG, filemode='w', format='\n%(asctime)s\n%(message)s')

LOAD = False
#Example 1
# X = np.genfromtxt('datasets/k_means_clustering.txt')
# count = X.shape[0]

#Example 2
#Try with different number of clusters
n_clusters = 7
count = 7

if not LOAD:
    X = np.empty((0, 2))
    rng = np.random.default_rng()
    i = 0
    while i < count:
        xy = rng.choice(count, 2, replace=False)
        if xy not in X:
            X = np.append(X, [xy], axis=0)
        else:
            i -= 1
        i += 1
        logging.debug(f'X:{X}')
        # print(i)
    # print(len(X), len(X[0]), type(X), type(X[0]), type(X[0][0]))
    with open('Examples/datasets/divisive_clustering.npy', 'wb') as f:
        np.save(f, X)
else:
    with open('Examples/datasets/divisive_clustering.npy', 'rb') as f:
        X = np.load(f)

start = perf_counter()
#create divisive clustering object
DC = DivisiveClustering()
#find the result clusters and 
result_clusters, result_centroids = DC.work(X, n_clusters, 7)
stop = perf_counter()
print(f'Elapsed time: {stop-start}')
print(f'len(result_centroids): {len(result_centroids)}')
print(f'len(result_clusters): {len(result_clusters)}')
visualize(result_clusters, result_centroids, n_clusters, count)

# start = perf_counter()
Z = linkage(X, 'median')
# stop = perf_counter()
plt.figure()
# print(f'Linkage done in {stop-start} time.')
dn = dendrogram(Z, n_clusters, 'lastp')
plt.show()