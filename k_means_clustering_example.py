from MLlib.models import KMeansClustering
import numpy as np

X = np.genfromtxt('MLlib/datasets/k_means_clustering.txt')

KMC = KMeansClustering()

KMC.work(X, 3, 7)
