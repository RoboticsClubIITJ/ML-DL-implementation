from MLlib.models import Agglomerative_clustering
import numpy as np

X = np.genfromtxt('datasets/agglomerative_clustering.txt')


model = Agglomerative_clustering()
model.work(X, 4)
model.plot(X)
