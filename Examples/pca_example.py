from MLlib.models import PCA
import numpy as np


iris = np.genfromtxt('datasets/Iris.csv', delimiter=',')
X = iris.iloc[:, 1:5]

pca = PCA()
X_new = pca.fit(X)

print(pca.transform(X))
