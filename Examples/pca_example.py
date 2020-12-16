from MLlib.models import PCA
import pandas as pd

iris = pd.read_csv('datasets/Iris.csv')
X, Y = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm',
             'PetalWidthCm']], iris[['Species']]

pca = PCA()
X_new = pca.fit(X)

print(pca.transform(X))
