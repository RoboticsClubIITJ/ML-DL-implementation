from MLlib.utils.knn_utils import read_KNN_dataFile
from MLlib.models import KNN
from matplotlib import pyplot as plt
import numpy as np

X = read_KNN_dataFile('datasets/knn_classification.txt')

model = KNN()

prediction = model.predict(X, X[1], num_neighbours=5, classify=True)

print(len(X[:,2]))
model.plot(X,X,5,26)

