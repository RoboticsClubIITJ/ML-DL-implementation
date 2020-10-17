from MLlib.utils.knn_utils import read_KNN_dataFile
from MLlib.models import KNN

X = read_KNN_dataFile('datasets/k_regression.txt')

model = KNN()

prediction = model.predict(X, X[1], num_neighbours=5, classify=False)

print(X[1][-1], prediction)
