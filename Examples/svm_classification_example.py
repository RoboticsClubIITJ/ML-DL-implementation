from MLlib.models import LinearSVC
import numpy as np
A = np.genfromtxt("datasets/svm_classification.txt")
X = A[:, 0:-1]
Y = A[:, -1].flatten()
clf = LinearSVC()
clf.plot(X, Y, epochs=1000)
