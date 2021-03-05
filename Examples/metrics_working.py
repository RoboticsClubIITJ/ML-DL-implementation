import numpy as np
from MLlib.metrics import matrix_evolution
from MLlib.utils.misc_utils import read_data
p, y = read_data("datasets/metrics_dataset.txt")
z = np.transpose(p)
x = z[0]
matrix = matrix_evolution.confusion_matrix(x, y[0])
print(matrix)
matrix_evolution.score_metrics(x, y[0])
