from MLlib.models import Gaussian_Naive_Bayes
import numpy as np

with open('MLlib/datasets/gaussian_naive_bayes_dataset.txt', 'r') as f:
    words = [[string.strip('\n')
              for string in line.split(',')] for line in f]

# for testing default label="outlook" (sunny,rainy or overcast)

x_label = np.array([words[i][0] for i in range(len(words))])
y_class = np.array([[float(words[i][j]) for j in range(1,len(words[0])) ]for i in range(len(words))])

GNB = Gaussian_Naive_Bayes()

# sample input for testing, belongs to female, can be replaced to have user input

X = [5.75, 130, 8]

print(GNB.predict(X, x_label, y_class))