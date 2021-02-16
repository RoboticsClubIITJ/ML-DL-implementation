from MLlib.models import BernoulliNB
import numpy as np

with open('/datasets/bernoulli_naive_bayes_dataset.txt', 'r') as f:
    words = [[string.strip('\n')
              for string in line.split(',')] for line in f]
for i in range(len(words)):
    words[i] = list(map(int, words[i]))

x = np.array([words[i] for i in range(len(words)-1)])
y_class = np.array(words[-1])

test = np.array([[1, 0, 0, 0, 1, 1], [1, 1, 1, 0, 0, 1]])
nb = BernoulliNB(alpha=1).fit(np.where(x > 0, 1, 0), y_class)
print(nb.predict(test))
