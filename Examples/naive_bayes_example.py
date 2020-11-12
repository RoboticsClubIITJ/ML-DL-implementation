from MLlib.models import Naive_Bayes
import numpy as np

with open('/datasets/naive_bayes_dataset.txt', 'r') as f:
    words = [[string.strip('\n')
              for string in line.split(',')] for line in f]

# for testing default label="outlook" (sunny,rainy or overcast)

x_label = np.array([words[i][0] for i in range(len(words))])
y_class = np.array([words[i][-1] for i in range(len(words))])

model = Naive_Bayes()

prediction = model.predict(x_label, y_class)

print(prediction)
