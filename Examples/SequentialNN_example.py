from MLlib import Tensor
import MLlib.optim as optim
import MLlib.nn as nn
from MLlib.models import Sequential
import numpy as np      # for features and target generation


np.random.seed(5322)

model = Sequential(
    nn.Linear(2, 6),
    nn.Linear(6, 8),
    nn.Linear(8, 1)
)


X = Tensor(np.random.randn(3, 2))       # (batch_size, features)
Y = Tensor(np.random.randn(3, 1))       # (batch_size, output)

nb_epochs = 800     # number of epochs

alpha = 0.01        # learning rate

optimizer = optim.SGD(model.parameters(), alpha)        # SGD optimizer

for i in range(nb_epochs):

    pred = model(X)

    loss = ((Y-pred)**2).sum()/3

    if (i+1) % 100 == 0:
        print('Loss after epoch {}: {}'.format(i+1, loss.data))

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()


p = model(X)
print('\nTarget: \n', Y, '\n')
print('Predicted: \n', p.data, '\n')


# TODO:
# -decide a naming convention for classes in functional.py
# -add activation functions in functional.py
#
# -create a .fit(...) method inside Sequential model class
#
# -create a Layer class inheriting from Module which can
#   apply activation function in forward pass which is passed
#   to it in __init__(...)
#
