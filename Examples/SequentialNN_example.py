from MLlib import Tensor
import MLlib.optim as optim
import MLlib.nn as nn
from MLlib.models import Sequential
from MLlib.activations import Relu
from MLlib.loss_func import MSELoss
import numpy as np      # for features and target generation


np.random.seed(5322)


model = Sequential(
    nn.Linear(4, 16, activation_fn=Relu),
    nn.Linear(16, 8, activation_fn=Relu),
    nn.Linear(8, 2)
)


X = Tensor(np.random.randn(10, 4))       # (batch_size, features)
Y = Tensor(np.random.randn(10, 2))       # (batch_size, output)


nb_epochs = 800     # number of epochs

alpha = 0.001        # learning rate

# SGD optimizer
optimizer = optim.SGDWithMomentum(model.parameters(), alpha, momentum=0.9)

loss_fn = MSELoss()     # Mean Squared Error loss

for i in range(nb_epochs):

    pred = model(X)

    loss = loss_fn(pred, Y)

    if (i+1) % 100 == 0:
        print('Loss after epoch {}: {}'.format(i+1, loss.data))

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()
