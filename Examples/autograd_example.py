"""
Demonstrates that the autograd implementation actually works.

class Linear: a linear layer of the neural network
class NN: our tiny toy neural network with no activation functions.

Why no activation functions are being used?
The purpose is of this example is not to produce a standard Neural Network but
to demonstrate the working and ability of autograd.
"""

from MLlib import Tensor
import numpy as np

np.random.seed(5322)


class Linear:
    def __init__(self, in_features, out_features):
        self.bias = Tensor(0., requires_grad=True)
        self.weight = Tensor.randn(out_features, in_features)
        self.weight.requires_grad = True

    def zero_grad(self):
        self.bias.grad = None
        self.weight.grad = None

    def forward(self, x):
        x = (x @ self.weight.T()) + self.bias
        return x

    def update(self, alpha):
        # Gradient Descent
        self.weight.data = self.weight.data - alpha*self.weight.grad.data
        self.bias.data = self.bias.data - alpha*self.bias.grad.data

    __call__ = forward


class NN:
    def __init__(self):
        self.l1 = Linear(2, 6)
        self.l2 = Linear(6, 8)
        self.l3 = Linear(8, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x

    def update(self, alpha):
        self.l1.update(alpha)
        self.l2.update(alpha)
        self.l3.update(alpha)

    def zero_grad(self):
        self.l1.zero_grad()
        self.l2.zero_grad()
        self.l3.zero_grad()

    __call__ = forward


model = NN()

X = Tensor(np.random.randn(3, 2))       # (batch_size, features)
Y = Tensor(np.random.randn(3, 1))       # (batch_size, output)

nb_epochs = 800     # number of epochs

alpha = 0.01        # learning rate

for i in range(nb_epochs):

    model.zero_grad()

    pred = model(X)

    loss = ((Y-pred)**2).sum()/3

    if (i+1) % 100 == 0:
        print('Loss after epoch {}: {}'.format(i+1, loss.data))

    loss.backward()

    model.update(alpha)

p = model(X)
print('\nTarget: \n', Y, '\n')
print('Predicted: \n', p.data, '\n')
