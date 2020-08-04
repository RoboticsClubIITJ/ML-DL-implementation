from loss_func import MeanSquaredError
import random


class GradientDescent():
    '''
    A classic gradient descent implementation.

    W = W - a * dm

    a - learning rate
    dm - derivative of loss function wrt x (parameter)
    W - Weights
    '''

    def __init__(self, learning_rate=0.01, loss_func=MeanSquaredError):
        self.learning_rate = learning_rate
        self.loss_func = loss_func

    def iterate(self, X, Y, W):
        return W - self.learning_rate * self.loss_func.derivative(X, Y, W)


class StochasticGradientDescent():

    def __init__(self, learning_rate=0.01, loss_func=MeanSquaredError):
        self.learning_rate = learning_rate
        self.loss_func = loss_func

    def iterate(self, X, Y, W):
        M, N = X.shape
        i = random.randint(0, M-1)
        x, y = X[i, :], Y[:, i]
        x.shape, y.shape = (1, N), (1, 1)
        return W - self.learning_rate * self.loss_func.derivative(x, y, W)


class SGD(StochasticGradientDescent):
    '''
    An abstract class to provide an alias to the
    really long class name StochasticGradientDescent
    '''
    pass
