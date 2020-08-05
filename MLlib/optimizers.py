from .loss_func import MeanSquaredError
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
    really long class name StochasticGradientDescent.
    '''
    pass


class MiniBatchGradientDescent():

    def __init__(
            self, learning_rate=0.01,
            loss_func=MeanSquaredError,
            batch_size=5
    ):
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.batch_size = batch_size

    def iterate(self, X, Y, W):
        M, N = X.shape
        index = [random.randint(0, M-1) for i in range(self.batch_size)]
        x = X[index, :]
        y = Y[:, index]
        x.shape = (self.batch_size, N)
        y.shape = (1, self.batch_size)
        return W - self.learning_rate * self.loss_func.derivative(x, y, W)


class MiniBatchGD(MiniBatchGradientDescent):
    '''
    An abstract class to provide an alias to the
    really long class name MiniBatchGradientDescent.
    '''
    pass


class MomentumGradientDescent():

    def __init__(
            self, learning_rate=0.01,
            loss_func=MeanSquaredError,
            batch_size=5,
            gamma=0.9
    ):
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.gamma = gamma
        self.Vp = 0
        self.Vc = 0

    def iterate(self, X, Y, W):

        M, N = X.shape
        index = [random.randint(0, M-1) for i in range(self.batch_size)]
        x = X[index, :]
        y = Y[:, index]
        x.shape = (self.batch_size, N)
        y.shape = (1, self.batch_size)

        self.Vc = self.gamma * self.Vp + \
            self.learning_rate * self.loss_func.derivative(x, y, W)

        W = W - self.Vc

        self.Vp = self.Vc

        return W


class MomentumGD(MomentumGradientDescent):
    '''
    An abstract class to provide an alias to the
    really long class name MomentumGradientDescent.
    '''
