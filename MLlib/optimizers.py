from MLlib.loss_func import MeanSquaredError
import numpy as np
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
    pass


class NesterovAcceleratedGradientDescent():

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
            self.learning_rate * \
            self.loss_func.derivative(x, y, W - self.gamma * self.Vp)

        W = W - self.Vc

        self.Vp = self.Vc

        return W


class NesterovAccGD(NesterovAcceleratedGradientDescent):
    '''
    An abstract class to provide an alias to the
    really long class name NesterovAcceleratedGradientDescent.
    '''
    pass


class Adagrad():

    def __init__(
            self, learning_rate=0.01,
            loss_func=MeanSquaredError,
            batch_size=5,
            epsilon=0.00000001
    ):

        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.S = 0

    def iterate(self, X, Y, W):

        M, N = X.shape
        index = [random.randint(0, M-1) for i in range(self.batch_size)]
        x = X[index, :]
        y = Y[:, index]
        x.shape = (self.batch_size, N)
        y.shape = (1, self.batch_size)

        derivative = self.loss_func.derivative(x, y, W)

        self.S += derivative * derivative

        W = W - self.learning_rate / \
            np.sqrt(self.S + self.epsilon) * derivative

        return W


class Adadelta():

    def __init__(
            self, learning_rate=0.01,
            loss_func=MeanSquaredError,
            batch_size=5,
            gamma=0.9,
            epsilon=0.00000001
    ):

        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.S = 0

    def iterate(self, X, Y, W):

        M, N = X.shape
        index = [random.randint(0, M-1) for i in range(self.batch_size)]
        x = X[index, :]
        y = Y[:, index]
        x.shape = (self.batch_size, N)
        y.shape = (1, self.batch_size)

        derivative = self.loss_func.derivative(x, y, W)

        self.S += self.gamma * self.S + \
            (1 - self.gamma) * derivative * derivative

        W = W - self.learning_rate / \
            np.sqrt(self.S + self.epsilon) * derivative

        return W


class Adam():

    def __init__(
            self, learning_rate=0.01,
            loss_func=MeanSquaredError,
            batch_size=5,
            epsilon=0.00000001,
            beta1=0.9,
            beta2=0.999
    ):

        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.S = 0
        self.Sc = 0
        self.V = 0
        self.Vc = 0

    def iterate(self, X, Y, W):

        M, N = X.shape
        index = [random.randint(0, M-1) for i in range(self.batch_size)]
        x = X[index, :]
        y = Y[:, index]
        x.shape = (self.batch_size, N)
        y.shape = (1, self.batch_size)

        derivative = self.loss_func.derivative(x, y, W)

        self.V = self.beta1 * self.V + (1 - self.beta1) * derivative
        self.S = self.beta2 * self.S + \
            (1 - self.beta2) * derivative * derivative

        self.Vc = self.V / (1 - self.beta1)
        self.Sc = self.S / (1 - self.beta2)

        W = W - self.learning_rate / \
            (np.sqrt(self.Sc) + self.epsilon) * self.Vc

        return W
