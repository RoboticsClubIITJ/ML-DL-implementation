from MLlib.loss_func import MeanSquaredError
import numpy as np
import random


class GradientDescent():
    """
    A classic gradient descent implementation.

    W = W - a * dm

    a : learning rate
    dm : derivative of loss function wrt x (parameter)
    W : Weights
    """

    def __init__(self, learning_rate=0.01, loss_func=MeanSquaredError):
        """
        Init GradientDescent.

        PARAMETERS
        ==========

        learning_rate: dtype=float
            Step size at each iteration while moving toward
            a minimum of a loss function

        loss_func: function
            Loss function to be implemented from loss_func.py
        """
        self.learning_rate = learning_rate
        self.loss_func = loss_func

    def iterate(self, X, Y, W):
        """
        Calculate Weights for Gradient Descent.

        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
            Input vector
        Y: ndarray(dtype=float)
            Output vector
        W: ndarray(dtype=float)
            Weights

        RETURNS
        =======

        W: ndarray(dtype=float)
            Optimized weights using suitable learning rate with Gradient
            Descent Algorithm
        """
        return W - self.learning_rate * self.loss_func.derivative(X, Y, W)


class StochasticGradientDescent():
    """
    A  stochastic gradient descent implementation.

    W = W - a * dm

     a : learning rate
    dm : derivative of loss function wrt x
    W : Weights
    """

    def __init__(self, learning_rate=0.01, loss_func=MeanSquaredError):
        """
        Init StochasticGradientDescent.

        PARAMETERS
        ==========

        learning_rate: dtype=float
            Step size at each iteration while moving toward
            a minimum of a loss function

        loss_func: function
            Loss function to be implemented from loss_func.py
        """
        self.learning_rate = learning_rate
        self.loss_func = loss_func

    def iterate(self, X, Y, W):
        """
        Calculate Weights for Stochastic Gradient Descent.

        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
            Input vector
        Y: ndarray(dtype=float)
            Output vector
        W: ndarray(dtype=float)
            Weights

        RETURNS
        =======

        W: ndarray(dtype=float)
           Optimized weights using suitable learning rate with
           Stochastic Gradient Descent Algorithm
        """
        M, N = X.shape
        i = random.randint(0, M - 1)
        x, y = X[i, :], Y[:, i]
        x.shape, y.shape = (1, N), (1, 1)
        return W - self.learning_rate * self.loss_func.derivative(x, y, W)


class SGD(StochasticGradientDescent):
    """
    An abstract class.

    Provide an alias to the really long class name
    StochasticGradientDescent.
    """

    pass


class MiniBatchGradientDescent():
    """
     A  mini batch gradient descent implementation.

    W = W - a * dm

    a : learning rate
    dm : derivative of loss function wrt x (of batch size)
    W : Weights
    """

    def __init__(
            self, learning_rate=0.01,
            loss_func=MeanSquaredError,
            batch_size=5
    ):
        """
        Init MiniBatchGradientDescent.

        PARAMETERS
        ==========

        learning_rate: dtype=float
            Step size at each iteration while moving toward
            a minimum of a loss function

        loss_func: function
            Loss function to be implemented from loss_func.py

        batch_size: dtype=int
            Size of batches into which training dataset
            is split for the algorithm
        """
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.batch_size = batch_size

    def iterate(self, X, Y, W):
        """
        Calculate Weights for MiniBatch Gradient Descent.

        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
            Input vector
        Y: ndarray(dtype=float)
            Output vector
        W: ndarray(dtype=float)
            Weights

        RETURNS
        =======

        W: ndarray(dtype=float)
            Optimized weights using suitable learning rate with
            MiniBatch Gradient Descent Algorithm
        """
        M, N = X.shape
        index = [random.randint(0, M - 1) for i in range(self.batch_size)]
        x = X[index, :]
        y = Y[:, index]
        x.shape = (self.batch_size, N)
        y.shape = (1, self.batch_size)
        return W - self.learning_rate * self.loss_func.derivative(x, y, W)


class MiniBatchGD(MiniBatchGradientDescent):
    """
    An abstract class.

    Provide an alias to the
    really long class name MiniBatchGradientDescent.
    """

    pass


class MomentumGradientDescent():
    """
    A  momentum gradient descent implementation.

    W = W - Vc

    Vc: current update vector
    Vc = gamma * Vp + a * dm
    a: learning_rate
    dm: derivative of loss function wrt x
    W : Weights
    """

    def __init__(
            self, learning_rate=0.01,
            loss_func=MeanSquaredError,
            batch_size=5,
            gamma=0.9
    ):
        """
        Init MomentumGradientDescent.

        PARAMETERS
        ==========

        learning_rate: dtype=float
            Step size at each iteration while moving toward
            a minimum of a loss function

        loss_func: function
            Loss function to be implemented from loss_func.py

        batch_size: dtype=int
            Size of batches into which training dataset
            is split for the algorithm

        gamma: dtype=float
            Part of past update vector (Vp) to be added
            in current update vector (Vc)
        """
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.gamma = gamma
        self.Vp = 0
        self.Vc = 0

    def iterate(self, X, Y, W):
        """
        Calculate Weights for Momentum Gradient Descent.

        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
           Input vector
        Y: ndarray(dtype=float)
           Output vector
        W: ndarray(dtype=float)
           Weights

        RETURNS
        =======

        W: ndarray(dtype=float)
           Optimized weights using suitable learning rate with
           Momentum Gradient Descent Algorithm
        """
        M, N = X.shape
        index = [random.randint(0, M - 1) for i in range(self.batch_size)]
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
    """
    An abstract class.

    Provide an alias to the
    really long class name MomentumGradientDescent.
    """

    pass


class NesterovAcceleratedGradientDescent():
    """
    A nesterov accelerated descent implementation.

    W = W - Vc

    Vc: current update vector
    Vc = gamma * Vp + a * dm
    a: learning_rate
    dm: derivative of loss function wrt x
    W : Weights
    """

    def __init__(
            self, learning_rate=0.01,
            loss_func=MeanSquaredError,
            batch_size=5,
            gamma=0.9
    ):
        """
        Init NesterovAcceleratedGradientDescent.

        PARAMETERS
        ==========

        learning_rate: dtype=float
            Step size at each iteration while moving toward
            a minimum of a loss function

        loss_func: function
            Loss function to be implemented from loss_func.py

        batch_size: dtype=int
            Size of batches into which training
            dataset is split for the algorithm

        gamma: dtype=float
            Part of past update vector (Vp) to be added
            in current update vector (Vc)
        """
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.gamma = gamma
        self.Vp = 0
        self.Vc = 0

    def iterate(self, X, Y, W):
        """
        Calculate Weights for Nesterov Accelerated Gradient Descent.

        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
            Input vector
        Y: ndarray(dtype=float)
            Output vector
        W: ndarray(dtype=float)
            Weights

        RETURNS
        =======

        W: ndarray(dtype=float)
            Optimized weights using suitable learning rate with
            Nesterov Accelerated Gradient Descent Algorithm
        """
        M, N = X.shape
        index = [random.randint(0, M - 1) for i in range(self.batch_size)]
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
    """
    An abstract class.

    Provide an alias to the
    really long class name NesterovAcceleratedGradientDescent.
    """

    pass


class Adagrad():
    """
    An adagrad implementation.

    W = W - learning_rate / sqrt(S + epsilon) * derivative

    W: Weights
    """

    def __init__(
            self, learning_rate=0.01,
            loss_func=MeanSquaredError,
            batch_size=5,
            epsilon=0.00000001
    ):
        """
        Init Adagrad.

        PARAMETERS
        ==========

        learning_rate: dtype=float
            Step size at each iteration while moving toward
            a minimum of a loss function

        loss_func: function
            Loss function to be implemented from loss_func.py

        batch_size: dtype=int
            Size of batches into which training
            dataset is split for the algorithm

        epsilon: dtype=float
            smoothing term , avoids division by zero
        """
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.S = 0

    def iterate(self, X, Y, W):
        """
        Calculate Weights for Adagrad algorithm.

        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
            Input vector
        Y: ndarray(dtype=float)
            Output vector
        W: ndarray(dtype=float)
            Weights

        RETURNS
        =======

        W: ndarray(dtype=float)
            Optimized weights using suitable learning rate with
            Adagrad algorithm
        """
        M, N = X.shape
        index = [random.randint(0, M - 1) for i in range(self.batch_size)]
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
    """
    An adagrad implementation.

    W = W - learning_rate / sqrt(S + epsilon) * derivative

    W: Weights
    """

    def __init__(
            self, learning_rate=0.01,
            loss_func=MeanSquaredError,
            batch_size=5,
            gamma=0.9,
            epsilon=0.00000001
    ):
        """
        Init Adadelta.

        PARAMETERS
        ==========

        learning_rate: dtype=float
            Step size at each iteration while moving toward a
            minimum of a loss function

        loss_func: function
            Loss function to be implemented from loss_func.py

        batch_size: dtype=int
            Size of batches into which training dataset is
             split for the algorithm

        epsilon: dtype=float
            Smoothing term , avoids division by zero
        """
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.S = 0

    def iterate(self, X, Y, W):
        """
        Calculate Weights for  Adadelta  algorithm.

        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
            Input vector
        Y: ndarray(dtype=float)
            Output vector
        W: ndarray(dtype=float)
            Weights

        RETURNS
        =======

        W: ndarray(dtype=float)
            Optimized weights using suitable learning rate
            with Adadelta algorithm
        """
        M, N = X.shape
        index = [random.randint(0, M - 1) for i in range(self.batch_size)]
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
    """
    An adaptive momentum estimation (Adam) implementation.

    W = W - learning_rate / (sqrt(Sc) + epsilon) * Vc

    Vc: current update vector
    W: Weights
    """

    def __init__(
            self, learning_rate=0.01,
            loss_func=MeanSquaredError,
            batch_size=5,
            epsilon=0.00000001,
            beta1=0.9,
            beta2=0.999
    ):
        """
        Init Adam.

        PARAMETERS
        ==========

        learning_rate: dtype=float
            Step size at each iteration while moving
            towards a minimum of a loss function

        loss_func: function
            Loss function to be implemented from loss_func.py

        batch_size: dtype=int
            Size of batches into which training dataset is split for
            the algorithm

        epsilon: dtype=float
            Smoothing term , avoids division by zero

        beta1,beta2: dtype=float
            Decay rates
        """
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
        """
        Calculate Weights for Adam algorithm.

        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
            Input vector
        Y: ndarray(dtype=float)
            Output vector
        W: ndarray(dtype=float)
            Weights

        RETURNS
        =======

        W: ndarray(dtype=float)
            Optimized weights using suitable learning rate with Adam algorithm
        """
        M, N = X.shape
        index = [random.randint(0, M - 1) for i in range(self.batch_size)]
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
