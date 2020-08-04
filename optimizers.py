from loss_func import MeanSquaredError


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
