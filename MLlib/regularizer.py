from MLlib import Tensor
import MLlib.nn as nn
from MLlib.loss_func import MSELoss
from MLlib.optim import SGDWithMomentum
import numpy as np
from MLlib.functional import absolute


class L1_Regularizer:
    """
    Implement L1 Regularizer a.k.a. Lasso Regression

    ATTRIBUTES
    ==========

    None

    METHODS
    =======

    get_loss(parameters, Lambda)
    Calculates and returns the L1 Regression Loss

    """

    def __init__(self, parameters, Lambda):
        """
        PARAMETERS
        ==========

        params: list or iterator
            Parameters which need to be regularized

        Lambda: float
            Regularization rate

        """
        if type(parameters).__name__ == 'Tensor':
            self.params = (parameters,)
        else:
            self.params = tuple(parameters)

        self.Lambda = Lambda

    def get_loss(self):
        """
        Calculates and returns the  L2 Regression Loss

        """
        reg_loss = Tensor(0., requires_grad=True)

        for param in self.params:
            reg_loss += absolute(param).sum()

        return (reg_loss * self.Lambda)


class L2_Regularizer:
    """
    Implement L2 Regularizer a.k.a. Ridge Regression

    ATTRIBUTES
    ==========

    None

    METHODS
    =======

    get_loss(parameters, Lambda)
    Calculates and returns the L2 Regression Loss

    """

    def __init__(self, parameters, Lambda):
        """
        PARAMETERS
        ==========

        params: list or iterator
            Parameters which need to be regularized

        Lambda: float
            Regularization rate

        """

        if type(parameters).__name__ == 'Tensor':
            self.params = (parameters,)
        else:
            self.params = tuple(parameters)

        self.Lambda = Lambda

    def get_loss(self):
        """
        Calculates the returns the  L2 Regression Loss

        """
        reg_loss = Tensor(0., requires_grad=True)
        for param in self.params:
            reg_loss += (param**2).sum()
        return (reg_loss*self.Lambda)


class LinearRegWith_Regularization(nn.Module):
    """"
    LinearRegWith_Regularization

    It inherits the Class Module

    It implements Linear Regression with different types of
    Regularizer(L1 and L2)

    """

    def __init__(self,
                 in_features,
                 regularizer,
                 loss_fn=MSELoss,
                 optimizer=SGDWithMomentum,
                 Lambda=10):
        """
        PARAMETERS
        ==========

        in_features: int
            number of features

        regularizer: class
            Class of one of the Regularizers like
            L1_Regularizer and L2_Regularizer

        optimizer: class
            Class of one of the Optimizers like
            SGD and SGDWithMomentum

        loss_fn: class
            Class of one of the loss functions like
            MSELoss

        Lambda: float
            Regularization rate

        """
        super().__init__()
        self.linear_layer = nn.Linear(in_features, 1)
        self.loss_fn = loss_fn()
        self.regularizer = regularizer(self.linear_layer.weights, Lambda)
        self.optimizer = optimizer(self.linear_layer.parameters())

    def forward(self, input):
        """
        Forward pass
        """
        return self.linear_layer(input)

    def fit(self, x, y, epochs=1):
        """
        Train LinearRegWith_Regularization Model
        by fitting its associated Regularizer

        PARAMETERS
        ==========

        x: list or iterator
           input Dataset

        y: list or iterator
           input Dataset

        epochs: int
            Number of times, the loop to calculate loss
            and optimize weights, will going to take
            place.

        RETURNS
        =======
        output: ndarray(dtype=float, ndim=1)
            Total loss

        """
        output = []
        for i in range(epochs):
            # for batch in train_batches:
            prediction = self(x)
            loss = self.loss_fn(prediction, y) \
                + self.regularizer.get_loss()/(2*prediction.shape[0])
            if (i+1) % 100 == 0:
                output.append(loss.data)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return np.array(output)
