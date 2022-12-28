import numpy as np
from MLlib.activations import Sigmoid
from MLlib import Tensor
from MLlib import autograd
from MLlib.utils.misc_utils import unbroadcast


class MeanSquaredError(autograd.Function):
    """
    Calculate Mean Squared Error.
    """

    __slots__ = ()

    @staticmethod
    def forward(ctx, prediction, target):
        if not (type(prediction).__name__ == 'Tensor' and
                type(target).__name__ == 'Tensor'):

            raise RuntimeError("Expected Tensors, got {} and {}. Please use "
                               ".loss() method for non-Tensor data"
                               .format(type(prediction).__name__,
                                       type(target).__name__))

        requires_grad = prediction.requires_grad

        batch_size = target.data.shape[0]

        out = prediction.data - target.data

        if requires_grad:
            ctx.derivative_core = out

        out = np.sum(np.power(out, 2)) / (2*batch_size)

        output = Tensor(out, requires_grad=requires_grad,
                        is_leaf=not requires_grad)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        derivative = ctx.derivative_core

        grad_prediction = (derivative / derivative.shape[0]) * grad_output.data

        return Tensor(unbroadcast(grad_prediction, derivative.shape))

    @staticmethod
    def loss(X, Y, W):
        """
        Calculate loss by mean square method.

        PARAMETERS
        ==========

        X:ndarray(dtype=float,ndim=1)
          input vector
        Y:ndarray(dtype=float)
          output vector
        W:ndarray(dtype=float)
          Weights

         RETURNS
         =======

         array of mean squared losses
        """
        M = X.shape[0]
        return np.sum((np.dot(X, W).T - Y) ** 2) / (2 * M)

    @staticmethod
    def derivative(X, Y, W):
        """
        Calculate derivative for mean square method.

        PARAMETERS
        ==========

        X:ndarray(dtype=float,ndim=1)
          input vector
        Y:ndarray(dtype=float)
          output vector
        W:ndarray(dtype=float)
          Weights

         RETURNS
         =======

         array of derivates
        """
        M = X.shape[0]
        return np.dot((np.dot(X, W).T - Y), X).T / M


class MSELoss(MeanSquaredError):
    pass


class LogarithmicError():
    """
    Calculate Logarithmic Error.
    """

    @staticmethod
    def loss(X, Y, W):
        """
        Calculate loss by logarithmic error method.

        PARAMETERS
        ==========

        X:ndarray(dtype=float,ndim=1)
          input vector
        Y:ndarray(dtype=float)
          output vector
        W:ndarray(dtype=float)
          Weights

         RETURNS
         =======

         array of logarithmic losses
        """
        M = X.shape[0]
        sigmoid = Sigmoid()
        H = sigmoid.activation(np.dot(X, W).T)
        return (1/M)*(np.sum((-Y)*np.log(H)-(1-Y)*np.log(1-H)))

    @staticmethod
    def derivative(X, Y, W):
        """
        Calculate derivative for logarithmic error method.

        PARAMETERS
        ==========

        X:ndarray(dtype=float,ndim=1)
          input vector
        Y:ndarray(dtype=float)
          output vector
        W:ndarray(dtype=float)
          Weights

         RETURNS
         =======

         array of derivates
        """
        M = X.shape[0]
        sigmoid = Sigmoid()
        H = sigmoid.activation(np.dot(X, W).T)
        return (1/M)*(np.dot(X.T, (H-Y).T))


class AbsoluteError():
    """
    Calculate Absolute Error.
    """

    @staticmethod
    def loss(X, Y, W):
        """
        Calculate loss by absolute error method.

        PARAMETERS
        ==========

        X:ndarray(dtype=float,ndim=1)
          input vector
        Y:ndarray(dtype=float)
          output vector
        W:ndarray(dtype=float)
          Weights

         RETURNS
         =======

         array of absolute losses
        """
        M = X.shape[0]
        return np.sum(np.absolute(np.dot(X, W).T - Y)) / M

    @staticmethod
    def derivative(X, Y, W):
        """
        Calculate derivative for absolute error method.

        PARAMETERS
        ==========

        X:ndarray(dtype=float,ndim=1)
          input vector
        Y:ndarray(dtype=float)
          output vector
        W:ndarray(dtype=float)
          Weights

         RETURNS
         =======

         array of derivates
        """
        M = X.shape[0]
        AbsError = (np.dot(X, W).T-Y)
        return np.dot(
            np.divide(
                AbsError,
                np.absolute(AbsError),
                out=np.zeros_like(AbsError),
                where=(np.absolute(AbsError)) != 0),
            X
        ).T/M


class CosineSimilarity():
    """
    Calculate Similarity between actual value and similarity value.
    """

    @staticmethod
    def loss(X, Y, W):
        """
        Calculate error by cosine similarity method

        PARAMETERS
        ==========

        X:ndarray(dtype=float,ndim=1)
          input vector
        Y:ndarray(dtype=float)
          output vector
        W:ndarray(dtype=float)
          Weights

         RETURNS
         =======

         Percentage of error in the actural value and predicted value
         """
        H = (np.dot(X, W).T)
        DP = np.sum(np.dot(H, Y))
        S = DP/((np.sum(np.square(H))**(0.5))*(np.sum(np.square(Y))**(0.5)))
        dissimilarity = 1-S
        return dissimilarity*(np.sum(np.square(Y))**(0.5))


class Log_cosh():

    @staticmethod
    def logcosh_loss(X, Y, W):
        """
        Calculate Error by log cosh method

        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
           Input Vector
        Y: ndarray (dtpye=float)
           Output Vector
        W:ndarray(dtype=float)
          Weights

        RETURNS
        =======

        Logarithm of the hyperbolic cosine of the prediction error
        """
        p = np.cosh(Y - np.dot(X, W).T)
        loss = np.log(p)
        error = np.sum(loss)
        return error

    @staticmethod
    def derivative_logcosh(X, Y, W):
        """
        Calculate the derivative of "log cosh" loss method

        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
           Actual values
        Y: ndarray (dtpye=float)
           Predicted values
        W:ndarray(dtype=float)
           Weights

        RETURNS
        =======

        Derivative of Log cosh prediction error
        """
        t = np.tanh(Y-np.dot(X, W).T) @ (-X)
        derivative = np.sum(t)
        return derivative


class Huber():
    """
    Calculate Huber loss.
    """

    @staticmethod
    def loss(X, Y, W, delta=1.0):

        """
        Calculate loss by Huber method.

        PARAMETERS
        ==========

        X:ndarray(dtype=float,ndim=1)
          input vector
        Y:ndarray(dtype=float)
          output vector
        W:ndarray(dtype=float)
          Weights

         RETURNS
         =======

         array of Huber loss
        """
        y_pred = np.dot(X, W).T
        M = X.shape[0]
        error = np.where(np.abs(Y - y_pred) <= delta,
                         0.5 * (Y - y_pred)**2,
                         delta * (np.abs(Y - y_pred)-0.5*delta))
        return np.sum(error) / M

    @staticmethod
    def derivative(X, Y, W, delta=1.0):

        """
        Calculate derivative for Huber method.

        PARAMETERS
        ==========

        X:ndarray(dtype=float,ndim=1)
          input vector
        Y:ndarray(dtype=float)
          output vector
        W:ndarray(dtype=float)
          Weights

         RETURNS
         =======

         array of derivates
        """
        y_pred = np.dot(X, W).T
        M = X.shape[0]
        der = 0
        for i in range(M):
            y = Y.transpose()
            yp = y_pred.transpose()
            if abs(y[i] - yp[i]) <= delta:
                der += -X[i] * (y[i] - yp[i])
            else:
                der += delta * X[i] * (y[i] - yp[i]) / abs(yp[i] - y[i])
        return der


class MeanSquaredLogLoss():
    """""
    Calcute Mean Squared Log Loss
    """

    @staticmethod
    def loss(X, Y, W):
        """
            Calculate  Mean Squared Log Loss

            PARAMETERS
            ==========

            X:ndarray(dtype=float,ndim=1)
              input vector
            Y:ndarray(dtype=float)
              output vector
            W:ndarray(dtype=float)
              Weights

            RETURNS
            =======

            array of mean of logarithmic losses
        """

        M = X.shape[0]
        sigmoid = Sigmoid()
        H = sigmoid.activations(np.dot(X, W).T)
        return np.sqrt((1 / M) * (np.sum(np.log(Y + 1) - np.log(H + 1))))


class MeanAbsolutePrecentageError():
    """""
    Calcute Mean Absolute Percentage Loss
    """

    @staticmethod
    def loss(X, Y, W):
        """
            Calculate  Mean Squared Log Loss

            PARAMETERS
            ==========

            X:ndarray(dtype=float,ndim=1)
              input vector
            Y:ndarray(dtype=float)
              output vector
            W:ndarray(dtype=float)
              Weights

            RETURNS
            =======

            array of Mean Absolute Percentage Loss
        """

        y_pred = np.dot(X, W).T
        L = np.sum(np.true_divide((np.abs(Y - y_pred) * 100), Y)) / X.shape[0]
        return L


class PoisonLoss():
    """
    Calculate Poisson Loss.
    """

    @staticmethod
    def loss(X, Y, W):
        """
        Calculate Poisson Loss.

        PARAMETERS
        ==========

        X: ndarray(dtype=float)
            Input vector
        Y: ndarray(dtype=float)
            Output vector
        W: ndarray(dtype=float)
            Weights

        RETURNS
        =======

        float or ndarray
            array of Poisson losses or
            float value of Poisson loss
            (depending on the parameters)
        """

        y_pred = np.dot(X, W).T
        return np.mean(y_pred - Y * np.log(y_pred), axis=-1)
