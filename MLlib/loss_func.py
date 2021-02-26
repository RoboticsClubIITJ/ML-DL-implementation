import numpy as np
from MLlib.activations import Sigmoid


class MeanSquaredError():
    """
    Calculate Mean Squared Error.
    """

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
        sigmoid = Sigmoid()
        H = sigmoid.activation(np.dot(X, W).T)
        DP = np.sum(np.dot(H, Y))
        S = DP/((np.sum(np.square(H))**(0.5))*(np.sum(np.square(Y))**(0.5)))
        dissimilarity = 1-S
        return dissimilarity*(np.sum(np.square(Y))**(0.5))
