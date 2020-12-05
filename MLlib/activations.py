import numpy as np


class Sigmoid():
    def activation(X):
        """
        Apply Sigmoid on X Vector.

        PARAMETERS
        ==========

        X: ndarray(dtype=float, ndim=1)
            Array containing Input Values.

        RETURNS
        =======

        ndarray(dtype=float,ndim=1)
            Output Vector after Vectorised Operation.
        """
        return 1 / (1 + np.exp(-X))

    def derivative(X):
        """
        Calculate derivative of Sigmoid on X Vector.

        PARAMETERS
        ==========

        X: ndarray(dtype=float, ndim=1)
            Array containing Input Values.

        RETURNS
        =======

        ndarray(dtype=float,ndim=1)
            Outputs array of derivatives.
        """
        s = 1 / (1 + np.exp(-X))
        ds = s * (1 - s)
        return ds


def tanh(X):
    """
    Apply Inverse of Tan on X Vector.

    PARAMETERS
    ==========

    X: ndarray(dtype=float, ndim=1)
        Array containing Input Values.

    RETURNS
    =======

    ndarray(dtype=float,ndim=1)
        Output Vector after Vectorised Operation.
    """
    return np.tanh(X)


def softmax(X):
    """
    Apply Softmax on X Vector.

    PARAMETERS
    ==========

    X: ndarray(dtype=float, ndim=1)
        Array containing Input Values.
    Sum: float
        Sum of values of Input Array.

    RETURNS
    =======

    ndarray(dtype=float,ndim=1)
        Output Vector after Vectorised Operation.
    """
    Sum = np.sum(np.exp(X))
    return np.exp(X) / Sum


def softsign(X):
    """
    Apply Softsign on X Vector.

    PARAMETERS
    ==========

    X: ndarray(dtype=float, ndim=1)
        Array containing Input Values.

    RETURNS
    =======

    ndarray(dtype=float,ndim=1)
        Output Vector after Vectorised Operation.
    """
    return X / (np.abs(X) + 1)


def relu(X):
    """
    Apply Rectified Linear Unit on X Vector.

    PARAMETERS
    ==========

    X: ndarray(dtype=float, ndim=1)
        Array containing Input Values.

    RETURNS
    =======

    ndarray(dtype=float,ndim=1)
        Output Vector after Vectorised Operation.
    """
    return np.maximum(0, X)


def leakyRelu(X, alpha=0.01):
    """
    Apply Leaky Rectified Linear Unit on X Vector.

    PARAMETERS
    ==========

    X: ndarray(dtype=float, ndim=1)
        Array containing Input Values.
    alpha: float
        Slope for Values of X less than 0.

    RETURNS
    =======

    ndarray(dtype=float,ndim=1)
        Output Vector after Vectorised Operation.
    """

    return np.maximum(alpha*X, X)


def elu(X, alpha=1.0):
    """
    Apply Exponential Linear Unit on X Vector.

    PARAMETERS
    ==========

    X: ndarray(dtype=float, ndim=1)
        Array containing Input Values.
    alpha: float
        Curve Constant for Values of X less than 0.

    RETURNS
    =======

    ndarray(dtype=float,ndim=1)
        Output Vector after Vectorised Operation.
    """
    assert (alpha > 0)
    return np.maximum(0, X) + np.minimum(0, alpha * (np.exp(X) - 1))
