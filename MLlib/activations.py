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


class TanH():
    def activation(X):
        """
        Apply hyperbolic tangent function on X Vector.

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

    def derivative(X):
        """
        Calculate derivative of hyperbolic tangent function on X Vector.

        PARAMETERS
        ==========

        X: ndarray(dtype=float, ndim=1)
            Array containing Input Values.

        RETURNS
        =======

        ndarray(dtype=float,ndim=1)
            Outputs array of derivatives.
        """
        return 1.0 - np.tanh(X)**2


class Softmax():
    def activation(X):
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
    
    def derivative(X):
        """
        Calculate derivative of Softmax on X Vector.

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
        X_vector = X.reshape(X.shape[0],1)
        X_matrix = np.tile(X_vector,X.shape[0])
        X_der = np.diag(X) - (X_matrix * np.transpose(X_matrix))
        return X_der


class Softsign():
    def activation(X):
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
    
    def derivative(X):
        """
        Calculate derivative of Softsign on X Vector.

        PARAMETERS
        ==========

        X: ndarray(dtype=float, ndim=1)
            Array containing Input Values.

        RETURNS
        =======

        ndarray(dtype=float,ndim=1)
            Output Vector after Vectorised Operation.
        """
        return 1 / (np.abs(X) + 1)**2


class Relu():
    def activation(X):
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

    def derivative(X):
        """
        Calculate derivative of Rectified Linear Unit on X Vector.

        PARAMETERS
        ==========

        X: ndarray(dtype=float, ndim=1)
            Array containing Input Values.

        RETURNS
        =======

        ndarray(dtype=float,ndim=1)
            Outputs array of derivatives.
        """
        return np.greater(X, 0).astype(int)


class LeakyRelu():
    def activation(X, alpha=0.01):
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

    def derivative(X, alpha=0.01):
        """
        Calculate derivative of Leaky Rectified Linear Unit on X Vector.

        PARAMETERS
        ==========

        X: ndarray(dtype=float, ndim=1)
            Array containing Input Values.
        alpha: float
            Slope for Values of X less than 0.

        RETURNS
        =======

        ndarray(dtype=float,ndim=1)
            Outputs array of derivatives.
        """
        dx = np.ones_like(X)
        dx[X < 0] = alpha
        return dx


class Elu():   
    def activation(X, alpha=1.0):
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

    def derivative(X, alpha=1.0):
        """
        Calculate derivative of Exponential Linear Unit on X Vector.

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
        return np.where(x > 0, np.ones_like(x), alpha * np.exp(x))


def unit_step(X):
    """
    Apply Binary Step Function on X Vector.

    PARAMETERS
    ==========

    X: ndarray(dtype=float, ndim=1)
        Array containing Input Values.


    RETURNS
    =======

    ndarray(dtype=float,ndim=1)
        Output Vector after Vectorised Operation.
    """
    return np.heaviside(X, 1)


class Swish():
    def activation(X, alpha=1.0):
        """
        Apply Swish activation function on X Vector.

        PARAMETERS
        ==========

        X: ndarray(dtype=float, ndim=1)
            Array containing Input Values.
        b: int or float
            Either constant or trainable parameter according to the model.

        RETURNS
        =======

        ndarray(dtype=float,ndim=1)
            Output Vector after Vectorised Operation.
        """
        return X / (1 + np.exp(-(alpha*X)))

    def derivative(X, alpha=1.0):
        """
        Calculate derivative of Swish activation function on X Vector.

        PARAMETERS
        ==========

        X: ndarray(dtype=float, ndim=1)
            Array containing Input Values.
        b: int or float
            Either constant or trainable parameter according to the model.

        RETURNS
        =======

        ndarray(dtype=float,ndim=1)
            Output Vector after Vectorised Operation.
        """
        s = 1 / (1 + np.exp(-X))
        f = X / (1 + np.exp(-(alpha*X)))
        df = f + (s * (1 - f))
        return df
