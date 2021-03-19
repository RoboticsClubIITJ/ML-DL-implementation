import MLlib
import numpy as np
from MLlib import autograd
from MLlib.utils.misc_utils import unbroadcast


class Sigmoid(autograd.Function):

    __slots__ = ()

    @staticmethod
    def forward(ctx, input):
        if not (type(input).__name__ == 'Tensor'):
            raise RuntimeError("Expected a Tensor, got {}. Please use "
                               "Sigmoid.activation() for non-Tensor data"
                               .format(type(input).__name__))

        requires_grad = input.requires_grad

        output = 1 / (1 + np.exp(-input.data))

        output = MLlib.Tensor(output, requires_grad=requires_grad,
                              is_leaf=not requires_grad)

        if requires_grad:
            ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        o = ctx.saved_tensors[0]

        grad_o = o.data * (1 - o.data) * grad_output.data
        grad_o = MLlib.Tensor(unbroadcast(grad_o, o.shape))

        return grad_o

    @staticmethod
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

    @staticmethod
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
    @staticmethod
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

    @staticmethod
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


class Softmax(autograd.Function):

    __slots__ = ()

    @staticmethod
    def forward(ctx, input):
        if not (type(input).__name__ == 'Tensor'):
            raise RuntimeError("Expected a Tensor, got {}. Please use "
                               "Softmax.activation() for non-Tensor data"
                               .format(type(input).__name__))

        if len(input.shape) != 2:
            raise RuntimeError("Expected a batch of data of size (m, classes)"
                               ", got {}".format(input.shape))

        requires_grad = input.requires_grad

        e_x = np.exp(input.data)
        output = e_x / np.sum(e_x, axis=1, keepdims=True)
        # axis=1 because we don't want to compute across batch dimension

        output = MLlib.Tensor(output, requires_grad=requires_grad,
                              is_leaf=not requires_grad)

        if requires_grad:
            ctx.save_for_backward(output)
            ctx.nb_elems = input.data.size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0].data

        o = -output[..., None] * output[:, None, :]
        diag_x, diag_y = np.diag_indices_from(o[0])
        o[:, diag_y, diag_x] = output * (1.0 - output)

        grad_o = o.sum(axis=1)

        grad_o = grad_o * grad_output.data
        grad_o = MLlib.Tensor(grad_o)

        return grad_o

    @staticmethod
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

    @staticmethod
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
        x_vector = X.reshape(X.shape[0], 1)
        x_matrix = np.tile(x_vector, X.shape[0])
        x_der = np.diag(X) - (x_matrix * np.transpose(x_matrix))
        return x_der


class Softsign():
    @staticmethod
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

    @staticmethod
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


class Relu(autograd.Function):

    __slots__ = ()

    @staticmethod
    def forward(ctx, input):
        if not (type(input).__name__ == 'Tensor'):
            raise RuntimeError("Expected a Tensor, got {}. Please use "
                               "Relu.activation() for non-Tensor data"
                               .format(type(input).__name__))

        requires_grad = input.requires_grad

        output = np.maximum(input.data, 0)

        output = MLlib.Tensor(output, requires_grad=requires_grad,
                              is_leaf=not requires_grad)

        if requires_grad:
            ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        o = ctx.saved_tensors[0]

        grad_o = np.greater(o.data, 0).astype(int) * grad_output.data
        grad_o = MLlib.Tensor(unbroadcast(grad_o, o.shape))

        return grad_o

    @staticmethod
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

    @staticmethod
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
    @staticmethod
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

    @staticmethod
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
        dx = np.greater(X, 0).astype(float)
        dx[X < 0] = -alpha
        return dx


class Elu():
    @staticmethod
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
        if (alpha <= 0):
            raise AssertionError
        return np.maximum(0, X) + np.minimum(0, alpha * (np.exp(X) - 1))


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
    @staticmethod
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

    @staticmethod
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
