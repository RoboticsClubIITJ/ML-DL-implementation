import numpy as np


class Tensor:
    """
    Tensor object which acts as a wrapper around a NumPy array.

    Args:
        data: the actual data of the tensor
        requires_grad (boolean): If true, accumulate gradient in `.grad`
        is_leaf (boolean): If true, this is a leaf tensor; see writeup.
        is_parameter (boolean): If true, data contains trainable params
    """

    def __init__(self, data, requires_grad=False, is_leaf=True,
                 is_parameter=False):

        if not (isinstance(data, np.ndarray)):
            data = np.array(data)
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.grad_fn = None  # Set during forward pass
        self.grad = None
        self.is_parameter = is_parameter

    # ----------------------
    # for printing tensors
    # ----------------------
    def __str__(self):
        sgf = self.grad_fn
        return "{}{}".format(
            str(self.data),
            ", grad_fn={}".format(
                self.grad_fn.__class__.__name__) if sgf is not None else ""
        )

    def __repr__(self):
        return self.__str__()

    # ----------------------
    # Tensor operations
    # ----------------------
    @property
    def shape(self):
        """
        Returns the shape of  data array in a tuple
        """
        return self.data.shape

    def copy(self, **kwargs):
        """
        Returns a copy of data associated with the tensor as a new tensor.
        Parameters of tensors like is_leaf and grad_fn can be associated
            with this copy by using appropriate **kwargs.
        """
        return Tensor(self.data, **kwargs)

    # ----------------------------------------------------------------
    # Tensor creation methods, can be used WITHOUT creating a tensor
    # ----------------------------------------------------------------
    @staticmethod
    def ones(*shape, **kwargs):
        return Tensor(np.ones(shape), **kwargs)

    @staticmethod
    def zeros(*shape, **kwargs):
        return Tensor(np.zeros(shape), **kwargs)

    @staticmethod
    def randn(*shape, **kwargs):
        return Tensor(np.random.randn(shape), **kwargs)

    @staticmethod
    def arange(*interval, **kwargs):
        return Tensor(np.arange(*interval), **kwargs)

    @staticmethod
    def empty(*shape, **kwargs):
        return Tensor(np.empty(shape), **kwargs)

    # ---------------------------------
    # Autograd backward initialization
    # ----------------------------------
    def backward(self):
        # TODO
        pass

    # --------------------------------------------------------------
    # Tensor operations that get reflected on the computation graph
    # --------------------------------------------------------------

    def __add__(self, other):
        # simple as `return self.data + other.data` right?
        # NO. we need to link this to our computational graph too
        pass

    def __sub__(self, other):
        pass

    def __matmul__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def __mul__(self, other):
        pass
