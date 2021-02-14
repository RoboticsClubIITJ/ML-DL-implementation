import numpy as np
import MLlib.functional as F


class Tensor:
    """
    Tensor object which acts as a wrapper around a NumPy array.
    """

    def __init__(self, data, requires_grad=False, is_leaf=True,
                 is_parameter=False, dtype='float32'):

        """
        PARAMETERS
        ==========

        data: list, tuple or np.array
                The actual data of the tensor

        requires_grad: boolean
                        If true, accumulate gradient in `.grad`

        is_leaf: boolean
                    If true, this is a leaf tensor.

        is_parameter: boolean
                    If true, data contains trainable params.


        RETURNS
        =======

        None
        """

        if not (isinstance(data, np.ndarray)):
            data = np.array(data, dtype)
        self.data = data
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.grad_fn = None  # Set during forward pass
        self.grad = None
        self.is_parameter = is_parameter

    def __getitem__(self, *args):
        # return a numpy array to make life simpler
        return self.data.__getitem__(*args)

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

    def numpy(self):
        """
        Returns the data stored in Tensor as np.array
        """
        return self.data

    # ----------------------------------------------------------------
    # Tensor creation methods, can be used WITHOUT creating a tensor
    # ----------------------------------------------------------------
    @staticmethod
    def ones(*shape, **kwargs):
        """
        Similar to np.ones(...)

        PARAMETERS
        ==========
        shape: int or tuple of ints
                Used for defining shape of the Tensor

        **kwargs

        RETURNS
        =======
        a Tensor filled with ones of given `shape`
        """
        return Tensor(np.ones(shape), **kwargs)

    @staticmethod
    def zeros(*shape, **kwargs):
        """
        Similar to np.zeros(...)

        PARAMETERS
        ==========
        shape: int or tuple of ints
                Used for defining shape of the Tensor

        **kwargs

        RETURNS
        =======
        a Tensor filled with zeros of given `shape`
        """
        return Tensor(np.zeros(shape), **kwargs)

    @staticmethod
    def randn(*shape, **kwargs):
        """
        Similar to np.random.randn(...)
        Generates a Tensor filled with ones.

        PARAMETERS
        ==========
        shape: int or tuple of ints
                Used for defining shape of the Tensor

        **kwargs

        RETURNS
        =======
        a Tensor of `shape` sampled from Gaussian Distribution with
        mu=0 and sigma=1
        """
        return Tensor(np.random.randn(shape), **kwargs)

    @staticmethod
    def arange(*interval, **kwargs):
        """
        Similar to np.arange(...)

        PARAMETERS
        ==========
        interval: int or tuple of ints
                Used to define interval for values inside Tensor

        **kwargs

        RETURNS
        =======
        a Tensor with values from `interval`
        """
        return Tensor(np.arange(*interval), **kwargs)

    @staticmethod
    def empty(*shape, **kwargs):
        """
        Similar to np.empty(...)
        Generates a Tensor with uninitialized data.

        PARAMETERS
        ==========
        shape: int or tuple of ints
                Used for defining shape of the Tensor

        **kwargs

        RETURNS
        =======
        a Tensor containing uninitialized with given shape and properties
        """
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

    def T(self):
        return F.Transpose.apply(self)

    def reshape(self, *shape):
        return F.Reshape.apply(self, shape)

    def __add__(self, other):
        # handles Tensor + other

        # simple as `return Tensor(self.data + other.data)` right?
        # NO. we need to link this to our computational graph too

        # Done to support operations with int and float data
        if type(other) == int:
            other = float(other)

        if type(other) == float:
            other = Tensor(other)

        return F.Add.apply(self, other)

    def __radd__(self, other):
        # handles other + Tensor
        return self.__add__(other)

    def __sub__(self, other):

        # Done to support operations with int and float data
        if type(other) == int:
            other = float(other)

        if type(other) == float:
            other = Tensor(other)

        return F.Sub.apply(self, other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __matmul__(self, other):
        return F.MatMul.apply(self, other)

    def __rmatmul__(self, other):
        return self.__matmul__(other)

    def __truediv__(self, other):

        # Done to support operations with int and float data
        if type(other) == int:
            other = float(other)

        if type(other) == float:
            other = Tensor(other)

        return F.Div.apply(self, other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __mul__(self, other):

        # Done to support operations with int and float data
        if type(other) == int:
            other = float(other)

        if type(other) == float:
            other = Tensor(other)

        return F.Mul.apply(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):

        # Done to support operations with int and float data
        if type(other) == int:
            other = float(other)

        if type(other) == float:
            other = Tensor(other)

        return F.Pow.apply(self, other)

    def __rpow__(self, other):

        # Done to support operations with int and float data
        if type(other) == int:
            other = float(other)

        if type(other) == float:
            other = Tensor(other)

        return F.Pow.apply(other, self)

    def dot(self, other):
        return F.Dot.apply(self, other)

    def sum(self, axis=None, keepdims=False):
        return F.Sum.apply(self, axis, keepdims)
