import numpy as np
import MLlib.functional as F
import MLlib.autograd as autograd


class Tensor:
    """
    Tensor object which acts as a wrapper around a NumPy array.
    """

    __slots__ = ('data', 'requires_grad', 'is_leaf', 'grad_fn', '_grad',
                 'is_parameter')

    def __init__(self, data, requires_grad=False, is_leaf=True,
                 is_parameter=False, dtype=None):

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

        A Tensor (MLlib.Tensor) object
        """

        if not (isinstance(data, np.ndarray)):
            data = np.array(data, dtype)
        self.data = data
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.grad_fn = None  # Set during forward pass
        self._grad = None
        self.is_parameter = is_parameter

    def __getitem__(self, *args):
        """
        Allows tensor to be accessed via indices. Indexing works just like
        numpy arrays. This operation is not connected to the computation graph.

        RETURNS
        -------
        a numpy array with desired elements.


        Example:
        >>> a = MLlib.Tensor([2., 4., 6.], requires_grad=True)
        >>> a[2]
        6.
        """
        return self.data.__getitem__(*args)

    def get_grad(self):
        return self._grad

    def del_grad(self):
        del self._grad

    def set_grad(self, val):
        if val is None or type(val).__name__ == 'Tensor':
            self._grad = val
        else:
            raise Exception("Expected the gradient to be NoneType object or a Tensor\
                            (got {})".format(type(val).__name__))

    grad = property(get_grad, set_grad, del_grad, 'The gradient of the tensor')
    # why do we need the grad as property?
    # because the user may set the grad property to 0 and we want our gradients
    # to be nothing else but Tensors. So, having grad as property helps us to
    # define a custom `setter` function for the _grad attribute that ensures
    # that only Tensors are stored in _grad

    # ----------------------
    # for printing tensors
    # ----------------------
    def __str__(self):
        """
        This function is called whenever we call print() on an instance of
        Tensor.
        """
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
    def ones(shape, **kwargs):
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
        return Tensor(np.random.randn(*shape), **kwargs)

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
        a Tensor containing uninitialized data with given shape and properties
        """
        return Tensor(np.empty(shape), **kwargs)

    # ---------------------------------
    # Autograd backward initialization
    # ----------------------------------
    def backward(self, grad_of_output=None):
        """
        This method is called to initiate the backward pass on the computation
        graph.

        This method initiates MLlib.autograd.backward() method which
        accumulates the gradient(s) to the Leaf Tensors (`is_leaf=True`) with
        `requires_grad=True` with respect to the root of the computation graph.

        PARAMETERS
        ==========
        grad_of_output: None or MLlib.Tensor
                        The gradient of root node with respect to the current
                        tensor('s node)

        \t\t When nothing is passed, the gradients are calculated with respect\
        \t to the tensor through which this method is being called. To put it
        \t simply, if nothing is passed this tensor is assumed to be the root
        \t of the computation graph.


        \t\t If the gradient is passed, it must be of the same shape as that\
        \t of the tensor through which this method is being called. If the
        \t gradient is passed, this tensor is assumed to be an intermediate
        \t node in the computation graph and the gradient is assumed to be\
            with respect to some root node.

        RETURNS
        =======
        `None`
        """
        if grad_of_output is None:
            grad_of_output = Tensor.ones(self.shape)

        if grad_of_output.shape != self.shape:
            # this block will be executed only when graient is supplied
            raise Exception('The shape of gradient and variable must match')

        if self.grad_fn is None:
            raise Exception('backward should not be called on tensors '
                            + 'without grad_fn')

        return autograd.backward(self.grad_fn, grad_of_output)

    # --------------------------------------------------------------
    # Tensor operations that get reflected on the computation graph
    # --------------------------------------------------------------
    @property
    def T(self):
        """
        Transposes a 2-D Tensor.

        Usage:
        >>> a = MLlib.Tensor([[2., 4.], [8., 6,]])
        >>> a.T
        [[2. 8.]
        [4. 6.]], grad_fn=BackwardFunction
        """
        return F.Transpose.apply(self)

    def reshape(self, *shape):
        """
        Reshapes a tensor to desired shape.

        Usage:
        >>> a = MLlib.Tensor.randn(5, 6, 8)
        >>> b = a.reshape(30, 8)
        """
        return F.Reshape.apply(self, shape)

    def __add__(self, other):
        """
        This function is called internally by python whenever addition
        operation (`+`) is performed and the left operand is an instance of
        MLlib.Tensor.

        If the right operand (`other` arguement passed to this function) is
        int or float we should convert that to a Tensor because our
        computation graph is built using Tensors.
        """
        if type(other) == int:
            other = float(other)

        if type(other) == float:
            other = Tensor(other)

        return F.Add.apply(self, other)

    def __radd__(self, other):
        """
        This function is called internally by python whenever addition
        operation (`+`) is performed and the right operand is an instance of
        MLlib.Tensor.

        Since `other + Tensor` should be equivalent to `Tensor + other`
        if the operation is valid, so we call __add__ method for the Tensor.
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        This function is called internally by python whenever subtraction
        operation (`-`) is performed and the left operand is an instance of
        MLlib.Tensor.

        If the right operand (`other` arguement passed to this function) is
        int or float we should convert that to a Tensor because our
        computation graph is built using Tensors.
        """
        if type(other) == int:
            other = float(other)

        if type(other) == float:
            other = Tensor(other)

        return F.Sub.apply(self, other)

    def __rsub__(self, other):
        """
        This function is called internally by python whenever subtraction
        operation (`-`) is performed and the right operand is an instance of
        MLlib.Tensor.

        If the left operand (`other` arguement passed to this function) is
        int or float we should convert that to a Tensor because our
        computation graph is built using Tensors.
        """
        if type(other) == int:
            other = float(other)

        if type(other) == float:
            other = Tensor(other)

        return F.Sub.apply(other, self)

    def __neg__(self):
        """
        This function is called internally by python whenever we perform
        `-a` operation where the variable a is an instance of MLlib.Tensor
        class
        """
        return (-1)*self

    def __matmul__(self, other):
        """
        This function is called internally by python whenever
        'matrix multiplication' (`@`) is performed and the left operand is an
        instance of MLlib.Tensor class.

        NOTE: We should only perform this operation on Tensors
        (MLlib.Tensor's instances)

        The __matmul__ operation is denoted by '@'
        >>> x @ y
        """
        return F.MatMul.apply(self, other)

    def __truediv__(self, other):
        """
        This function is called internally by python whenever division
        operation (`/`) is performed and the left operand is an instance of
        MLlib.Tensor.

        If the right operand (`other` arguement passed to this function) is
        int or float we should convert that to a Tensor because our
        computation graph is built using Tensors.
        """
        if type(other) == int:
            other = float(other)

        if type(other) == float:
            other = Tensor(other)

        return F.Div.apply(self, other)

    def __rtruediv__(self, other):
        """
        This function is called internally by python whenever division
        operation (`/`) is performed and the right operand is an instance of
        MLlib.Tensor.

        If the left operand (`other` arguement passed to this function) is
        int or float we should convert that to a Tensor because our
        computation graph is built using Tensors.
        """
        if type(other) == int:
            other = float(other)

        if type(other) == float:
            other = Tensor(other)

        return F.Div.apply(other, self)

    def __mul__(self, other):
        """
        This function is called internally by python whenever multiplication
        operation (`*`) is performed and the left operand is an instance of
        MLlib.Tensor.

        If the right operand (`other` arguement passed to this function) is
        int or float we should convert that to a Tensor because our
        computation graph is built using Tensors.
        """
        if type(other) == int:
            other = float(other)

        if type(other) == float:
            other = Tensor(other)

        return F.Mul.apply(self, other)

    def __rmul__(self, other):
        """
        This function is called internally by python whenever multiplication
        operation (`*`) is performed and the right operand is an instance of
        MLlib.Tensor.

        Since `other * Tensor` should be equivalent to `Tensor * other`
        if the operation is valid, so we call __mul__ method for the Tensor.
        """
        return self.__mul__(other)

    def __pow__(self, other):
        """
        This function is called internally by python whenever power
        operation (`**`) is performed and the left operand is an instance of
        MLlib.Tensor.

        Internally, np.power(...) method is used.

        If the right operand (`other` arguement passed to this function) is
        int or float we should convert that to a Tensor because our
        computation graph is built using Tensors.

        >>> a**2 or a**b
        """
        if type(other) == int:
            other = float(other)

        if type(other) == float:
            other = Tensor(other)

        return F.Pow.apply(self, other)

    def __rpow__(self, other):
        """
        This function is called internally by python whenever power
        operation (`**`) is performed and the right operand is an instance of
        MLlib.Tensor.

        Internally, np.power(...) method is used.

        If the left operand (`other` arguement passed to this function) is
        int or float we should convert that to a Tensor because our
        computation graph is built using Tensors.

        >>> 2**b or a**b
        """
        if type(other) == int:
            other = float(other)

        if type(other) == float:
            other = Tensor(other)

        return F.Pow.apply(other, self)

    def dot(self, other):
        """
        Vector dot product.

        PARAMETERS
        ==========
        other: MLlib.Tensor
               The Tensor with which the dot product is to be computed.


        RETURNS
        =======
        MLlib.Tensor which is a dot product of given input.

        NOTE: Should be used only for vectors (Tensors of shape `(n,)`).

        For matrices and n-dimensional tensors, usage of `@` (matmul operation)
        is recommended.
        """
        return F.Dot.apply(self, other)

    def sum(self, axis=None, keepdims=False):
        """
        Computes the sum of elements of the Tensor.

        PARAMETERS
        ==========
        axis: int
              index of axis of Tensor along which sum of elements is to be
              computed.

        keepdims: boolean
                  if True, the shape of Tensor is retained.

        RETURNS
        =======
        A Tensor with the sum of elements along the given axis having shape
        governed by the `keepdims` arguement.
        """
        return F.Sum.apply(self, axis, keepdims)

    def log(self):
        """
        Returns the element-wise log of the Tensor.
        """
        return F.Log.apply(self)
