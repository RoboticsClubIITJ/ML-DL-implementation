from MLlib import tensor


def backward(grad_fn, grad_of_output):
    """
    Recursive DFS that traverses the computation graph backward.

    PARAMETERS
    ==========
    grad_fn: node of the current Tensor.

    grad_of_output: gradient of final output with respect to the current
                    Tensor.
    """

    # obtain gradients to be passed:
    #   - the return of None object by grad_fn.apply() implies that the current
    #     Tensor is a leaf and has `grad_fn` as AccumulateGrad, this means
    #     it has no parent nodes and hence we don't need to pass gradients
    #     any further.
    #   - grad_fn.apply() might return a single Tensor object whcih, we know,
    #     corresponds to backward() method of some UNARY operation.
    out_grads = grad_fn.apply(grad_of_output)

    if out_grads is not None:
        out_grads = (out_grads,) if isinstance(out_grads, tensor.Tensor)\
            else tuple(out_grads)

        # pass them
        parent_nodes = grad_fn.next_functions

        for i in range(len(parent_nodes)):
            if (parent_nodes[i] is not None):
                backward(parent_nodes[i], out_grads[i])


class ContextManager:
    """
    Used to pass variables between function's .forward() and .backward()
    Passed as arguement ctx to functions

    1. To store tensors: simply use ctx.save_for_backward(tensor1, tensor2,...)
    2. To store any other data: save as new attribute
    """

    __slots__ = ('saved_tensors', '__dict__')

    def __init__(self):
        self.saved_tensors = []

    def save_for_backward(self, *args):
        for obj in args:
            if type(obj).__name__ != 'Tensor':
                raise Exception('Expected a Tensor but got {}. \n \
                    This method can be used to save Tensors only. For saving \
                    other types, just save directly as a new \
                    attribute'.format(type(obj)))
            self.saved_tensors.append(obj.copy())


class Function:
    """
    Superclass that links all the nodes to computational graph.
    All functions which have to do with computational graph must inherit from \
        this class
    """

    __slots__ = ()

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *args):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        """
        Runs .forward() of subclass and links node to the comp graph.
        """

        backward_function = BackwardFunction(cls)

        output_tensor = cls.forward(backward_function.ctx, *args, **kwargs)

        # grad_fn contains the node object

        # adding parental nodes to the comp graph
        for obj in args:
            if type(obj).__name__ == 'Tensor':      # parent tensor
                if obj.requires_grad and obj.is_leaf:
                    # if True,gradients must be stored in nodes during backprop
                    if obj.grad_fn is None:
                        obj.grad_fn = AccumulateGrad(obj)
                backward_function.next_functions.append(obj.grad_fn)

        # store the node on current tensor inside grad_fn
        output_tensor.grad_fn = backward_function

        return output_tensor


class BackwardFunction:
    """
    Represents an intermediate node in the comp graph
    """

    __slots__ = ('ctx', '_forward_cls', 'next_functions', 'function_name')

    def __init__(self, cls):
        self.ctx = ContextManager()
        self._forward_cls = cls

        # nodes of parents, should be populated by Function.apply()
        self.next_functions = []

        # name of the function, for debugging purposes
        self.function_name = cls.__name__

    def apply(self, *args, **kwagrs):

        return self._forward_cls.backward(self.ctx, *args)
        # this ctx was already supplied to the forward function in .apply()


class AccumulateGrad:
    """
    Represents a node where gradient must be accumulated.
    """

    __slots__ = ('variable')

    def __init__(self, tensor):
        self.variable = tensor

    def apply(self, arg):
        """
        Accumulates the provided gradient.
        """
        # if no grad stored yet, initialize. otherwise +=
        if self.variable.grad is None:
            self.variable.grad = tensor.Tensor(arg.data)
        else:
            self.variable.grad.data += arg.data

        # Some tests to make sure valid grads were stored.
        shape = self.variable.shape
        grad_shape = self.variable.grad.shape
        assert shape == grad_shape, (shape, grad_shape)
