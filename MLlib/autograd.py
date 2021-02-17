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

    # obtain gradients to be passed
    out_grads = grad_fn.apply(grad_of_output)

    if out_grads is not None:
        out_grads = [tnsr for tnsr in out_grads]

    # pass them
    parent_nodes = grad_fn.next_functions

    if len(parent_nodes) > 0:
        for i in range(len(parent_nodes)):
            if parent_nodes[i] is not None:
                # print('now calling ', parent_nodes[i]) for debugging
                backward(parent_nodes[i], out_grads[i])


class ContextManager:
    """
    Used to pass variables between function's .forward() and .backward()
    Passed as arguement ctx to functions

    1. To store tensors: simply use ctx.save_for_backward(tensor1, tensor2,...)
    2. To store any other data: save as new attribute
    """

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

        # store the none on current tensor inside grad_fn
        output_tensor.grad_fn = backward_function

        return output_tensor


class BackwardFunction:
    """
    Represents an intermediate node in the comp graph
    """

    def __init__(self, cls):
        self.ctx = ContextManager()
        self._forward_cls = cls

        # nodes of parents, should be populated by Function.apply()
        self.next_functions = []

        self.function_name = cls.__name__   # name of the function

    def apply(self, *args, **kwagrs):

        return self._forward_cls.backward(self.ctx, *args)
        # this ctx was already supplied to the forward function in .apply()


class AccumulateGrad:
    """
    Represents a node where gradient must be accumulated.
    """
    def __init__(self, tensor):
        self.variable = tensor

        self.next_functions = []  # nodes of current node's parents (empty)
        # exists just to be consistent in format
        #  with BackwardFunction

        self.function_name = 'AccumulateGrad'  # just for convenience

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
