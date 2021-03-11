from MLlib import tensor


def backward(grad_fn, grad_of_output):
    """
    Recursive DFS that traverses the computation graph backward passing and
    accumulating the gradients along the way.

    PARAMETERS
    ==========
    grad_fn: node of the current Tensor.

    grad_of_output: gradient of final output with respect to the current
                    Tensor.

    The `grad_fn` might be an instance of the `MLlib.autograd.BackwardFunction`
    class or that of the `MLlib.autograd.AccumulateGrad` class (why? see
    MLlib.autograd.Function class) and we have defined the method `.apply()`
    for both of them, so we have to just keep calling `grad_fn.apply(...)`
    recursively with correct parameters until we pass the gradients through/to
    all the nodes of the computation graph which have `requires_grad=True`.
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
    Superclass that links all the nodes to the computation graph.
    All functions which have to do with computational graph must inherit from \
    this class
    """

    __slots__ = ()

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        This method performs the required operation(s) on the Tensor(s) and is
        called by `.apply()` method.
        """
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *args):
        """
        This method takes the gradient of the root of computation
        graph with respect to the output of the operation as
        input and returns the gradient of root of the computation
        graph with repect to the operands of the operation.
        """
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        """
        Runs .forward() of subclass and links node to the comp graph.
        The node is stored on the Tensor as `grad_fn` attribute.


        FLOW:

        -Create an empty node by creating an instance of the `BackwardFunction`
        class and store it into `backward_function` (temporarily variable).

        -Obtain the output Tensor by calling `.forward(...)` method which will
        perform the required operation(s) on the Tensor(s)

        -We store the `grad_fn` (node object) of every 'Parent Tensor' which
        is passed to this function to perform the desired operation(s) into the
        node obeject of the output Tensor: we store the parental node objects
        into `backward_function.next_functions` variable. The parent Tensors
        might not already have a node stored into them.
        If such parent Tensors are leaf Tensors (have `is_leaf=True`) and
        require gradient to be passed to them (have `requires_grad=True`),
        we create a node and store it into the Tensor: this node is an
        instance of `MLlib.autograd.AccumulateGrad` class which is used to
        accumulate the gradients. Else None (their `grad_fn` by default) gets
        stored.

        -We store the `backward_function` as the output Tensor's node
        (grad_fn)
        """

        backward_function = BackwardFunction(cls)

        output_tensor = cls.forward(backward_function.ctx, *args, **kwargs)

        # do we really need to backpropagate through the output tensor?
        # the requires_grad flag is set inside .forward() method above

        if output_tensor.requires_grad:

            # adding parental nodes to the comp graph

            for obj in args:
                if type(obj).__name__ == 'Tensor':          # parent "Tensor"
                    if (obj.requires_grad and obj.is_leaf
                            and (obj.grad_fn is None)):
                        # if True, grads must be stored in nodes for backprop

                        obj.grad_fn = AccumulateGrad(obj)

                    backward_function.next_functions.append(obj.grad_fn)

            # store the node on current tensor inside grad_fn
            # grad_fn contains the node object
            output_tensor.grad_fn = backward_function

        else:
            # if output_tensor had requires_grad=True then the .forward() must
            # have had used the `ctx` object passed to it. But if not, it
            # won't have used it because the .forward() method is built like
            # that. So, we should delete the backward_function variable
            # (to save memory) which would have served the purpose of
            # output_tensor.grad_fn if it were required to be backpropagated.

            del backward_function

        return output_tensor

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)


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
