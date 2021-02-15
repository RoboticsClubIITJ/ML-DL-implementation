
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
        #TODO: linking node to the comp graph.
        """

        backward_function = BackwardFunction(cls)

        output_tensor = cls.forward(backward_function.ctx, *args, **kwargs)

        # TODO:  add parental nodes
        #       store current node in output

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

        return self._forward_cls.backward(self.ctx, args)
        # this ctx was already supplied to the forward function in .apply()
