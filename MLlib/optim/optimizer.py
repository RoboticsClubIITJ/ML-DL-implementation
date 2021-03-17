
class Optimizer:
    """
    Base class for all optimizers.
    """

    __slots__ = ('params')

    def __init__(self, parameters):
        self.params = tuple(parameters)

    def step(self):
        """
        Updates the all the parameters. Should be called only after the
        backward pass.
        """
        raise NotImplementedError

    def zero_grad(self):
        """
        Sets the gradient of all parameters to none. Should be called after
        the parameters have been updated to stop the unwanted accumulation of
        the gradients in parameters.
        """
        for param in self.params:
            param.grad = None
