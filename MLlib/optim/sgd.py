import MLlib.optim as optim


class SGD(optim.Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    #TODO: Include momentum

    Usage:
    >>> optimizer = SGD(model.parameters(), lr=0.1)

    PARAMETERS
    ==========
    params: list or iterator
            Parameters to update

    lr: float
        Learning rate (eta/alpha)
    """
    def __init__(self, parameters, lr=0.001):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data = param.data - self.lr * param.grad.data
