import MLlib.optim as optim


class SGD(optim.Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Usage:
    >>> optimizer = SGD(model.parameters(), lr=0.1)

    PARAMETERS
    ==========
    params: list or iterator
            Parameters to update

    lr: float
        Learning rate (eta/alpha)
    """

    __slots__ = ('lr')

    def __init__(self, parameters, lr=0.001):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data = param.data - self.lr * param.grad.data


class SGDWithMomentum(optim.Optimizer):
    """
    Stochastic Gradient Descent optimizer with Momentum.

    Usage:
    >>> optimizer = SGDWithMomentum(model.parameters(),
                                    lr=0.1,
                                    momentum=0.95)

    PARAMETERS
    ==========
    params: list or iterator
            Parameters to update

    lr: float
        Learning rate (eta/alpha)

    momentum: float
              The momentum (beta)
    """

    __slots__ = ('beta', 'lr', 'v')

    def __init__(self, parameters, lr=0.001, momentum=0.9):
        super().__init__(parameters)
        self.beta = momentum
        self.lr = lr

        # list(last velocity of params, initialized to zero)
        self.v = [0] * len(self.params)

    def step(self):
        for i in range(len(self.params)):
            self.v[i] = self.beta*self.v[i] + self.lr*self.params[i].grad.data
            self.params[i].data = self.params[i].data - self.v[i]
