from MLlib import Tensor


class Module:
    """
    Base class (superclass) for all components of an NN.

    Layer classes and even full Model classes should inherit from this Module.
    Inheritance gives the subclass all the functions/variables below

    NOTE: We shouldn't ever need to instantiate Module() directly.
    """

    __slots__ = ('_parameters', '_submodules', 'is_train')

    def __init__(self):
        self._submodules = {}  # Submodules of the class
        self._parameters = {}  # Trainable params in module and its submodules

        self.is_train = True  # Is the module being trained?

    def train(self):
        """
        Activates training mode for the Module and all of its sub-modules.
        """
        self.is_train = True
        for module in self._submodules.values():
            module.train()

    def eval(self):
        """
        Activates evaluation mode for the Module and all of its sub-modules.
        """
        self.is_train = False
        for module in self._submodules.values():
            module.eval()

    def forward(self, *args, **kwargs):
        """
        Forward pass of the module
        """
        raise NotImplementedError("Subclasses of Module must implement\
                                   their own forward method")

    def is_parameter(self, obj):
        """
        Checks if input object is a Tensor of trainable param(s).
        """
        return isinstance(obj, Tensor) and obj.is_parameter

    def parameters(self):
        """
        Returns an interator over stored params.
        Includes submodules' params too
        """
        self._ensure_is_initialized()
        for name, parameter in self._parameters.items():
            yield parameter
        for name, module in self._submodules.items():
            for parameter in module.parameters():
                yield parameter

    def register_parameter(self, name, value):
        """
        Stores the parameters
        """
        self._ensure_is_initialized()
        self._parameters[name] = value

    def register_module(self, name, value):
        """
        Stores module and its params
        """
        self._ensure_is_initialized()
        self._submodules[name] = value

    def apply(self, function):
        """
        Applies `function` recursively to each submodule and self

        PARAMETERS
        ==========
        function: Callable
                  The function to be applied

        RETURNS
        =======
        Module: self
        """

        for module in self._submodules.values():
            module.apply(function)
        function(self)
        return self

    def __setattr__(self, name, value):
        """
        Method that stores params or modules that we provide.

        `__setattr__(...)` is called by python whenever we try to set a new
        attribute in a class.
        """
        if self.is_parameter(value):
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            self.register_module(name, value)

        object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        """
        Runs self.forward(*args, **kwargs) of the module.
        """
        return self.forward(*args, **kwargs)

    def _ensure_is_initialized(self):
        """
        Ensures that subclass's __init__() method ran super().__init__()
        """
        if self.__slots__[1] is None:
            raise Exception("Module not intialized yet. "
                            "Did you forget to call super().__init__()?")
