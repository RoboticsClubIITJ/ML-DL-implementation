import numpy as np
import MLlib
import MLlib.autograd as autograd
from MLlib.utils.misc_utils import unbroadcast

"""
Contains the Functions which are called whenever an operation concerning
Tensors and computaion graph occurs.

All the functions are derived from the base class `MLlib.autograd.Function`
which defines a `.apply()` method which calls the `.forward()` method and
the `.backward()` method (not directly though) of these functions as and when
required.

`__slots__`: defined to reduce memory overhead. This should be an empty tuple
                unless a function explicitly requires a variable to be stored
                in the class itself. If such case arises:the function must
                define its own `__init__()` method and put the class variable's
                name in `__slots__`

`.forward(...)`: this method performs the required operation and is called by
                    `.apply()`. To find out more about `.apply()` method,
                    please checkout `autograd.py`

`.backward(...)`: this method takes the gradient of the root of computation
                    graph with respect to the output of the operation as
                    input and returns the gradient of root of the computation
                    graph with repect to the operands of the operation.

Why unbroadcast(...) is being used?
numpy broadcasts the input in order to perform different operations and the
gradients are returned in the shape used for performing the operation. So we
need a way to reshape those gradients back to the shape of the original
operand, and the `unbroadcast()` utility does that for us.

"""


class Transpose(autograd.Function):

    __slots__ = ()

    @staticmethod
    def forward(ctx, a):

        if not (type(a).__name__ == 'Tensor'):
            raise Exception("The arg must be Tensor, got \
                {} instead".format(type(a).__name__))

        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor, \
                got {}".format(a.shape))

        requires_grad = a.requires_grad

        b = MLlib.Tensor(a.data.T, requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return b

    @staticmethod
    def backward(ctx, grad_output):
        return MLlib.Tensor(grad_output.data.T)


class Reshape(autograd.Function):

    __slots__ = ()

    @staticmethod
    def forward(ctx, a, shape):

        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor, got\
                         {}".format(type(a).__name__))

        requires_grad = a.requires_grad

        if requires_grad:
            ctx.shape = a.shape

        c = MLlib.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        return MLlib.Tensor(grad_output.data.reshape(ctx.shape))


class Add(autograd.Function):

    __slots__ = ()

    @staticmethod
    def forward(ctx, a, b):

        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors, got \
                {}, {} instead".format(type(a).__name__, type(b).__name__))

        requires_grad = a.requires_grad or b.requires_grad

        if requires_grad:
            ctx.shape_a = a.shape
            ctx.shape_b = b.shape

        c = MLlib.Tensor(a.data + b.data, requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        shape_a, shape_b = ctx.shape_a, ctx.shape_b

        # dL/da = (dout/da)*dL/dout
        grad_a = np.ones(shape_a) * grad_output.data
        grad_b = np.ones(shape_b) * grad_output.data

        grad_a = MLlib.Tensor(unbroadcast(grad_a, shape_a))
        grad_b = MLlib.Tensor(unbroadcast(grad_b, shape_b))

        return grad_a, grad_b


class Sub(autograd.Function):

    __slots__ = ()

    @staticmethod
    def forward(ctx, a, b):

        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors, got \
                {}, {} instead".format(type(a).__name__, type(b).__name__))

        requires_grad = a.requires_grad or b.requires_grad

        if requires_grad:
            ctx.shape_a = a.shape
            ctx.shape_b = b.shape

        c = MLlib.Tensor(a.data - b.data, requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        shape_a, shape_b = ctx.shape_a, ctx.shape_b

        grad_a = np.ones(shape_a) * grad_output.data
        grad_b = np.ones(shape_b) * grad_output.data * (-1)

        grad_a = MLlib.Tensor(unbroadcast(grad_a, shape_a))
        grad_b = MLlib.Tensor(unbroadcast(grad_b, shape_b))

        return grad_a, grad_b


class Mul(autograd.Function):

    __slots__ = ()

    @staticmethod
    def forward(ctx, a, b):

        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors, got \
                {}, {} instead".format(type(a).__name__, type(b).__name__))

        requires_grad = a.requires_grad or b.requires_grad

        if requires_grad:
            ctx.save_for_backward(a, b)

        c = MLlib.Tensor(a.data * b.data, requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        grad_a = b.data * grad_output.data
        grad_b = a.data * grad_output.data

        grad_a = MLlib.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = MLlib.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b


class Div(autograd.Function):

    __slots__ = ()

    @staticmethod
    def forward(ctx, a, b):

        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors, got \
                {}, {} instead".format(type(a).__name__, type(b).__name__))

        requires_grad = a.requires_grad or b.requires_grad

        if requires_grad:
            ctx.save_for_backward(a, b)

        c = MLlib.Tensor(a.data / b.data, requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        grad_a = grad_output.data / b.data
        grad_b = (-1)*a.data * grad_output.data / (b.data**2)

        grad_a = MLlib.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = MLlib.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b


class MatMul(autograd.Function):

    __slots__ = ()

    @staticmethod
    def forward(ctx, a, b):

        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors, got \
                {}, {} instead".format(type(a).__name__, type(b).__name__))

        requires_grad = a.requires_grad or b.requires_grad

        if requires_grad:
            ctx.save_for_backward(a, b)

        c = MLlib.Tensor(np.matmul(a.data, b.data),
                         requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):

        grad_output = grad_output.data
        a, b = ctx.saved_tensors

        grad_a = (grad_output) @ (b.data.T)
        grad_b = (a.data.T) @ (grad_output)

        grad_a = MLlib.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = MLlib.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b


class Pow(autograd.Function):

    __slots__ = ()

    @staticmethod
    def forward(ctx, a, b):

        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors, got \
                {}, {} instead".format(type(a).__name__, type(b).__name__))

        requires_grad = a.requires_grad or b.requires_grad

        c = MLlib.Tensor(np.power(a.data, b.data), requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        if requires_grad:
            ctx.save_for_backward(a, b, c)
            ctx.a_req_grad = a.requires_grad
            ctx.b_req_grad = b.requires_grad

        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b, output = ctx.saved_tensors

        grad_a = grad_b = None

        if ctx.a_req_grad:
            grad_a = b.data * np.power(a.data, b.data-1) * grad_output.data
            grad_a = MLlib.Tensor(unbroadcast(grad_a, a.shape))

        if ctx.b_req_grad:
            grad_b = output.data * np.log(a.data) * grad_output.data
            grad_b = MLlib.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b


class Dot(autograd.Function):

    __slots__ = ()

    @staticmethod
    def forward(ctx, a, b):

        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors, got \
                {}, {} instead".format(type(a).__name__, type(b).__name__))

        requires_grad = a.requires_grad or b.requires_grad

        if requires_grad:
            ctx.save_for_backward(a, b)

        c = MLlib.Tensor(np.dot(a.data, b.data), requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.data
        a, b = ctx.saved_tensors

        if len(grad_output.shape) > 0:
            grad_a = (grad_output).dot(b.data.T)
            grad_b = (a.data.T).dot(grad_output)

            grad_a = MLlib.Tensor(unbroadcast(grad_a, a.shape))
            grad_b = MLlib.Tensor(unbroadcast(grad_b, b.shape))

        else:
            grad_a = (grad_output) * (b.data.T)
            grad_b = (a.data.T) * (grad_output)

            grad_a = MLlib.Tensor(unbroadcast(grad_a, a.shape))
            grad_b = MLlib.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b


class Sum(autograd.Function):

    __slots__ = ()

    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only sum of tensor is supported")

        requires_grad = a.requires_grad

        if requires_grad:
            ctx.axis = axis
            ctx.shape = a.shape

            if axis is not None:
                ctx.len = a.shape[axis]

            ctx.keepdims = keepdims

        c = MLlib.Tensor(a.data.sum(axis=axis, keepdims=keepdims),
                         requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        grad_out = grad_output.data

        if (ctx.axis is not None) and (not ctx.keepdims):
            grad_out = np.expand_dims(grad_output.data, axis=ctx.axis)
        else:
            grad_out = grad_output.data.copy()

        grad = np.ones(ctx.shape) * grad_out

        assert grad.shape == ctx.shape

        return MLlib.Tensor(grad)


class Log(autograd.Function):

    __slots__ = ()

    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor, got\
                            {}".format(type(a).__name__))

        ctx.save_for_backward(a)

        requires_grad = a.requires_grad

        c = MLlib.Tensor(np.log(a.data), requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]

        return MLlib.Tensor(grad_output.data / a.data)
