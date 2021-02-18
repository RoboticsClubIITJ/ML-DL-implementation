import numpy as np
import MLlib
import MLlib.autograd as autograd
from MLlib.utils.misc_utils import unbroadcast


class Transpose(autograd.Function):

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

        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b, output = ctx.saved_tensors

        grad_a = b.data * np.power(a.data, b.data-1) * grad_output.data
        grad_a = MLlib.Tensor(unbroadcast(grad_a, a.shape))

        grad_b = output.data * np.log(a.data) * grad_output.data
        grad_b = MLlib.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b


class Dot(autograd.Function):

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

        grad_a = grad_output.dot(b.data.T)
        grad_b = (b.data.T).dot(grad_output)

        grad_a = MLlib.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = MLlib.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b


class Sum(autograd.Function):

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
