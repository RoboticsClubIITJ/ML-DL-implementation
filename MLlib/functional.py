import numpy as np
import MLlib
import MLlib.autograd as autograd


class Transpose(autograd.Function):

    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor, \
                got {}".format(a.shape))

        requires_grad = a.requires_grad

        b = MLlib.Tensor(a.data.T, requires_grad=requires_grad,
                         is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        pass


class Add(autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        # a, b are tensors: must be checked in Tensor.__add__()

        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad

        c = MLlib.Tensor(a.data + b.data, requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        pass


class Sub(autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        # a, b are tensors: must be checked in Tensor.__sub__()

        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad

        c = MLlib.Tensor(a.data - b.data, requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        pass


class Mul(autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        # a, b are tensors: must be checked in Tensor.__mul__()

        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad

        c = MLlib.Tensor(a.data * b.data, requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        pass


class MatMul(autograd.Function):

    @staticmethod
    def forward(ctx, a, b):

        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad

        c = MLlib.Tensor(np.matmul(a, b), requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        pass


class Dot(autograd.Function):

    @staticmethod
    def forward(ctx, a, b):

        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad

        c = MLlib.Tensor(np.dot(a, b), requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        pass
