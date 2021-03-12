import MLlib
import MLlib.nn as nn


class Linear(nn.Module):

    __slots__ = ('weights', 'bias', 'actvn_fn')

    def __init__(self,
                 in_features,
                 out_features,
                 activation_fn=None):

        super().__init__()
        self.bias = MLlib.Tensor.randn(out_features,
                                       requires_grad=True,
                                       is_parameter=True)

        self.weights = MLlib.Tensor.randn(out_features,
                                          in_features,
                                          requires_grad=True,
                                          is_parameter=True)

        self.actvn_fn = activation_fn() if activation_fn is not None else None

    def forward(self, x):
        x = (x @ self.weights.T) + self.bias
        if self.actvn_fn is not None:
            x = self.actvn_fn.apply(x)
        return x
