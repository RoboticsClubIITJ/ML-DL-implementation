import MLlib
import MLlib.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.bias = MLlib.Tensor(0.,
                                 requires_grad=True,
                                 is_parameter=True)

        self.weight = MLlib.Tensor.randn(out_features,
                                         in_features,
                                         is_parameter=True)

        self.weight.requires_grad = True

    def forward(self, x):
        x = (x @ self.weight.T) + self.bias
        return x
