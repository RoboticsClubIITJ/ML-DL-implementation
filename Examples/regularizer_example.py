from MLlib import Tensor
from MLlib.regularizer import LinearRegWith_Regularization
from MLlib.regularizer import L1_Regularizer
from MLlib.optim import SGDWithMomentum
from MLlib.utils.misc_utils import printmat
import numpy as np

np.random.seed(5322)

x = Tensor.randn(10, 8)       # (batch_size, features)

y = Tensor.randn(10, 1)

reg = LinearRegWith_Regularization(8, L1_Regularizer,
                                   optimizer=SGDWithMomentum,
                                   Lambda=7)

# Regularizer,optimizer and Lambda as per user's choice

printmat("Total Loss", reg.fit(x, y, 800))
