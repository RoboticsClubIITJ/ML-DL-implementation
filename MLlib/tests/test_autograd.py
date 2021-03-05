from MLlib.utils.misc_utils import unbroadcast
from MLlib import Tensor
import numpy as np

# -------------------
# UTILITY FUNCTIONS
# -------------------


def gen_mT(*args):
    # generates Tensor from np.arrays with requires_grad=True
    tnsrs = list()
    for na in args:
        tnsrs.append(Tensor(na, requires_grad=True))
    return tuple(tnsrs)


def not_close(a, b):
    return not (np.allclose(a, b, equal_nan=True))


# ------
# TESTS
# ------


def test_Power():
    a = np.abs(np.random.randn(5, 6))
    b = np.random.randn(5, 6)

    ma, mb = gen_mT(a, b)

    mk = ma**mb
    mk.backward()

    if not_close(mb.grad.data, (ma.data**mb.data)*np.log(ma.data)):
        raise AssertionError

    if not_close(ma.grad.data, mb.data*np.power(ma.data, mb.data - 1)):
        raise AssertionError


def test_Log():
    a = np.random.randn(5, 6)
    b = np.random.randn(5, 6)

    ma, mb = gen_mT(a, b)

    mo = (ma + mb).log()

    mo.backward()

    if not_close(mb.grad.data, 1/(ma.data + mb.data)):
        raise AssertionError

    if not_close(ma.grad.data, 1/(ma.data + mb.data)):
        raise AssertionError


def test_MulSum():
    a = np.random.randn(5, 6, 8)
    b = np.random.randn(5, 6, 8)

    ma, mb = gen_mT(a, b)

    mo = (ma * mb).sum()

    mo.backward()

    if not_close(mb.grad.data, ma.data):
        raise AssertionError

    if not_close(ma.grad.data, mb.data):
        raise AssertionError


def test_MatmulTranspose():
    a = np.random.randn(8, 6)
    b = np.random.randn(8, 6)

    ma, mb = gen_mT(a, b)

    mo = ma @ mb.T - 8

    mo.backward()

    if not_close(mb.grad.data, np.ones(mo.shape) @ ma.data):
        raise AssertionError

    if not_close(ma.grad.data, np.ones(mo.shape) @ mb.data):
        raise AssertionError


def test_DivSum():
    a = np.random.randn(4, 6, 8)
    b = np.random.randn(6, 8)

    ma, mb = gen_mT(a, b)

    mo = (ma / mb).sum()

    mo.backward()

    if not_close(mb.grad.data, unbroadcast(-a / (b**2), b.shape)):
        raise AssertionError

    if not_close(ma.grad.data, 1/(mb.data)):
        raise AssertionError


def test_ReshapeSub():
    a = np.random.randn(5, 6, 8)
    b = np.random.randn(5, 6, 8)

    ma, mb = gen_mT(a, b)

    ma_i, mb_i = ma.reshape(30, 8), mb.reshape(30, 8)

    mo = (ma_i * mb_i - ma_i).sum()

    mo.backward()

    if not_close(mb.grad.data, ma.data):
        raise AssertionError

    if not_close(ma.grad.data, mb.data - 1):
        raise AssertionError


def test_Dot():
    a = np.random.randn(6)
    b = np.random.randn(6)

    ma, mb = gen_mT(a, b)

    mo = ma.dot(mb)

    mo.backward()

    if not_close(mb.grad.data, ma.data.T):
        raise AssertionError

    if not_close(ma.grad.data, mb.data.T):
        raise AssertionError
