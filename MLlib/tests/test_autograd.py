from MLlib import Tensor
import torch
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


def gen_tT(*args):
    # generates torch.Tensor from np.arrays with requires_grad=True
    tnsrs = list()
    for na in args:
        t = torch.from_numpy(na)
        t.requires_grad = True
        tnsrs.append(t)
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
    ta, tb = gen_tT(a, b)

    mk = ma**mb
    mk.backward()

    tk = ta**tb
    tk.backward(torch.ones(tk.shape))

    if not_close(mb.grad.data, tb.grad.numpy()):
        raise AssertionError

    if not_close(ma.grad.data, ta.grad.numpy()):
        raise AssertionError


def test_Log():
    a = np.random.randn(5, 6)
    b = np.random.randn(5, 6)

    ma, mb = gen_mT(a, b)
    ta, tb = gen_tT(a, b)

    mo = (ma + mb).log()
    to = torch.log(ta + tb)

    mo.backward()
    to.backward(torch.ones(to.shape))

    if not_close(mb.grad.data, tb.grad.numpy()):
        raise AssertionError

    if not_close(ma.grad.data, ta.grad.numpy()):
        raise AssertionError


def test_MulSum():
    a = np.random.randn(5, 6, 8)
    b = np.random.randn(5, 6, 8)

    ma, mb = gen_mT(a, b)
    ta, tb = gen_tT(a, b)

    mo = (ma * mb).sum()
    to = (ta * tb).sum()

    mo.backward()
    to.backward()

    if not_close(mb.grad.data, tb.grad.numpy()):
        raise AssertionError

    if not_close(ma.grad.data, ta.grad.numpy()):
        raise AssertionError


def test_MatmulTranspose():
    a = np.random.randn(8, 6)
    b = np.random.randn(8, 6)

    ma, mb = gen_mT(a, b)
    ta, tb = gen_tT(a, b)

    mo = ma @ mb.T() - 8
    to = ta @ tb.T - 8

    mo.backward()
    to.backward(torch.ones(to.shape))

    if not_close(mb.grad.data, tb.grad.numpy()):
        raise AssertionError

    if not_close(ma.grad.data, ta.grad.numpy()):
        raise AssertionError


def test_DivSum():
    a = np.random.randn(4, 6, 8)
    b = np.random.randn(6, 8)

    ma, mb = gen_mT(a, b)
    ta, tb = gen_tT(a, b)

    mo = (ma / mb).sum()
    to = (ta / tb).sum()

    mo.backward()
    to.backward()

    if not_close(mb.grad.data, tb.grad.numpy()):
        raise AssertionError

    if not_close(ma.grad.data, ta.grad.numpy()):
        raise AssertionError


def test_ReshapeSub():
    a = np.random.randn(5, 6, 8)
    b = np.random.randn(5, 6, 8)

    ma, mb = gen_mT(a, b)
    ta, tb = gen_tT(a, b)

    ma_i, mb_i = ma.reshape(30, 8), mb.reshape(30, 8)
    ta_i, tb_i = ta.reshape(30, 8), tb.reshape(30, 8)

    mo = (ma_i * mb_i - ma_i).sum()
    to = (ta_i * tb_i - ta_i).sum()

    mo.backward()
    to.backward()

    if not_close(mb.grad.data, tb.grad.numpy()):
        raise AssertionError

    if not_close(ma.grad.data, ta.grad.numpy()):
        raise AssertionError


def test_Dot():
    a = np.random.randn(6)
    b = np.random.randn(6)

    ma, mb = gen_mT(a, b)
    ta, tb = gen_tT(a, b)

    mo = ma.dot(mb)
    to = ta.dot(tb)

    mo.backward()
    to.backward(torch.ones(to.shape))

    if not_close(mb.grad.data, tb.grad.numpy()):
        raise AssertionError

    if not_close(ma.grad.data, ta.grad.numpy()):
        raise AssertionError
