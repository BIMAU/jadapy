import numpy
import math

def dot(x, y):
    try:
        return x.T.conj() @ y
    except AttributeError:
        return x.dot(y)

def sqrt(x):
    if isinstance(x, numpy.ndarray):
        assert x.shape[0] == 1 and (len(x.shape) < 2 or x.shape[1] < 2)
        return math.sqrt(x.item())

    return math.sqrt(x)

def norm(x, M=None):
    def applyM(x):
        if M is not None:
            return M @ x
        return x

    if len(x.shape) < 2 or x.shape[1] < 2:
        return sqrt(dot(x, applyM(x)).real)

    s = 0
    for i in range(x.shape[1]):
        s += dot(x[:, i], applyM(x[:, i])).real
    return sqrt(s)

def eps(x):
    return numpy.finfo(x.dtype).eps * 10
