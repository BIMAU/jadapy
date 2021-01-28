from math import sqrt

import numpy

def dot(x, y):
    try:
        return x.T.conj() @ y
    except AttributeError:
        return x.dot(y)

def norm(x):
    if len(x.shape) < 2 or x.shape[1] < 2:
        return sqrt(dot(x, x).real)

    s = 0
    for i in range(x.shape[1]):
        s += dot(x[:, i], x[:, i]).real
    return sqrt(s)

def eps(x):
    return numpy.finfo(x.dtype).eps * 10
