from math import sqrt

import numpy

def dot(x, y):
    return x.T.conj() @ y

def norm(x):
    return numpy.linalg.norm(x)

def eps(x):
    return numpy.finfo(x.dtype).eps * 10
