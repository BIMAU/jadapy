import numpy

class NumPyInterface:

    def __init__(self, n, dtype=None):
        self.n = n
        self.dtype = dtype

    def vector(self, k=None):
        if k:
            return numpy.zeros((self.n, k), self.dtype)
        else:
            return numpy.zeros(self.n, self.dtype)
