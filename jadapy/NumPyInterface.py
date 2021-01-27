import numpy

REAL_DTYPES = [numpy.float32, numpy.float64]
COMPLEX_DTYPES = [numpy.complex64, numpy.complex128]
DTYPES = REAL_DTYPES + COMPLEX_DTYPES

def generate_random_dtype_array(shape, dtype):
    if dtype in COMPLEX_DTYPES:
        return (numpy.random.rand(*shape)
                + numpy.random.rand(*shape) * 1.0j).astype(dtype)
    return numpy.random.rand(*shape).astype(dtype)

class NumPyInterface:

    def __init__(self, n, dtype=None):
        self.n = n
        self.dtype = dtype

    def vector(self, k=None):
        if k:
            return numpy.zeros((self.n, k), self.dtype)
        else:
            return numpy.zeros(self.n, self.dtype)

    def random(self):
        return generate_random_dtype_array([self.n], self.dtype)
