import numpy
import warnings

from scipy.sparse import linalg

REAL_DTYPES = [numpy.float32, numpy.float64]
COMPLEX_DTYPES = [numpy.complex64, numpy.complex128]
DTYPES = REAL_DTYPES + COMPLEX_DTYPES

def generate_random_dtype_array(shape, dtype):
    if dtype in COMPLEX_DTYPES:
        return (numpy.random.rand(*shape) + numpy.random.rand(*shape) * 1.0j).astype(dtype, copy=False)
    return numpy.random.rand(*shape).astype(dtype, copy=False)

class NumPyInterface:

    def __init__(self, n, dtype=None):
        self.n = n
        self.dtype = numpy.dtype(dtype)

    def vector(self, k=None):
        if k:
            return numpy.zeros((self.n, k), self.dtype)
        else:
            return numpy.zeros(self.n, self.dtype)

    def random(self):
        return generate_random_dtype_array([self.n], self.dtype)

    def solve(self, op, x, tol, maxit):
        if op.dtype.char != op.dtype.char.upper():
            # Real case
            if abs(op.alpha.real) < abs(op.alpha.imag):
                op.alpha = op.alpha.imag
            else:
                op.alpha = op.alpha.real
            op.beta = op.beta.real

        out = x.copy()
        for i in range(x.shape[1]):
            out[:, i], info = linalg.gmres(op, x[:, i], restart=100, maxiter=maxit, tol=tol, atol=0)
            if info < 0:
                raise Exception('GMRES returned ' + str(info))
            elif info > 0 and maxit > 1:
                warnings.warn('GMRES did not converge in ' + str(info) + ' iterations')
        return out
