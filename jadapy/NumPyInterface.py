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
            out[:, i], info, iterations = gmres(op, x[:, i], maxit, tol)
            if info < 0:
                raise Exception('GMRES returned ' + str(info))
            elif info > 0 and maxit > 10:
                warnings.warn('GMRES did not converge in ' + str(iterations) + ' iterations')
        return out


def gmres(A, b, maxit, tol, restart=None, prec=None):
    iterations = 0

    def callback(_r):
        nonlocal iterations
        iterations += 1

    if restart is None:
        restart = min(maxit, 100)

    maxiter = (maxit - 1) // restart + 1

    try:
        y, info = linalg.gmres(A, b, restart=restart, maxiter=maxiter,
                               rtol=tol, atol=0, M=prec,
                               callback=callback, callback_type='pr_norm')
    except TypeError:
        # Compatibility with SciPy <= 1.11
        y, info = linalg.gmres(A, b, restart=restart, maxiter=maxiter,
                               tol=tol, atol=0, M=prec,
                               callback=callback, callback_type='pr_norm')

    return y, info, iterations
