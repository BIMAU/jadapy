import pytest
import numpy
import scipy

from math import sqrt

from numpy.testing import assert_equal, assert_allclose

from jadapy import jdqz
from jadapy import Target

REAL_DTYPES = [numpy.float32, numpy.float64]
COMPLEX_DTYPES = [numpy.complex64, numpy.complex128]
DTYPES = REAL_DTYPES + COMPLEX_DTYPES

def generate_random_dtype_array(shape, dtype):
    if dtype in COMPLEX_DTYPES:
        return (numpy.random.rand(*shape)
                + numpy.random.rand(*shape) * 1.0j).astype(dtype)
    return numpy.random.rand(*shape).astype(dtype)

@pytest.mark.parametrize('dtype', COMPLEX_DTYPES)
def test_jdqz(dtype):
    atol = numpy.finfo(dtype).eps * 100
    n = 10
    k = 5
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)

    alpha, beta = jdqz.jdqz(a, b, num=k, tol=atol)

    eigs = scipy.linalg.eig(a, b, right=False, left=False)

    eigs = sorted(eigs, key=lambda x: abs(x))
    assert_allclose(alpha / beta, eigs[0:k], rtol=0, atol=atol)
