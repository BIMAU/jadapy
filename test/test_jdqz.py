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

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz(dtype):
    numpy.random.seed(1234)
    atol = numpy.finfo(dtype).eps * 1000
    n = 20
    k = 5
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)

    alpha, beta = jdqz.jdqz(a, b, num=k, tol=atol)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    eigs = scipy.linalg.eig(a, b, right=False, left=False)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)
