import pytest
import numpy

from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
                           assert_allclose, assert_almost_equal,
                           assert_array_equal)

from jadapy import generalized_schur
from jadapy import Target

REAL_DTYPES = [numpy.float32, numpy.float64]
COMPLEX_DTYPES = [numpy.complex64, numpy.complex128]
DTYPES = REAL_DTYPES + COMPLEX_DTYPES

def generate_random_dtype_array(shape, dtype):
    if dtype in COMPLEX_DTYPES:
        return (numpy.random.rand(*shape)
                + numpy.random.rand(*shape)*1.0j).astype(dtype)
    return numpy.random.rand(*shape).astype(dtype)

@pytest.mark.parametrize('dtype', DTYPES)
def test_generalized_schur(dtype):
    n = 20
    a = generate_random_dtype_array([n, n], dtype=dtype)
    b = generate_random_dtype_array([n, n], dtype=dtype)
    s, t, q, z = generalized_schur.generalized_schur(a, b)

    atol = numpy.finfo(dtype).eps*100
    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', COMPLEX_DTYPES)
def test_generalized_schur_sort(dtype):
    n = 20
    a = generate_random_dtype_array([n, n], dtype=dtype)
    b = generate_random_dtype_array([n, n], dtype=dtype)
    s, t, q, z = generalized_schur.generalized_schur(a, b)

    idx = range(n)
    idx1 = min(idx, key=lambda i: abs(s[i, i] / t[i, i]))
    assert idx1 != 0

    d1 = s[0, 0] / t[0, 0]
    d2 = s[idx1, idx1] / t[idx1, idx1]

    s, t, q, z = generalized_schur.generalized_schur_sort(s, t, q, z, Target.SmallestMagnitude)
    assert s[0, 0] / t[0, 0] != d1
    assert_almost_equal(s[0, 0] / t[0, 0], d2)

    atol = numpy.finfo(dtype).eps*100
    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)
