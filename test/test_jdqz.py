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
def test_jdqz_smallest_magnitude(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 150
    atol = tol * 10
    n = 20
    k = 5
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)

    alpha, beta = jdqz.jdqz(a, b, num=k, tol=tol)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    eigs = scipy.linalg.eig(a, b, right=False, left=False)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_largest_magnitude(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 150
    atol = tol * 10
    n = 20
    k = 5
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)

    alpha, beta = jdqz.jdqz(a, b, k, Target.LargestMagnitude, tol=tol)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: -abs(x)))

    eigs = scipy.linalg.eig(a, b, right=False, left=False)
    eigs = numpy.array(sorted(eigs, key=lambda x: -abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)


@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_smallest_real(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 150
    atol = tol * 10
    n = 20
    k = 5
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)

    alpha, beta = jdqz.jdqz(a, b, k, Target.SmallestRealPart, tol=tol)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: x.real))

    eigs = scipy.linalg.eig(a, b, right=False, left=False)
    eigs = numpy.array(sorted(eigs, key=lambda x: x.real))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_largest_real(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 150
    atol = tol * 10
    n = 20
    k = 5
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)

    alpha, beta = jdqz.jdqz(a, b, k, Target.LargestRealPart, tol=tol)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: -x.real))

    eigs = scipy.linalg.eig(a, b, right=False, left=False)
    eigs = numpy.array(sorted(eigs, key=lambda x: -x.real))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_smallest_imag(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 150
    atol = tol * 10
    n = 20
    k = 5
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)

    alpha, beta = jdqz.jdqz(a, b, k, Target.SmallestImaginaryPart, tol=tol)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: x.imag))

    eigs = scipy.linalg.eig(a, b, right=False, left=False)
    eigs = numpy.array(sorted(eigs, key=lambda x: x.imag))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_largest_imag(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 150
    atol = tol * 10
    n = 20
    k = 5
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)

    alpha, beta = jdqz.jdqz(a, b, k, Target.LargestImaginaryPart, tol=tol)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: -x.imag))

    eigs = scipy.linalg.eig(a, b, right=False, left=False)
    eigs = numpy.array(sorted(eigs, key=lambda x: -x.imag))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_target(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 150
    atol = tol * 10
    n = 20
    k = 5
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)

    target = Target.Target(complex(2, 2))
    alpha, beta = jdqz.jdqz(a, b, k, target, tol=tol)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x - target.target)))

    eigs = scipy.linalg.eig(a, b, right=False, left=False)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x - target.target)))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)


@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_prec(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 150
    atol = tol * 10
    n = 20
    k = 5
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)

    ctype = numpy.dtype(numpy.dtype(dtype).char.upper())
    inv = scipy.sparse.linalg.spilu(scipy.sparse.csc_matrix(a, dtype=ctype))

    def _prec(x):
        return inv.solve(x)

    alpha, beta = jdqz.jdqz(a, b, num=k, tol=tol, prec=_prec)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    eigs = scipy.linalg.eig(a, b, right=False, left=False)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)
