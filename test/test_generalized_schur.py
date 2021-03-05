import pytest
import numpy
import scipy

from numpy.testing import assert_allclose

from jadapy import generalized_schur
from jadapy import Target

REAL_DTYPES = [numpy.float32, numpy.float64]
COMPLEX_DTYPES = [numpy.complex64, numpy.complex128]
DTYPES = REAL_DTYPES + COMPLEX_DTYPES

def generate_random_dtype_array(shape, dtype):
    if dtype in COMPLEX_DTYPES:
        return (numpy.random.rand(*shape) + numpy.random.rand(*shape) * 1.0j).astype(dtype)
    return numpy.random.rand(*shape).astype(dtype)

@pytest.mark.parametrize('dtype', DTYPES)
def test_generalized_schur(dtype):
    n = 20
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)
    s, t, q, z = generalized_schur.generalized_schur(a, b)

    atol = numpy.finfo(dtype).eps * 100
    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)

def _get_ev(a, b, i):
    n = a.shape[0]
    if i > 0 and a[i, i - 1] != 0:
        return scipy.linalg.eig(a[i-1:i+1, i-1:i+1], b[i-1:i+1, i-1:i+1])[0][1]
    elif i < n - 1 and a[i + 1, i] != 0:
        return scipy.linalg.eig(a[i:i+2, i:i+2], b[i:i+2, i:i+2])[0][0]
    return a[i, i] / b[i, i]

@pytest.mark.parametrize('dtype', DTYPES)
def test_generalized_schur_sort_smallest_magnitude(dtype):
    atol = numpy.finfo(dtype).eps * 100
    n = 10
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)
    s, t, q, z = generalized_schur.generalized_schur(a, b)

    idx = min(range(n), key=lambda i: abs(_get_ev(s, t, i)))
    assert idx != 0

    wanted = _get_ev(s, t, idx)

    s, t, q, z = generalized_schur.generalized_schur_sort(s, t, q, z, Target.SmallestMagnitude)

    assert_allclose(_get_ev(s, t, 0).real, wanted.real, rtol=0, atol=atol)
    assert_allclose(abs(_get_ev(s, t, 0).imag), abs(wanted.imag), rtol=0, atol=atol)

    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_generalized_schur_sort_largest_magnitude(dtype):
    atol = numpy.finfo(dtype).eps * 100
    n = 10
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)
    s, t, q, z = generalized_schur.generalized_schur(a, b)

    idx = max(range(n), key=lambda i: abs(_get_ev(s, t, i)))

    wanted = _get_ev(s, t, idx)

    s, t, q, z = generalized_schur.generalized_schur_sort(s, t, q, z, Target.LargestMagnitude)

    assert_allclose(_get_ev(s, t, 0).real, wanted.real, rtol=0, atol=atol)
    assert_allclose(abs(_get_ev(s, t, 0).imag), abs(wanted.imag), rtol=0, atol=atol)

    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_generalized_schur_sort_smallest_real(dtype):
    atol = numpy.finfo(dtype).eps * 100
    n = 10
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)
    s, t, q, z = generalized_schur.generalized_schur(a, b)

    idx = min(range(n), key=lambda i: _get_ev(s, t, i).real)

    wanted = _get_ev(s, t, idx)

    s, t, q, z = generalized_schur.generalized_schur_sort(s, t, q, z, Target.SmallestRealPart)

    assert_allclose(_get_ev(s, t, 0).real, wanted.real, rtol=0, atol=atol)
    assert_allclose(abs(_get_ev(s, t, 0).imag), abs(wanted.imag), rtol=0, atol=atol)

    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_generalized_schur_sort_largest_real(dtype):
    atol = numpy.finfo(dtype).eps * 100
    n = 10
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)
    s, t, q, z = generalized_schur.generalized_schur(a, b)

    idx = max(range(n), key=lambda i: _get_ev(s, t, i).real)

    wanted = _get_ev(s, t, idx)

    s, t, q, z = generalized_schur.generalized_schur_sort(s, t, q, z, Target.LargestRealPart)

    assert_allclose(_get_ev(s, t, 0).real, wanted.real, rtol=0, atol=atol)
    assert_allclose(abs(_get_ev(s, t, 0).imag), abs(wanted.imag), rtol=0, atol=atol)

    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_generalized_schur_sort_smallest_imag(dtype):
    atol = numpy.finfo(dtype).eps * 100
    n = 10
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)
    s, t, q, z = generalized_schur.generalized_schur(a, b)

    idx = min(range(n), key=lambda i: _get_ev(s, t, i).imag)

    wanted = _get_ev(s, t, idx)

    s, t, q, z = generalized_schur.generalized_schur_sort(s, t, q, z, Target.SmallestImaginaryPart)

    assert_allclose(_get_ev(s, t, 0).real, wanted.real, rtol=0, atol=atol)
    assert_allclose(abs(_get_ev(s, t, 0).imag), abs(wanted.imag), rtol=0, atol=atol)

    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_generalized_schur_sort_largest_imag(dtype):
    atol = numpy.finfo(dtype).eps * 100
    n = 10
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)
    s, t, q, z = generalized_schur.generalized_schur(a, b)

    idx = max(range(n), key=lambda i: _get_ev(s, t, i).imag)

    wanted = _get_ev(s, t, idx)

    s, t, q, z = generalized_schur.generalized_schur_sort(s, t, q, z, Target.LargestImaginaryPart)

    assert_allclose(_get_ev(s, t, 0).real, wanted.real, rtol=0, atol=atol)
    assert_allclose(abs(_get_ev(s, t, 0).imag), abs(wanted.imag), rtol=0, atol=atol)

    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_generalized_schur_sort_target(dtype):
    atol = numpy.finfo(dtype).eps * 100
    n = 10
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)
    s, t, q, z = generalized_schur.generalized_schur(a, b)

    target = Target.Target(complex(2, 2))
    idx = min(range(n), key=lambda i: abs(_get_ev(s, t, i) - target))

    wanted = _get_ev(s, t, idx)

    s, t, q, z = generalized_schur.generalized_schur_sort(s, t, q, z, target)

    assert_allclose(_get_ev(s, t, 0).real, wanted.real, rtol=0, atol=atol)
    assert_allclose(abs(_get_ev(s, t, 0).imag), abs(wanted.imag), rtol=0, atol=atol)

    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_generalized_schur_sort_target_complex(dtype):
    atol = numpy.finfo(dtype).eps * 100
    n = 10
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)
    s, t, q, z = generalized_schur.generalized_schur(a, b)

    target = 2.0 + 2.0j
    idx = min(range(n), key=lambda i: abs(_get_ev(s, t, i) - target))

    wanted = _get_ev(s, t, idx)

    s, t, q, z = generalized_schur.generalized_schur_sort(s, t, q, z, target)

    assert_allclose(_get_ev(s, t, 0).real, wanted.real, rtol=0, atol=atol)
    assert_allclose(abs(_get_ev(s, t, 0).imag), abs(wanted.imag), rtol=0, atol=atol)

    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_generalized_schur_sort_target_real(dtype):
    atol = numpy.finfo(dtype).eps * 100
    n = 10
    a = generate_random_dtype_array([n, n], dtype)
    b = generate_random_dtype_array([n, n], dtype)
    s, t, q, z = generalized_schur.generalized_schur(a, b)

    target = 2.0
    idx = min(range(n), key=lambda i: abs(_get_ev(s, t, i) - target))

    wanted = _get_ev(s, t, idx)

    s, t, q, z = generalized_schur.generalized_schur_sort(s, t, q, z, target)

    assert_allclose(_get_ev(s, t, 0).real, wanted.real, rtol=0, atol=atol)
    assert_allclose(abs(_get_ev(s, t, 0).imag), abs(wanted.imag), rtol=0, atol=atol)

    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)
