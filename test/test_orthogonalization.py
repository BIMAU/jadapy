import pytest

import numpy
from numpy.linalg import norm

from numpy.testing import assert_equal, assert_allclose

from jadapy import orthogonalization

REAL_DTYPES = [numpy.float32, numpy.float64]
COMPLEX_DTYPES = [numpy.complex64, numpy.complex128]
DTYPES = REAL_DTYPES + COMPLEX_DTYPES

def generate_random_dtype_array(shape, dtype):
    if dtype in COMPLEX_DTYPES:
        return (numpy.random.rand(*shape)
                + numpy.random.rand(*shape) * 1.0j).astype(dtype)
    return numpy.random.rand(*shape).astype(dtype)

def dot(x, y):
    return x.T.conj() @ y

@pytest.mark.parametrize('dtype', DTYPES)
def test_normalization(dtype):
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    x = generate_random_dtype_array([n], dtype)
    assert norm(x) > 1
    orthogonalization.normalize(x)
    assert_allclose(norm(x), 1, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_orthonormalization(dtype):
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    x = generate_random_dtype_array([n], dtype)
    orthogonalization.normalize(x)

    y = generate_random_dtype_array([n], dtype)
    orthogonalization.orthonormalize(x, y)
    assert_allclose(dot(x, y), 0, rtol=0, atol=atol)
    assert_allclose(norm(y), 1, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_orthonormalization_multiple_vectors(dtype):
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    k = 5
    x = generate_random_dtype_array([n, k], dtype)
    orthogonalization.orthonormalize(x)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            assert_allclose(dot(x[:, i], x[:, j]), 0, rtol=0, atol=atol)
        assert_allclose(norm(x[:, i]), 1, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_orthonormalization_multiple_vectors_twice(dtype):
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    k = 5
    x = generate_random_dtype_array([n, k], dtype)
    orthogonalization.orthonormalize(x)

    y = generate_random_dtype_array([n, k], dtype)
    orthogonalization.orthonormalize(x, y)
    for i in range(k):
        for j in range(k):
            assert_allclose(dot(x[:, i], y[:, j]), 0, rtol=0, atol=atol)
            if i == j:
                continue
            assert_allclose(dot(y[:, i], y[:, j]), 0, rtol=0, atol=atol)
        assert_allclose(norm(y[:, i]), 1, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_orthogonalization(dtype):
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    x = generate_random_dtype_array([n], dtype)
    orthogonalization.normalize(x)

    y = generate_random_dtype_array([n], dtype)
    orthogonalization.orthogonalize(x, y)
    assert_allclose(dot(x, y), 0, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_orthogonalization_multiple_vectors(dtype):
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    k = 5
    x = generate_random_dtype_array([n, k], dtype)
    orthogonalization.orthonormalize(x)

    y = generate_random_dtype_array([n], dtype)
    orthogonalization.orthogonalize(x, y)
    for i in range(k):
        assert_allclose(dot(x[:, i], y), 0, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_orthogonalization_no_vectors(dtype):
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    x = numpy.ndarray(())

    y = generate_random_dtype_array([n], dtype)
    orthogonalization.orthogonalize(x, y)
    assert norm(y) > 1
