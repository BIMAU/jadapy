import pytest

import numpy
from numpy.linalg import norm

from numpy.testing import assert_equal, assert_allclose

from jadapy import orthogonalization

REAL_DTYPES = [numpy.float32, numpy.float64]
COMPLEX_DTYPES = [numpy.complex64, numpy.complex128]
DTYPES = REAL_DTYPES + COMPLEX_DTYPES
OTYPES = ['DGKS', 'MGS']

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
@pytest.mark.parametrize('otype', OTYPES)
def test_orthonormalization(dtype, otype):
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    x = generate_random_dtype_array([n], dtype)
    orthogonalization.normalize(x)

    y = generate_random_dtype_array([n], dtype)
    orthogonalization.orthonormalize(x, y, method=otype)
    assert_allclose(dot(x, y), 0, rtol=0, atol=atol)
    assert_allclose(norm(y), 1, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('otype', OTYPES)
def test_orthonormalization_multiple_vectors(dtype, otype):
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    k = 5
    x = generate_random_dtype_array([n, k], dtype)
    orthogonalization.orthonormalize(x, method=otype)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            assert_allclose(dot(x[:, i], x[:, j]), 0, rtol=0, atol=atol)
        assert_allclose(norm(x[:, i]), 1, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('otype', OTYPES)
def test_orthonormalization_multiple_vectors_twice(dtype, otype):
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    k = 5
    x = generate_random_dtype_array([n, k], dtype)
    orthogonalization.orthonormalize(x, method=otype)

    y = generate_random_dtype_array([n, k], dtype)
    orthogonalization.orthonormalize(x, y, method=otype)
    for i in range(k):
        for j in range(k):
            assert_allclose(dot(x[:, i], y[:, j]), 0, rtol=0, atol=atol)
            if i == j:
                continue
            assert_allclose(dot(y[:, i], y[:, j]), 0, rtol=0, atol=atol)
        assert_allclose(norm(y[:, i]), 1, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('otype', OTYPES)
def test_orthogonalization(dtype, otype):
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    x = generate_random_dtype_array([n], dtype)
    orthogonalization.normalize(x)

    y = generate_random_dtype_array([n], dtype)
    orthogonalization.orthogonalize(x, y, method=otype)
    assert_allclose(dot(x, y), 0, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('otype', OTYPES)
def test_orthogonalization_multiple_vectors(dtype, otype):
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    k = 5
    x = generate_random_dtype_array([n, k], dtype)
    orthogonalization.orthogonalize(x, method=otype)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            assert_allclose(dot(x[:, i], x[:, j]), 0, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('otype', OTYPES)
def test_orthogonalization_multiple_vectors_twice(dtype, otype):
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    k = 5
    x = generate_random_dtype_array([n, k], dtype)
    orthogonalization.orthonormalize(x, method=otype)

    y = generate_random_dtype_array([n, k], dtype)
    orthogonalization.orthogonalize(x, y, method=otype)
    for i in range(k):
        for j in range(k):
            assert_allclose(dot(x[:, i], y[:, j]), 0, rtol=0, atol=atol)
            if i == j:
                continue
            assert_allclose(dot(y[:, i], y[:, j]), 0, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('otype', OTYPES)
def test_orthogonalization_no_vectors(dtype, otype):
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    x = numpy.ndarray(())

    y = generate_random_dtype_array([n], dtype)
    orthogonalization.orthogonalize(x, y, method=otype)
    assert norm(y) > 1

@pytest.mark.parametrize('otype', OTYPES)
def test_orthonormalization_epetra(otype):
    from PyTrilinos import Epetra
    from jadapy import EpetraInterface

    dtype = numpy.float64
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    k = 5

    comm = Epetra.PyComm()
    map = Epetra.Map(n, 0, comm)
    x = EpetraInterface.Vector(map, k)
    x.Random()

    orthogonalization.orthogonalize(x, method=otype)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            assert_allclose(x[:, i].dot(x[:, j]), 0, rtol=0, atol=atol)
