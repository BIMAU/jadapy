import pytest

import numpy

from numpy.testing import assert_allclose

from jadapy import orthogonalization
from jadapy.utils import norm

REAL_DTYPES = [numpy.float32, numpy.float64]
COMPLEX_DTYPES = [numpy.complex64, numpy.complex128]
DTYPES = REAL_DTYPES + COMPLEX_DTYPES
OTYPES = ['DGKS', 'MGS', 'Repeated MGS']

def generate_random_dtype_array(shape, dtype):
    if dtype in COMPLEX_DTYPES:
        return (numpy.random.rand(*shape) + numpy.random.rand(*shape) * 1.0j).astype(dtype)
    return numpy.random.rand(*shape).astype(dtype)

def generate_random_mass_matrix(shape, dtype):
    x = numpy.zeros(shape, dtype)
    numpy.fill_diagonal(x, numpy.random.rand(shape[0]))
    return x

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
def test_orthonormalization_multiple_vectors_with_mass(dtype, otype):
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    k = 5
    x = generate_random_dtype_array([n, k], dtype)
    M = generate_random_mass_matrix([n, n], dtype)
    orthogonalization.orthonormalize(x, method=otype, M=M)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            assert_allclose(dot(x[:, i], M @ x[:, j]), 0, rtol=0, atol=atol)
        assert_allclose(norm(x[:, i], M), 1, rtol=0, atol=atol)

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
def test_orthonormalization_multiple_vectors_twice_with_mass(dtype, otype):
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    k = 5
    x = generate_random_dtype_array([n, k], dtype)
    M = generate_random_mass_matrix([n, n], dtype)
    orthogonalization.orthonormalize(x, method=otype, M=M)

    y = generate_random_dtype_array([n, k], dtype)
    orthogonalization.orthonormalize(x, y, method=otype, M=M)
    for i in range(k):
        for j in range(k):
            assert_allclose(dot(x[:, i], M @ y[:, j]), 0, rtol=0, atol=atol)
            if i == j:
                continue
            assert_allclose(dot(y[:, i], M @ y[:, j]), 0, rtol=0, atol=atol)
        assert_allclose(norm(y[:, i], M), 1, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('otype', OTYPES)
def test_orthonormalization_multiple_vectors_twice_with_mass_optim(dtype, otype):
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    k = 5
    x = generate_random_dtype_array([n, k], dtype)
    M = generate_random_mass_matrix([n, n], dtype)
    orthogonalization.orthonormalize(x, method=otype, M=M)

    Mx = M @ x
    y = generate_random_dtype_array([n, k], dtype)
    orthogonalization.orthonormalize(x, y, method=otype, M=M, MV=Mx)
    for i in range(k):
        for j in range(k):
            assert_allclose(dot(x[:, i], M @ y[:, j]), 0, rtol=0, atol=atol)
            if i == j:
                continue
            assert_allclose(dot(y[:, i], M @ y[:, j]), 0, rtol=0, atol=atol)
        assert_allclose(norm(y[:, i], M), 1, rtol=0, atol=atol)

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
            # Not orthogonalized with respect to itself
            # assert_allclose(dot(y[:, i], y[:, j]), 0, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('otype', OTYPES)
def test_orthogonalization_multiple_vectors_twice_with_mass(dtype, otype):
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    k = 5
    x = generate_random_dtype_array([n, k], dtype)
    M = generate_random_mass_matrix([n, n], dtype)
    orthogonalization.orthonormalize(x, method=otype, M=M)

    y = generate_random_dtype_array([n, k], dtype)
    orthogonalization.orthogonalize(x, y, method=otype, M=M)
    for i in range(k):
        for j in range(k):
            assert_allclose(dot(x[:, i], M @ y[:, j]), 0, rtol=0, atol=atol)
            if i == j:
                continue
            # Not orthogonalized with respect to itself
            # assert_allclose(dot(y[:, i], y[:, j]), 0, rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('otype', OTYPES)
def test_orthogonalization_no_vectors(dtype, otype):
    n = 20
    x = numpy.ndarray(())

    y = generate_random_dtype_array([n], dtype)
    orthogonalization.orthogonalize(x, y, method=otype)
    assert norm(y) > 1

@pytest.mark.parametrize('otype', OTYPES)
def test_orthonormalization_multiple_vectors_epetra(otype):
    try:
        from PyTrilinos import Epetra
        from jadapy import EpetraInterface
    except ImportError:
        pytest.skip("Trilinos not found")

    dtype = numpy.float64
    atol = numpy.finfo(dtype).eps * 100
    n = 20
    k = 5

    comm = Epetra.PyComm()
    map = Epetra.Map(n, 0, comm)
    x = EpetraInterface.Vector(map, k)
    x.Random()

    orthogonalization.orthonormalize(x, method=otype)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            assert_allclose(x[:, i].dot(x[:, j]), 0, rtol=0, atol=atol)
        assert_allclose(norm(x[:, i]), 1, rtol=0, atol=atol)
