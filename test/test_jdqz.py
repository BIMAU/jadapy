import pytest
import numpy
import scipy

from math import sqrt

from numpy.testing import assert_equal, assert_allclose

from jadapy import jdqz
from jadapy import Target
from jadapy.utils import norm

REAL_DTYPES = [numpy.float32, numpy.float64]
COMPLEX_DTYPES = [numpy.complex64, numpy.complex128]
DTYPES = REAL_DTYPES + COMPLEX_DTYPES

def generate_random_dtype_array(shape, dtype):
    if dtype in COMPLEX_DTYPES:
        return (numpy.random.rand(*shape)
                + numpy.random.rand(*shape) * 1.0j).astype(dtype)
    return numpy.random.rand(*shape).astype(dtype)

def generate_test_matrix(shape, dtype):
    a = generate_random_dtype_array(shape, dtype)
    a += 3 * numpy.diag(numpy.ones([shape[0]], dtype))
    return a

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_smallest_magnitude(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)
    b = generate_test_matrix([n, n], dtype)

    alpha, beta = jdqz.jdqz(a, b, num=k, tol=tol)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    eigs = scipy.linalg.eigvals(a, b)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_largest_magnitude(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)
    b = generate_test_matrix([n, n], dtype)

    alpha, beta = jdqz.jdqz(a, b, k, Target.LargestMagnitude, tol=tol)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: -abs(x)))

    eigs = scipy.linalg.eigvals(a, b)
    eigs = numpy.array(sorted(eigs, key=lambda x: -abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)


@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_smallest_real(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)
    b = generate_test_matrix([n, n], dtype)

    alpha, beta = jdqz.jdqz(a, b, k, Target.SmallestRealPart, tol=tol)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: x.real))

    eigs = scipy.linalg.eigvals(a, b)
    eigs = numpy.array(sorted(eigs, key=lambda x: x.real))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_largest_real(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)
    b = generate_test_matrix([n, n], dtype)

    alpha, beta = jdqz.jdqz(a, b, k, Target.LargestRealPart, tol=tol)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: -x.real))

    eigs = scipy.linalg.eigvals(a, b)
    eigs = numpy.array(sorted(eigs, key=lambda x: -x.real))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_smallest_imag(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)
    b = generate_test_matrix([n, n], dtype)

    alpha, beta = jdqz.jdqz(a, b, k, Target.SmallestImaginaryPart, tol=tol, arithmetic='complex')
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: x.imag))

    eigs = scipy.linalg.eigvals(a, b)
    eigs = numpy.array(sorted(eigs, key=lambda x: x.imag))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_smallest_imag_real(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 6
    a = generate_test_matrix([n, n], dtype)
    b = generate_test_matrix([n, n], dtype)

    alpha, beta = jdqz.jdqz(a, b, k, Target.SmallestImaginaryPart, tol=tol)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: x.imag))

    eigs = scipy.linalg.eigvals(a, b)
    eigs = numpy.array(sorted(eigs, key=lambda x: x.imag))
    eigs = eigs[:k]

    # In the real case, we store complex conjugate eigenpairs, so only
    # at least half of the eigenvalues are correct
    eigs = eigs[:k // 2]
    jdqz_eigs = jdqz_eigs[:k // 2]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_largest_imag(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)
    b = generate_test_matrix([n, n], dtype)

    alpha, beta = jdqz.jdqz(a, b, k, Target.LargestImaginaryPart, tol=tol, arithmetic='complex')
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: -x.imag))

    eigs = scipy.linalg.eigvals(a, b)
    eigs = numpy.array(sorted(eigs, key=lambda x: -x.imag))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_largest_imag_real(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 6
    a = generate_test_matrix([n, n], dtype)
    b = generate_test_matrix([n, n], dtype)

    alpha, beta = jdqz.jdqz(a, b, k, Target.LargestImaginaryPart, tol=tol)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: -x.imag))

    eigs = scipy.linalg.eigvals(a, b)
    eigs = numpy.array(sorted(eigs, key=lambda x: -x.imag))
    eigs = eigs[:k]

    # In the real case, we store complex conjugate eigenpairs, so only
    # at least half of the eigenvalues are correct
    eigs = eigs[:k // 2]
    jdqz_eigs = jdqz_eigs[:k // 2]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_target(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)
    b = generate_test_matrix([n, n], dtype)

    target = Target.Target(complex(2, 1))
    alpha, beta = jdqz.jdqz(a, b, k, target, tol=tol, arithmetic='complex')
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x - target.target)))

    eigs = scipy.linalg.eigvals(a, b)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x - target.target)))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_target_real(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 6
    a = generate_test_matrix([n, n], dtype)
    b = generate_test_matrix([n, n], dtype)

    target = Target.Target(complex(2, 1))
    alpha, beta = jdqz.jdqz(a, b, k, target, tol=tol)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x - target.target)))

    eigs = scipy.linalg.eigvals(a, b)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x - target.target)))
    eigs = eigs[:k]

    # In the real case, we store complex conjugate eigenpairs, so only
    # at least half of the eigenvalues are correct
    eigs = eigs[:k // 2]
    jdqz_eigs = jdqz_eigs[:k // 2]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_prec(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)
    b = generate_test_matrix([n, n], dtype)

    inv = scipy.sparse.linalg.spilu(scipy.sparse.csc_matrix(a))
    def _prec(x, *args):
        return inv.solve(x)

    alpha, beta = jdqz.jdqz(a, b, num=k, tol=tol, prec=_prec)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    eigs = scipy.linalg.eigvals(a, b)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_petrov(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)
    b = generate_test_matrix([n, n], dtype)

    alpha, beta = jdqz.jdqz(a, b, num=k, tol=tol, testspace='Petrov')
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    eigs = scipy.linalg.eigvals(a, b)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_variable_petrov(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    # Apparently this is really bad?
    atol = 1e-2
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)
    b = generate_test_matrix([n, n], dtype)

    alpha, beta = jdqz.jdqz(a, b, num=k, tol=tol, testspace='Variable Petrov')
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    eigs = scipy.linalg.eigvals(a, b)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_largest_magnitude_lowdim(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)
    b = generate_test_matrix([n, n], dtype)

    alpha, beta = jdqz.jdqz(a, b, k, Target.LargestMagnitude, tol=tol, subspace_dimensions=[10, 18])
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: -abs(x)))

    eigs = scipy.linalg.eigvals(a, b)
    eigs = numpy.array(sorted(eigs, key=lambda x: -abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqz_smallest_magnitude_eigenvectors(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)
    b = generate_test_matrix([n, n], dtype)

    alpha, beta, v = jdqz.jdqz(a, b, num=k, tol=tol, return_eigenvectors=True)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))
    jdqz_eigs = jdqz_eigs[:k]

    eigs = scipy.linalg.eigvals(a, b)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

    i = 0
    while i < k:
        ctype = numpy.dtype(numpy.dtype(dtype).char.upper())
        if dtype != ctype and alpha[i].imag:
            assert norm(beta[i].real * a @ v[:, i]   - alpha[i].real * b @ v[:, i] + alpha[i].imag * b @ v[:, i+1]) < atol
            assert norm(beta[i].real * a @ v[:, i+1] - alpha[i].imag * b @ v[:, i] - alpha[i].real * b @ v[:, i+1]) < atol
            i += 2
        else:
            assert norm(beta[i] * a @ v[:, i] - alpha[i] * b @ v[:, i]) < atol
            i += 1

def generate_Epetra_test_matrix(map, shape, dtype):
    from PyTrilinos import Epetra
    from jadapy import EpetraInterface

    a1 = generate_test_matrix(shape, dtype)
    a2 = EpetraInterface.CrsMatrix(Epetra.Copy, map, shape[1])

    for i in range(shape[0]):
        for j in range(shape[1]):
            a2[i, j] = a1[i, j]
    a2.FillComplete()

    return a1, a2

def test_Epetra():
    from PyTrilinos import Epetra
    from jadapy import EpetraInterface

    dtype = numpy.float64
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5

    comm = Epetra.PyComm()
    map = Epetra.Map(n, 0, comm)
    a1, a2 = generate_Epetra_test_matrix(map, [n, n], dtype)
    b1, b2 = generate_Epetra_test_matrix(map, [n, n], dtype)

    interface = EpetraInterface.EpetraInterface(map)

    alpha, beta = jdqz.jdqz(a2, b2, k, tol=tol, interface=interface)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    eigs = scipy.linalg.eigvals(a1, b1)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

def test_Epetra_lowdim():
    from PyTrilinos import Epetra
    from jadapy import EpetraInterface

    dtype = numpy.float64
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5

    comm = Epetra.PyComm()
    map = Epetra.Map(n, 0, comm)
    a1, a2 = generate_Epetra_test_matrix(map, [n, n], dtype)
    b1, b2 = generate_Epetra_test_matrix(map, [n, n], dtype)

    interface = EpetraInterface.EpetraInterface(map)

    alpha, beta = jdqz.jdqz(a2, b2, k, Target.LargestMagnitude, tol=tol, subspace_dimensions=[10, 18], interface=interface)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: -abs(x)))

    eigs = scipy.linalg.eigvals(a1, b1)
    eigs = numpy.array(sorted(eigs, key=lambda x: -abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

def test_Epetra_eigenvectors():
    from PyTrilinos import Epetra
    from jadapy import EpetraInterface

    dtype = numpy.float64
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5

    comm = Epetra.PyComm()
    map = Epetra.Map(n, 0, comm)
    a1, a2 = generate_Epetra_test_matrix(map, [n, n], dtype)
    b1, b2 = generate_Epetra_test_matrix(map, [n, n], dtype)

    interface = EpetraInterface.EpetraInterface(map)

    alpha, beta, v = jdqz.jdqz(a2, b2, num=k, tol=tol, return_eigenvectors=True, interface=interface)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))
    jdqz_eigs = jdqz_eigs[:k]

    eigs = scipy.linalg.eigvals(a1, b1)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqz_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

    i = 0
    while i < k:
        if alpha[i].imag:
            assert norm(a2 @ v[:, i] * beta[i].real   - b2 @ v[:, i] * alpha[i].real + b2 @ v[:, i+1] * alpha[i].imag) < atol
            assert norm(a2 @ v[:, i+1] * beta[i].real - b2 @ v[:, i] * alpha[i].imag - b2 @ v[:, i+1] * alpha[i].real) < atol
            i += 2
        else:
            assert norm(a2 @ v[:, i] * beta[i].real - b2 @ v[:, i] * alpha[i].real) < atol
            i += 1
