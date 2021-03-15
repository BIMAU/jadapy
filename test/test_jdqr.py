import pytest
import numpy
import scipy

from numpy.testing import assert_allclose

from jadapy import jdqr
from jadapy import Target
from jadapy.utils import norm

REAL_DTYPES = [numpy.float32, numpy.float64]
COMPLEX_DTYPES = [numpy.complex64, numpy.complex128]
DTYPES = REAL_DTYPES + COMPLEX_DTYPES

def generate_random_dtype_array(shape, dtype):
    if dtype in COMPLEX_DTYPES:
        return (numpy.random.rand(*shape) + numpy.random.rand(*shape) * 1.0j).astype(dtype)
    return numpy.random.rand(*shape).astype(dtype)

def generate_mass_matrix(shape, dtype):
    x = numpy.zeros(shape, dtype)
    numpy.fill_diagonal(x, numpy.random.rand(shape[0]))
    return x

def generate_test_matrix(shape, dtype):
    a = generate_random_dtype_array(shape, dtype)
    a += 3 * numpy.diag(numpy.ones([shape[0]], dtype))
    return a

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqr_smallest_magnitude(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)

    alpha = jdqr.jdqr(a, num=k, tol=tol)
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: abs(x)))

    eigs = scipy.linalg.eigvals(a)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqr_smallest_magnitude_with_mass(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)
    m = generate_mass_matrix([n, n], dtype)

    alpha = jdqr.jdqr(a, num=k, tol=tol, M=m)
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: abs(x)))

    eigs = scipy.linalg.eigvals(a, m)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqr_largest_magnitude(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)

    alpha = jdqr.jdqr(a, k, Target.LargestMagnitude, tol=tol)
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: -abs(x)))

    eigs = scipy.linalg.eigvals(a)
    eigs = numpy.array(sorted(eigs, key=lambda x: -abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqr_largest_magnitude_with_mass(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)
    m = generate_mass_matrix([n, n], dtype)

    alpha = jdqr.jdqr(a, k, Target.LargestMagnitude, tol=tol, M=m)
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: -abs(x)))

    eigs = scipy.linalg.eigvals(a, m)
    eigs = numpy.array(sorted(eigs, key=lambda x: -abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqr_smallest_real(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)

    alpha = jdqr.jdqr(a, k, Target.SmallestRealPart, tol=tol)
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: x.real))

    eigs = scipy.linalg.eigvals(a)
    eigs = numpy.array(sorted(eigs, key=lambda x: x.real))
    eigs = eigs[:k]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqr_largest_real(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)

    alpha = jdqr.jdqr(a, k, Target.LargestRealPart, tol=tol)
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: -x.real))

    eigs = scipy.linalg.eigvals(a)
    eigs = numpy.array(sorted(eigs, key=lambda x: -x.real))
    eigs = eigs[:k]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqr_smallest_imag(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)

    alpha = jdqr.jdqr(a, k, Target.SmallestImaginaryPart, tol=tol, arithmetic='complex')
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: x.imag))

    eigs = scipy.linalg.eigvals(a)
    eigs = numpy.array(sorted(eigs, key=lambda x: x.imag))
    eigs = eigs[:k]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqr_smallest_imag_real(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 6
    a = generate_test_matrix([n, n], dtype)

    alpha = jdqr.jdqr(a, k, Target.SmallestImaginaryPart, tol=tol)
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: x.imag))

    eigs = scipy.linalg.eigvals(a)
    eigs = numpy.array(sorted(eigs, key=lambda x: x.imag))
    eigs = eigs[:k]

    # In the real case, we store complex conjugate eigenpairs, so only
    # at least half of the eigenvalues are correct
    eigs = eigs[:k // 2]
    jdqr_eigs = jdqr_eigs[:k // 2]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqr_largest_imag(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)

    alpha = jdqr.jdqr(a, k, Target.LargestImaginaryPart, tol=tol, arithmetic='complex')
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: -x.imag))

    eigs = scipy.linalg.eigvals(a)
    eigs = numpy.array(sorted(eigs, key=lambda x: -x.imag))
    eigs = eigs[:k]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqr_largest_imag_real(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 6
    a = generate_test_matrix([n, n], dtype)

    alpha = jdqr.jdqr(a, k, Target.LargestImaginaryPart, tol=tol)
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: -x.imag))

    eigs = scipy.linalg.eigvals(a)
    eigs = numpy.array(sorted(eigs, key=lambda x: -x.imag))
    eigs = eigs[:k]

    # In the real case, we store complex conjugate eigenpairs, so only
    # at least half of the eigenvalues are correct
    eigs = eigs[:k // 2]
    jdqr_eigs = jdqr_eigs[:k // 2]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqr_target(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)

    target = 2 + 1j
    alpha = jdqr.jdqr(a, k, target, tol=tol, arithmetic='complex')
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: abs(x - target)))

    eigs = scipy.linalg.eigvals(a)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x - target)))
    eigs = eigs[:k]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqr_target_real(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 6
    a = generate_test_matrix([n, n], dtype)

    target = 2 + 1j
    alpha = jdqr.jdqr(a, k, target, tol=tol)
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: abs(x - target)))

    eigs = scipy.linalg.eigvals(a)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x - target)))
    eigs = eigs[:k]

    # In the real case, we store complex conjugate eigenpairs, so only
    # at least half of the eigenvalues are correct
    eigs = eigs[:k // 2]
    jdqr_eigs = jdqr_eigs[:k // 2]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqr_prec(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)

    inv = scipy.sparse.linalg.spilu(scipy.sparse.csc_matrix(a))

    def _prec(x, *args):
        return inv.solve(x)

    alpha = jdqr.jdqr(a, num=k, tol=tol, prec=_prec)
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: abs(x)))

    eigs = scipy.linalg.eigvals(a)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqr_prec_with_mass(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)
    m = generate_mass_matrix([n, n], dtype)

    inv = scipy.sparse.linalg.spilu(scipy.sparse.csc_matrix(a))

    def _prec(x, *args):
        return inv.solve(x)

    alpha = jdqr.jdqr(a, num=k, tol=tol, prec=_prec, M=m)
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: abs(x)))

    eigs = scipy.linalg.eigvals(a, m)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqr_largest_magnitude_lowdim(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 2
    a = generate_test_matrix([n, n], dtype)

    alpha = jdqr.jdqr(a, k, Target.LargestMagnitude, tol=tol, subspace_dimensions=[10, 18])
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: -abs(x)))

    eigs = scipy.linalg.eigvals(a)
    eigs = numpy.array(sorted(eigs, key=lambda x: -abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqr_largest_magnitude_lowdim_with_mass(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 2
    a = generate_test_matrix([n, n], dtype)
    m = generate_mass_matrix([n, n], dtype)

    alpha = jdqr.jdqr(a, k, Target.LargestMagnitude, tol=tol, subspace_dimensions=[10, 18], M=m)
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: -abs(x)))

    eigs = scipy.linalg.eigvals(a, m)
    eigs = numpy.array(sorted(eigs, key=lambda x: -abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqr_smallest_magnitude_eigenvectors(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)

    alpha, v = jdqr.jdqr(a, num=k, tol=tol, return_eigenvectors=True)
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: abs(x)))
    jdqr_eigs = jdqr_eigs[:k]

    eigs = scipy.linalg.eigvals(a)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

    i = 0
    while i < k:
        ctype = numpy.dtype(numpy.dtype(dtype).char.upper())
        if dtype != ctype and alpha[i].imag:
            assert norm(a @ v[:, i] - alpha[i].real * v[:, i] + alpha[i].imag * v[:, i+1]) < atol
            assert norm(a @ v[:, i+1] - alpha[i].imag * v[:, i] - alpha[i].real * v[:, i+1]) < atol
            i += 2
        else:
            assert norm(a @ v[:, i] - alpha[i] * v[:, i]) < atol
            i += 1

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqr_smallest_magnitude_eigenvectors_with_mass(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)
    m = generate_mass_matrix([n, n], dtype)

    alpha, v = jdqr.jdqr(a, num=k, tol=tol, M=m, return_eigenvectors=True)
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: abs(x)))
    jdqr_eigs = jdqr_eigs[:k]

    eigs = scipy.linalg.eigvals(a, m)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

    i = 0
    while i < k:
        ctype = numpy.dtype(numpy.dtype(dtype).char.upper())
        if dtype != ctype and alpha[i].imag:
            assert norm(a @ v[:, i] - alpha[i].real * m @ v[:, i] + alpha[i].imag * m @ v[:, i+1]) < atol
            assert norm(a @ v[:, i+1] - alpha[i].imag * m @ v[:, i] - alpha[i].real * m @ v[:, i+1]) < atol
            i += 2
        else:
            assert norm(a @ v[:, i] - alpha[i] * m @ v[:, i]) < atol
            i += 1

@pytest.mark.parametrize('dtype', DTYPES)
def test_jdqr_smallest_magnitude_initial_subspace(dtype):
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5
    a = generate_test_matrix([n, n], dtype)

    alpha, q = jdqr.jdqr(a, num=k, tol=tol, return_subspace=True)

    alpha = jdqr.jdqr(a, num=k, tol=tol, initial_subspace=q)
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: abs(x)))
    jdqr_eigs = jdqr_eigs[:k]

    eigs = scipy.linalg.eigvals(a)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

def generate_Epetra_test_matrix(map, shape, dtype):
    try:
        from PyTrilinos import Epetra
        from jadapy import EpetraInterface
    except ImportError:
        pytest.skip("Trilinos not found")

    a1 = generate_test_matrix(shape, dtype)
    a2 = EpetraInterface.CrsMatrix(Epetra.Copy, map, shape[1])

    for i in range(shape[0]):
        for j in range(shape[1]):
            a2[i, j] = a1[i, j]
    a2.FillComplete()

    return a1, a2

def test_Epetra():
    try:
        from PyTrilinos import Epetra
        from jadapy import EpetraInterface
    except ImportError:
        pytest.skip("Trilinos not found")

    dtype = numpy.float64
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5

    comm = Epetra.PyComm()
    map = Epetra.Map(n, 0, comm)
    a1, a2 = generate_Epetra_test_matrix(map, [n, n], dtype)

    interface = EpetraInterface.EpetraInterface(map)

    alpha = jdqr.jdqr(a2, k, tol=tol, interface=interface)
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: abs(x)))

    eigs = scipy.linalg.eigvals(a1)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

def test_Epetra_lowdim():
    try:
        from PyTrilinos import Epetra
        from jadapy import EpetraInterface
    except ImportError:
        pytest.skip("Trilinos not found")

    dtype = numpy.float64
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 2

    comm = Epetra.PyComm()
    map = Epetra.Map(n, 0, comm)
    a1, a2 = generate_Epetra_test_matrix(map, [n, n], dtype)

    interface = EpetraInterface.EpetraInterface(map)

    alpha = jdqr.jdqr(a2, k, Target.LargestMagnitude, tol=tol, subspace_dimensions=[10, 18], interface=interface)
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: -abs(x)))

    eigs = scipy.linalg.eigvals(a1)
    eigs = numpy.array(sorted(eigs, key=lambda x: -abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

def test_Epetra_eigenvectors():
    try:
        from PyTrilinos import Epetra
        from jadapy import EpetraInterface
    except ImportError:
        pytest.skip("Trilinos not found")

    dtype = numpy.float64
    numpy.random.seed(1234)
    tol = numpy.finfo(dtype).eps * 1e3
    atol = tol * 10
    n = 20
    k = 5

    comm = Epetra.PyComm()
    map = Epetra.Map(n, 0, comm)
    a1, a2 = generate_Epetra_test_matrix(map, [n, n], dtype)

    interface = EpetraInterface.EpetraInterface(map)

    alpha, v = jdqr.jdqr(a2, num=k, tol=tol, return_eigenvectors=True, interface=interface)
    jdqr_eigs = numpy.array(sorted(alpha, key=lambda x: abs(x)))
    jdqr_eigs = jdqr_eigs[:k]

    eigs = scipy.linalg.eigvals(a1)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    eigs = eigs[:k]

    assert_allclose(jdqr_eigs.real, eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqr_eigs.imag), abs(eigs.imag), rtol=0, atol=atol)

    i = 0
    while i < k:
        if alpha[i].imag:
            assert norm(a2 @ v[:, i] - v[:, i] * alpha[i].real + v[:, i+1] * alpha[i].imag) < atol
            assert norm(a2 @ v[:, i+1] - v[:, i] * alpha[i].imag - v[:, i+1] * alpha[i].real) < atol
            i += 2
        else:
            assert norm(a2 @ v[:, i] - v[:, i] * alpha[i].real) < atol
            i += 1
