import pytest

import numpy
from numpy.linalg import norm

from jadapy import orthogonalization

def test_normalization():
    n = 20
    x = numpy.random.rand(n)
    assert norm(x) > 1
    orthogonalization.normalize(x)
    assert norm(x) == pytest.approx(1)

def test_orthonormalization():
    n = 20
    x = numpy.random.rand(n)
    orthogonalization.normalize(x)

    y = numpy.random.rand(n)
    orthogonalization.orthonormalize(x, y)
    assert x.dot(y) == pytest.approx(0)
    assert norm(y) == pytest.approx(1)

def test_orthonormalization_multiple_vectors():
    n = 20
    k = 5
    x = numpy.random.rand(n, k)
    orthogonalization.orthonormalize(x)

    y = numpy.random.rand(n)
    orthogonalization.orthogonalize(x, y)
    for i in range(k):
        assert x[:, i].dot(y) == pytest.approx(0)

def test_orthogonalization():
    n = 20
    x = numpy.random.rand(n)
    orthogonalization.normalize(x)

    y = numpy.random.rand(n)
    orthogonalization.orthogonalize(x, y)
    assert x.dot(y) == pytest.approx(0)
    assert norm(y) > 1

def test_orthogonalization_multiple_vectors():
    n = 20
    k = 5
    x = numpy.random.rand(n, k)
    orthogonalization.orthonormalize(x)
    assert x[:, 1].dot(x[:, 3]) == pytest.approx(0)
    assert norm(x) > 1
