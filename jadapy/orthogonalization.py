from math import sqrt

from numpy.linalg import norm

def _proj(x, y):
    try:
        y -= x * x.conj().dot(y)
    except ValueError:
        y -= x @ (x.T.conj() @ y)

def DGKS(V, w):
    if V.ndim > 1 and V.shape[1] < 1:
        return norm(w)

    prev_nrm = norm(w)
    _proj(V, w)

    nrm = norm(w)

    eta = 1 / sqrt(2)
    while nrm < eta * prev_nrm:
        _proj(V, w)
        prev_nrm = nrm
        nrm = norm(w)

    return nrm

def modified_gs(V, w):
    for i in range(V.shape[1]):
        _proj(V[:, i], w)

    return None

def normalize(w, nrm=None):
    if nrm is None:
        nrm = norm(w)
    w /= nrm

def orthogonalize(V, w=None, method='DGKS'):
    if w is None:
        for i in range(V.shape[1]):
            orthogonalize(V[:, 0:i], V[:, i])
        return

    if method == 'Modified Gram-Schmidt':
        return modified_gs(V, w)
    return DGKS(V, w)

def orthonormalize(V, w=None, method='DGKS'):
    if w is None:
        for i in range(V.shape[1]):
            orthonormalize(V[:, 0:i], V[:, i])
        return

    nrm = orthogonalize(V, w, method)
    normalize(w, nrm)
