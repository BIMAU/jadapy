from math import sqrt

from jadapy.utils import dot, norm

def _proj(x, y):
    if x is None:
        return

    try:
        y -= x @ dot(x, y)
    except ValueError:
        y -= x * x.conj().dot(y)

def DGKS(V, w, W=None):
    prev_nrm = norm(w)
    _proj(V, w)
    _proj(W, w)

    nrm = norm(w)

    eta = 1 / sqrt(2)
    while nrm < eta * prev_nrm:
        _proj(V, w)
        _proj(W, w)
        prev_nrm = nrm
        nrm = norm(w)

    return nrm

def modified_gs(V, w, W=None):
    if V is not None:
        for i in range(V.shape[1]):
            _proj(V[:, i], w)

    if W is not None:
        for i in range(W.shape[1]):
            _proj(W[:, i], w)

    return None

def normalize(w, nrm=None):
    if nrm is None:
        nrm = norm(w)
    w /= nrm

def orthogonalize(V, w=None, W=None, method='DGKS'):
    if w is None:
        w = V
        V = None

    if len(w.shape) > 1:
        nrms = [0] * w.shape[1]
        for i in range(w.shape[1]):
            nrms[i] = orthogonalize(V, w[:, i], w[:, 0:i])
            w[:, i] /= nrms[i]
        for i in range(w.shape[1]):
            w[:, i] *= nrms[i]
        return

    if method == 'Modified Gram-Schmidt':
        return modified_gs(V, w, W)
    return DGKS(V, w, W)

def orthonormalize(V, w=None, W=None, method='DGKS'):
    if w is None:
        w = V
        V = None

    if len(w.shape) > 1:
        for i in range(w.shape[1]):
            orthonormalize(V, w[:, i], w[:, 0:i])
        return

    nrm = orthogonalize(V, w, W, method)
    normalize(w, nrm)
