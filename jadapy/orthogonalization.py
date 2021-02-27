from math import sqrt

from jadapy.utils import dot, norm, eps

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
        if len(V.shape) > 1:
            for i in range(V.shape[1]):
                _proj(V[:, i], w)
        else:
            _proj(V, w)

    if W is not None:
        if len(W.shape) > 1:
            for i in range(W.shape[1]):
                _proj(W[:, i], w)
        else:
            _proj(W, w)

    return None

def normalize(w, nrm=None, verbose=True):
    if nrm is None:
        nrm = norm(w)

    if verbose and nrm < eps(w):
        # print('Warning: norm during normalization is nearly zero: %e' % nrm)
        raise Exception('Norm during normalization is nearly zero: %e' % nrm)

    w /= nrm
    return nrm

def orthogonalize(V, w=None, W=None, method='DGKS'):
    # Orthogonalize the whole space
    if w is None and len(V.shape) > 1 and V.shape[1] > 1:
        nrms = [0] * V.shape[1]
        for i in range(V.shape[1]):
            nrm = orthogonalize(None, V[:, i], V[:, 0:i])
            nrms[i] = normalize(V[:, i], nrm, verbose=False)
        for i in range(V.shape[1]):
            V[:, i] *= nrms[i]
        return

    # Orthogonalize with respect to the basis, not itself
    if len(w.shape) > 1 and w.shape[1] > 1:
        for i in range(w.shape[1]):
            orthogonalize(V, w[:, i])
        return

    if method == 'Modified Gram-Schmidt' or method == 'MGS':
        return modified_gs(V, w, W)
    return DGKS(V, w, W)

def orthonormalize(V, w=None, W=None, method='DGKS'):
    if w is None:
        w = V
        V = None

    # Orthonormalize with respect to the basis and itself
    if len(w.shape) > 1 and w.shape[1] > 1:
        for i in range(w.shape[1]):
            orthonormalize(V, w[:, i], w[:, 0:i])
        return

    nrm = orthogonalize(V, w, W, method)
    normalize(w, nrm)
