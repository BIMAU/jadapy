from math import sqrt

from jadapy.utils import dot, norm, eps

def _proj(x, y, M):
    if x is None:
        return

    z = x
    if M is not None:
        z = M @ x

    try:
        y -= x @ dot(z, y)
    except ValueError:
        y -= x * z.conj().dot(y)

def DGKS(V, w, W=None, M=None):
    prev_nrm = norm(w, M)
    _proj(V, w, M)
    _proj(W, w, M)

    nrm = norm(w, M)

    eta = 1 / sqrt(2)
    while nrm < eta * prev_nrm:
        _proj(V, w, M)
        _proj(W, w, M)
        prev_nrm = nrm
        nrm = norm(w, M)

    return nrm

def modified_gs(V, w, W=None, M=None):
    if V is not None:
        if len(V.shape) > 1:
            for i in range(V.shape[1]):
                _proj(V[:, i], w, M)
        else:
            _proj(V, w, M)

    if W is not None:
        if len(W.shape) > 1:
            for i in range(W.shape[1]):
                _proj(W[:, i], w, M)
        else:
            _proj(W, w, M)

    return None

def normalize(w, nrm=None, M=None, verbose=True):
    if nrm is None:
        nrm = norm(w, M)

    if verbose and nrm < eps(w):
        # print('Warning: norm during normalization is nearly zero: %e' % nrm)
        raise Exception('Norm during normalization is nearly zero: %e' % nrm)

    w /= nrm
    return nrm

def orthogonalize(V, w=None, W=None, M=None, method='DGKS'):
    # Orthogonalize the whole space
    if w is None and len(V.shape) > 1 and V.shape[1] > 1:
        nrms = [0] * V.shape[1]
        for i in range(V.shape[1]):
            nrm = orthogonalize(None, V[:, i], V[:, 0:i], M, method)
            nrms[i] = normalize(V[:, i], nrm, M, verbose=False)
        for i in range(V.shape[1]):
            V[:, i] *= nrms[i]
        return

    # Orthogonalize with respect to the basis, not itself
    if len(w.shape) > 1 and w.shape[1] > 1:
        for i in range(w.shape[1]):
            orthogonalize(V, w[:, i], W, M, method)
        return

    if method == 'Modified Gram-Schmidt' or method == 'MGS':
        return modified_gs(V, w, W, M)
    return DGKS(V, w, W, M)

def orthonormalize(V, w=None, W=None, M=None, method='DGKS'):
    if w is None:
        w = V
        V = None

    # Orthonormalize with respect to the basis and itself
    if len(w.shape) > 1 and w.shape[1] > 1:
        for i in range(w.shape[1]):
            orthonormalize(V, w[:, i], w[:, 0:i], M, method)
        return

    nrm = orthogonalize(V, w, W, M, method)
    normalize(w, nrm, M)
