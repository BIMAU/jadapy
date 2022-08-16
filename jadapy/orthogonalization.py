import warnings

from math import sqrt

from jadapy.utils import dot, norm, eps

def _proj(x, y, z=None):
    if x is None:
        return

    if z is None:
        z = x

    try:
        y -= x @ dot(z, y)
    except ValueError:
        y -= x * z.conj().dot(y)

def gram_schmidt(V, w, W=None, M=None, MV=None, MW=None):
    _proj(V, w, MV)
    _proj(W, w, MW)

def DGKS(V, w, W=None, M=None, MV=None, MW=None):
    prev_nrm = norm(w, M)
    gram_schmidt(V, w, W, M, MV, MW)
    nrm = norm(w, M)

    eta = 1 / sqrt(2)
    it = 0
    maxit = 3
    while nrm < eta * prev_nrm and it < maxit:
        gram_schmidt(V, w, W, M, MV, MW)
        prev_nrm = nrm
        nrm = norm(w, M)
        it += 1

    return nrm

def modified_gs(V, w, W=None, M=None, MV=None, MW=None):
    if V is not None:
        if len(V.shape) > 1:
            for i in range(V.shape[1]):
                if MV is not None:
                    _proj(V[:, i:i+1], w, MV[:, i:i+1])
                else:
                    _proj(V[:, i:i+1], w)
        else:
            _proj(V, w, MV)

    if W is not None:
        if len(W.shape) > 1:
            for i in range(W.shape[1]):
                if MW is not None:
                    _proj(W[:, i:i+1], w, MW[:, i:i+1])
                else:
                    _proj(W[:, i:i+1], w)
        else:
            _proj(W, w, MW)

    return None

def repeated_mgs(V, w, W=None, M=None, MV=None, MW=None):
    prev_nrm = norm(w, M)
    modified_gs(V, w, W, M, MV, MW)
    nrm = norm(w, M)

    it = 0
    maxit = 3

    eta = 1 / sqrt(2)
    while nrm < eta * prev_nrm and it < maxit:
        modified_gs(V, w, W, M, MV, MW)
        prev_nrm = nrm
        nrm = norm(w, M)
        it += 1

    return nrm

def normalize(w, nrm=None, M=None, verbose=True, interface=None):
    if nrm is None:
        nrm = norm(w, M)

    if verbose and nrm < eps(w):
        if not interface:
            raise Exception('Norm during normalization is nearly zero: %e' % nrm)
        else:
            warnings.warn('Warning: norm during normalization is nearly zero: %e. Taking a random vector.' % nrm)

            if len(w.shape) > 1:
                w[:, 0] = interface.random()
            else:
                w[:] = interface.random()

            w /= norm(w, M)
            return w

    w /= nrm
    return nrm

def orthogonalize(V, w, W=None, M=None, MV=None, MW=None, method='Repeated MGS'):
    if M is not None and V is not None and MV is None:
        MV = M @ V

    if M is not None and MW is not None and MW is None:
        MW = M @ W

    # Orthogonalize with respect to the basis, not itself
    if len(w.shape) > 1 and w.shape[1] > 1:
        for i in range(w.shape[1]):
            orthogonalize(V, w[:, i], W, M, MV, MW, method)
        return

    if method == 'Modified Gram-Schmidt' or method == 'MGS':
        return modified_gs(V, w, W, M, MV, MW)

    if method == 'Repeated Modified Gram-Schmidt' or method == 'Repeated MGS':
        return repeated_mgs(V, w, W, M, MV, MW)

    return DGKS(V, w, W, M, MV, MW)

def orthonormalize(V, w=None, W=None, M=None, MV=None, MW=None, method='Repeated MGS', interface=None):
    if w is None:
        w = V
        V = None
        MV = None

    if M is not None and V is not None and MV is None:
        MV = M @ V

    if M is not None and W is not None and MW is None:
        MW = M @ W

    # Orthonormalize with respect to the basis and itself
    if len(w.shape) > 1 and w.shape[1] > 1:
        for i in range(w.shape[1]):
            orthonormalize(V, w[:, i], w[:, 0:i], M, MV, MW, method)
        return

    nrm = orthogonalize(V, w, W, M, MV, MW, method)
    normalize(w, nrm, M, interface=interface)
