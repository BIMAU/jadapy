import warnings

from math import sqrt

from jadapy.utils import dot, norm, eps

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

            nrm = norm(w, M)
            w /= nrm
            return nrm

    w /= nrm
    return nrm

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

def DGKS(V, w, W=None, M=None, MV=None, MW=None, normalized=False, interface=None):
    prev_nrm = None
    nrm = norm(w, M)

    if normalized:
        normalize(w, nrm, M, interface=interface)

    eta = 1 / sqrt(2)
    while prev_nrm is None or nrm < eta * prev_nrm:
        gram_schmidt(V, w, W, M, MV, MW)
        prev_nrm = nrm
        nrm = norm(w, M)

        if normalized:
            normalize(w, nrm, M, interface=interface)
            prev_nrm = 1

    if normalized:
        return 1

    return nrm

def modified_gs(V, w, W=None, M=None, MV=None, MW=None, normalized=False, interface=None):
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
        modified_gs(W, w, None, M, MW, None)

    if normalized:
        normalize(w, M=M, interface=interface)
        return 1

    return None

def repeated_mgs(V, w, W=None, M=None, MV=None, MW=None, normalized=False, interface=None):
    prev_nrm = None
    nrm = norm(w, M)

    if normalized:
        normalize(w, nrm, M, interface=interface)

    eta = 1 / sqrt(2)
    while prev_nrm is None or nrm < eta * prev_nrm:
        modified_gs(V, w, W, M, MV, MW)
        prev_nrm = nrm
        nrm = norm(w, M)

        if normalized:
            normalize(w, nrm, M, interface=interface)
            prev_nrm = 1

    if normalized:
        return 1

    return nrm

def orthogonalize(V, w, W=None, M=None, MV=None, MW=None, method='Repeated MGS', normalized=False, interface=None):
    if M is not None and V is not None and MV is None:
        MV = M @ V

    if M is not None and W is not None and MW is None:
        MW = M @ W

    # Orthogonalize with respect to the basis, not itself
    if len(w.shape) > 1 and w.shape[1] > 1:
        for i in range(w.shape[1]):
            orthogonalize(V, w[:, i], W, M, MV, MW, method, normalized, interface)
        return

    if method == 'Modified Gram-Schmidt' or method == 'MGS':
        return modified_gs(V, w, W, M, MV, MW, normalized, interface)

    if method == 'Repeated Modified Gram-Schmidt' or method == 'Repeated MGS':
        return repeated_mgs(V, w, W, M, MV, MW, normalized, interface)

    return DGKS(V, w, W, M, MV, MW, normalized, interface)

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
            orthonormalize(V, w[:, i], w[:, 0:i], M, MV, MW, method, interface)
        return

    nrm = orthogonalize(V, w, W, M, MV, MW, method, True, interface)
    normalize(w, nrm, M, interface=interface)
