from math import sqrt

from numpy.linalg import norm

def DGKS(V, w):
    prev_nrm = norm(w)
    w -= V * V.dot(w)

    nrm = norm(w)

    eta = 1 / sqrt(2)
    while nrm < eta * prev_nrm:
        w -= V * V.dot(w)
        prev_nrm = nrm
        nrm = norm(w)

    return nrm

def modified_gs(V, w):
    for i in range(V.shape[0]):
        w -= V[:, i] * V[:, i].dot(w)

    return None

def normalize(w, nrm=None):
    if nrm is None:
        nrm = norm(w)
    w /= nrm

def orthogonalize(V, w, method='DGKS'):
    if method == 'Modified Gram-Schmidt':
        return modified_gs(V, w)
    return DGKS(V, w)

def orthonormalize(V, w, method='DGKS'):
    nrm = orthogonalize(V, w, method)
    normalize(w, nrm)
