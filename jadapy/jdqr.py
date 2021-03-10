import sys
import numpy
import scipy

from jadapy import Target
from jadapy.schur import schur, schur_sort
from jadapy.orthogonalization import orthogonalize, orthonormalize
from jadapy.correction_equation import solve_correction_equation
from jadapy.utils import dot, norm
from jadapy.NumPyInterface import NumPyInterface

def _prec(x, *args):
    return x

def jdqr(A, num=5, target=Target.SmallestMagnitude, tol=1e-8, prec=None,
         maxit=1000, subspace_dimensions=(20, 40), arithmetic='real',
         return_eigenvectors=False, interface=None):

    if arithmetic not in ['real', 'complex', 'r', 'c']:
        raise ValueError("argument must be 'real', or 'complex'")

    if not prec:
        prec = _prec

    solver_tolerance = 1.0

    n = A.shape[0]

    subspace_dimensions = (min(subspace_dimensions[0], n // 2), min(subspace_dimensions[1], n))

    it = 1
    k = 0 # Number of eigenvalues found
    m = 0 # Size of the search subspace
    nev = 1 # Amount of eigenvalues currently converging

    alpha = None
    evs = None

    dtype = A.dtype
    ctype = numpy.dtype(dtype.char.upper())

    if arithmetic in ['complex', 'c']:
        dtype = ctype

    if not interface:
        interface = NumPyInterface(n, dtype)

    extra = 0
    if dtype != ctype:
        # Allocate extra space in case a complex eigenpair may exist for a real matrix
        extra = 1

    # Eigenvalues
    aconv = numpy.zeros(num + extra, ctype)

    # Schur matrices
    R = numpy.zeros((num + extra, num + extra), dtype)

    # Schur vectors
    Q = interface.vector(num + extra)
    H = numpy.zeros((num + extra, num + extra), dtype)

    # Orthonormal search subspace
    V = interface.vector(subspace_dimensions[1])
    # Preconditioned orthonormal search subspace
    Y = interface.vector(subspace_dimensions[1])
    # AV = A*V without orthogonalization
    AV = interface.vector(subspace_dimensions[1])

    # Residual vector
    r = interface.vector(1 + extra)

    # Low-dimensional projection: VAV = V'*A*V
    VAV = numpy.zeros((subspace_dimensions[1], subspace_dimensions[1]), dtype)

    while k < num and it <= maxit:
        if it == 1:
            V[:, 0] = interface.random()
        else:
            solver_maxit = 100
            sigma = evs[0]

            # Build an initial search subspace in an inexpensive way
            # and as close to the target as possible
            if m < subspace_dimensions[0]:
                solver_tolerance = 0.5
                solver_maxit = 1
                if target != 0.0:
                    sigma = target

            V[:, m:m+nev] = solve_correction_equation(
                A, prec, Q[:, 0:k+nev], Y[:, 0:k+nev], H[0:k+nev, 0:k+nev],
                sigma, r[:, 0:nev], solver_tolerance, solver_maxit, interface)

        orthonormalize(V[:, 0:m], V[:, m:m+nev])

        AV[:, m:m+nev] = A @ V[:, m:m+nev]

        # Update VAV = V' * A * V
        for i in range(m):
            VAV[i, m:m+nev] = dot(V[:, i], AV[:, m:m+nev])
            VAV[m:m+nev, i] = dot(V[:, m:m+nev], AV[:, i])
        VAV[m:m+nev, m:m+nev] = dot(V[:, m:m+nev], AV[:, m:m+nev])

        m += nev

        [S, U] = schur(VAV[0:m, 0:m])

        found = True
        while found:
            [S, U] = schur_sort(S, U, target)

            nev = 1
            if dtype != ctype and S.shape[0] > 1 and abs(S[1, 0]) > 0.0:
                # Complex eigenvalue in real arithmetic
                nev = 2

            alpha = S[0:nev, 0:nev]

            evcond = norm(alpha)

            Q[:, k:k+nev] = V[:, 0:m] @ U[:, 0:nev]
            Y[:, k:k+nev] = prec(Q[:, k:k+nev], alpha)

            for i in range(k):
                H[i, k:k+nev] = dot(Q[:, i], Y[:, k:k+nev])
                H[k:k+nev, i] = dot(Q[:, k:k+nev], Y[:, i])
            H[k:k+nev, k:k+nev] = dot(Q[:, k:k+nev], Y[:, k:k+nev])

            r[:, 0:nev] = A @ Q[:, k:k+nev] - Q[:, k:k+nev] @ alpha
            orthogonalize(Q[:, 0:k+nev], r[:, 0:nev])

            rnorm = norm(r[:, 0:nev]) / evcond

            evs = scipy.linalg.eigvals(alpha)
            ev_est = evs[0]
            print("Step: %4d, subspace dimension: %3d, eigenvalue estimate: %13.6e + %13.6ei, residual norm: %e" %
                  (it, m, ev_est.real, ev_est.imag, rnorm))
            sys.stdout.flush()

            # Store converged Ritz pairs
            if rnorm <= tol:
                # Compute R so we can compute the eigenvectors
                if return_eigenvectors:
                    for i in range(k):
                        R[i, k:k+nev] = dot(Q[:, i], A @ Q[:, k:k+nev])
                    R[k:k+nev, k:k+nev] = alpha

                # Store the converged eigenvalues
                for i in range(nev):
                    print("Found an eigenvalue:", evs[i])
                    sys.stdout.flush()

                    aconv[k] = evs[i]
                    k += 1

                if k >= num:
                    break

                # Reset the iterative solver tolerance
                solver_tolerance = 1.0

                # Remove the eigenvalue from the search space
                V[:, 0:m-nev] = V[:, 0:m] @ U[:, nev:m]
                AV[:, 0:m-nev] = AV[:, 0:m] @ U[:, nev:m]

                VAV[0:m-nev, 0:m-nev] = S[nev:m, nev:m]

                S = VAV[0:m-nev, 0:m-nev]

                U = numpy.identity(m-nev, dtype)

                m -= nev
            else:
                found = False

        solver_tolerance = max(solver_tolerance / 2, tol)

        if m >= min(subspace_dimensions[1], n - k) and k < num:
            # Maximum search space dimension has been reached.
            new_m = min(subspace_dimensions[0], n - k)

            print("Shrinking the search space from %d to %d" % (m, new_m))
            sys.stdout.flush()

            V[:, 0:new_m] = V[:, 0:m] @ U[:, 0:new_m]
            AV[:, 0:new_m] = AV[:, 0:m] @ U[:, 0:new_m]

            VAV[0:new_m, 0:new_m] = S[0:new_m, 0:new_m]

            m = new_m
        elif m + nev - 1 >= min(subspace_dimensions[1], n - k):
            # Only one extra vector fits in the search space.
            nev = 1

        it += 1

    if return_eigenvectors:
        evs, v = scipy.linalg.eig(R[0:k, 0:k], left=False, right=True)

        if ctype == dtype:
            return evs, Q[:, 0:k] @ v

        i = 0
        while i < k:
            Y[:, i] = Q[:, 0:k] @ v[:, i].real
            if evs[i].imag:
                Y[:, i+1] = Q[:, 0:k] @ v[:, i].imag
                i += 1
            i += 1
        return evs, Y

    return aconv[0:num]
