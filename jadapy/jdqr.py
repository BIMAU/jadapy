import numpy
import scipy

from jadapy import Target
from jadapy.schur import schur, schur_sort
from jadapy.orthogonalization import normalize, orthogonalize, orthonormalize
from jadapy.correction_equation import solve_correction_equation, solve_generalized_correction_equation
from jadapy.utils import dot, norm
from jadapy.NumPyInterface import NumPyInterface

def _prec(x, *args):
    return x

def jdqr(A, num=5, target=Target.SmallestMagnitude, tol=1e-8, lock_tol=None, M=None, prec=None,
         maxit=1000, subspace_dimensions=(20, 40), initial_subspace=None, arithmetic='real',
         return_eigenvectors=False, return_subspace=False,
         interface=None):

    if arithmetic not in ['real', 'complex', 'r', 'c']:
        raise ValueError("argument must be 'real', or 'complex'")

    if not prec:
        prec = _prec

    if not lock_tol:
        lock_tol = tol * 1e2

    solver_tolerance = 1.0

    n = A.shape[0]

    subspace_dimensions = (min(subspace_dimensions[0], n // 2), min(subspace_dimensions[1], n))

    it = 1
    k = 0 # Number of eigenvalues found
    m = 0 # Size of the search subspace
    nev = 1 # Amount of eigenvalues currently converging

    alpha = None
    evs = None
    sort_target = target

    dtype = A.dtype
    if interface:
        dtype = interface.dtype

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
    # Preconditioned Q
    Y = interface.vector(num + extra)
    H = numpy.zeros((num + extra, num + extra), dtype)

    MQ = Q
    if M is not None:
        MQ = interface.vector(num + extra)

    # Orthonormal search subspace
    V = interface.vector(subspace_dimensions[1])
    # AV = A*V without orthogonalization
    AV = interface.vector(subspace_dimensions[1])

    # MV = M*V without orthogonalization
    MV = None
    if M is not None:
        MV = interface.vector(subspace_dimensions[1])

    # Residual vector
    r = interface.vector(1 + extra)

    # Low-dimensional projection: VAV = V'*A*V
    VAV = numpy.zeros((subspace_dimensions[1], subspace_dimensions[1]), dtype)

    while k < num and it <= maxit:
        if it == 1:
            if initial_subspace is not None:
                nev = min(initial_subspace.shape[1], subspace_dimensions[1])
                V[:, 0:nev] = initial_subspace[:, 0:nev]
            else:
                V[:, 0] = interface.random()
                normalize(V[:, 0], M=M)
        else:
            solver_maxit = 100
            sigma = evs[0]

            # Build an initial search subspace in an inexpensive way
            # and as close to the target as possible
            if m < subspace_dimensions[0]:
                solver_tolerance = 0.5
                solver_maxit = 10
                if target != 0.0:
                    sigma = target

            if M is not None:
                V[:, m:m+nev] = solve_generalized_correction_equation(
                    A, M, prec, MQ[:, 0:k+nev], Q[:, 0:k+nev], Y[:, 0:k+nev], H[0:k+nev, 0:k+nev],
                    sigma, 1.0, r[:, 0:nev], solver_tolerance, solver_maxit, interface)
            else:
                V[:, m:m+nev] = solve_correction_equation(
                    A, prec, Q[:, 0:k+nev], Y[:, 0:k+nev], H[0:k+nev, 0:k+nev],
                    sigma, r[:, 0:nev], solver_tolerance, solver_maxit, interface)

            orthonormalize(V[:, 0:m], V[:, m:m+nev], M=M, MV=None if MV is None else MV[:, 0:m], interface=interface)

        AV[:, m:m+nev] = A @ V[:, m:m+nev]
        if M is not None:
            MV[:, m:m+nev] = M @ V[:, m:m+nev]

        # Update VAV = V' * A * V
        for i in range(m):
            VAV[i, m:m+nev] = dot(V[:, i], AV[:, m:m+nev])
            VAV[m:m+nev, i] = dot(V[:, m:m+nev], AV[:, i])
        VAV[m:m+nev, m:m+nev] = dot(V[:, m:m+nev], AV[:, m:m+nev])

        m += nev

        [S, U] = schur(VAV[0:m, 0:m])

        found = True
        while found:
            [S, U] = schur_sort(S, U, sort_target)

            nev = 1
            if dtype != ctype and S.shape[0] > 1 and abs(S[1, 0]) > 0.0:
                # Complex eigenvalue in real arithmetic
                nev = 2

            alpha = S[0:nev, 0:nev]

            evcond = norm(alpha)

            Q[:, k:k+nev] = V[:, 0:m] @ U[:, 0:nev]
            Y[:, k:k+nev] = prec(Q[:, k:k+nev], alpha)

            if M is not None:
                MQ[:, k:k+nev] = MV[:, 0:m] @ U[:, 0:nev]

            for i in range(k):
                H[i, k:k+nev] = dot(MQ[:, i], Y[:, k:k+nev])
                H[k:k+nev, i] = dot(MQ[:, k:k+nev], Y[:, i])
            H[k:k+nev, k:k+nev] = dot(MQ[:, k:k+nev], Y[:, k:k+nev])

            r[:, 0:nev] = A @ Q[:, k:k+nev] - MQ[:, k:k+nev] @ alpha
            orthogonalize(MQ[:, 0:k+nev], r[:, 0:nev], M=None, MV=Q[:, 0:k+nev])

            rnorm = norm(r[:, 0:nev]) / evcond

            evs = scipy.linalg.eigvals(alpha)
            ev_est = evs[0]
            print("Step: %4d, subspace dimension: %3d, eigenvalue estimate: %13.6e + %13.6ei, residual norm: %e" %
                  (it, m, ev_est.real, ev_est.imag, rnorm), flush=True)

            if rnorm <= lock_tol:
                sort_target = ev_est

            # Store converged Ritz pairs
            if rnorm <= tol and m > nev:
                # Compute R so we can compute the eigenvectors
                if return_eigenvectors:
                    if k > 0:
                        AQ = AV[:, 0:m] @ U[:, 0:nev]
                        for i in range(k):
                            R[i, k:k+nev] = dot(Q[:, i], AQ)
                    R[k:k+nev, k:k+nev] = alpha

                # Store the converged eigenvalues
                for i in range(nev):
                    print("Found an eigenvalue:", evs[i], flush=True)

                    aconv[k] = evs[i]
                    k += 1

                if k >= num:
                    break

                # Reset the iterative solver tolerance
                solver_tolerance = 1.0

                # Unlock the target
                sort_target = target

                # Remove the eigenvalue from the search space
                V[:, 0:m-nev] = V[:, 0:m] @ U[:, nev:m]
                AV[:, 0:m-nev] = AV[:, 0:m] @ U[:, nev:m]

                if M is not None:
                    MV[:, 0:m-nev] = MV[:, 0:m] @ U[:, nev:m]

                VAV[0:m-nev, 0:m-nev] = S[nev:m, nev:m]

                S = VAV[0:m-nev, 0:m-nev]

                U = numpy.identity(m-nev, dtype)

                m -= nev
            else:
                found = False

        solver_tolerance = max(solver_tolerance / 2, tol / 100)

        if m >= min(subspace_dimensions[1], n - k) and k < num:
            # Maximum search space dimension has been reached.
            new_m = min(subspace_dimensions[0], n - k)

            print("Shrinking the search space from %d to %d" % (m, new_m), flush=True)

            V[:, 0:new_m] = V[:, 0:m] @ U[:, 0:new_m]
            AV[:, 0:new_m] = AV[:, 0:m] @ U[:, 0:new_m]

            if M is not None:
                MV[:, 0:new_m] = MV[:, 0:m] @ U[:, 0:new_m]

            VAV[0:new_m, 0:new_m] = S[0:new_m, 0:new_m]

            m = new_m
        elif m + nev - 1 >= min(subspace_dimensions[1], n - k):
            # Only one extra vector fits in the search space.
            nev = 1

        it += 1

    if return_eigenvectors:
        evs, v = scipy.linalg.eig(R[0:k, 0:k], left=False, right=True)

        if ctype == dtype:
            if return_subspace:
                return evs, Q[:, 0:k] @ v, Q[:, 0:k]
            return evs, Q[:, 0:k] @ v

        i = 0
        while i < k:
            Y[:, i] = Q[:, 0:k] @ v[:, i].real
            if evs[i].imag:
                Y[:, i+1] = Q[:, 0:k] @ v[:, i].imag
                i += 1
            i += 1
        if return_subspace:
            return evs, Y[:, 0:k], Q[:, 0:k]
        return evs, Y[:, 0:k]

    if return_subspace:
        return aconv[0:num], Q[:, 0:k]

    return aconv[0:num]
