import sys
import numpy
import scipy

from math import sqrt

from jadapy import Target
from jadapy.generalized_schur import generalized_schur, generalized_schur_sort
from jadapy.orthogonalization import orthogonalize, orthonormalize
from jadapy.correction_equation import solve_generalized_correction_equation
from jadapy.utils import dot, norm
from jadapy.NumPyInterface import NumPyInterface

def _prec(x, *args):
    return x

def _set_testspace(testspace, target, alpha, beta, dtype, ctype):
    gamma = sqrt(1 + abs(target) ** 2)

    if testspace == 'Harmonic Petrov':
        nu = 1 / gamma
        mu = -target / gamma
    elif testspace == 'Petrov' or (testspace == 'Variable Petrov' and alpha is None):
        nu = target.conj() / gamma
        mu = 1 / gamma
    elif testspace == 'Variable Petrov':
        nu = alpha.T.conj()
        mu = beta.T.conj()
    else:
        raise Exception('Invalid testspace value')

    if not isinstance(mu, numpy.ndarray):
        if dtype != ctype:
            nu = numpy.array([[nu.real, nu.imag], [-nu.imag, nu.real]])
            mu = numpy.array([[mu.real, mu.imag], [-mu.imag, mu.real]])
        else:
            nu = numpy.array([[nu]])
            mu = numpy.array([[mu]])

    return nu, mu

def jdqz(A, B, num=5, target=Target.SmallestMagnitude, tol=1e-8, prec=None,
         maxit=1000, subspace_dimensions=(20, 40), arithmetic='real', testspace='Harmonic Petrov',
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
    beta = None
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

    # Generalized eigenvalues
    aconv = numpy.zeros(num + extra, ctype)
    bconv = numpy.zeros(num + extra, dtype)

    # Generalized Schur matrices
    RA = numpy.zeros((num + extra, num + extra), dtype)
    RB = numpy.zeros((num + extra, num + extra), dtype)

    # Generalized Schur vectors
    Q = interface.vector(num + extra)
    Z = interface.vector(num + extra)
    QZ = numpy.zeros((num + extra, num + extra), dtype)

    # Orthonormal search subspace
    V = interface.vector(subspace_dimensions[1])
    # Orthonormal test subspace
    W = interface.vector(subspace_dimensions[1])
    # Preconditioned orthonormal search subspace
    Y = interface.vector(subspace_dimensions[1])
    # AV = A*V without orthogonalization
    AV = interface.vector(subspace_dimensions[1])
    # BV = B*V without orthogonalization
    BV = interface.vector(subspace_dimensions[1])

    # Residual vector
    r = interface.vector(1 + extra)

    # Low-dimensional projections: WAV = W'*A*V, WBV = W'*B*V
    WAV = numpy.zeros((subspace_dimensions[1], subspace_dimensions[1]), dtype)
    WBV = numpy.zeros((subspace_dimensions[1], subspace_dimensions[1]), dtype)

    while k < num and it <= maxit:
        if it == 1:
            V[:, 0] = interface.random()
        else:
            solver_maxit = 100
            sigma_a = evs[0, 0]
            sigma_b = evs[1, 0]

            # Build an initial search subspace in an inexpensive way
            # and as close to the target as possible
            if m < subspace_dimensions[0]:
                solver_tolerance = 0.5
                solver_maxit = 1
                if target != 0.0:
                    sigma_a = target
                    sigma_b = 1.0

            V[:, m:m+nev] = solve_generalized_correction_equation(
                A, B, prec, Q[:, 0:k+nev], Z[:, 0:k+nev], Y[:, 0:k+nev], QZ[0:k+nev, 0:k+nev],
                sigma_a, sigma_b, r[:, 0:nev], solver_tolerance, solver_maxit, interface)

        orthonormalize(V[:, 0:m], V[:, m:m+nev])

        AV[:, m:m+nev] = A @ V[:, m:m+nev]
        BV[:, m:m+nev] = B @ V[:, m:m+nev]

        nu, mu = _set_testspace(testspace, target, alpha, beta, dtype, ctype)
        W[:, m:m+nev] = AV[:, m:m+nev] @ nu[0:nev, 0:nev] + BV[:, m:m+nev] @ mu[0:nev, 0:nev]

        orthogonalize(Z[:, 0:k], W[:, m:m+nev])
        orthonormalize(W[:, 0:m], W[:, m:m+nev])

        # Update WAV = W' * A * V
        for i in range(m):
            WAV[i, m:m+nev] = dot(W[:, i], AV[:, m:m+nev])
            WAV[m:m+nev, i] = dot(W[:, m:m+nev], AV[:, i])
        WAV[m:m+nev, m:m+nev] = dot(W[:, m:m+nev], AV[:, m:m+nev])

        # Update WBV = W' * B * V
        for i in range(m):
            WBV[i, m:m+nev] = dot(W[:, i], BV[:, m:m+nev])
            WBV[m:m+nev, i] = dot(W[:, m:m+nev], BV[:, i])
        WBV[m:m+nev, m:m+nev] = dot(W[:, m:m+nev], BV[:, m:m+nev])

        m += nev

        [S, T, UL, UR] = generalized_schur(WAV[0:m, 0:m], WBV[0:m, 0:m])

        found = True
        while found:
            [S, T, UL, UR] = generalized_schur_sort(S, T, UL, UR, target)

            nev = 1
            if dtype != ctype and S.shape[0] > 1 and abs(S[1, 0]) > 0.0:
                # Complex eigenvalue in real arithmetic
                nev = 2

            alpha = S[0:nev, 0:nev]
            beta = T[0:nev, 0:nev]

            evcond = sqrt(norm(alpha) ** 2 + norm(beta) ** 2)

            Q[:, k:k+nev] = V[:, 0:m] @ UR[:, 0:nev]
            orthonormalize(Q[:, 0:k], Q[:, k:k+nev])

            Z[:, k:k+nev] = W[:, 0:m] @ UL[:, 0:nev]
            orthonormalize(Z[:, 0:k], Z[:, k:k+nev])

            Y[:, k:k+nev] = prec(Z[:, k:k+nev], alpha, beta)

            for i in range(k):
                QZ[i, k:k+nev] = dot(Q[:, i], Y[:, k:k+nev])
                QZ[k:k+nev, i] = dot(Q[:, k:k+nev], Y[:, i])
            QZ[k:k+nev, k:k+nev] = dot(Q[:, k:k+nev], Y[:, k:k+nev])

            r[:, 0:nev] = A @ Q[:, k:k+nev] @ beta - B @ Q[:, k:k+nev] @ alpha
            orthogonalize(Z[:, 0:k+nev], r[:, 0:nev])

            rnorm = norm(r[:, 0:nev]) / evcond

            evs = scipy.linalg.eigvals(alpha, beta, homogeneous_eigvals=True)
            ev_est = evs[0, 0] / evs[1, 0]
            print("Step: %4d, subspace dimension: %3d, eigenvalue estimate: %13.6e + %13.6ei, residual norm: %e" %
                  (it, m, ev_est.real, ev_est.imag, rnorm))
            sys.stdout.flush()

            # Store converged Petrov pairs
            if rnorm <= tol:
                # Compute RA and RB so we can compute the eigenvectors
                if return_eigenvectors:
                    for i in range(k):
                        RA[i, k:k+nev] = dot(Z[:, i], A @ Q[:, k:k+nev])
                        RB[i, k:k+nev] = dot(Z[:, i], B @ Q[:, k:k+nev])

                    RA[k:k+nev, k:k+nev] = alpha
                    RB[k:k+nev, k:k+nev] = beta

                # Store the converged eigenvalues
                for i in range(nev):
                    print("Found an eigenvalue:", evs[0, i] / evs[1, i])
                    sys.stdout.flush()

                    aconv[k] = evs[0, i]
                    bconv[k] = evs[1, i].real
                    k += 1

                if k >= num:
                    break

                # Reset the iterative solver tolerance
                solver_tolerance = 1.0

                # Remove the eigenvalue from the search space
                V[:, 0:m-nev] = V[:, 0:m] @ UR[:, nev:m]
                AV[:, 0:m-nev] = AV[:, 0:m] @ UR[:, nev:m]
                BV[:, 0:m-nev] = BV[:, 0:m] @ UR[:, nev:m]
                W[:, 0:m-nev] = W[:, 0:m] @ UL[:, nev:m]

                WAV[0:m-nev, 0:m-nev] = S[nev:m, nev:m]
                WBV[0:m-nev, 0:m-nev] = T[nev:m, nev:m]

                S = WAV[0:m-nev, 0:m-nev]
                T = WBV[0:m-nev, 0:m-nev]

                UL = numpy.identity(m-nev, dtype)
                UR = numpy.identity(m-nev, dtype)

                m -= nev
            else:
                found = False

        solver_tolerance = max(solver_tolerance / 2, tol)

        if m >= min(subspace_dimensions[1], n - k) and k < num:
            # Maximum search space dimension has been reached.
            new_m = min(subspace_dimensions[0], n - k)

            print("Shrinking the search space from %d to %d" % (m, new_m))
            sys.stdout.flush()

            V[:, 0:new_m] = V[:, 0:m] @ UR[:, 0:new_m]
            AV[:, 0:new_m] = AV[:, 0:m] @ UR[:, 0:new_m]
            BV[:, 0:new_m] = BV[:, 0:m] @ UR[:, 0:new_m]
            W[:, 0:new_m] = W[:, 0:m] @ UL[:, 0:new_m]

            WAV[0:new_m, 0:new_m] = S[0:new_m, 0:new_m]
            WBV[0:new_m, 0:new_m] = T[0:new_m, 0:new_m]

            m = new_m
        elif m + nev - 1 >= min(subspace_dimensions[1], n - k):
            # Only one extra vector fits in the search space.
            nev = 1

        it += 1

    if return_eigenvectors:
        evs, v = scipy.linalg.eig(RA[0:k, 0:k], RB[0:k, 0:k], left=False, right=True, homogeneous_eigvals=True)

        if ctype == dtype:
            return evs[0], evs[1], Q[:, 0:k] @ v

        i = 0
        while i < k:
            Z[:, i] = Q[:, 0:k] @ v[:, i].real
            if evs[0][i].imag:
                Z[:, i+1] = Q[:, 0:k] @ v[:, i].imag
                i += 1
            i += 1
        return evs[0], evs[1], Z

    return aconv[0:num], bconv[0:num]
