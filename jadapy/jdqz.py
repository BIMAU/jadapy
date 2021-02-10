import numpy
import scipy

from math import sqrt

from jadapy import Target
from jadapy.generalized_schur import generalized_schur, generalized_schur_sort
from jadapy.orthogonalization import orthogonalize, orthonormalize
from jadapy.correction_equation import solve_generalized_correction_equation
from jadapy.utils import dot, norm
from jadapy.NumPyInterface import NumPyInterface

def _prec(x):
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
         interface=None):

    if arithmetic not in ['real', 'complex', 'r', 'c']:
        raise ValueError("argument must be 'real', or 'complex'")

    if not prec:
        prec = _prec

    solver_tolerance = 1.0

    n = A.shape[0]

    _subspace_dimensions = (min(subspace_dimensions[0], n // 2), min(subspace_dimensions[1], n))

    it = 1
    k = 0 # Number of eigenvalues found
    m = 0 # Size of the search subspace
    nev = 1 # Amount of eigenvalues currently converging

    alpha = None
    beta = None

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

    aconv = numpy.zeros(num + extra, ctype)
    bconv = numpy.zeros(num + extra, dtype)

    # Generalized Schur vectors
    Q = interface.vector(num + extra)
    Z = interface.vector(num + extra)
    QZ = numpy.zeros((num + extra, num + extra), dtype)

    # Orthonormal search subspace
    V = interface.vector(_subspace_dimensions[1])
    # Orthonormal test subspace
    W = interface.vector(_subspace_dimensions[1])
    # Preconditioned orthonormal search subspace
    Y = interface.vector(_subspace_dimensions[1])
    # AV = A*V without orthogonalization
    AV = interface.vector(_subspace_dimensions[1])
    # BV = B*V without orthogonalization
    BV = interface.vector(_subspace_dimensions[1])

    # Low-dimensional projections: WAV = W'*A*V, WBV = W'*B*V
    WAV = numpy.zeros((_subspace_dimensions[1], _subspace_dimensions[1]), dtype)
    WBV = numpy.zeros((_subspace_dimensions[1], _subspace_dimensions[1]), dtype)

    while k < num and it <= maxit:
        solver_tolerance /= 2

        if it == 1:
            V[:, 0] = interface.random()
        else:
            V[:, m:m+nev] = solve_generalized_correction_equation(
                A, B, prec, Q[:, 0:k+nev], Z[:, 0:k+nev], Y[:, 0:k+nev], QZ[0:k+nev, 0:k+nev],
                evs[0, 0], evs[1, 0], r, solver_tolerance, interface)

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

            Y[:, k:k+nev] = prec(Z[:, k:k+nev])

            for i in range(k):
                QZ[i, k:k+nev] = dot(Q[:, i], Y[:, k:k+nev])
                QZ[k:k+nev, i] = dot(Q[:, k:k+nev], Y[:, i])
            QZ[k:k+nev, k:k+nev] = dot(Q[:, k:k+nev], Y[:, k:k+nev])

            r = A @ Q[:, k:k+nev] @ beta - B @ Q[:, k:k+nev] @ alpha
            orthogonalize(Z[:, 0:k+nev], r)

            rnorm = norm(r) / evcond

            evs = scipy.linalg.eigvals(alpha, beta, homogeneous_eigvals=True)
            ev_est = evs[0, 0] / evs[1, 0]
            print("Step: %4d, subspace dimension: %3d, eigenvalue estimate: %13.6e + %13.6ei, residual norm: %e" % (it, m, ev_est.real, ev_est.imag, rnorm))

            # Store converged Petrov pairs
            if rnorm <= tol:
                for i in range(nev):
                    print("Found an eigenvalue:", evs[0, i] / evs[1, i])

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

        if m >= min(_subspace_dimensions[1], n - k):
            # Maximum search space dimension has been reached.
            new_m = min(_subspace_dimensions[0], n - k)

            print("Shrinking the search space from %d to %d" % (m, new_m))

            V[:, 0:new_m] = V[:, 0:m] @ UR[:, 0:new_m]
            AV[:, 0:new_m] = AV[:, 0:m] @ UR[:, 0:new_m]
            BV[:, 0:new_m] = BV[:, 0:m] @ UR[:, 0:new_m]
            W[:, 0:new_m] = W[:, 0:m] @ UL[:, 0:new_m]

            WAV[0:new_m, 0:new_m] = S[0:new_m, 0:new_m]
            WBV[0:new_m, 0:new_m] = T[0:new_m, 0:new_m]

            m = new_m
        elif m + nev - 1 >= min(_subspace_dimensions[1], n - k):
            # Only one extra vector fits in the search space.
            nev = 1
            r = r[:, 0:1]

        it += 1

    return aconv[0:num], bconv[0:num]
