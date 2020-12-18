import numpy
from math import sqrt

from jadapy import Target
from jadapy.generalized_schur import generalized_schur, generalized_schur_sort
from jadapy.orthogonalization import orthogonalize, orthonormalize
from jadapy.correction_equation import solve_generalized_correction_equation

def prec(x):
    return x

def norm(x):
    return sqrt(x.dot(x))

def view(V, begin, end):
    if end < begin:
        return V[:, 0:0]
    return V[:, begin:end]

def jdqz(A, B, pairs=5, target=Target.SmallestMagnitude, tol=1e-8,
         maxit=1000, subspace_dimensions=[20, 40], testspace='Harmonic Petrov'):
    solver_tolerance = 1.0

    n = A.shape[0]
    it = 1
    k = 0 # Number of eigenvalues found
    m = 0 # Size of the search subspace

    if testspace == 'Harmonic Petrov':
        gamma = sqrt(1 + abs(target) ** 2)
        nu = 1 / gamma
        mu = -target / gamma
    else:
        gamma = sqrt(1 + abs(target) ** 2)
        nu = conj(target) / gamma
        mu = 1 / gamma

    aconv = numpy.zeros(pairs)
    bconv = numpy.zeros(pairs)

    # Generalized Schur vectors
    Q = numpy.zeros((n, pairs))
    Z = numpy.zeros((n, pairs))
    QZ = numpy.zeros((pairs, pairs))
    # Orthonormal search subspace
    V = numpy.zeros((n, subspace_dimensions[1]))
    # Orthonormal test subspace
    W = numpy.zeros((n, subspace_dimensions[1]))
    # Preconditioned orthonormal search subspace
    Y = numpy.zeros((n, subspace_dimensions[1]))
    # AV = A*V without orthogonalization
    AV = numpy.zeros((n, subspace_dimensions[1]))
    # BV = B*V without orthogonalization
    BV = numpy.zeros((n, subspace_dimensions[1]))

    # Low-dimensional projections: WAV = W'*A*V, WBV = W'*B*V
    WAV = numpy.zeros((subspace_dimensions[1], subspace_dimensions[1]))
    WBV = numpy.zeros((subspace_dimensions[1], subspace_dimensions[1]))

    while k < pairs and it <= maxit:
        solver_tolerance /= 2

        if it == 1:
            V[:, 0] = numpy.random.rand(n)
        else:
            V[:, m] = solve_generalized_correction_equation(A, B, prec, Q[:, 0:k+1], Y[:, 0:k+1], QZ[0:k+1, 0:k+1],
                                                            alpha, beta, r, solver_tolerance)

        orthonormalize(V[:, 0:m-1], V[:, m])

        AV[:, m] = A @ V[:, m]
        BV[:, m] = B @ V[:, m]
        W[:, m] = nu * AV[:, m] + mu * BV[:, m]

        orthogonalize(view(Z, 0, k-1), W[:, m])
        orthonormalize(view(W, 0, m-1), W[:, m])

        # Update WAV = W' * A * V
        for i in range(m):
            WAV[i, m] = W[:, i].dot(AV[:, m])
            WAV[m, i] = W[:, m].dot(AV[:, i])
        WAV[m, m] = W[:, m].dot(AV[:, m])

        # Update WBV = W' * B * V
        for i in range(m):
            WBV[i, m] = W[:, i].dot(BV[:, m])
            WBV[m, i] = W[:, m].dot(BV[:, i])
        WBV[m, m] = W[:, m].dot(BV[:, m])

        found = True

        while found:
            [S, T, UL, UR] = generalized_schur(WAV[0:m+1, 0:m+1], WBV[0:m+1, 0:m+1])
            [S, T, UL, UR] = generalized_schur_sort(S, T, UL, UR, target)

            alpha = S[0, 0]
            beta = T[0, 0]

            Q[:, k] = view(V, 0, m+1) @ UR[:, 0]
            orthonormalize(view(Q, 0, k-1), Q[:, k])

            Z[:, k] = view(W, 0, m+1) @ UL[:, 0]
            orthonormalize(view(Z, 0, k-1), Z[:, k])

            Y[:, k] = prec(Z[:, k])

            r = beta * A @ Q[:, k] - alpha * B @ Q[:, k]
            orthogonalize(Z[:, 0:k+1], r)

            for i in range(k - 1):
                QZ[i, k] = Q[:, i].dot(Y[:, k])
                QZ[k, i] = Q[:, k].dot(Y[:, i])
            QZ[k, k] = Q[:, k].dot(Y[:, k])

            rnorm = norm(r)
            print("Residual norm in step %d: %e" % (it, rnorm))

            # Store converged Petrov pairs
            if rnorm <= tol:
                print("Found an eigenvalue:", alpha / beta)

                aconv[k] = alpha
                bconv[k] = beta
                k += 1

                if k == pairs:
                    break

                # Reset the iterative solver tolerance
                solver_tolerance = 1.0

                # Remove the eigenvalue from the search space
                V[:, 0:m] = view(V, 0, m+1) @ view(UR, 1, m+1)
                AV[:, 0:m] = view(AV, 0, m+1) @ view(UR, 1, m+1)
                BV[:, 0:m] = view(BV, 0, m+1) @ view(UR, 1, m+1)
                W[:, 0:m] = view(W, 0, m+1) @ view(UL, 1, m+1)

                WAV[0:m, 0:m] = S[1:m+1, 1:m+1]
                WBV[0:m, 0:m] = T[1:m+1, 1:m+1]

                m -= 1
            else:
                found = False

        m += 1

        if m == subspace_dimensions[1]:
            print("Shrinking the search space from %d to %d" % (m, subspace_dimensions[0]))

            V[:, 0:subspace_dimensions[0]] = view(V, 0, m) @ view(UR, 0, subspace_dimensions[0])
            AV[:, 0:subspace_dimensions[0]] = view(AV, 0, m) @ view(UR, 0, subspace_dimensions[0])
            BV[:, 0:subspace_dimensions[0]] = view(BV, 0, m) @ view(UR, 0, subspace_dimensions[0])
            W[:, 0:subspace_dimensions[0]] = view(W, 0, m) @ view(UL, 0, subspace_dimensions[0])

            WAV[0:subspace_dimensions[0], 0:subspace_dimensions[0]] = S[0:subspace_dimensions[0], 0:subspace_dimensions[0]]
            WBV[0:subspace_dimensions[0], 0:subspace_dimensions[0]] = T[0:subspace_dimensions[0], 0:subspace_dimensions[0]]

            m = subspace_dimensions[0]

        it += 1

    return aconv, bconv
