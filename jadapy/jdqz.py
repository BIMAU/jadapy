import numpy
from math import sqrt

from jadapy import Target
from jadapy.generalized_schur import generalized_schur, generalized_schur_sort
from jadapy.orthogonalization import orthogonalize, orthonormalize
from jadapy.correction_equation import solve_generalized_correction_equation

REAL_DTYPES = [numpy.float32, numpy.float64]
COMPLEX_DTYPES = [numpy.complex64, numpy.complex128]
DTYPES = REAL_DTYPES + COMPLEX_DTYPES

def generate_random_dtype_array(shape, dtype):
    if dtype in COMPLEX_DTYPES:
        return (numpy.random.rand(*shape)
                + numpy.random.rand(*shape) * 1.0j).astype(dtype)
    return numpy.random.rand(*shape).astype(dtype)

def prec(x):
    return x

def dot(x, y):
    return x.T.conj() @ y

def norm(x):
    return sqrt(dot(x, x).real)

def jdqz(A, B, num=5, target=Target.SmallestMagnitude, tol=1e-8,
         maxit=1000, subspace_dimensions=[20, 40], testspace='Harmonic Petrov'):
    solver_tolerance = 1.0

    n = A.shape[0]
    it = 1
    k = 0 # Number of eigenvalues found
    m = 0 # Size of the search subspace

    dtype = A.dtype

    if testspace == 'Harmonic Petrov':
        gamma = sqrt(1 + abs(target) ** 2)
        nu = 1 / gamma
        mu = -target / gamma
    else:
        gamma = sqrt(1 + abs(target) ** 2)
        nu = conj(target) / gamma
        mu = 1 / gamma

    aconv = numpy.zeros(num, dtype)
    bconv = numpy.zeros(num, dtype)

    # Generalized Schur vectors
    Q = numpy.zeros((n, num), dtype)
    Z = numpy.zeros((n, num), dtype)
    QZ = numpy.zeros((num, num), dtype)
    # Orthonormal search subspace
    V = numpy.zeros((n, subspace_dimensions[1]), dtype)
    # Orthonormal test subspace
    W = numpy.zeros((n, subspace_dimensions[1]), dtype)
    # Preconditioned orthonormal search subspace
    Y = numpy.zeros((n, subspace_dimensions[1]), dtype)
    # AV = A*V without orthogonalization
    AV = numpy.zeros((n, subspace_dimensions[1]), dtype)
    # BV = B*V without orthogonalization
    BV = numpy.zeros((n, subspace_dimensions[1]), dtype)

    # Low-dimensional projections: WAV = W'*A*V, WBV = W'*B*V
    WAV = numpy.zeros((subspace_dimensions[1], subspace_dimensions[1]), dtype)
    WBV = numpy.zeros((subspace_dimensions[1], subspace_dimensions[1]), dtype)

    while k < num and it <= maxit:
        solver_tolerance /= 2

        if it == 1:
            V[:, 0] = generate_random_dtype_array([n], dtype)
        else:
            V[:, m] = solve_generalized_correction_equation(A, B, prec, Q[:, 0:k+1], Y[:, 0:k+1], QZ[0:k+1, 0:k+1],
                                                            alpha, beta, r, solver_tolerance)

        orthonormalize(V[:, 0:m], V[:, m])

        AV[:, m] = A @ V[:, m]
        BV[:, m] = B @ V[:, m]
        W[:, m] = nu * AV[:, m] + mu * BV[:, m]

        orthogonalize(Z[:, 0:k], W[:, m])
        orthonormalize(W[:, 0:m], W[:, m])

        # Update WAV = W' * A * V
        for i in range(m):
            WAV[i, m] = dot(W[:, i], AV[:, m])
            WAV[m, i] = dot(W[:, m], AV[:, i])
        WAV[m, m] = dot(W[:, m], AV[:, m])

        # Update WBV = W' * B * V
        for i in range(m):
            WBV[i, m] = dot(W[:, i], BV[:, m])
            WBV[m, i] = dot(W[:, m], BV[:, i])
        WBV[m, m] = dot(W[:, m], BV[:, m])

        found = True

        while found:
            [S, T, UL, UR] = generalized_schur(WAV[0:m+1, 0:m+1], WBV[0:m+1, 0:m+1])
            [S, T, UL, UR] = generalized_schur_sort(S, T, UL, UR, target)

            alpha = S[0, 0]
            beta = T[0, 0]

            Q[:, k] = V[:, 0:m+1] @ UR[:, 0]
            orthonormalize(Q[:, 0:k], Q[:, k])

            Z[:, k] = W[:, 0:m+1] @ UL[:, 0]
            orthonormalize(Z[:, 0:k], Z[:, k])

            Y[:, k] = prec(Z[:, k])

            r = beta * A @ Q[:, k] - alpha * B @ Q[:, k]
            orthogonalize(Z[:, 0:k+1], r)

            for i in range(k - 1):
                QZ[i, k] = dot(Q[:, i], Y[:, k])
                QZ[k, i] = dot(Q[:, k], Y[:, i])
            QZ[k, k] = dot(Q[:, k], Y[:, k])

            rnorm = norm(r)
            print("Residual norm in step %d: %e" % (it, rnorm))

            # Store converged Petrov num
            if rnorm <= tol:
                print("Found an eigenvalue:", alpha / beta)

                aconv[k] = alpha
                bconv[k] = beta
                k += 1

                if k == num:
                    break
 
                # Reset the iterative solver tolerance
                solver_tolerance = 1.0

                # Remove the eigenvalue from the search space
                V[:, 0:m] = V[:, 0:m+1] @ UR[:, 1:m+1]
                AV[:, 0:m] = AV[:, 0:m+1] @ UR[:, 1:m+1]
                BV[:, 0:m] = BV[:, 0:m+1] @ UR[:, 1:m+1]
                W[:, 0:m] = W[:, 0:m+1] @ UL[:, 1:m+1]

                WAV[0:m, 0:m] = S[1:m+1, 1:m+1]
                WBV[0:m, 0:m] = T[1:m+1, 1:m+1]

                m -= 1
            else:
                found = False

        m += 1

        if m == subspace_dimensions[1]:
            print("Shrinking the search space from %d to %d" % (m, subspace_dimensions[0]))

            V[:, 0:subspace_dimensions[0]] = V[:, 0:m] @ UR[:, 0:subspace_dimensions[0]]
            AV[:, 0:subspace_dimensions[0]] = AV[:, 0:m] @ UR[:, 0:subspace_dimensions[0]]
            BV[:, 0:subspace_dimensions[0]] = BV[:, 0:m] @ UR[:, 0:subspace_dimensions[0]]
            W[:, 0:subspace_dimensions[0]] = W[:, 0:m] @ UL[:, 0:subspace_dimensions[0]]

            WAV[0:subspace_dimensions[0], 0:subspace_dimensions[0]] = S[0:subspace_dimensions[0], 0:subspace_dimensions[0]]
            WBV[0:subspace_dimensions[0], 0:subspace_dimensions[0]] = T[0:subspace_dimensions[0], 0:subspace_dimensions[0]]

            m = subspace_dimensions[0]

        it += 1

    return aconv, bconv
