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

def _prec(x):
    return x

def dot(x, y):
    return x.T.conj() @ y

def norm(x):
    return sqrt(dot(x, x).real)

def jdqz(A, B, num=5, target=Target.SmallestMagnitude, tol=1e-8, prec=None,
         maxit=1000, subspace_dimensions=[20, 40], testspace='Harmonic Petrov'):

    if not prec:
        prec = _prec

    solver_tolerance = 1.0

    n = A.shape[0]

    subspace_dimensions[0] = min(subspace_dimensions[0], n // 2)
    subspace_dimensions[1] = min(subspace_dimensions[1], n)

    it = 1
    k = 0 # Number of eigenvalues found
    m = 0 # Size of the search subspace

    dtype = A.dtype
    ctype = numpy.dtype(dtype.char.upper())

    if testspace == 'Harmonic Petrov':
        gamma = sqrt(1 + abs(target) ** 2)
        nu = 1 / gamma
        mu = -target / gamma
    else:
        gamma = sqrt(1 + abs(target) ** 2)
        nu = conj(target) / gamma
        mu = 1 / gamma

    aconv = numpy.zeros(num, ctype)
    bconv = numpy.zeros(num, ctype)

    # Generalized Schur vectors
    Q = numpy.zeros((n, num), ctype)
    Z = numpy.zeros((n, num), ctype)
    QZ = numpy.zeros((num, num), ctype)
    # Orthonormal search subspace
    V = numpy.zeros((n, subspace_dimensions[1]), ctype)
    # Orthonormal test subspace
    W = numpy.zeros((n, subspace_dimensions[1]), ctype)
    # Preconditioned orthonormal search subspace
    Y = numpy.zeros((n, subspace_dimensions[1]), ctype)
    # AV = A*V without orthogonalization
    AV = numpy.zeros((n, subspace_dimensions[1]), ctype)
    # BV = B*V without orthogonalization
    BV = numpy.zeros((n, subspace_dimensions[1]), ctype)

    # Low-dimensional projections: WAV = W'*A*V, WBV = W'*B*V
    WAV = numpy.zeros((subspace_dimensions[1], subspace_dimensions[1]), ctype)
    WBV = numpy.zeros((subspace_dimensions[1], subspace_dimensions[1]), ctype)

    while k < num and it <= maxit:
        solver_tolerance /= 2

        if it == 1:
            V[:, 0] = generate_random_dtype_array([n], ctype)
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

        [S, T, UL, UR] = generalized_schur(WAV[0:m+1, 0:m+1], WBV[0:m+1, 0:m+1])

        found = True
        while found:
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

            for i in range(k):
                QZ[i, k] = dot(Q[:, i], Y[:, k])
                QZ[k, i] = dot(Q[:, k], Y[:, i])
            QZ[k, k] = dot(Q[:, k], Y[:, k])

            rnorm = norm(r)
            ev_est = alpha / beta
            print("Step: %4d, eigenvalue estimate: %13.6e + %13.6ei, residual norm: %e" % (it, ev_est.real, ev_est.imag, rnorm))

            # Store converged Petrov num
            if rnorm <= tol:
                print("Found an eigenvalue:", ev_est)

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

                S = WAV[0:m, 0:m]
                T = WBV[0:m, 0:m]

                UL = numpy.identity(m, ctype)
                UR = numpy.identity(m, ctype)

                m -= 1
            else:
                found = False

        m += 1

        if m >= min(subspace_dimensions[1], n - k):
            new_m = min(subspace_dimensions[0], n - k - 1)

            print("Shrinking the search space from %d to %d" % (m, new_m))

            V[:, 0:new_m] = V[:, 0:m] @ UR[:, 0:new_m]
            AV[:, 0:new_m] = AV[:, 0:m] @ UR[:, 0:new_m]
            BV[:, 0:new_m] = BV[:, 0:m] @ UR[:, 0:new_m]
            W[:, 0:new_m] = W[:, 0:m] @ UL[:, 0:new_m]

            WAV[0:new_m, 0:new_m] = S[0:new_m, 0:new_m]
            WBV[0:new_m, 0:new_m] = T[0:new_m, 0:new_m]

            m = new_m

        it += 1

    return aconv, bconv
