import numpy

from jadapy.utils import dot

class linear_operator(object):
    def __init__(self, A, prec, Q, Y, H, alpha):
        self.A = A
        self.prec = prec
        self.Q = Q
        self.Y = Y
        self.H = H
        self.alpha = alpha
        self.beta = 1.0

        self.dtype = self.Q.dtype
        self.shape = self.A.shape

        # self.lu, self.piv = scipy.linalg.lu_factor(H)

    def matvec(self, x):
        y = self.proj(x)
        y = self.A @ y - y * self.alpha
        y = self.prec(y, self.alpha)
        return self.proj(y)

    def proj(self, x):
        y = dot(self.Q, x)
        y = numpy.linalg.solve(self.H, y)
        # y = scipy.linalg.lu_solve((self.lu, self.piv), y)
        y = self.Y @ y
        return x - y

def solve_correction_equation(A, prec, Q, Y, H, alpha, r, tolerance, maxit, interface):
    op = linear_operator(A, prec, Q, Y, H, alpha)
    r = prec(r, alpha)
    r = op.proj(r)
    return interface.solve(op, -r, tolerance, maxit)

class generalized_linear_operator(object):
    def __init__(self, A, B, prec, Q, Z, Y, H, alpha, beta):
        self.A = A
        self.B = B
        self.prec = prec
        self.Q = Q
        self.Z = Z
        self.Y = Y
        self.H = H
        self.alpha = alpha
        self.beta = beta

        self.dtype = self.Q.dtype
        self.shape = self.A.shape

        # self.lu, self.piv = scipy.linalg.lu_factor(H)

    def matvec(self, x):
        y = self.proj(x)
        y = (self.A @ y) * self.beta - (self.B @ y) * self.alpha
        y = self.prec(y, self.alpha, self.beta)
        return self.proj(y)

    def proj(self, x):
        y = dot(self.Q, x)
        y = numpy.linalg.solve(self.H, y)
        # y = scipy.linalg.lu_solve((self.lu, self.piv), y)
        y = self.Y @ y
        return x - y

def solve_generalized_correction_equation(A, B, prec, Q, Z, Y, H, alpha, beta, r, tolerance, maxit, interface):
    op = generalized_linear_operator(A, B, prec, Q, Z, Y, H, alpha, beta)
    r = prec(r, alpha, beta)
    r = op.proj(r)
    return interface.solve(op, -r, tolerance, maxit)
