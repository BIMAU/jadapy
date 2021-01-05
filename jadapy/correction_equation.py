import numpy
import scipy

from scipy.sparse import linalg

class generalized_linear_operator(object):
    def __init__(self, A, B, prec, Q, Y, H, alpha, beta):
        self.A = A
        self.B = B
        self.prec = prec
        self.Q = Q
        self.Y = Y
        self.H = H
        self.alpha = alpha
        self.beta = beta

        self.dtype = self.A.dtype
        self.shape = self.A.shape

        # self.lu, self.piv = scipy.linalg.lu_factor(H)

    def matvec(self, x):
        y = self.proj(x)
        y = self.beta * (self.A @ y) - self.alpha * (self.B @ y)
        y = self.prec(y)
        return self.proj(y)

    def proj(self, x):
        y = self.Q.T.conj() @ x
        y = numpy.linalg.solve(self.H, y)
        # y = scipy.linalg.lu_solve((self.lu, self.piv), y)
        y = self.Y @ y
        return x - y

def solve_generalized_correction_equation(A, B, prec, Q, Y, H, alpha, beta, r, tolerance):
    op = generalized_linear_operator(A, B, prec, Q, Y, H, alpha, beta)
    r = prec(r)
    r = op.proj(r)
    v, info = linalg.gmres(op, -r, tol=tolerance, atol=0)
    if info != 0:
        raise Exception('GMRES returned ' + str(info))
    return v
