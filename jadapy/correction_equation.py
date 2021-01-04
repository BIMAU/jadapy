import numpy
from scipy.sparse import linalg

def _proj(Q, Y, H, x):
    y = Q.T.conj() @ x
    y = numpy.linalg.solve(H, y)
    y = Y @ y
    return x - y

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

        self.shape = self.A.shape

    def matvec(self, x):
        y = _proj(self.Q, self.Y, self.H, x)
        y = self.beta * (self.A @ y) - self.alpha * (self.B @ y)
        y = self.prec(y)
        return _proj(self.Q, self.Y, self.H, y)

def solve_generalized_correction_equation(A, B, prec, Q, Y, H, alpha, beta, r, tolerance):
    op = generalized_linear_operator(A, B, prec, Q, Y, H, alpha, beta)
    r = _proj(Q, Y, H, r)
    r = prec(r)
    v, info = linalg.gmres(op, -r, tol=tolerance, atol=0)
    if info != 0:
        raise Exception('GMRES returned ' + str(info))
    return v
