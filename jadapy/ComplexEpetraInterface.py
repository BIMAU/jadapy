import numpy

from PyTrilinos import Epetra

from jadapy import EpetraInterface

class ComplexVector:

    def __init__(self, *args, **kwargs):
        self.real = None
        self.imag = None

        if len(args) > 1 and isinstance(args[0], EpetraInterface.Vector) and isinstance(args[1], EpetraInterface.Vector):
            self.real = args[0]
            self.imag = args[1]
        elif len(args) > 0 and isinstance(args[0], ComplexVector):
            self.real = EpetraInterface.Vector(args[0].real, *args[1:], **kwargs)
            self.imag = EpetraInterface.Vector(args[0].imag, *args[1:], **kwargs)
        elif len(args) > 1 and isinstance(args[1], ComplexVector):
            self.real = EpetraInterface.Vector(args[0], args[1].real, *args[1:], **kwargs)
            self.imag = EpetraInterface.Vector(args[0], args[1].imag, *args[1:], **kwargs)
        else:
            self.real = EpetraInterface.Vector(*args, **kwargs)
            self.imag = EpetraInterface.Vector(*args, **kwargs)

        self.shape = self.real.shape
        self.dtype = numpy.dtype(self.real.dtype.char.upper())

    def __getitem__(self, key):
        real = self.real[key]
        if real is None:
            return None

        return ComplexVector(real, self.imag[key])

    def __setitem__(self, key, val):
        self.real[key] = val.real
        self.imag[key] = val.imag

    def dot(self, x):
        local_map = Epetra.LocalMap(self.shape[1], 0, self.Comm())

        tmp_real = Epetra.MultiVector(local_map, x.shape[1])
        tmp_real.Multiply('T', 'N', 1.0, self.real, x.real, 0.0)
        tmp_real.Multiply('T', 'N', 1.0, self.imag, x.imag, 1.0)

        tmp_imag = Epetra.MultiVector(local_map, x.shape[1])
        tmp_imag.Multiply('T', 'N', 1.0, self.real, x.imag, 0.0)
        tmp_imag.Multiply('T', 'N', -1.0, self.imag, x.real, 1.0)

        if self.shape[1] == 1 or x.shape[1] == 1:
            # Numpy expects a 1D array in this case
            return tmp_real.array.copy().flatten() + tmp_imag.array.copy().flatten() * 1j
        return tmp_real.array.copy().T + tmp_imag.array.copy().T * 1j

    def random(self):
        self.real.random()
        self.imag.random()

    def copy(self):
        return ComplexVector(self)

    def Comm(self):
        return self.real.Comm()

    def conj(self):
        return ComplexVector(self.real, -self.imag)

    def __mul__(self, x):
        tmp = ComplexVector(self)
        tmp *= x
        return tmp

    def __add__(self, x):
        tmp = ComplexVector(self)
        tmp += x
        return tmp

    def __sub__(self, x):
        tmp = ComplexVector(self)
        tmp -= x
        return tmp

    def __matmul__(self, x):
        if isinstance(x, numpy.ndarray):
            local_map = Epetra.LocalMap(x.shape[0], 0, self.Comm())
            x = ComplexVector(EpetraInterface.Vector(local_map, x.T.real), EpetraInterface.Vector(local_map, x.T.imag))

        tmp = ComplexVector(self.real.Map(), x.shape[1])

        tmp.real.Multiply('N', 'N', 1.0, self.real, x.real, 0.0)
        tmp.real.Multiply('N', 'N', -1.0, self.imag, x.imag, 1.0)

        tmp.imag.Multiply('N', 'N', 1.0, self.real, x.imag, 0.0)
        tmp.imag.Multiply('N', 'N', 1.0, self.imag, x.real, 1.0)

        return tmp

    def __neg__(self):
        return self * -1.0

    def __imul__(self, x):
        if isinstance(x, numpy.ndarray):
            assert len(x.shape) == 1 and x.shape[0] == 1
            x = x[0]

        if x == 0.0:
            self.real.PutScalar(0.0)
            self.imag.PutScalar(0.0)
            return self

        if x.imag == 0.0:
            self.real.Scale(x.real)
            self.imag.Scale(x.real)
            return self

        tmp = ComplexVector(self)
        self.real.Update(-x.imag, tmp.imag, x.real)
        self.imag.Update(x.imag, tmp.real, x.real)
        return self

    def __iadd__(self, x):
        self.real += x.real
        self.imag += x.imag
        return self

    def __isub__(self, x):
        self.real -= x.real
        self.imag -= x.imag
        return self

    def __itruediv__(self, x):
        self *= 1.0 / x
        return self

    def Multiply(self, scalar_ab, a, b, scalar_self):
        ret = 0
        ret += self.real.Multiply(scalar_ab, a, b.real, scalar_self)
        ret += self.imag.Multiply(scalar_ab, a, b.imag, scalar_self)
        return ret

class CrsMatrix(Epetra.CrsMatrix):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dtype = numpy.dtype('d')
        self.shape = [self.NumGlobalRows(), self.NumGlobalCols()]

    def __matmul__(self, x):
        tmp = ComplexVector(self.RangeMap(), x.shape[1])
        self.Apply(x.real, tmp.real)
        self.Apply(x.imag, tmp.imag)
        return tmp

class Operator(Epetra.Operator):

    def __init__(self, op):
        super().__init__()

        self.op = op

    def Apply(self, x, y):
        assert x.NumVectors() == 2
        # Create a view here because this is an Epetra.MultiVector.
        x = EpetraInterface.Vector(Epetra.View, x, 0, x.NumVectors())
        y = EpetraInterface.Vector(Epetra.View, y, 0, y.NumVectors())

        x = ComplexVector(x[:, 0], x[:, 1])
        z = self.op.matvec(x)
        ret = y[:, 0].Update(1.0, z.real, 0.0)
        ret += y[:, 1].Update(1.0, z.imag, 0.0)
        return ret

    def ApplyInverse(self, x, y):
        # Just copy the vector. This is used when no preconditioning
        # is required in the solver itself.
        return y.Update(1.0, x, 0.0)

    def Comm(self):
        return self.op.A.Comm()

    def OperatorDomainMap(self):
        return self.op.A.OperatorDomainMap()

    def OperatorRangeMap(self):
        return self.op.A.OperatorRangeMap()

    def HasNormInf(self):
        return False

    def UseTranspose(self):
        return False

    def Label(self):
        return 'EpetraInterfaceOperator'

class ComplexEpetraInterface:

    def __init__(self, map):
        self.map = map
        self.dtype = numpy.dtype('D')

    def vector(self, k=None):
        if k:
            return ComplexVector(self.map, k)
        else:
            return ComplexVector(self.map, 1)

    def random(self):
        tmp = self.vector()
        tmp.random()
        return tmp

    def solve(self, op, rhs, tol, maxit):
        print('Warning: this is not meant for actual applications')
        return rhs
