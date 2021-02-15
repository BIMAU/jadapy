import numpy

from PyTrilinos import Epetra
from PyTrilinos import AztecOO

class Vector(Epetra.MultiVector):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shape = [self.shape[1], self.shape[0]]

    def __getitem__(self, key):
        if isinstance(key, tuple) and key[0] == slice(None, None, None):
            if isinstance(key[1], slice):
                num = key[1].stop - key[1].start
                if num < 1:
                    return None
                return Vector(Epetra.View, self, key[1].start, num)
            return Vector(Epetra.View, self, key[1], 1)

        raise Exception('Only full vectors are supported')

    def __setitem__(self, key, val):
        if isinstance(key, tuple) and key[0] == slice(None, None, None):
            if isinstance(key[1], slice):
                tmp = Vector(Epetra.View, self, key[1].start, key[1].stop - key[1].start)
            else:
                tmp = Vector(Epetra.View, self, key[1], 1)
            tmp.Update(1.0, val, 0.0)
            return

        raise Exception('Only full vectors are supported')

    def dot(self, x):
        local_map = Epetra.LocalMap(self.shape[1], 0, self.Comm())
        tmp = Epetra.MultiVector(local_map, x.shape[1])
        tmp.Multiply('T', 'N', 1.0, self, x, 0.0)

        if self.shape[1] == 1 or x.shape[1] == 1:
            # Numpy expects a 1D array in this case
            return tmp.array.copy().flatten()
        return tmp.array.copy().T

    def conj(self):
        return self

    def __mul__(self, x):
        if isinstance(x, numpy.ndarray):
            assert len(x.shape) == 1 and x.shape[0] == 1
            x = x[0]

        tmp = Vector(self)
        tmp.Scale(x)
        return tmp

    def __matmul__(self, x):
        if isinstance(x, numpy.ndarray):
            local_map = Epetra.LocalMap(x.shape[0], 0, self.Comm())
            x = Epetra.MultiVector(local_map, x.T)

        tmp = Vector(self.Map(), x.NumVectors())
        tmp.Multiply('N', 'N', 1.0, self, x, 0.0)
        return tmp

    def __neg__(self):
        return self * -1.0

    def __imul__(self, x):
        self.Scale(x)
        return self

    def __isub__(self, x):
        self.Update(-1.0, x, 1.0)
        return self

    def __itruediv__(self, x):
        self.Scale(1.0 / x)
        return self

class CrsMatrix(Epetra.CrsMatrix):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dtype = numpy.dtype('d')
        self.shape = [self.NumGlobalRows(), self.NumGlobalCols()]

    def __matmul__(self, x):
        tmp = Vector(self.RangeMap(), x.NumVectors())
        self.Apply(x, tmp)
        return tmp

class Operator(Epetra.Operator):

    def __init__(self, op):
        super().__init__()

        self.op = op

    def Apply(self, x, y):
        # Create a view here because this is an Epetra.MultiVector.
        x = Vector(Epetra.View, x, 0, x.NumVectors())
        z = self.op.matvec(x)
        return y.Update(1.0, z, 0.0)

    def ApplyInverse(self, x, y):
        # Just copy the vector. This is used when no preconditioning
        # is required in the solver itself.
        return y.Update(1.0, x, 0.0)

    def Comm(self):
        return self.op.Q.Comm()

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

class EpetraInterface:

    def __init__(self, map):
        self.map = map

    def vector(self, k=None):
        if k:
            return Vector(self.map, k)
        else:
            return Vector(self.map, 1)

    def random(self):
        tmp = self.vector()
        tmp.Random()
        return tmp

    def solve(self, op, rhs, tol):
        epetra_op = Operator(op)
        x = Vector(rhs)
        solver = AztecOO.AztecOO(epetra_op, x, rhs)
        solver.SetParameters({"Solver": "GMRES",
                              "Precond": "None",
                              "Output": -3}) # Warnings. See az_aztec_defs.h
        info = solver.Iterate(100, tol)
        if info != 0:
            raise Exception('AztecOO returned ' + str(info))
        return x
