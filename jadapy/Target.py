from numpy import complex128

class Target(complex128):
    def __new__(cls, *args, **kwargs):
        return complex128.__new__(cls, *args, **kwargs)


SmallestMagnitude = Target(0.0)
LargestMagnitude = Target(0.0)
SmallestRealPart = Target(0.0)
LargestRealPart = Target(0.0)
SmallestImaginaryPart = Target(0.0)
LargestImaginaryPart = Target(0.0)
