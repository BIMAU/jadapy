from numpy import float32

class Target(float32):
    def __new__(cls, *args, **kwargs):
        return float32.__new__(cls, *args, **kwargs)


SmallestMagnitude = Target(0.0)
LargestMagnitude = Target(0.0)
SmallestRealPart = Target(0.0)
LargestRealPart = Target(0.0)
SmallestImaginaryPart = Target(0.0)
LargestImaginaryPart = Target(0.0)
