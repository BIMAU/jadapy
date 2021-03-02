class Target:
    def __init__(self, target):
        self.target = target

    def __abs__(self):
        return abs(self.target)

    def __complex__(self):
        return self.target

    def __neg__(self):
        return -self.target

    def __call__(self, target):
        self.target = target

    def conj(self):
        try:
            return self.target.conj()
        except AttributeError:
            return self.target

SmallestMagnitude = Target(0.0)
LargestMagnitude = Target(0.0)
SmallestRealPart = Target(0.0)
LargestRealPart = Target(0.0)
SmallestImaginaryPart = Target(0.0)
LargestImaginaryPart = Target(0.0)
