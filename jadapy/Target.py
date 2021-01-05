class _TargetType:
    @staticmethod
    def __abs__():
        return 0.0

    @staticmethod
    def __complex__():
        return 0.0

    @staticmethod
    def __neg__():
        return 0.0

SmallestMagnitude = _TargetType()
LargestMagnitude = _TargetType()
SmallestRealPart = _TargetType()
LargestRealPart = _TargetType()
SmallestImaginaryPart = _TargetType()
LargestImaginaryPart = _TargetType()

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
