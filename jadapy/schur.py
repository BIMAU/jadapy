import numpy
import scipy

from scipy.linalg import schur
from scipy.linalg.lapack import _compute_lwork

from jadapy import Target

__all__ = ['schur', 'schur_sort']

def _is_target(target, target_type):
    try:
        return target is target_type or isinstance(target, target_type)
    except TypeError:
        return False

def _get_ev(a, i):
    if numpy.iscomplexobj(a):
        return a[i, i]

    n = a.shape[0]
    if i > 0 and a[i, i - 1] != 0:
        return scipy.linalg.eigvals(a[i-1:i+1, i-1:i+1])[1]
    elif i < n - 1 and a[i + 1, i] != 0:
        return scipy.linalg.eigvals(a[i:i+2, i:i+2])[0]
    return a[i, i]

def _select(start, end, a, target):
    idx = -1
    idx_list = range(start, end)
    if _is_target(target, Target.SmallestMagnitude):
        idx = min(idx_list, key=lambda i: abs(_get_ev(a, i)))
    elif _is_target(target, Target.LargestMagnitude):
        idx = max(idx_list, key=lambda i: abs(_get_ev(a, i)))
    elif _is_target(target, Target.SmallestRealPart):
        idx = min(idx_list, key=lambda i: _get_ev(a, i).real)
    elif _is_target(target, Target.LargestRealPart):
        idx = max(idx_list, key=lambda i: _get_ev(a, i).real)
    elif _is_target(target, Target.SmallestImaginaryPart):
        idx = min(idx_list, key=lambda i: _get_ev(a, i).imag)
    elif _is_target(target, Target.LargestImaginaryPart):
        idx = max(idx_list, key=lambda i: _get_ev(a, i).imag)
    else:
        idx = min(idx_list, key=lambda i: abs(_get_ev(a, i) - target))
    return idx

def schur_sort(a, q, target):
    n = a.shape[0]

    try:
        trexc, = scipy.linalg.get_lapack_funcs(('trexc',), (a,))
        for i in range(n):
            if i > 0 and a[i, i - 1] != 0:
                # Complex conjugate eigenpair
                continue

            idx = _select(i, n, a, target)

            if idx == i:
                continue

            result = trexc(a, q, idx, i)
            assert result[-1] == 0

            a = result[0]
            q = result[1]

        return a, q
    except ValueError:
        trsen, trsen_lwork = scipy.linalg.get_lapack_funcs(('trsen', 'trsen_lwork',), (a,))
        idx = _select(0, n, a, target)

        select = numpy.zeros(n)
        select[idx] = 1

        lwork = _compute_lwork(trsen_lwork, select, a)
        result = trsen(select, a, q, lwork=lwork)
        assert result[-1] == 0

        return result[0], result[1]
