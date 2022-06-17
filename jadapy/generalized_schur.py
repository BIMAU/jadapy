import numpy
import scipy

from jadapy import Target

_double_precision = ['i', 'l', 'd']

__all__ = ['generalized_schur', 'generalized_schur_sort']


def _datacopied(arr, original):
    """
    Strict check for `arr` not sharing any data with `original`,
    under the assumption that arr = asarray(original)
    """
    if arr is original:
        return False
    if not isinstance(original, numpy.ndarray) and hasattr(original, '__array__'):
        return False
    return arr.base is None


def generalized_schur(a, b, output='real', lwork=None, overwrite_a=False, overwrite_b=False, sort=None,
                      check_finite=True):

    if output not in ['real', 'complex', 'r', 'c']:
        raise ValueError("argument must be 'real', or 'complex'")

    if check_finite:
        a1 = numpy.asarray_chkfinite(a)
        b1 = numpy.asarray_chkfinite(b)
    else:
        a1 = numpy.asarray(a)
        b1 = numpy.asarray(b)

    if len(a1.shape) != 2 or (a1.shape[0] != a1.shape[1]) or \
       len(b1.shape) != 2 or (b1.shape[0] != b1.shape[1]):
        raise ValueError('expected square matrix')

    typ = a1.dtype.char
    if output in ['complex', 'c'] and typ not in ['F', 'D']:
        if typ in _double_precision:
            a1 = a1.astype('D')
            b1 = b1.astype('D')
            typ = 'D'
        else:
            a1 = a1.astype('F')
            b1 = b1.astype('F')
            typ = 'F'

    overwrite_a = overwrite_a or (_datacopied(a1, a))
    overwrite_b = overwrite_b or (_datacopied(b1, b))
    gges, = scipy.linalg.get_lapack_funcs(('gges',), (a1, b1,))
    if lwork is None or lwork == -1:
        # get optimal work array
        result = gges(lambda x: None, a1, b1, lwork=-1)
        lwork = result[-2][0].real.astype(numpy.int_)

    if sort is None:
        sort_t = 0
        sfunction = lambda x: None
    else:
        sort_t = 1
        if callable(sort):
            sfunction = sort
        elif sort == 'lhp':
            sfunction = lambda x: (x.real < 0.0)
        elif sort == 'rhp':
            sfunction = lambda x: (x.real >= 0.0)
        elif sort == 'iuc':
            sfunction = lambda x: (abs(x) <= 1.0)
        elif sort == 'ouc':
            sfunction = lambda x: (abs(x) > 1.0)
        else:
            raise ValueError("'sort' parameter must either be 'None', or a "
                             "callable, or one of ('lhp','rhp','iuc','ouc')")

    result = gges(sfunction, a1, b1, lwork=lwork, overwrite_a=overwrite_a, overwrite_b=overwrite_b,
                  sort_t=sort_t)

    info = result[-1]
    if info < 0:
        raise ValueError('illegal value in {}-th argument of internal gges'
                         ''.format(-info))
    elif info == a1.shape[0] + 1:
        raise scipy.linalg.LinAlgError('Eigenvalues could not be separated for reordering.')
    elif info == a1.shape[0] + 2:
        raise scipy.linalg.LinAlgError('Leading eigenvalues do not satisfy sort condition.')
    elif info > 0:
        raise scipy.linalg.LinAlgError("Schur form not found. Possibly ill-conditioned.")

    if sort_t == 0:
        return result[0], result[1], result[-4], result[-3]
    else:
        return result[0], result[1], result[-4], result[-3], result[2]

def _is_target(target, target_type):
    try:
        return target is target_type or isinstance(target, target_type)
    except TypeError:
        return False

def _get_ev(a, b, i):
    if b[i, i] == 0.0:
        return numpy.inf

    if numpy.iscomplexobj(a):
        return a[i, i] / b[i, i]

    n = a.shape[0]
    if i > 0 and a[i, i - 1] != 0:
        return scipy.linalg.eigvals(a[i-1:i+1, i-1:i+1], b[i-1:i+1, i-1:i+1])[1]
    elif i < n - 1 and a[i + 1, i] != 0:
        return scipy.linalg.eigvals(a[i:i+2, i:i+2], b[i:i+2, i:i+2])[0]
    return a[i, i] / b[i, i]

def _select(start, end, a, b, target):
    idx = -1
    idx_list = range(start, end)
    if _is_target(target, Target.SmallestMagnitude):
        idx = min(idx_list, key=lambda i: abs(_get_ev(a, b, i)))
    elif _is_target(target, Target.LargestMagnitude):
        idx = max(idx_list, key=lambda i: abs(_get_ev(a, b, i)))
    elif _is_target(target, Target.SmallestRealPart):
        idx = min(idx_list, key=lambda i: _get_ev(a, b, i).real)
    elif _is_target(target, Target.LargestRealPart):
        idx = max(idx_list, key=lambda i: _get_ev(a, b, i).real)
    elif _is_target(target, Target.SmallestImaginaryPart):
        idx = min(idx_list, key=lambda i: _get_ev(a, b, i).imag)
    elif _is_target(target, Target.LargestImaginaryPart):
        idx = max(idx_list, key=lambda i: _get_ev(a, b, i).imag)
    else:
        idx = min(idx_list, key=lambda i: abs(_get_ev(a, b, i) - target))
    return idx

def generalized_schur_sort(a, b, q, z, target):
    n = a.shape[0]

    try:
        tgexc, = scipy.linalg.get_lapack_funcs(('tgexc',), (a, b,))
        for i in range(n):
            if i > 0 and a[i, i - 1] != 0:
                # Complex conjugate eigenpair
                continue

            idx = _select(i, n, a, b, target)

            if idx == i:
                continue

            result = tgexc(a, b, q, z, idx, i)
            assert result[-1] >= 0

            a = result[0]
            b = result[1]
            q = result[2]
            z = result[3]

            if result[-1] != 0:
                break

        return a, b, q, z
    except ValueError:
        tgsen, = scipy.linalg.get_lapack_funcs(('tgsen',), (a, b,))
        idx = _select(0, n, a, b, target)

        select = numpy.zeros(n)
        select[idx] = 1

        result = tgsen(select, a, b, q, z, lwork=-1)
        assert result[-1] == 0
        lwork = result[-3][0].real.astype(numpy.int_) + 1

        result = tgsen(select, a, b, q, z, lwork=lwork)
        assert result[-1] == 0

        return result[0], result[1], result[-9], result[-8]
