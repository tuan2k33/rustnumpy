from . import _core
from ._core import asarray, ndarray
from ._indexing import _norm_axis


def _unique_flat(a, return_index, return_inverse, return_counts, equal_nan):
    if a.dtype.kind == "c" or (a.dtype.kind == "f" and (not equal_nan or bool(_core.any(_core.isnan(a))))):
        return _unique_by_sort(a, equal_nan)
    res = _core.unique_all(a)
    return res.values, res.indices, res.inverse_indices, res.counts


def _unique_by_sort(a, equal_nan):
    n = a.size
    order = _core.argsort(a, 0)
    s = a[order]
    starts = _core.concatenate([_core.ones((min(n, 1),), "bool"), _core.not_equal(s[1:], s[:-1])])
    if equal_nan and n:
        nan = _core.isnan(s)
        if bool(_core.any(nan)):
            first_nan = int(_core.argmax(nan))
            keep = _core.arange(n) == first_nan
            starts = _core.where(nan, keep, starts)
    positions = _core.arange(n)[starts]
    group = _core.subtract(_core.cumsum(starts.astype("int64"), 0), 1)
    inverse = _core.empty((n,), "int64")
    inverse[order] = group
    counts = _core.subtract(_core.concatenate([positions[1:], _core.full((1,), n, "int64")]), positions)
    return s[positions], order[positions], inverse, counts


def _rows_unique(a, axis, return_index, return_inverse, return_counts):
    a = asarray(a)
    axis = _norm_axis(axis, a.ndim)
    moved = _core.moveaxis(a, axis, 0)
    n = moved.shape[0]
    flat = moved.reshape((n, -1)) if n else moved.reshape((0, 0))
    if n == 0 or flat.shape[1] == 0:
        keep = _core.arange(min(n, 1))
        values = moved[keep]
        inverse = _core.zeros((n,), "int64")
        counts = _core.full((int(keep.size),), n, "int64")
        return _core.moveaxis(values, 0, axis), keep, inverse, counts
    cols = [flat[:, j] for j in range(flat.shape[1])]
    from ._manip import lexsort

    order = lexsort(cols[::-1])
    sorted_rows = flat[order]
    diff = _core.any(_core.not_equal(sorted_rows[1:], sorted_rows[:-1]), 1)
    starts = _core.concatenate([_core.ones((1,), "bool"), diff])
    group = _core.subtract(_core.cumsum(starts.astype("int64"), 0), 1)
    first_pos = _core.arange(n)[starts]
    indices = order[first_pos]
    values = moved[indices]
    inverse = _core.empty((n,), "int64")
    inverse[order] = group
    counts = _core.subtract(_core.concatenate([first_pos[1:], _core.full((1,), n, "int64")]), first_pos)
    return _core.moveaxis(values, 0, axis), indices, inverse, counts


def unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None, *, equal_nan=True, sorted=True):
    ar = asarray(ar)
    if axis is not None:
        values, indices, inverse, counts = _rows_unique(ar, axis, return_index, return_inverse, return_counts)
        inv_shape = inverse
    else:
        flat = ar.reshape((-1,))
        values, indices, inverse, counts = _unique_flat(flat, return_index, return_inverse, return_counts, equal_nan)
        if not equal_nan and ar.dtype.kind in "fc":
            pass
        inv_shape = inverse.reshape(tuple(ar.shape))
    out = [values]
    if return_index:
        out.append(indices)
    if return_inverse:
        out.append(inv_shape)
    if return_counts:
        out.append(counts)
    return out[0] if len(out) == 1 else tuple(out)


def _prep_pair(ar1, ar2):
    return asarray(ar1), asarray(ar2)


def isin(element, test_elements, assume_unique=False, invert=False, *, kind=None):
    element = asarray(element)
    test = asarray(test_elements).reshape((-1,))
    shape = tuple(element.shape)
    if test.size == 0:
        res = _core.zeros(shape, "bool")
        return _core.logical_not(res) if invert else res
    common = _core.result_type(element.dtype, test.dtype)
    e = element.astype(common).reshape((-1,))
    t = _core.sort(test.astype(common), 0)
    pos = _core.minimum(_core.searchsorted(t, e, "left"), t.size - 1)
    found = _core.equal(t[pos], e)
    res = found.reshape(shape)
    return _core.logical_not(res) if invert else res


def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    ar1, ar2 = _prep_pair(ar1, ar2)
    if not assume_unique:
        if return_indices:
            u1, i1, _, _ = _unique_flat(ar1.reshape((-1,)), True, False, False, True)
            u2, i2, _, _ = _unique_flat(ar2.reshape((-1,)), True, False, False, True)
            ar1, ar2 = u1, u2
        else:
            ar1 = unique(ar1)
            ar2 = unique(ar2)
    else:
        ar1, ar2 = ar1.reshape((-1,)), ar2.reshape((-1,))
    aux = _core.concatenate([ar1, ar2])
    if return_indices:
        order = _core.argsort(aux, 0)
        aux_sorted = aux[order]
        mask = _core.equal(aux_sorted[1:], aux_sorted[:-1])
        int1d = aux_sorted[:-1][mask]
        ar1_indices = order[:-1][mask]
        ar2_indices = _core.subtract(order[1:][mask], ar1.size)
        if not assume_unique:
            ar1_indices = i1[ar1_indices]
            ar2_indices = i2[ar2_indices]
        return int1d, ar1_indices, ar2_indices
    aux = _core.sort(aux, 0)
    mask = _core.equal(aux[1:], aux[:-1])
    return aux[:-1][mask]


def union1d(ar1, ar2):
    return unique(_core.concatenate([asarray(ar1).reshape((-1,)), asarray(ar2).reshape((-1,))]))


def setdiff1d(ar1, ar2, assume_unique=False):
    ar1 = asarray(ar1)
    if assume_unique:
        ar1 = ar1.reshape((-1,))
    else:
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    return ar1[isin(ar1, ar2, assume_unique=True, invert=True)]


def setxor1d(ar1, ar2, assume_unique=False):
    if not assume_unique:
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    aux = _core.sort(_core.concatenate([asarray(ar1).reshape((-1,)), asarray(ar2).reshape((-1,))]), 0)
    if aux.size == 0:
        return aux
    flag = _core.concatenate([_core.ones((1,), "bool"), _core.not_equal(aux[1:], aux[:-1]), _core.ones((1,), "bool")])
    mask = _core.logical_and(flag[1:], flag[:-1])
    return aux[mask]


def unique_all(x):
    return _core.unique_all(asarray(x))


def unique_counts(x):
    return _core.unique_counts(asarray(x))


def unique_inverse(x):
    r = _core.unique_inverse(asarray(x))
    return r


def unique_values(x):
    return _core.unique_values(asarray(x))
