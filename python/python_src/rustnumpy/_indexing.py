from . import _core
from ._core import asarray, ndarray
from ._ufunc import _NoValue, _write_out


def _norm_axis(axis, ndim):
    if isinstance(axis, bool) or not hasattr(axis, "__index__"):
        raise TypeError("'%s' object cannot be interpreted as an integer" % type(axis).__name__)
    axis = axis.__index__()
    if axis < -ndim or axis >= ndim:
        raise ValueError("axis %d is out of bounds for array of dimension %d" % (axis, ndim))
    return axis % ndim if ndim else 0


def _check_int_indices(idx, what="indices"):
    if idx.dtype.kind not in "iu" and idx.dtype.name != "bool":
        raise TypeError("Cannot cast array data from dtype('%s') to dtype('int64') according to the rule 'safe'" % idx.dtype.name)
    return idx.astype("int64") if idx.dtype.name != "int64" else idx


def _apply_mode(idx, n, mode, axis_desc="axis 0"):
    if mode == "raise":
        bad = _core.logical_or(_core.greater_equal(idx, n), _core.less(idx, -n))
        if bool(_core.any(bad)):
            first = int(idx[bad][0]) if idx.ndim else int(idx)
            raise IndexError("index %d is out of bounds for axis %s with size %d" % (first, axis_desc, n))
        return _core.where(_core.less(idx, 0), _core.add(idx, n), idx)
    if mode == "wrap":
        return _core.remainder(idx, n) if n else idx
    if mode == "clip":
        return _core.clip(idx, 0, n - 1) if n else idx
    raise ValueError("clipmode must be one of 'clip', 'raise', or 'wrap' (got %r)" % (mode,))


def take(a, indices, axis=None, out=None, mode="raise"):
    a = asarray(a)
    idx = _check_int_indices(asarray(indices))
    if axis is None:
        a = a.reshape((-1,))
        axis = 0
    else:
        axis = _norm_axis(axis, a.ndim)
    n = a.shape[axis]
    if n == 0 and idx.size:
        raise IndexError("cannot do a non-empty take from an empty axes.")
    idx = _apply_mode(idx, n, mode, str(axis))
    res = a[(slice(None),) * axis + (idx,)]
    if out is not None:
        return _write_out(res, out, True, "same_kind" if mode == "raise" else "unsafe", "take")
    return res


def _unravel_flat(flat, shape):
    idx = []
    rem = flat
    for s in reversed(shape):
        idx.append(_core.remainder(rem, s) if s else _core.zeros_like(rem))
        rem = _core.floor_divide(rem, s) if s else rem
    return tuple(reversed(idx))


def unravel_index(indices, shape, order="C"):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(int(s) for s in shape)
    idx = asarray(indices)
    if idx.dtype.kind not in "iu":
        raise TypeError("only int indices permitted")
    total = 1
    for s in shape:
        total *= s
    if idx.size and (bool(_core.any(_core.less(idx, 0))) or bool(_core.any(_core.greater_equal(idx, total)))):
        raise ValueError("index %d is out of bounds for array with size %d" % (int(_core.max(idx)) if bool(_core.any(_core.greater_equal(idx, total))) else int(_core.min(idx)), total))
    flat = idx.astype("int64")
    if order == "F":
        return tuple(reversed(_unravel_flat(flat, tuple(reversed(shape)))))
    if order != "C":
        raise ValueError("only 'C' or 'F' order is permitted")
    return _unravel_flat(flat, shape)


def ravel_multi_index(multi_index, dims, mode="raise", order="C"):
    dims = tuple(dims) if not isinstance(dims, int) else (dims,)
    arrs = [asarray(m) for m in multi_index]
    if len(arrs) != len(dims):
        raise ValueError("parameter multi_index must be a sequence of length %d" % len(dims))
    modes = mode if isinstance(mode, (tuple, list)) else (mode,) * len(dims)
    fixed = []
    for a, d, m in zip(arrs, dims, modes):
        if a.dtype.kind not in "iu":
            raise TypeError("only int indices permitted")
        a = a.astype("int64")
        if m == "raise":
            if bool(_core.any(_core.logical_or(_core.less(a, 0), _core.greater_equal(a, d)))):
                raise ValueError("invalid entry in coordinates array")
        elif m == "wrap":
            a = _core.remainder(a, d)
        elif m == "clip":
            a = _core.clip(a, 0, d - 1)
        else:
            raise ValueError("clipmode must be one of 'clip', 'raise', or 'wrap' (got %r)" % (m,))
        fixed.append(a)
    order_dims = dims if order == "C" else tuple(reversed(dims))
    order_arrs = fixed if order == "C" else list(reversed(fixed))
    res = _core.zeros_like(order_arrs[0])
    stride = 1
    for a, d in zip(reversed(order_arrs), reversed(order_dims)):
        res = _core.add(res, _core.multiply(a, stride))
        stride *= d
    return res


def nonzero(a):
    a = asarray(a)
    if a.ndim == 0:
        raise ValueError("Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead.")
    mask = _core.not_equal(a.reshape((-1,)), 0)
    flat = _core.arange(a.size)[mask]
    if a.ndim == 1:
        return (flat,)
    return _unravel_flat(flat, tuple(a.shape))


def flatnonzero(a):
    return nonzero(asarray(a).reshape((-1,)))[0]


def argwhere(a):
    a = asarray(a)
    if a.ndim == 0:
        a = a.reshape((1,))
        idx = nonzero(a)
        return _core.stack(idx, axis=1)[:0] if not bool(a) else _core.zeros((1, 0), "int64")
    idx = nonzero(a)
    if not idx or idx[0].size == 0:
        return _core.zeros((0, a.ndim), "int64")
    return _core.stack(idx, axis=1)


def where(condition, x=_NoValue, y=_NoValue):
    if x is _NoValue and y is _NoValue:
        return nonzero(condition)
    if x is _NoValue or y is _NoValue:
        raise ValueError("either both or neither of x and y should be given")
    return _core.where(condition, x, y)


def compress(condition, a, axis=None, out=None):
    cond = asarray(condition)
    if cond.ndim != 1:
        raise ValueError("condition must be a 1-d array")
    a = asarray(a)
    if axis is None:
        a = a.reshape((-1,))
        axis = 0
    else:
        axis = _norm_axis(axis, a.ndim)
    idx = flatnonzero(cond)
    if idx.size and int(_core.max(idx)) >= a.shape[axis]:
        raise IndexError("index %d is out of bounds for axis %d with size %d" % (int(_core.max(idx)), axis, a.shape[axis]))
    return take(a, idx, axis=axis, out=out)


def extract(condition, arr):
    arr = asarray(arr).reshape((-1,))
    cond = asarray(condition).reshape((-1,))
    return take(arr, flatnonzero(cond))


def put(a, ind, v, mode="raise"):
    if not isinstance(a, ndarray):
        raise TypeError("argument 1 must be numpy.ndarray, not %s" % type(a).__name__)
    idx = _check_int_indices(asarray(ind)).reshape((-1,))
    vals = asarray(v).reshape((-1,))
    if idx.size == 0:
        return None
    if vals.size == 0:
        raise ValueError("cannot put with empty values")
    n = a.size
    idx = _apply_mode(idx, n, mode, "0")
    if vals.size != idx.size:
        reps = -(-idx.size // vals.size)
        vals = _core.concatenate([vals] * reps)[: idx.size]
    a[_unravel_flat(idx, tuple(a.shape))] = vals
    return None


def putmask(a, mask, values):
    mask = asarray(mask, "bool")
    vals = asarray(values).reshape((-1,))
    if vals.size == 0:
        return None
    flatmask = _core.broadcast_to(mask, tuple(a.shape)).reshape((-1,))
    idx = _core.arange(a.size)[flatmask]
    if idx.size == 0:
        return None
    take_vals = vals[_core.remainder(idx, vals.size)]
    a[_unravel_flat(idx, tuple(a.shape))] = take_vals
    return None


def place(arr, mask, vals):
    mask = _core.broadcast_to(asarray(mask, "bool"), tuple(arr.shape)).reshape((-1,))
    idx = _core.arange(arr.size)[mask]
    vals = asarray(vals).reshape((-1,))
    if idx.size == 0:
        return None
    if vals.size == 0:
        raise ValueError("Cannot insert from an empty array!")
    src = vals[_core.remainder(_core.arange(idx.size), vals.size)]
    arr[_unravel_flat(idx, tuple(arr.shape))] = src
    return None


def copyto(dst, src, casting="same_kind", where=True):
    if not isinstance(dst, ndarray):
        raise TypeError("copyto() argument 1 must be numpy.ndarray, not %s" % type(dst).__name__)
    s = asarray(src)
    if isinstance(src, (int, float, complex, bool)):
        pass
    elif not _core.can_cast(s.dtype, dst.dtype, casting):
        raise TypeError("Cannot cast array data from dtype('%s') to dtype('%s') according to the rule '%s'" % (s.dtype.name, dst.dtype.name, casting))
    s = _core.broadcast_to(s.astype(dst.dtype), tuple(dst.shape))
    if where is True:
        dst[...] = s
    else:
        m = _core.broadcast_to(asarray(where, "bool"), tuple(dst.shape))
        dst[m] = s[m]
    return None


def _along_axis_index(arr_shape, indices, axis):
    nd = len(arr_shape)
    if nd != indices.ndim:
        raise ValueError("`indices` and `arr` must have the same number of dimensions")
    fancy = []
    for dim in range(nd):
        if dim == axis:
            fancy.append(indices)
        else:
            shape = [1] * nd
            shape[dim] = indices.shape[dim]
            fancy.append(_core.arange(arr_shape[dim]).reshape(tuple(shape)) if indices.shape[dim] == arr_shape[dim] or indices.shape[dim] == 1 else _core.arange(arr_shape[dim]).reshape(tuple(shape)))
    return tuple(fancy)


def take_along_axis(arr, indices, axis=-1):
    arr = asarray(arr)
    indices = asarray(indices)
    if indices.dtype.kind not in "iu":
        raise IndexError("`indices` must be an integer array")
    if axis is None:
        if indices.ndim != 1:
            raise ValueError("when axis=None, `indices` must have a single dimension.")
        arr = arr.reshape((-1,))
        axis = 0
    else:
        axis = _norm_axis(axis, arr.ndim)
    if arr.ndim != indices.ndim:
        raise ValueError("`indices` and `arr` must have the same number of dimensions")
    idx = _core.where(_core.less(indices, 0), _core.add(indices.astype("int64"), arr.shape[axis]), indices.astype("int64"))
    fancy = []
    for dim in range(arr.ndim):
        if dim == axis:
            fancy.append(idx)
        else:
            shape = [1] * arr.ndim
            shape[dim] = arr.shape[dim]
            fancy.append(_core.arange(arr.shape[dim]).reshape(tuple(shape)))
    bshape = _core.broadcast_arrays(*[_core.zeros(tuple(f.shape), "bool") for f in fancy])[0].shape
    if idx.shape[axis] and bool(_core.any(_core.logical_or(_core.less(idx, 0), _core.greater_equal(idx, arr.shape[axis])))):
        raise IndexError("index %d is out of bounds for axis %d with size %d" % (int(_core.max(idx)), axis, arr.shape[axis]))
    return arr[tuple(fancy)]


def put_along_axis(arr, indices, values, axis):
    indices = asarray(indices)
    if axis is None:
        if indices.ndim != 1:
            raise ValueError("when axis=None, `indices` must have a single dimension.")
        target = arr.reshape((-1,))
        axis = 0
    else:
        target = arr
        axis = _norm_axis(axis, arr.ndim)
    if target.ndim != indices.ndim:
        raise ValueError("`indices` and `arr` must have the same number of dimensions")
    idx = _core.where(_core.less(indices, 0), _core.add(indices.astype("int64"), target.shape[axis]), indices.astype("int64"))
    fancy = []
    for dim in range(target.ndim):
        if dim == axis:
            fancy.append(idx)
        else:
            shape = [1] * target.ndim
            shape[dim] = target.shape[dim]
            fancy.append(_core.arange(target.shape[dim]).reshape(tuple(shape)))
    target[tuple(fancy)] = values
    return None


def choose(a, choices, out=None, mode="raise"):
    a = asarray(a)
    if a.dtype.kind not in "iu" and a.dtype.name != "bool":
        raise TypeError("Cannot cast array data from dtype('%s') to dtype('int64') according to the rule 'safe'" % a.dtype.name)
    res = _core.choose(a.astype("int64"), choices, mode)
    return _write_out(res, out, True, "same_kind", "choose") if out is not None else res


def fill_diagonal(a, val, wrap=False):
    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    end = None
    if a.ndim == 2:
        step = a.shape[1] + 1
        if not wrap:
            end = a.shape[1] * a.shape[1]
    else:
        if not len(set(a.shape)) == 1:
            raise ValueError("All dimensions of input must be of equal length")
        step = 1 + sum(_prod(a.shape[i + 1:]) for i in range(a.ndim - 1))
    flat_idx = _core.arange(0, a.size if end is None else min(end, a.size), step)
    a[_unravel_flat(flat_idx, tuple(a.shape))] = val
    return None


def _prod(t):
    r = 1
    for x in t:
        r *= x
    return r
