from . import _core
from ._core import asarray, ndarray
from ._indexing import _norm_axis, take_along_axis
from ._ufunc import _NoValue, _write_out


def _item(x):
    return x.item() if hasattr(x, "item") else x


def _seq(x):
    return isinstance(x, (list, tuple))


def ndim(a):
    return a.ndim if isinstance(a, ndarray) else asarray(a).ndim


def shape(a):
    return tuple(a.shape) if isinstance(a, ndarray) else tuple(asarray(a).shape)


def size(a, axis=None):
    a = asarray(a)
    if axis is None:
        return a.size
    return a.shape[axis]


def isscalar(element):
    return isinstance(element, (int, float, complex, bool, str, bytes)) and not isinstance(element, ndarray)


def iterable(y):
    try:
        iter(y)
    except TypeError:
        return False
    return True


def reshape(a, /, shape=None, order="C", *, newshape=None, copy=None):
    if newshape is not None:
        if shape is not None:
            raise TypeError("You cannot specify 'newshape' and 'shape' arguments at the same time.")
        shape = newshape
    a = asarray(a)
    if order not in ("C", "F", "A"):
        raise ValueError("order must be one of 'C', 'F', or 'A'")
    if order == "F":
        t = _core.transpose(a)
        dims = shape if _seq(shape) else (shape,)
        res = _core.transpose(_core.reshape(t, tuple(reversed(tuple(dims)))) if -1 not in dims else _core.reshape(t, tuple(reversed(tuple(dims)))))
        return res
    if copy is True:
        a = a.copy()
    return _core.reshape(a, shape)


def ravel(a, order="C"):
    a = asarray(a)
    if order == "F":
        return _core.ravel(_core.transpose(a))
    return _core.ravel(a)


def transpose(a, axes=None):
    return _core.transpose(asarray(a), axes)


def matrix_transpose(x, /):
    x = asarray(x)
    if x.ndim < 2:
        raise ValueError("Input array must be at least 2-dimensional, but it is %d" % x.ndim)
    return _core.swapaxes(x, -1, -2)


def moveaxis(a, source, destination):
    return _core.moveaxis(asarray(a), source, destination)


def rollaxis(a, axis, start=0):
    a = asarray(a)
    n = a.ndim
    axis = _norm_axis(axis, n)
    if start < 0:
        start += n
    if not 0 <= start < n + 1:
        raise ValueError("'start' arg requires %d <= start < %d, but %d was passed in" % (-n, n + 1, start))
    if axis < start:
        start -= 1
    if axis == start:
        return a[...]
    axes = list(range(n))
    axes.remove(axis)
    axes.insert(start, axis)
    return _core.transpose(a, tuple(axes))


def swapaxes(a, axis1, axis2):
    return _core.swapaxes(asarray(a), axis1, axis2)


def squeeze(a, axis=None):
    return _core.squeeze(asarray(a), axis)


def expand_dims(a, axis):
    return _core.expand_dims(asarray(a), axis)


def flip(m, axis=None):
    return _core.flip(asarray(m), axis)


def fliplr(m):
    m = asarray(m)
    if m.ndim < 2:
        raise ValueError("Input must be >= 2-d.")
    return _core.flip(m, 1)


def flipud(m):
    m = asarray(m)
    if m.ndim < 1:
        raise ValueError("Input must be >= 1-d.")
    return _core.flip(m, 0)


def rot90(m, k=1, axes=(0, 1)):
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")
    m = asarray(m)
    if axes[0] == axes[1] or abs(axes[0] - axes[1]) == m.ndim:
        raise ValueError("Axes must be different.")
    if axes[0] >= m.ndim or axes[0] < -m.ndim or axes[1] >= m.ndim or axes[1] < -m.ndim:
        raise ValueError("Axes={} out of range for array of ndim={}.".format(axes, m.ndim))
    k %= 4
    if k == 0:
        return m[:]
    if k == 2:
        return _core.flip(_core.flip(m, axes[0]), axes[1])
    axes_list = list(range(m.ndim))
    axes_list[axes[0]], axes_list[axes[1]] = axes_list[axes[1]], axes_list[axes[0]]
    if k == 1:
        return _core.transpose(_core.flip(m, axes[1]), tuple(axes_list))
    return _core.flip(_core.transpose(m, tuple(axes_list)), axes[1])


def roll(a, shift, axis=None):
    return _core.roll(asarray(a), shift, axis)


def repeat(a, repeats, axis=None):
    return _core.repeat(asarray(a), repeats, axis)


def broadcast_to(array, shape, subok=False):
    return _core.broadcast_to(asarray(array), shape)


def broadcast_arrays(*args, subok=False):
    return tuple(_core.broadcast_arrays(*[asarray(a) for a in args]))


def broadcast_shapes(*args):
    arrays = [_core.empty(a if isinstance(a, tuple) else (a,), "bool") for a in args]
    if not arrays:
        return ()
    return tuple(_core.broadcast_arrays(*arrays)[0].shape)


def atleast_1d(*arys):
    res = [asarray(a).reshape((1,)) if asarray(a).ndim == 0 else asarray(a) for a in arys]
    return res[0] if len(res) == 1 else tuple(res)


def atleast_2d(*arys):
    out = []
    for a in arys:
        a = asarray(a)
        if a.ndim == 0:
            a = a.reshape((1, 1))
        elif a.ndim == 1:
            a = a[None, :]
        out.append(a)
    return out[0] if len(out) == 1 else tuple(out)


def atleast_3d(*arys):
    out = []
    for a in arys:
        a = asarray(a)
        if a.ndim == 0:
            a = a.reshape((1, 1, 1))
        elif a.ndim == 1:
            a = a[None, :, None]
        elif a.ndim == 2:
            a = a[:, :, None]
        out.append(a)
    return out[0] if len(out) == 1 else tuple(out)


def _cat(arrays, axis, dtype, casting, out):
    arrays = [asarray(a) for a in arrays]
    if not arrays:
        raise ValueError("need at least one array to concatenate")
    if dtype is not None:
        dt = _core.dtype(dtype)
        for a in arrays:
            if not _core.can_cast(a.dtype, dt, casting):
                raise TypeError("Cannot cast array data from dtype('%s') to dtype('%s') according to the rule '%s'" % (a.dtype.name, dt.name, casting))
        arrays = [a.astype(dt) for a in arrays]
    res = _core.concatenate(arrays, axis)
    return _write_out(res, out, True, casting, "concatenate") if out is not None else res


def concatenate(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
    return _cat(arrays, axis, dtype, casting, out)


concat = concatenate


def stack(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
    arrays = [asarray(a) for a in arrays]
    if not arrays:
        raise ValueError("need at least one array to stack")
    if dtype is not None:
        dt = _core.dtype(dtype)
        arrays = [a.astype(dt) for a in arrays]
    res = _core.stack(arrays, axis)
    return _write_out(res, out, True, casting, "stack") if out is not None else res


def vstack(tup, *, dtype=None, casting="same_kind"):
    return _cat(atleast_2d(*[asarray(t) for t in tup]) if len(tup) != 1 else [atleast_2d(tup[0])], 0, dtype, casting, None)


def hstack(tup, *, dtype=None, casting="same_kind"):
    arrs = [atleast_1d(asarray(t)) for t in tup]
    if arrs and arrs[0].ndim == 1:
        return _cat(arrs, 0, dtype, casting, None)
    return _cat(arrs, 1, dtype, casting, None)


def dstack(tup):
    arrs = [atleast_3d(asarray(t)) for t in tup]
    return _cat(arrs, 2, None, "same_kind", None)


def column_stack(tup):
    arrays = []
    for v in tup:
        arr = asarray(v)
        if arr.ndim < 2:
            arr = arr.reshape((-1, 1)) if arr.ndim == 1 else arr.reshape((1, 1))
        arrays.append(arr)
    return _cat(arrays, 1, None, "same_kind", None)


def block(arrays):
    def depth(x):
        if isinstance(x, tuple):
            raise TypeError("arrays is a tuple. Only lists can be used to arrange blocks, and np.block does not allow implicit conversion from tuple to ndarray.")
        d = 0
        while isinstance(x, list):
            if not x:
                raise ValueError("List at arrays with index () is empty")
            x = x[0]
            d += 1
        return d

    def rec(x, level, maxdepth, top):
        if isinstance(x, list):
            parts = [rec(i, level + 1, maxdepth, top) for i in x]
            nd = max(p.ndim for p in parts)
            nd = max(nd, maxdepth)
            parts = [p.reshape((1,) * (nd - p.ndim) + tuple(p.shape)) for p in parts]
            axis = nd - maxdepth + level
            return _core.concatenate(parts, axis)
        return asarray(x)

    d = depth(arrays)
    if not isinstance(arrays, list):
        return asarray(arrays).copy()
    return rec(arrays, 0, d, True)


def _split_axis(ary, sections_or_indices, axis, equal):
    ary = asarray(ary)
    axis = _norm_axis(axis, ary.ndim)
    n = ary.shape[axis]
    if hasattr(sections_or_indices, "__len__") and not isinstance(sections_or_indices, int):
        idxs = [int(i) for i in sections_or_indices]
        bounds = [0] + idxs + [n]
    else:
        sections = int(sections_or_indices)
        if sections <= 0:
            raise ValueError("number sections must be larger than 0.")
        if equal:
            if n % sections:
                raise ValueError("array split does not result in an equal division")
            each = n // sections
            sizes = [each] * sections
        else:
            each, extras = divmod(n, sections)
            sizes = [each + 1] * extras + [each] * (sections - extras)
        bounds = [0]
        for s in sizes:
            bounds.append(bounds[-1] + s)
    out = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        out.append(ary[(slice(None),) * axis + (slice(lo, hi),)])
    return out


def array_split(ary, indices_or_sections, axis=0):
    return _split_axis(ary, indices_or_sections, axis, False)


def split(ary, indices_or_sections, axis=0):
    return _split_axis(ary, indices_or_sections, axis, True)


def hsplit(ary, indices_or_sections):
    ary = asarray(ary)
    if ary.ndim == 0:
        raise ValueError("hsplit only works on arrays of 1 or more dimensions")
    return split(ary, indices_or_sections, 1 if ary.ndim > 1 else 0)


def vsplit(ary, indices_or_sections):
    if asarray(ary).ndim < 2:
        raise ValueError("vsplit only works on arrays of 2 or more dimensions")
    return split(ary, indices_or_sections, 0)


def dsplit(ary, indices_or_sections):
    if asarray(ary).ndim < 3:
        raise ValueError("dsplit only works on arrays of 3 or more dimensions")
    return split(ary, indices_or_sections, 2)


def unstack(x, /, *, axis=0):
    return tuple(_core.unstack(asarray(x), axis))


def tile(A, reps):
    A = asarray(A)
    reps = tuple(reps) if _seq(reps) else (reps,)
    reps = tuple(int(r) for r in reps)
    d = len(reps)
    if d < A.ndim:
        reps = (1,) * (A.ndim - d) + reps
    elif d > A.ndim:
        A = A.reshape((1,) * (d - A.ndim) + tuple(A.shape))
    nd = len(reps)
    total = A.size
    for r in reps:
        total *= r
    if total > 2**33:
        raise MemoryError("Unable to allocate a tiled array of %d elements" % total)
    res = A
    for ax in range(nd):
        if reps[ax] != 1:
            res = _core.concatenate([res] * reps[ax], ax) if reps[ax] > 0 else _core.take_zero(res, ax)
    if any(r == 0 for r in reps):
        shape = tuple(s * r for s, r in zip(A.shape, reps))
        return _core.empty(shape, A.dtype)
    return res.copy() if res is A else res


def append(arr, values, axis=None):
    arr = asarray(arr)
    if axis is None:
        arr = arr.reshape((-1,))
        values = asarray(values).reshape((-1,))
        axis = 0
    return _core.concatenate([arr, asarray(values)], axis)


def delete(arr, obj, axis=None):
    arr = asarray(arr)
    if axis is None:
        arr = arr.reshape((-1,))
        axis = 0
    else:
        axis = _norm_axis(axis, arr.ndim)
    n = arr.shape[axis]
    if isinstance(obj, slice):
        keep = _core.ones((n,), "bool")
        keep[obj] = False
    else:
        o = asarray(obj)
        if o.dtype.name == "bool":
            if o.shape != (n,):
                raise ValueError("boolean array argument obj to delete must be one dimensional and match the axis length of %d" % n)
            keep = _core.logical_not(o)
        else:
            if o.dtype.kind not in "iu":
                if o.size == 0:
                    o = o.astype("int64")
                else:
                    raise IndexError("arrays used as indices must be of integer (or boolean) type")
            o = o.reshape((-1,)).astype("int64")
            if o.size and (bool(_core.any(_core.greater_equal(o, n))) or bool(_core.any(_core.less(o, -n)))):
                bad = int(o[_core.logical_or(_core.greater_equal(o, n), _core.less(o, -n))][0])
                raise IndexError("index %d is out of bounds for axis %d with size %d" % (bad, axis, n))
            o = _core.where(_core.less(o, 0), _core.add(o, n), o)
            keep = _core.ones((n,), "bool")
            keep[o] = False
    return arr[(slice(None),) * axis + (keep,)]


def insert(arr, obj, values, axis=None):
    arr = asarray(arr)
    if axis is None:
        arr = arr.reshape((-1,))
        axis = 0
    else:
        axis = _norm_axis(axis, arr.ndim)
    n = arr.shape[axis]
    if isinstance(obj, slice):
        indices = _core.arange(*obj.indices(n))
    else:
        indices = asarray(obj)
        if indices.dtype.name == "bool":
            raise TypeError("boolean insert indices are not supported")
        if indices.dtype.kind not in "iu":
            if indices.size == 0:
                indices = indices.astype("int64")
            else:
                raise IndexError("index arrays are required to be integers")
    scalar_index = indices.ndim == 0 and not isinstance(obj, slice)
    idx = indices.reshape((-1,)).astype("int64")
    if idx.size and (bool(_core.any(_core.greater(idx, n))) or bool(_core.any(_core.less(idx, -n)))):
        bad = int(idx[_core.logical_or(_core.greater(idx, n), _core.less(idx, -n))][0])
        raise IndexError("index %d is out of bounds for axis %d with size %d" % (bad, axis, n))
    idx = _core.where(_core.less(idx, 0), _core.add(idx, n), idx)
    vals = asarray(values).astype(arr.dtype)
    if scalar_index:
        vals = _core.moveaxis(vals, 0, axis) if vals.ndim > 1 or (vals.ndim == 1 and False) else vals
        new_shape = list(arr.shape)
        new_shape[axis] = 1
        if vals.ndim == 0:
            vals = _core.broadcast_to(vals, tuple(new_shape))
        elif vals.ndim == arr.ndim:
            pass
        else:
            v = vals
            if v.ndim == arr.ndim - 1:
                vs = list(v.shape)
                vs.insert(axis, 1)
                v = v.reshape(tuple(vs))
            vals = _core.broadcast_to(v, tuple(new_shape)) if v.ndim == arr.ndim else _core.broadcast_to(v.reshape((1,) * (arr.ndim - v.ndim) + tuple(v.shape)), tuple(new_shape))
        i = int(idx[0])
        return _core.concatenate([arr[(slice(None),) * axis + (slice(0, i),)], vals, arr[(slice(None),) * axis + (slice(i, None),)]], axis)
    k = idx.size
    order = _core.argsort(idx)
    idx_sorted = idx[order]
    new_n = n + k
    new_shape = list(arr.shape)
    new_shape[axis] = new_n
    res = _core.empty(tuple(new_shape), arr.dtype)
    positions = _core.add(idx_sorted, _core.arange(k))
    mask = _core.ones((new_n,), "bool")
    mask[positions] = False
    res[(slice(None),) * axis + (mask,)] = arr
    v_shape = list(arr.shape)
    v_shape[axis] = k
    if vals.ndim == 0:
        vals = _core.broadcast_to(vals, tuple(v_shape))
    else:
        if vals.ndim == 1 and arr.ndim > 1:
            vs = [1] * arr.ndim
            vs[axis] = vals.shape[0]
            vals = vals.reshape(tuple(vs))
        vals = _core.broadcast_to(vals, tuple(v_shape))
    res[(slice(None),) * axis + (positions,)] = vals[(slice(None),) * axis + (order,)]
    return res


def resize(a, new_shape):
    a = asarray(a)
    if isinstance(new_shape, int):
        new_shape = (new_shape,)
    new_shape = tuple(new_shape)
    n = 1
    for s in new_shape:
        if s < 0:
            raise ValueError("all elements of `new_shape` must be non-negative")
        n *= s
    if a.size == 0 or n == 0:
        return _core.zeros(new_shape, a.dtype)
    flat = a.reshape((-1,))
    reps = -(-n // flat.size)
    return _core.concatenate([flat] * reps)[:n].reshape(new_shape)


def trim_zeros(filt, trim="fb", axis=None):
    filt = asarray(filt)
    trim = trim.lower()
    if axis is None:
        axes = tuple(range(filt.ndim))
    elif isinstance(axis, int):
        axes = (axis,)
    else:
        axes = tuple(axis)
    non = _core.not_equal(filt, 0)
    sl = [slice(None)] * filt.ndim
    for ax in axes:
        other = tuple(i for i in range(filt.ndim) if i != ax)
        lane = non
        for o in sorted(other, reverse=True):
            lane = _core.any(lane, o)
        idx = _core.arange(filt.shape[ax])[lane] if filt.shape[ax] else _core.arange(0)
        if idx.size == 0:
            sl[ax] = slice(0, 0)
            continue
        lo = int(idx[0]) if "f" in trim else 0
        hi = int(idx[-1]) + 1 if "b" in trim else filt.shape[ax]
        sl[ax] = slice(lo, hi)
    return filt[tuple(sl)]


def _pad_pair(pw, nd):
    pw = asarray(pw)
    if pw.dtype.kind not in "iu":
        raise TypeError("`pad_width` must be of integral type.")
    arr = _core.broadcast_to(pw.astype("int64"), (nd, 2)) if pw.ndim < 2 else pw.astype("int64")
    if pw.ndim == 1 and pw.shape[0] == 2:
        arr = _core.broadcast_to(pw.astype("int64").reshape((1, 2)), (nd, 2))
    elif pw.ndim == 1 and pw.shape[0] == 1:
        arr = _core.broadcast_to(pw.astype("int64").reshape((1, 1)), (nd, 2))
    elif pw.ndim == 0:
        arr = _core.broadcast_to(pw.astype("int64").reshape((1, 1)), (nd, 2))
    if bool(_core.any(_core.less(arr, 0))):
        raise ValueError("index can't contain negative values")
    return [(int(arr[i, 0]), int(arr[i, 1])) for i in range(nd)]


def pad(array, pad_width, mode="constant", **kwargs):
    array = asarray(array)
    nd = array.ndim
    pads = _pad_pair(pad_width, nd)
    if callable(mode):
        raise NotImplementedError("callable pad modes are not supported")
    allowed = {
        "constant": {"constant_values"}, "edge": set(), "linear_ramp": {"end_values"}, "maximum": {"stat_length"},
        "mean": {"stat_length"}, "median": {"stat_length"}, "minimum": {"stat_length"}, "reflect": {"reflect_type"},
        "symmetric": {"reflect_type"}, "wrap": set(), "empty": set(),
    }
    if mode not in allowed:
        raise ValueError("mode '%s' is not supported" % mode)
    extra = set(kwargs) - allowed[mode]
    if extra:
        raise ValueError("unsupported keyword arguments for mode '%s': %s" % (mode, extra))
    if mode != "constant" and array.size == 0 and any(p != (0, 0) for p, s in zip(pads, array.shape) if s == 0) and mode not in ("empty",):
        for (lo, hi), s in zip(pads, array.shape):
            if s == 0 and (lo or hi):
                raise ValueError("can't extend empty axis %d using modes other than 'constant' or 'empty'" % list(array.shape).index(0))
    res = array
    if mode == "constant":
        cv = kwargs.get("constant_values", 0)
        cvv = asarray(cv)
        cv_pairs = _pad_pair(cvv, nd) if cvv.ndim else [(cv, cv)] * nd
        if cvv.ndim == 0:
            cv_pairs = [(cv, cv)] * nd
        else:
            arr2 = _core.broadcast_to(cvv, (nd, 2)) if cvv.ndim < 2 else cvv
            if cvv.ndim == 1 and cvv.shape[0] == 2:
                arr2 = _core.broadcast_to(cvv.reshape((1, 2)), (nd, 2))
            cv_pairs = [(_item(arr2[i, 0]), _item(arr2[i, 1])) for i in range(nd)]
        for ax in range(nd):
            lo, hi = pads[ax]
            if lo == 0 and hi == 0:
                continue
            parts = []
            shp = list(res.shape)
            if lo:
                shp[ax] = lo
                parts.append(_core.full(tuple(shp), cv_pairs[ax][0], res.dtype))
            parts.append(res)
            if hi:
                shp[ax] = hi
                parts.append(_core.full(tuple(shp), cv_pairs[ax][1], res.dtype))
            res = _core.concatenate(parts, ax)
        return res
    if mode == "empty":
        shape = tuple(s + lo + hi for s, (lo, hi) in zip(array.shape, pads))
        out = _core.zeros(shape, array.dtype)
        out[tuple(slice(lo, lo + s) for s, (lo, _) in zip(array.shape, pads))] = array
        return out
    for ax in range(nd):
        lo, hi = pads[ax]
        if lo == 0 and hi == 0:
            continue
        res = _pad_axis(res, ax, lo, hi, mode, kwargs, array)
    return res


def _stat_len(stat_length, ax, nd, n):
    if stat_length is None:
        return (n, n)
    sl = asarray(stat_length)
    if sl.ndim == 0:
        v = int(sl)
        return (v, v)
    if sl.ndim == 1:
        if sl.shape[0] == 1:
            return (int(sl[0]), int(sl[0]))
        return (int(sl[0]), int(sl[1]))
    return (int(sl[ax, 0]), int(sl[ax, 1]))


def _pad_axis(res, ax, lo, hi, mode, kwargs, orig):
    n = res.shape[ax]
    take_ = lambda sl: res[(slice(None),) * ax + (sl,)]
    nd = res.ndim
    if mode == "edge":
        left = _core.concatenate([take_(slice(0, 1))] * lo, ax) if lo else None
        right = _core.concatenate([take_(slice(n - 1, n))] * hi, ax) if hi else None
    elif mode in ("maximum", "minimum", "mean", "median"):
        sl_lo, sl_hi = _stat_len(kwargs.get("stat_length"), ax, nd, n)
        if sl_lo <= 0 or sl_hi <= 0:
            raise ValueError("stat_length of 0 yields no value for padding")
        sl_lo, sl_hi = min(sl_lo, n), min(sl_hi, n)
        from . import _reductions as R

        def stat(seg):
            if mode == "maximum":
                v = _core.max(seg, ax, True)
            elif mode == "minimum":
                v = _core.min(seg, ax, True)
            elif mode == "mean":
                v = _core.rint(R.mean(seg, axis=ax, keepdims=True)) if res.dtype.kind in "iu" else R.mean(seg, axis=ax, keepdims=True)
            else:
                v = R.median(seg, axis=ax, keepdims=True)
                if res.dtype.kind in "iu":
                    v = _core.rint(v)
            return asarray(v).astype(res.dtype)

        left = _core.concatenate([stat(take_(slice(0, sl_lo)))] * lo, ax) if lo else None
        right = _core.concatenate([stat(take_(slice(n - sl_hi, n)))] * hi, ax) if hi else None
    elif mode == "linear_ramp":
        ev = asarray(kwargs.get("end_values", 0))
        e_lo, e_hi = (_item(ev), _item(ev)) if ev.ndim == 0 else ((_item(ev[0]), _item(ev[1])) if ev.ndim == 1 and ev.shape[0] == 2 else (_item(ev[ax, 0]), _item(ev[ax, 1])))
        wide = _core.result_type(res.dtype, "float64") if res.dtype.kind in "iu" else res.dtype

        def ramp(edge_val, end_val, length, from_left):
            start = asarray(end_val, wide)
            stop = edge_val.astype(wide)
            steps = _core.arange(length, dtype="float64")
            t = _core.linspace(0.0, 1.0, length, endpoint=False)
            shp = [1] * nd
            shp[ax] = length
            t = t.reshape(tuple(shp)).astype(wide)
            vals = _core.add(_core.multiply(_core.subtract(stop, start), t), start)
            return vals

        left = ramp(take_(slice(0, 1)), e_lo, lo, True).astype(res.dtype) if lo else None
        if hi:
            rr = ramp(take_(slice(n - 1, n)), e_hi, hi, False)
            right = _core.flip(rr, ax).astype(res.dtype)
        else:
            right = None
        if res.dtype.kind in "iu":
            left = _core.rint(left) if left is not None and False else left
    elif mode in ("reflect", "symmetric"):
        odd = kwargs.get("reflect_type", "even") == "odd"
        include_edge = mode == "symmetric"
        left = _reflect_side(res, ax, lo, True, include_edge, odd) if lo else None
        right = _reflect_side(res, ax, hi, False, include_edge, odd) if hi else None
        if lo and hi:
            pass
    elif mode == "wrap":
        left = _wrap_side(res, ax, lo, True) if lo else None
        right = _wrap_side(res, ax, hi, False) if hi else None
    else:
        raise ValueError("mode '%s' is not supported" % mode)
    parts = [p for p in (left, res, right) if p is not None]
    return _core.concatenate(parts, ax)


def _index_along(res, ax, idx):
    return res[(slice(None),) * ax + (idx,)]


def _reflect_side(res, ax, count, left, include_edge, odd):
    n = res.shape[ax]
    if n == 1 and not include_edge:
        edge = _index_along(res, ax, slice(0, 1))
        return _core.concatenate([edge] * count, ax)
    period = 2 * (n - 1) if not include_edge else 2 * n
    k = _core.arange(1, count + 1) if left else _core.arange(1, count + 1)
    if not include_edge:
        if left:
            pos = _core.subtract(0, k)
        else:
            pos = _core.add(n - 1, k)
        m = _core.remainder(pos, period)
        m = _core.where(_core.greater(m, n - 1), _core.subtract(period, m), m)
    else:
        if left:
            pos = _core.subtract(-1, _core.subtract(k, 1))
        else:
            pos = _core.add(n, _core.subtract(k, 1))
        m = _core.remainder(pos, period)
        m = _core.where(_core.greater(m, n - 1), _core.subtract(period - 1, m), m)
    if left:
        m = _core.flip(m, 0)
    vals = _index_along(res, ax, m.astype("int64"))
    if odd:
        edge_idx = 0 if left else n - 1
        edge = _index_along(res, ax, slice(edge_idx, edge_idx + 1))
        vals = _core.subtract(_core.multiply(edge, 2), vals)
    return vals


def _wrap_side(res, ax, count, left):
    n = res.shape[ax]
    k = _core.arange(1, count + 1)
    if left:
        pos = _core.remainder(_core.subtract(0, k), n)
        pos = _core.flip(pos, 0)
    else:
        pos = _core.remainder(_core.subtract(k, 1), n)
    return _index_along(res, ax, pos.astype("int64"))


def sort(a, axis=-1, kind=None, order=None, *, stable=None):
    a = asarray(a)
    if axis is None:
        a = a.reshape((-1,))
        axis = -1
    return _core.sort(a, axis)


def argsort(a, axis=-1, kind=None, order=None, *, stable=None):
    a = asarray(a)
    if axis is None:
        a = a.reshape((-1,))
        axis = -1
    return _core.argsort(a, axis)


def sort_complex(a):
    a = asarray(a)
    if a.dtype.kind != "c":
        a = a.astype("complex64" if a.dtype.itemsize <= 2 and a.dtype.kind in "iu" or a.dtype.name in ("float16", "float32", "bool") else "complex128")
    return _core.sort(a.reshape((-1,)) if a.ndim == 0 else a, -1)


def searchsorted(a, v, side="left", sorter=None):
    a = asarray(a)
    if sorter is not None:
        a = a[asarray(sorter).astype("int64")]
    return _core.searchsorted(a, v, side)


def lexsort(keys, axis=-1):
    if isinstance(keys, ndarray):
        keys = [keys[i] for i in range(keys.shape[0])]
    keys = [asarray(k) for k in keys]
    if not keys:
        raise TypeError("need sequence of keys with len > 0 in lexsort")
    first = keys[0]
    if first.ndim == 0:
        return _core.zeros((), "int64")
    axis = _norm_axis(axis, first.ndim)
    n = first.shape[axis]
    shape = list(first.shape)
    shp = [1] * first.ndim
    shp[axis] = n
    idx = _core.broadcast_to(_core.arange(n).reshape(tuple(shp)), tuple(first.shape)).copy()
    for key in keys:
        if tuple(key.shape) != tuple(first.shape):
            raise ValueError("all keys need to be the same shape")
        cur = take_along_axis(key, idx, axis)
        order = _core.argsort(cur, axis)
        idx = take_along_axis(idx, order, axis)
    return idx


def partition(a, kth, axis=-1, kind="introselect", order=None):
    a = asarray(a)
    if axis is None:
        a = a.reshape((-1,))
        axis = -1
    n = a.shape[axis]
    for k in (kth if _seq(kth) or isinstance(kth, ndarray) else [kth]):
        k = int(k)
        if k < -n or k >= n:
            raise ValueError("kth(=%d) out of bounds (%d)" % (k, n))
    return _core.sort(a, axis)


def argpartition(a, kth, axis=-1, kind="introselect", order=None):
    a = asarray(a)
    if axis is None:
        a = a.reshape((-1,))
        axis = -1
    n = a.shape[axis]
    for k in (kth if _seq(kth) or isinstance(kth, ndarray) else [kth]):
        k = int(k)
        if k < -n or k >= n:
            raise ValueError("kth(=%d) out of bounds (%d)" % (k, n))
    return _core.argsort(a, axis)


def diagonal(a, offset=0, axis1=0, axis2=1):
    a = asarray(a)
    if a.ndim < 2:
        raise ValueError("diag requires an array of at least two dimensions")
    ax1, ax2 = _norm_axis(axis1, a.ndim), _norm_axis(axis2, a.ndim)
    if ax1 == ax2:
        raise ValueError("axis1 and axis2 cannot be the same")
    if a.ndim == 2 and (ax1, ax2) == (0, 1):
        return _core.diagonal(a, offset)
    m = _core.moveaxis(a, (ax1, ax2), (-2, -1))
    n1, n2 = m.shape[-2], m.shape[-1]
    if offset >= 0:
        length = max(0, min(n1, n2 - offset))
        rows, cols = _core.arange(length), _core.arange(length) + offset
    else:
        length = max(0, min(n1 + offset, n2))
        rows, cols = _core.arange(length) - offset, _core.arange(length)
    return m[..., rows, cols]


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    a = asarray(a)
    if a.ndim < 2:
        raise ValueError("diag requires an array of at least two dimensions")
    d = diagonal(a, offset, axis1, axis2)
    from . import _reductions as R

    return R.sum(d, axis=-1, dtype=dtype, out=out)


def swap_and_ravel(a):
    return a.reshape((-1,))
