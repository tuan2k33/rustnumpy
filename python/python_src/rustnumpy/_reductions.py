import builtins as _b
import warnings

from . import _core
from ._core import asarray, ndarray
from ._ufunc import _NoValue, _write_out

_SIZE_TYPES = (int,)


def _axes(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    raw = axis if isinstance(axis, tuple) else (axis,)
    out = []
    for a in raw:
        if isinstance(a, bool) or not hasattr(a, "__index__"):
            raise TypeError("'%s' object cannot be interpreted as an integer" % type(a).__name__)
        a = a.__index__()
        if a < -ndim or a >= ndim:
            if ndim == 0 and a in (0, -1):
                continue
            raise ValueError("axis %d is out of bounds for array of dimension %d" % (a, ndim))
        out.append(a % ndim if ndim else 0)
    if len(set(out)) != len(out):
        raise ValueError("duplicate value in 'axis'")
    return tuple(out)


def _merge(a, axes):
    nd = a.ndim
    keep = [i for i in range(nd) if i not in axes]
    perm = keep + list(axes)
    t = _core.transpose(a, tuple(perm)) if perm != list(range(nd)) else a
    lead = tuple(a.shape[i] for i in keep)
    n = 1
    for i in axes:
        n *= a.shape[i]
    return t.reshape(lead + (n,)), lead


def _kept_shape(shape, axes):
    return tuple(1 if i in axes else s for i, s in enumerate(shape))


def _deliver(res, out, keepdims, shape, axes, casting="same_kind", name="reduce"):
    if keepdims or out is not None:
        res = asarray(res)
    if keepdims:
        res = asarray(res).reshape(_kept_shape(shape, axes))
    if out is not None:
        return _write_out(res, out, True, casting, name)
    return res


def _mask(where, shape):
    return _core.broadcast_to(asarray(where, "bool"), tuple(shape))


def _native_reduce(native, a, axis):
    axes = _axes(axis, a.ndim)
    if axis is None or (a.ndim == 0):
        return native(a) if a.ndim else native(a), axes
    res = a
    for ax in sorted(axes, reverse=True):
        res = native(res, ax)
    return res, axes


def sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=_NoValue, where=True):
    a = asarray(a)
    if dtype is not None:
        a = a.astype(dtype)
    shape = tuple(a.shape)
    if where is not True:
        a = _core.where(_mask(where, shape), a, asarray(0, a.dtype))
    res, axes = _native_reduce(_core.sum, a, axis)
    if initial is not _NoValue:
        res = _core.add(res, initial)
    if dtype is not None and asarray(res).dtype != asarray(0, dtype).dtype:
        res = asarray(res).astype(dtype)
    return _deliver(res, out, keepdims, shape, axes, name="sum")


def prod(a, axis=None, dtype=None, out=None, keepdims=False, initial=_NoValue, where=True):
    a = asarray(a)
    if dtype is not None:
        a = a.astype(dtype)
    shape = tuple(a.shape)
    if where is not True:
        a = _core.where(_mask(where, shape), a, asarray(1, a.dtype))
    res, axes = _native_reduce(_core.prod, a, axis)
    if initial is not _NoValue:
        res = _core.multiply(res, initial)
    if dtype is not None and asarray(res).dtype != asarray(0, dtype).dtype:
        res = asarray(res).astype(dtype)
    return _deliver(res, out, keepdims, shape, axes, name="prod")


def _extreme(name, native, combine, a, axis, out, keepdims, initial, where):
    a = asarray(a)
    shape = tuple(a.shape)
    axes = _axes(axis, a.ndim)
    if where is not True:
        if initial is _NoValue:
            raise ValueError("reduction operation '%s' does not have an identity, so to use a where mask one has to specify 'initial'" % name)
        a = _core.where(_mask(where, shape), a, asarray(initial, a.dtype))
    if a.size == 0 and initial is _NoValue and _b.any(a.shape[i] == 0 for i in axes):
        raise ValueError("zero-size array to reduction operation %s which has no identity" % name)
    if a.size == 0 and initial is not _NoValue:
        lead = tuple(s for i, s in enumerate(shape) if i not in axes)
        res = _core.full(lead, initial, _core.result_type(a.dtype, initial))
    else:
        res, axes = _native_reduce(native, a, axis)
        if initial is not _NoValue:
            res = combine(res, initial)
    return _deliver(res, out, keepdims, shape, axes, name=name)


def max(a, axis=None, out=None, keepdims=False, initial=_NoValue, where=True):
    return _extreme("maximum", _core.max, _core.maximum, a, axis, out, keepdims, initial, where)


def min(a, axis=None, out=None, keepdims=False, initial=_NoValue, where=True):
    return _extreme("minimum", _core.min, _core.minimum, a, axis, out, keepdims, initial, where)


amax = max
amin = min


def any(a, axis=None, out=None, keepdims=False, *, where=True):
    a = asarray(a)
    shape = tuple(a.shape)
    if where is not True:
        a = _core.where(_mask(where, shape), a.astype("bool"), False)
    res, axes = _native_reduce(_core.any, a, axis)
    return _deliver(res, out, keepdims, shape, axes, name="any")


def all(a, axis=None, out=None, keepdims=False, *, where=True):
    a = asarray(a)
    shape = tuple(a.shape)
    if where is not True:
        a = _core.where(_mask(where, shape), a.astype("bool"), True)
    res, axes = _native_reduce(_core.all, a, axis)
    return _deliver(res, out, keepdims, shape, axes, name="all")


def count_nonzero(a, axis=None, *, keepdims=False):
    a = asarray(a)
    if axis is None and not keepdims:
        return _core.count_nonzero(a)
    return sum(a.astype("bool"), axis=axis, dtype="int64", keepdims=keepdims)


def ptp(a, axis=None, out=None, keepdims=False):
    a = asarray(a)
    hi = max(a, axis=axis, keepdims=keepdims)
    lo = min(a, axis=axis, keepdims=keepdims)
    res = _core.subtract(hi, lo) if a.dtype.name != "bool" else _core.logical_xor(hi, lo)
    if out is not None:
        return _write_out(res, out, True, "same_kind", "ptp")
    return res


def _count(shape, axes, where):
    if where is True:
        n = 1
        for i in axes:
            n *= shape[i]
        return n
    return sum(_mask(where, shape), axis=tuple(axes) if axes else None, dtype="int64", keepdims=True)


def _float_result_dtype(a, dtype):
    if dtype is not None:
        return _core.dtype(dtype)
    k = a.dtype.kind
    return _core.dtype("float64") if k in "biu" else a.dtype


def mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    a = asarray(a)
    shape = tuple(a.shape)
    axes = _axes(axis, a.ndim)
    if dtype is not None:
        a = a.astype(dtype)
    if where is not True or a.dtype.kind == "c":
        mask = None if where is True else _mask(where, shape)
        data = a if mask is None else _core.where(mask, a, asarray(0, a.dtype))
        if a.dtype.kind in "biu":
            data = data.astype("float64")
        total = sum(data, axis=axis, keepdims=True)
        cnt = _count(shape, axes, where)
        if data.dtype.kind in "fc":
            cnt = asarray(cnt).astype(data.dtype) if isinstance(cnt, ndarray) else cnt
        res = asarray(_core.divide(total, cnt))
        res = res.reshape(tuple(s for i, s in enumerate(shape) if i not in axes)) if not keepdims else res
        return _deliver(res, out, False, shape, axes, name="mean")
    if a.dtype.name == "float16":
        pass
    if axis is None or a.ndim == 0:
        res = _core.mean(a)
    else:
        merged, _ = _merge(a, axes)
        res = _core.mean(merged, -1)
    return _deliver(res, out, keepdims, shape, axes, name="mean")


def _var_composite(a, axes, ddof, where, complex_ok=True):
    shape = tuple(a.shape)
    mask = None if where is True else _mask(where, shape)
    cnt = _count(shape, axes, where)
    data = a
    if data.dtype.kind in "biu":
        data = data.astype("float64")
    if mask is not None:
        data = _core.where(mask, data, asarray(0, data.dtype))
    total = asarray(sum(data, axis=tuple(axes), keepdims=True) if axes else data)
    if data.dtype.kind in "fc" and isinstance(cnt, ndarray):
        cnt = cnt.astype(data.dtype if data.dtype.kind == "f" else data.dtype)
    avg = _core.divide(total, cnt)
    dev = _core.subtract(data, avg)
    if data.dtype.kind == "c":
        sq = _core.add(_core.multiply(_core.real(dev), _core.real(dev)), _core.multiply(_core.imag(dev), _core.imag(dev)))
    else:
        sq = _core.multiply(dev, dev)
    sq = asarray(sq)
    if mask is not None:
        sq = _core.where(mask, sq, asarray(0, sq.dtype))
    ssq = asarray(sum(sq, axis=tuple(axes), keepdims=True) if axes else sq)
    denom = _core.maximum(_core.subtract(cnt, ddof), 0)
    if ssq.dtype.kind == "f":
        denom = denom.astype(ssq.dtype) if isinstance(denom, ndarray) else asarray(denom, ssq.dtype)
    return _core.divide(ssq, denom)


def _var_std(a, axis, dtype, out, ddof, keepdims, where, take_sqrt, name):
    a = asarray(a)
    shape = tuple(a.shape)
    axes = _axes(axis, a.ndim)
    if dtype is not None:
        a = a.astype(dtype)
    if isinstance(ddof, float) and ddof != int(ddof) or where is not True or a.dtype.kind == "c" or ddof < 0:
        res = _var_composite(a, axes, ddof, where)
        if take_sqrt:
            res = _core.sqrt(res)
        res = asarray(res)
        lead = tuple(s for i, s in enumerate(shape) if i not in axes)
        if not keepdims:
            res = res.reshape(lead)
        return _deliver(res, out, False, shape, axes, name=name)
    ddof = int(ddof)
    native = _core.std if take_sqrt else _core.var
    if axis is None or a.ndim == 0:
        res = native(a, None, ddof)
    else:
        merged, _ = _merge(a, axes)
        res = native(merged, -1, ddof)
    return _deliver(res, out, keepdims, shape, axes, name=name)


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, mean=_NoValue, correction=_NoValue):
    if correction is not _NoValue:
        ddof = correction
    return _var_std(a, axis, dtype, out, ddof, keepdims, where, False, "var")


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, mean=_NoValue, correction=_NoValue):
    if correction is not _NoValue:
        ddof = correction
    return _var_std(a, axis, dtype, out, ddof, keepdims, where, True, "std")


def _lanes(a, axis):
    a = asarray(a)
    shape = tuple(a.shape)
    axes = _axes(axis, a.ndim)
    if axis is None or a.ndim == 0:
        merged, lead = a.reshape((-1,)), ()
    else:
        merged, lead = _merge(a, axes)
    return merged, lead, shape, axes


def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    a = asarray(a)
    shape = tuple(a.shape)
    axes = _axes(axis, a.ndim)
    if a.dtype.kind == "c":
        merged, lead, _, _ = _lanes(a, axis)
        s = _core.sort(merged, -1)
        n = s.shape[-1]
        if n == 0:
            res = _core.full(lead, float("nan"), a.dtype)
        else:
            lo, hi = (n - 1) // 2, n // 2
            res = _core.divide(_core.add(s[..., lo], s[..., hi]), 2)
        return _deliver(res, out, keepdims, shape, axes, name="median")
    if a.dtype.name == "bool":
        a = a.astype("uint8")
    if axis is None or a.ndim == 0:
        res = _core.median(a)
    else:
        merged, _ = _merge(a, axes)
        res = _core.median(merged, -1)
    return _deliver(res, out, keepdims, shape, axes, name="median")


def _nan_mask(a):
    return _core.isnan(a) if a.dtype.kind in "fc" else None


def _identity_replace(a, ident):
    m = _nan_mask(a)
    if m is None:
        return a, None
    return _core.where(m, asarray(ident, a.dtype), a), m


def nansum(a, axis=None, dtype=None, out=None, keepdims=False, initial=_NoValue, where=True):
    a = asarray(a)
    a, _ = _identity_replace(a, 0)
    return sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where)


def nanprod(a, axis=None, dtype=None, out=None, keepdims=False, initial=_NoValue, where=True):
    a = asarray(a)
    a, _ = _identity_replace(a, 1)
    return prod(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where)


def nanmean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    a = asarray(a)
    m = _nan_mask(a)
    if m is None:
        return mean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)
    valid = _core.logical_not(m)
    if where is not True:
        valid = _core.logical_and(valid, _mask(where, a.shape))
    return mean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=valid)


def _nan_var_std(a, axis, dtype, out, ddof, keepdims, where, take_sqrt):
    a = asarray(a)
    m = _nan_mask(a)
    if m is None:
        return _var_std(a, axis, dtype, out, ddof, keepdims, where, take_sqrt, "nanvar")
    valid = _core.logical_not(m)
    if where is not True:
        valid = _core.logical_and(valid, _mask(where, a.shape))
    res = _var_std(a, axis, dtype, None, ddof, keepdims, valid, take_sqrt, "nanvar")
    cnt = sum(valid, axis=axis, dtype="int64", keepdims=keepdims)
    res = asarray(_core.where(_core.less_equal(_core.subtract(cnt, ddof), 0), asarray(float("nan"), asarray(res).dtype), res))
    return _write_out(res, out, True, "same_kind", "nanvar") if out is not None else res


def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, mean=_NoValue, correction=_NoValue):
    if correction is not _NoValue:
        ddof = correction
    return _nan_var_std(a, axis, dtype, out, ddof, keepdims, where, False)


def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, mean=_NoValue, correction=_NoValue):
    if correction is not _NoValue:
        ddof = correction
    return _nan_var_std(a, axis, dtype, out, ddof, keepdims, where, True)


def _nan_extreme(a, axis, out, keepdims, initial, where, want_max):
    a = asarray(a)
    m = _nan_mask(a)
    name = "nanmax" if want_max else "nanmin"
    fn = max if want_max else min
    if m is None:
        return fn(a, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)
    fill = float("-inf") if want_max else float("inf")
    filled = _core.where(m, asarray(fill, a.dtype), a)
    res = fn(filled, axis=axis, keepdims=True, initial=initial, where=where)
    all_nan = all(m, axis=axis, keepdims=True)
    if bool(any(all_nan)):
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=3)
    res = _core.where(all_nan, asarray(float("nan"), a.dtype), res)
    axes = _axes(axis, a.ndim)
    if not keepdims:
        res = res.reshape(tuple(s for i, s in enumerate(a.shape) if i not in axes))
    if out is not None:
        return _write_out(res, out, True, "same_kind", name)
    return res


def nanmax(a, axis=None, out=None, keepdims=False, initial=_NoValue, where=True):
    return _nan_extreme(a, axis, out, keepdims, initial, where, True)


def nanmin(a, axis=None, out=None, keepdims=False, initial=_NoValue, where=True):
    return _nan_extreme(a, axis, out, keepdims, initial, where, False)


def _nan_arg(a, axis, out, keepdims, want_max):
    a = asarray(a)
    m = _nan_mask(a)
    fn = argmax if want_max else argmin
    if m is None:
        return fn(a, axis=axis, out=out, keepdims=keepdims)
    if bool(any(all(m, axis=axis))):
        raise ValueError("All-NaN slice encountered")
    fill = float("-inf") if want_max else float("inf")
    return fn(_core.where(m, asarray(fill, a.dtype), a), axis=axis, out=out, keepdims=keepdims)


def nanargmax(a, axis=None, out=None, *, keepdims=_NoValue):
    return _nan_arg(a, axis, out, False if keepdims is _NoValue else keepdims, True)


def nanargmin(a, axis=None, out=None, *, keepdims=_NoValue):
    return _nan_arg(a, axis, out, False if keepdims is _NoValue else keepdims, False)


def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=_NoValue):
    a = asarray(a)
    keepdims = False if keepdims is _NoValue else keepdims
    shape = tuple(a.shape)
    axes = _axes(axis, a.ndim)
    if a.dtype.kind == "c":
        raise _core.Unsupported("complex nanmedian is not supported")
    if a.dtype.name == "bool":
        a = a.astype("uint8")
    if axis is None or a.ndim == 0:
        res = _core.nanmedian(a)
    else:
        merged, _ = _merge(a, axes)
        res = _core.nanmedian(merged, -1)
    return _deliver(res, out, keepdims, shape, axes, name="nanmedian")


def _arg(native, a, axis, out, keepdims):
    a = asarray(a)
    shape = tuple(a.shape)
    if axis is not None:
        (ax,) = _axes(axis, a.ndim) if a.ndim else (0,)
        res = native(a, ax) if a.ndim else native(a)
        axes = (ax,)
    else:
        res = native(a)
        axes = tuple(range(a.ndim))
    if keepdims:
        res = asarray(res).reshape(_kept_shape(shape, axes))
    if out is not None:
        return _write_out(res, out, True, "same_kind", "argmax")
    return res


def argmax(a, axis=None, out=None, *, keepdims=False):
    return _arg(_core.argmax, a, axis, out, keepdims)


def argmin(a, axis=None, out=None, *, keepdims=False):
    return _arg(_core.argmin, a, axis, out, keepdims)


def _cumulative(native, a, axis, dtype, out):
    a = asarray(a)
    if dtype is not None:
        a = a.astype(dtype)
    if axis is None:
        res = native(a.reshape((-1,)) if a.ndim != 1 else a, 0) if a.size or a.ndim else native(a)
    else:
        (ax,) = _axes(axis, a.ndim) if a.ndim else (0,)
        res = native(a, ax)
    if dtype is not None and asarray(res).dtype != a.dtype:
        res = asarray(res).astype(a.dtype)
    return _write_out(res, out, True, "same_kind", "cumulative") if out is not None else res


def cumsum(a, axis=None, dtype=None, out=None):
    return _cumulative(_core.cumsum, a, axis, dtype, out)


def cumprod(a, axis=None, dtype=None, out=None):
    return _cumulative(_core.cumprod, a, axis, dtype, out)


def nancumsum(a, axis=None, dtype=None, out=None):
    a = asarray(a)
    a, _ = _identity_replace(a, 0)
    return cumsum(a, axis=axis, dtype=dtype, out=out)


def nancumprod(a, axis=None, dtype=None, out=None):
    a = asarray(a)
    a, _ = _identity_replace(a, 1)
    return cumprod(a, axis=axis, dtype=dtype, out=out)


def _cumulative_api(fn, x, axis, dtype, out, include_initial):
    x = asarray(x)
    if axis is None:
        if x.ndim > 1:
            raise ValueError("axis must be specified for arrays with more than one dimension")
        axis = 0
    res = fn(x, axis=axis, dtype=dtype)
    if include_initial:
        ident = 0 if fn is cumsum else 1
        shape = list(res.shape)
        shape[axis] = 1
        res = _core.concatenate([_core.full(tuple(shape), ident, res.dtype), res], axis=axis)
    return _write_out(res, out, True, "same_kind", "cumulative") if out is not None else res


def cumulative_sum(x, /, *, axis=None, dtype=None, out=None, include_initial=False):
    return _cumulative_api(cumsum, x, axis, dtype, out, include_initial)


def cumulative_prod(x, /, *, axis=None, dtype=None, out=None, include_initial=False):
    return _cumulative_api(cumprod, x, axis, dtype, out, include_initial)


def _weights_check(a, weights, axis):
    return asarray(weights)


def average(a, axis=None, weights=None, returned=False, *, keepdims=False):
    a = asarray(a)
    if weights is None:
        avg = mean(a, axis=axis, keepdims=keepdims)
        if returned:
            n = a.size // avg_size(avg) if isinstance(avg, ndarray) else a.size
            scl = _core.full(tuple(asarray(avg).shape), n, asarray(avg).dtype) if isinstance(avg, ndarray) else float(n)
            return avg, scl
        return avg
    wgt = asarray(weights)
    if a.dtype.kind in "biu" or wgt.dtype.kind in "biu":
        result_dtype = _core.result_type(a.dtype, wgt.dtype, "float64")
    else:
        result_dtype = _core.result_type(a.dtype, wgt.dtype)
    if axis is None:
        if tuple(a.shape) != tuple(wgt.shape):
            raise TypeError("Axis must be specified when shapes of a and weights differ.")
        axes = tuple(range(a.ndim))
    else:
        axes = _axes(axis, a.ndim)
        if tuple(a.shape) != tuple(wgt.shape):
            if wgt.ndim != 1:
                raise TypeError("1D weights expected when shapes of a and weights differ.")
            if len(axes) != 1:
                raise NotImplementedError("Axis must be specified when shapes of a and weights differ.")
            if wgt.shape[0] != a.shape[axes[0]]:
                raise ValueError("Length of weights not compatible with specified axis.")
            new_shape = [1] * a.ndim
            new_shape[axes[0]] = wgt.shape[0]
            wgt = wgt.reshape(tuple(new_shape))
    wgt = wgt.astype(result_dtype)
    scl = sum(wgt, axis=axis if axis is None else axes, dtype=result_dtype, keepdims=True)
    if bool(any(_core.equal(scl, 0))):
        raise ZeroDivisionError("Weights sum to zero, can't be normalized")
    prod_ = _core.multiply(a.astype(result_dtype), wgt)
    avg = _core.divide(sum(prod_, axis=axis if axis is None else axes, keepdims=True), scl)
    if not keepdims:
        lead = tuple(s for i, s in enumerate(a.shape) if i not in axes)
        avg = asarray(avg).reshape(lead)
        scl = asarray(scl)
        scl = _core.broadcast_to(scl.reshape(tuple(s for i, s in enumerate(scl.shape) if i not in axes)), lead) if lead else scl.reshape(())
    if returned:
        if tuple(asarray(scl).shape) != tuple(asarray(avg).shape):
            scl = _core.broadcast_to(asarray(scl), tuple(asarray(avg).shape)).copy()
        return avg, scl
    return avg


def avg_size(x):
    return asarray(x).size


_METHODS = ("inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf", "hazen", "weibull",
            "linear", "median_unbiased", "normal_unbiased", "lower", "higher", "midpoint", "nearest")


def _virtual_index(method, n, q):
    def general(alpha, beta):
        return n * q + (alpha + q * (1 - alpha - beta)) - 1

    if method == "linear":
        return (n - 1) * q
    if method == "inverted_cdf":
        return n * q - 1
    if method == "averaged_inverted_cdf":
        return n * q - 1
    if method == "closest_observation":
        return n * q - 0.5
    if method == "interpolated_inverted_cdf":
        return general(0, 1)
    if method == "hazen":
        return general(0.5, 0.5)
    if method == "weibull":
        return general(0, 0)
    if method == "median_unbiased":
        return general(1 / 3.0, 1 / 3.0)
    if method == "normal_unbiased":
        return general(3 / 8.0, 3 / 8.0)
    x = (n - 1) * q
    if method == "lower":
        return _core.floor(x)
    if method == "higher":
        return _core.ceil(x)
    if method == "midpoint":
        return 0.5 * (_core.floor(x) + _core.ceil(x))
    if method == "nearest":
        return _core.rint(x)
    raise ValueError("%r is not a valid method. Use one of: %s" % (method, _METHODS))


def _quantile_lanes(s, n, qs, method):
    n = asarray(n, "float64")
    nm1 = _core.subtract(n, 1)
    results = []
    for qv in qs:
        vi = asarray(_virtual_index(method, n, qv), "float64")
        if method in ("inverted_cdf", "closest_observation"):
            index = vi if method == "inverted_cdf" else vi
            previous = _core.floor(index)
            gamma = _core.subtract(index, previous)
            if method == "inverted_cdf":
                keep_previous = _core.equal(gamma, 0)
            else:
                keep_previous = _core.logical_and(_core.equal(gamma, 0), _core.equal(_core.remainder(previous, 2), 0))
            pick = _core.where(keep_previous, previous, _core.add(previous, 1))
            pick = _core.minimum(_core.maximum(pick, 0), nm1)
            results.append(_take_last(s, pick.astype("int64")))
            continue
        if method in ("lower", "higher", "nearest"):
            pick = _core.minimum(_core.maximum(vi, 0), nm1)
            results.append(_take_last(s, asarray(pick).astype("int64")))
            continue
        previous = _core.floor(vi)
        above = _core.greater_equal(vi, nm1)
        below = _core.less(vi, 0)
        prev_i = _core.where(above, nm1, _core.where(below, 0.0, previous))
        next_i = _core.where(above, nm1, _core.where(below, 0.0, _core.add(previous, 1)))
        gamma = _core.subtract(vi, previous)
        if method == "averaged_inverted_cdf":
            gamma = _core.where(_core.equal(gamma, 0), 0.5, 1.0)
        elif method == "midpoint":
            gamma = _core.where(_core.equal(_core.remainder(vi, 1), 0), 0.0, 0.5)
        lo = _take_last(s, asarray(prev_i).astype("int64"))
        hi = _take_last(s, asarray(next_i).astype("int64"))
        results.append(_lerp(lo, hi, asarray(gamma)))
    return results


def _take_last(s, idx):
    idx = asarray(idx)
    if idx.ndim == 0:
        return s[..., int(idx)]
    from ._indexing import take_along_axis

    return take_along_axis(s, idx.reshape(tuple(idx.shape) + (1,)), axis=-1).reshape(tuple(idx.shape))


def _lerp(a, b, t):
    diff = _core.subtract(b, a)
    lerp = _core.add(a, _core.multiply(diff, t))
    alt = _core.subtract(b, _core.multiply(diff, _core.subtract(1, t)))
    return _core.where(_core.greater_equal(t, 0.5), alt, lerp)


def _quantile(a, q, axis, out, method, keepdims, weights, is_percentile, nan_policy):
    a = asarray(a)
    q = asarray(q)
    q = asarray(_core.divide(q, 100)) if is_percentile else q
    qf = q.astype("float64") if q.dtype.kind != "f" else q
    if q.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")
    if q.size and (bool(any(_core.less(qf, 0))) or bool(any(_core.greater(qf, 1))) or bool(any(_core.isnan(qf)))):
        raise ValueError("Percentiles must be in the range [0, 100]" if is_percentile else "Quantiles must be in the range [0, 1]")
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")
    if a.dtype.kind in "biu":
        work = a.astype("float64")
        out_dtype = _core.dtype("float64")
    else:
        work = a
        out_dtype = a.dtype
    if weights is not None:
        return _weighted_quantile(a, qf, axis, method, keepdims, weights, out)
    shape = tuple(a.shape)
    axes = _axes(axis, a.ndim)
    merged, lead, _, _ = _lanes(work, axis)
    s = _core.sort(merged, -1)
    n_total = s.shape[-1]
    if n_total == 0:
        raise IndexError("index -1 is out of bounds for axis 0 with size 0")
    if nan_policy == "propagate":
        counts = _core.full(lead, n_total, "int64") if lead else asarray(n_total, "int64")
        if a.dtype.kind == "f":
            has_nan = _core.isnan(s[..., -1])
        else:
            has_nan = None
    else:
        valid = _core.logical_not(_core.isnan(s)) if a.dtype.kind == "f" else _core.ones(tuple(s.shape), "bool")
        counts = sum(valid, axis=-1, dtype="int64")
        has_nan = None
        if a.dtype.kind == "f":
            empty = _core.equal(counts, 0)
            if bool(any(empty)):
                warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=3)
    q_shape = tuple(qf.shape)
    qs = qf.reshape((-1,)).tolist() if qf.ndim else [float(qf)]
    if nan_policy == "omit":
        counts_safe = _core.maximum(counts, 1)
    else:
        counts_safe = counts
    per_q = _quantile_lanes(s, counts_safe, qs, method)
    per_q = [asarray(r).astype(out_dtype) for r in per_q]
    if nan_policy == "propagate" and has_nan is not None:
        per_q = [asarray(_core.where(has_nan, asarray(float("nan"), out_dtype), r)) for r in per_q]
    if nan_policy == "omit" and a.dtype.kind == "f":
        per_q = [asarray(_core.where(_core.equal(counts, 0), asarray(float("nan"), out_dtype), r)) for r in per_q]
    if qf.ndim == 0:
        res = per_q[0]
    else:
        stacked = _core.stack(per_q, axis=0)
        res = stacked.reshape(q_shape + tuple(lead))
    if keepdims:
        res = asarray(res).reshape(q_shape + _kept_shape(shape, axes))
    if out is not None:
        return _write_out(res, out, True, "same_kind", "percentile")
    return res


def _weighted_quantile(a, qf, axis, method, keepdims, weights, out):
    if method != "inverted_cdf":
        raise ValueError("Only method 'inverted_cdf' supports weights. Got: %s." % method)
    shape = tuple(a.shape)
    axes = _axes(axis, a.ndim)
    w = asarray(weights)
    if tuple(w.shape) != shape:
        if w.ndim != 1 or len(axes) != 1 or w.shape[0] != shape[axes[0]]:
            raise TypeError("Axis must be specified when shapes of a and weights differ.")
        new = [1] * a.ndim
        new[axes[0]] = w.shape[0]
        w = _core.broadcast_to(w.reshape(tuple(new)), shape)
    if bool(any(_core.less(w, 0))):
        raise ValueError("Weights must be non-negative.")
    work = a.astype("float64") if a.dtype.kind in "biu" else a
    merged, lead = _merge(work, axes) if axes else (work, ())
    wm, _ = _merge(w.astype("float64"), axes) if axes else (w, ())
    order = _core.argsort(merged, -1)
    from ._indexing import take_along_axis

    s = take_along_axis(merged, order, axis=-1)
    ws = take_along_axis(wm, order, axis=-1)
    cdf = _core.cumsum(ws, -1)
    total = cdf[..., -1:]
    cdf = _core.divide(cdf, total)
    qs = qf.reshape((-1,)).tolist() if qf.ndim else [float(qf)]
    outs = []
    n = s.shape[-1]
    for qv in qs:
        idx = sum(_core.less(cdf, qv), axis=-1, dtype="int64")
        idx = _core.minimum(idx, n - 1)
        outs.append(_take_last(s, idx))
    res = outs[0] if qf.ndim == 0 else _core.stack(outs, axis=0).reshape(tuple(qf.shape) + tuple(lead))
    if keepdims:
        res = asarray(res).reshape(tuple(qf.shape) + _kept_shape(shape, axes))
    if out is not None:
        return _write_out(res, out, True, "same_kind", "percentile")
    return res


def percentile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, *, weights=None, interpolation=None):
    return _quantile(a, q, axis, out, method, keepdims, weights, True, "propagate")


def quantile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, *, weights=None, interpolation=None):
    return _quantile(a, q, axis, out, method, keepdims, weights, False, "propagate")


def nanpercentile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=_NoValue, *, weights=None, interpolation=None):
    return _quantile(a, q, axis, out, method, False if keepdims is _NoValue else keepdims, weights, True, "omit")


def nanquantile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=_NoValue, *, weights=None, interpolation=None):
    return _quantile(a, q, axis, out, method, False if keepdims is _NoValue else keepdims, weights, False, "omit")
