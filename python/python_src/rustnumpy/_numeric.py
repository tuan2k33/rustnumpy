import math as _math
import warnings

from . import _core
from ._core import asarray, ndarray
from ._indexing import _norm_axis
from ._ufunc import _NoValue, _write_out
from . import _reductions as R


def round(a, decimals=0, out=None):
    a = asarray(a)
    if a.dtype.kind == "b":
        res = a.astype("int8") if False else a
        return res
    if a.dtype.kind in "iu":
        if decimals >= 0:
            res = a.copy()
        else:
            f = 10 ** (-decimals)
            res = _core.multiply(_core.rint(_core.divide(a.astype("float64"), f)), f).astype(a.dtype)
        return _write_out(res, out, True, "same_kind", "round") if out is not None else res
    if a.dtype.kind == "c":
        res = _core.add(round(_core.real(a), decimals), _core.multiply(round(_core.imag(a), decimals), 1j))
        return res.astype(a.dtype)
    if decimals >= 0:
        f = 10.0**decimals
        res = _core.divide(_core.rint(_core.multiply(a, f)), f)
    else:
        f = 10.0 ** (-decimals)
        res = _core.multiply(_core.rint(_core.divide(a, f)), f)
    res = res.astype(a.dtype)
    return _write_out(res, out, True, "same_kind", "round") if out is not None else res


around = round


def fix(x, out=None):
    x = asarray(x)
    res = _core.trunc(x) if x.dtype.kind in "fiub" else x
    if x.dtype.kind == "b":
        res = x
    return _write_out(res, out, True, "same_kind", "fix") if out is not None else res


def clip(a, a_min=_NoValue, a_max=_NoValue, out=None, *, min=_NoValue, max=_NoValue, **kwargs):
    if min is not _NoValue:
        if a_min is not _NoValue:
            raise TypeError("clip() got multiple values for argument 'a_min'")
        a_min = min
    if max is not _NoValue:
        if a_max is not _NoValue:
            raise TypeError("clip() got multiple values for argument 'a_max'")
        a_max = max
    lo = None if a_min is _NoValue else a_min
    hi = None if a_max is _NoValue else a_max
    a = asarray(a)
    res = a.copy() if lo is None and hi is None else a
    if lo is not None:
        res = _core.maximum(res, lo)
    if hi is not None:
        res = _core.minimum(res, hi)
    res = asarray(res)
    return _write_out(res, out, True, "same_kind", "clip") if out is not None else res


def diff(a, n=1, axis=-1, prepend=_NoValue, append=_NoValue):
    if n == 0:
        return a
    if n < 0:
        raise ValueError("order must be non-negative but got " + repr(n))
    a = asarray(a)
    nd = a.ndim
    if nd == 0:
        raise ValueError("diff requires input that is at least one dimensional")
    axis = _norm_axis(axis, nd)
    combined = []
    if prepend is not _NoValue:
        prepend = asarray(prepend)
        if prepend.ndim == 0:
            shp = list(a.shape)
            shp[axis] = 1
            prepend = _core.broadcast_to(prepend, tuple(shp))
        combined.append(prepend)
    combined.append(a)
    if append is not _NoValue:
        append = asarray(append)
        if append.ndim == 0:
            shp = list(a.shape)
            shp[axis] = 1
            append = _core.broadcast_to(append, tuple(shp))
        combined.append(append)
    if len(combined) > 1:
        a = _core.concatenate(combined, axis)
    s1 = [slice(None)] * nd
    s2 = [slice(None)] * nd
    s1[axis] = slice(1, None)
    s2[axis] = slice(None, -1)
    s1, s2 = tuple(s1), tuple(s2)
    op = _core.not_equal if a.dtype.name == "bool" else _core.subtract
    for _ in range(n):
        a = op(a[s1], a[s2])
    return a


def ediff1d(ary, to_end=None, to_begin=None):
    ary = asarray(ary).reshape((-1,))
    res = _core.subtract(ary[1:], ary[:-1]) if ary.size else ary[:0]
    parts = []
    if to_begin is not None:
        parts.append(asarray(to_begin).reshape((-1,)).astype(res.dtype))
    parts.append(res)
    if to_end is not None:
        parts.append(asarray(to_end).reshape((-1,)).astype(res.dtype))
    return _core.concatenate(parts) if len(parts) > 1 else res


def trapezoid(y, x=None, dx=1.0, axis=-1):
    y = asarray(y)
    nd = y.ndim
    axis = _norm_axis(axis, nd)
    if x is None:
        d = dx
    else:
        x = asarray(x)
        if x.ndim == 1:
            d = diff(x)
            shp = [1] * nd
            shp[axis] = d.shape[0]
            d = d.reshape(tuple(shp))
        else:
            d = diff(x, axis=axis)
    s1 = [slice(None)] * nd
    s2 = [slice(None)] * nd
    s1[axis] = slice(1, None)
    s2[axis] = slice(None, -1)
    return R.sum(_core.divide(_core.multiply(d, _core.add(y[tuple(s1)], y[tuple(s2)])), 2.0), axis=axis)


def gradient(f, *varargs, axis=None, edge_order=1):
    f = asarray(f)
    N = f.ndim
    if axis is None:
        axes = tuple(range(N))
    elif isinstance(axis, int):
        axes = (_norm_axis(axis, N),)
    else:
        axes = tuple(_norm_axis(a, N) for a in axis)
        if len(set(axes)) != len(axes):
            raise ValueError("duplicate value in 'axis'")
    len_axes = len(axes)
    n = len(varargs)
    if n == 0:
        dx = [1.0] * len_axes
    elif n == 1 and asarray(varargs[0]).ndim == 0:
        dx = list(varargs) * len_axes
    elif n == len_axes:
        dx = list(varargs)
        for i, distances in enumerate(dx):
            distances = asarray(distances)
            if distances.ndim == 0:
                dx[i] = distances
                continue
            if distances.ndim != 1:
                raise ValueError("distances must be either scalars or 1d")
            if distances.shape[0] != f.shape[axes[i]]:
                raise ValueError("when 1d, distances must match the length of the corresponding dimension")
            if distances.dtype.kind in "iu":
                distances = distances.astype("float64")
            dx[i] = diff(distances)
            if bool(R.any(_core.equal(dx[i], 0))) and False:
                pass
    else:
        raise TypeError("invalid number of arguments")
    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")
    outvals = []
    otype = f.dtype
    if otype.kind in "biu":
        otype = _core.dtype("float64")
    f = f.astype(otype) if f.dtype.kind in "biu" else f
    for ax, ddx in zip(axes, dx):
        if f.shape[ax] < edge_order + 1:
            raise ValueError("Shape of array too small to calculate a numerical gradient, at least (edge_order + 1) elements are required.")
        out = _core.empty(tuple(f.shape), otype)
        nd = f.ndim

        def sl(s):
            idx = [slice(None)] * nd
            idx[ax] = s
            return tuple(idx)

        ddx = asarray(ddx)
        uniform = ddx.ndim == 0
        if uniform:
            out[sl(slice(1, -1))] = _core.divide(_core.subtract(f[sl(slice(2, None))], f[sl(slice(None, -2))]), _core.multiply(2.0, ddx))
        else:
            shp = [1] * nd
            shp[ax] = -1
            dx1 = ddx[:-1].reshape(tuple(shp))
            dx2 = ddx[1:].reshape(tuple(shp))
            a = _core.divide(_core.negative(dx2), _core.multiply(dx1, _core.add(dx1, dx2)))
            b = _core.divide(_core.subtract(dx2, dx1), _core.multiply(dx1, dx2))
            c = _core.divide(dx1, _core.multiply(dx2, _core.add(dx1, dx2)))
            out[sl(slice(1, -1))] = _core.add(_core.add(_core.multiply(a, f[sl(slice(None, -2))]), _core.multiply(b, f[sl(slice(1, -1))])), _core.multiply(c, f[sl(slice(2, None))]))
        if edge_order == 1:
            if uniform:
                out[sl(0)] = _core.divide(_core.subtract(f[sl(1)], f[sl(0)]), ddx)
                out[sl(-1)] = _core.divide(_core.subtract(f[sl(-1)], f[sl(-2)]), ddx)
            else:
                out[sl(0)] = _core.divide(_core.subtract(f[sl(1)], f[sl(0)]), ddx[0])
                out[sl(-1)] = _core.divide(_core.subtract(f[sl(-1)], f[sl(-2)]), ddx[-1])
        else:
            if uniform:
                out[sl(0)] = _core.divide(_core.add(_core.add(_core.multiply(-1.5, f[sl(0)]), _core.multiply(2.0, f[sl(1)])), _core.multiply(-0.5, f[sl(2)])), ddx)
                out[sl(-1)] = _core.divide(_core.add(_core.add(_core.multiply(0.5, f[sl(-3)]), _core.multiply(-2.0, f[sl(-2)])), _core.multiply(1.5, f[sl(-1)])), ddx)
            else:
                dx1, dx2 = ddx[0], ddx[1]
                a = _core.negative(_core.divide(_core.add(_core.multiply(2.0, dx1), dx2), _core.multiply(dx1, _core.add(dx1, dx2))))
                b = _core.divide(_core.add(dx1, dx2), _core.multiply(dx1, dx2))
                c = _core.negative(_core.divide(dx1, _core.multiply(dx2, _core.add(dx1, dx2))))
                out[sl(0)] = _core.add(_core.add(_core.multiply(a, f[sl(0)]), _core.multiply(b, f[sl(1)])), _core.multiply(c, f[sl(2)]))
                dx1, dx2 = ddx[-2], ddx[-1]
                a = _core.divide(dx2, _core.multiply(dx1, _core.add(dx1, dx2)))
                b = _core.negative(_core.divide(_core.add(dx2, dx1), _core.multiply(dx1, dx2)))
                c = _core.divide(_core.add(_core.multiply(2.0, dx2), dx1), _core.multiply(dx2, _core.add(dx1, dx2)))
                out[sl(-1)] = _core.add(_core.add(_core.multiply(a, f[sl(-3)]), _core.multiply(b, f[sl(-2)])), _core.multiply(c, f[sl(-1)]))
        outvals.append(out)
    if len_axes == 1:
        return outvals[0]
    return tuple(outvals)


def interp(x, xp, fp, left=None, right=None, period=None):
    x = asarray(x)
    xp = asarray(xp)
    fp = asarray(fp)
    if xp.ndim != 1 or fp.ndim != 1:
        raise ValueError("object too deep for desired array" if xp.ndim > 1 else "object of too small depth for desired array")
    if xp.shape[0] != fp.shape[0]:
        raise ValueError("fp and xp are not of the same length")
    if xp.shape[0] == 0:
        raise ValueError("array of sample points is empty")
    if fp.dtype.kind == "c":
        re = interp(x, xp, _core.real(fp), None if left is None else asarray(left).real, None if right is None else asarray(right).real, period)
        im = interp(x, xp, _core.imag(fp), None if left is None else asarray(left).imag, None if right is None else asarray(right).imag, period)
        return _core.add(re, _core.multiply(im, 1j))
    xf = x.astype("float64")
    xpf = xp.astype("float64")
    fpf = fp.astype("float64")
    if period is not None:
        if period == 0:
            raise ValueError("period must be a non-zero value")
        period = abs(period)
        left = None
        right = None
        xf = _core.remainder(xf, period)
        xpf = _core.remainder(xpf, period)
        order = _core.argsort(xpf, 0)
        xpf = xpf[order]
        fpf = fpf[order]
        xpf = _core.concatenate([xpf[-1:] - period, xpf, xpf[:1] + period])
        fpf = _core.concatenate([fpf[-1:], fpf, fpf[:1]])
    n = xpf.shape[0]
    lval = fpf[0] if left is None else float(left)
    rval = fpf[-1] if right is None else float(right)
    j = _core.subtract(_core.searchsorted(xpf, xf, "right"), 1)
    jc = _core.clip(j, 0, max(n - 2, 0)) if n > 1 else _core.zeros_like(j)
    if n > 1:
        x0, x1 = xpf[jc], xpf[_core.add(jc, 1)]
        f0, f1 = fpf[jc], fpf[_core.add(jc, 1)]
        slope = _core.divide(_core.subtract(f1, f0), _core.subtract(x1, x0))
        res = _core.add(_core.multiply(slope, _core.subtract(xf, x0)), f0)
        bad = _core.isnan(res)
        alt = _core.add(_core.multiply(slope, _core.subtract(xf, x1)), f1)
        res = _core.where(bad, alt, res)
        bad2 = _core.isnan(res)
        res = _core.where(bad2, f0, res)
        res = _core.where(_core.equal(xf, xpf[-1]), fpf[-1], res)
    else:
        res = _core.full(tuple(xf.shape), float(fpf[0]), "float64")
    res = _core.where(_core.less(xf, xpf[0]), lval, res)
    res = _core.where(_core.greater(xf, xpf[-1]), rval, res)
    res = _core.where(_core.isnan(xf), _core.multiply(xf, 1.0), res) if False else res
    if x.dtype.kind == "f":
        res = _core.where(_core.isnan(xf), xf, res)
    return res


def _mode_check(mode):
    if mode in (0, 1, 2):
        mode = ("valid", "same", "full")[mode]
    if mode not in ("full", "same", "valid"):
        raise ValueError("mode must be one of 'full', 'same', or 'valid'")
    return mode


def _conv_full(a, v):
    n, m = a.shape[0], v.shape[0]
    full = _core.zeros((n + m - 1,), a.dtype)
    for j in builtins_range(m):
        full[j : j + n] = _core.add(full[j : j + n], _core.multiply(a, v[j]))
    return full


def _check_1d(a, v):
    if a.ndim != 1 or v.ndim != 1:
        raise ValueError("object too deep for desired array" if max(a.ndim, v.ndim) > 1 else "object of too small depth for desired array")
    if a.shape[0] == 0:
        raise ValueError("a cannot be empty")
    if v.shape[0] == 0:
        raise ValueError("v cannot be empty")


def convolve(a, v, mode="full"):
    a, v = asarray(a), asarray(v)
    _check_1d(a, v)
    mode = _mode_check(mode)
    if v.shape[0] > a.shape[0]:
        a, v = v, a
    n, m = a.shape[0], v.shape[0]
    dt = _core.result_type(a.dtype, v.dtype)
    full = _conv_full(a.astype(dt), v.astype(dt))
    if mode == "full":
        return full
    if mode == "valid":
        return full[m - 1 : n]
    start = (m - 1) // 2
    return full[start : start + n]


def correlate(a, v, mode="valid"):
    a, v = asarray(a), asarray(v)
    _check_1d(a, v)
    mode = _mode_check(mode)
    n, m = a.shape[0], v.shape[0]
    dt = _core.result_type(a.dtype, v.dtype)
    w = _core.conjugate(v.astype(dt))[::-1]
    full = _conv_full(a.astype(dt), w)
    if mode == "full":
        return full
    if mode == "valid":
        return full[min(n, m) - 1 : max(n, m)]
    start = (m - 1) // 2 if n >= m else n // 2
    return full[start : start + max(n, m)]


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return bool(R.all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)))


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    x, y = asarray(a), asarray(b)
    if x.dtype.kind in "biu" and y.dtype.kind in "biu":
        pass
    dt = _core.result_type(x.dtype, y.dtype)
    if dt.kind in "biu":
        xf, yf = x.astype("float64"), y.astype("float64")
    else:
        xf, yf = x, y
    with_inf = _core.logical_or(_core.isinf(xf), _core.isinf(yf))
    finite = _core.logical_not(with_inf)
    within = _core.less_equal(_core.absolute(_core.subtract(xf, yf)), _core.add(atol, _core.multiply(rtol, _core.absolute(yf))))
    fin_ok = _core.logical_and(finite, within)
    inf_ok = _core.logical_and(with_inf, _core.equal(xf, yf))
    res = _core.logical_or(fin_ok, inf_ok)
    if equal_nan:
        res = _core.logical_or(res, _core.logical_and(_core.isnan(xf), _core.isnan(yf)))
    return res


def array_equal(a1, a2, equal_nan=False):
    try:
        a1, a2 = asarray(a1), asarray(a2)
    except Exception:
        return False
    if tuple(a1.shape) != tuple(a2.shape):
        return False
    if not equal_nan:
        return bool(R.all(_core.equal(a1, a2)))
    eq = _core.equal(a1, a2)
    if a1.dtype.kind in "fc" and a2.dtype.kind in "fc":
        eq = _core.logical_or(eq, _core.logical_and(_core.isnan(a1), _core.isnan(a2)))
    return bool(R.all(eq))


def array_equiv(a1, a2):
    try:
        a1, a2 = asarray(a1), asarray(a2)
        _core.broadcast_shapes if False else None
        eq = _core.equal(a1, a2)
    except Exception:
        return False
    try:
        return bool(R.all(_core.equal(a1, a2))) and tuple(_core.broadcast_arrays(a1, a2)[0].shape) == tuple(eq.shape)
    except ValueError:
        return False


def isreal(x):
    x = asarray(x)
    return _core.equal(_core.imag(x), 0)


def iscomplex(x):
    x = asarray(x)
    return _core.not_equal(_core.imag(x), 0)


def isrealobj(x):
    return asarray(x).dtype.kind != "c"


def iscomplexobj(x):
    return asarray(x).dtype.kind == "c"


def real_if_close(a, tol=100):
    a = asarray(a)
    if a.dtype.kind != "c":
        return a
    if tol > 1:
        eps = finfo(a.dtype).eps.item()
        tol = tol * eps
    if bool(R.all(_core.less(_core.absolute(_core.imag(a)), tol))):
        return _core.real(a)
    return a


def angle(z, deg=False):
    z = asarray(z)
    res = _core.arctan2(_core.imag(z), _core.real(z)) if z.dtype.kind == "c" else _core.arctan2(_core.zeros_like(z) if z.dtype.kind in "fb" else _core.multiply(z, 0), z)
    if z.dtype.kind in "iub":
        res = _core.arctan2(_core.zeros_like(z), z)
    if deg:
        res = _core.multiply(res, 180.0 / _math.pi)
    return res


def unwrap(p, discont=None, axis=-1, *, period=2 * _math.pi):
    p = asarray(p)
    nd = p.ndim
    dd = diff(p, axis=axis)
    if discont is None:
        discont = period / 2
    slice1 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)
    dtype = _core.result_type(dd.dtype, period)
    if dtype.kind in "iu":
        interval_high, rem = divmod(period, 2)
        boundary_ambiguous = rem == 0
    else:
        interval_high = period / 2
        boundary_ambiguous = True
    interval_low = -interval_high
    ddmod = _core.add(_core.remainder(_core.subtract(dd, interval_low), period), interval_low)
    if boundary_ambiguous:
        ddmod = _core.where(_core.logical_and(_core.equal(ddmod, interval_low), _core.greater(dd, 0)), interval_high, ddmod)
    ph_correct = _core.subtract(ddmod, dd)
    ph_correct = _core.where(_core.less(_core.absolute(dd), discont), 0, ph_correct)
    up = _core.array(p, dtype=dtype)
    up[slice1] = _core.add(p[slice1], _core.cumsum(ph_correct, axis))
    return up


def sinc(x):
    x = asarray(x)
    if x.dtype.kind in "biu":
        x = x.astype("float64")
    y = _core.multiply(_math.pi, _core.where(_core.equal(x, 0), 1.0e-20, x))
    return _core.divide(_core.sin(y), y)


def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    x = asarray(x)
    if x.dtype.kind not in "fc":
        return x.copy() if copy else x
    if x.dtype.kind == "c":
        re = nan_to_num(_core.real(x), nan=nan, posinf=posinf, neginf=neginf)
        im = nan_to_num(_core.imag(x), nan=nan, posinf=posinf, neginf=neginf)
        return _core.add(re, _core.multiply(im, 1j)).astype(x.dtype)
    info = finfo(x.dtype)
    pinf = info.max.item() if posinf is None else posinf
    ninf = info.min.item() if neginf is None else neginf
    res = _core.where(_core.isnan(x), asarray(nan, x.dtype), x)
    res = _core.where(_core.logical_and(_core.isinf(res), _core.greater(res, 0)), asarray(pinf, x.dtype), res)
    res = _core.where(_core.logical_and(_core.isinf(res), _core.less(res, 0)), asarray(ninf, x.dtype), res)
    if not copy and isinstance(x, ndarray):
        x[...] = res
        return x
    return res


def bincount(x, /, weights=None, minlength=0):
    x = asarray(x)
    if x.ndim != 1:
        raise ValueError("object too deep for desired array")
    if x.dtype.kind not in "iu" and x.dtype.name != "bool":
        raise TypeError("Cannot cast array data from dtype('%s') to dtype('int64') according to the rule 'safe'" % x.dtype.name)
    if minlength < 0:
        raise ValueError("'minlength' must not be negative")
    xi = x.astype("int64")
    if x.size and int(_core.min(xi)) < 0:
        raise ValueError("'list' argument must have no negative elements")
    n = max(int(_core.max(xi)) + 1 if x.size else 0, minlength)
    if weights is None:
        res = _core.zeros((n,), "int64")
        if x.size:
            u = _core.unique_all(xi)
            res[u.values] = u.counts.astype("int64")
        return res
    w = asarray(weights)
    if tuple(w.shape) != tuple(x.shape):
        raise ValueError("The weights and list don't have the same length.")
    w = w.astype("complex128" if w.dtype.kind == "c" else "float64")
    res = _core.zeros((n,), w.dtype)
    if x.size:
        order = _core.argsort(xi, 0)
        xs = xi[order]
        ws = w[order]
        cs = _core.cumsum(ws, 0)
        last = _core.concatenate([_core.not_equal(xs[1:], xs[:-1]), _core.ones((1,), "bool")])
        ends = cs[last]
        keys = xs[last]
        prev = _core.concatenate([_core.zeros((1,), w.dtype), ends[:-1]])
        res[keys] = _core.subtract(ends, prev)
    return res


def digitize(x, bins, right=False):
    x = asarray(x)
    bins = asarray(bins)
    if bins.ndim != 1:
        raise ValueError("object too deep for desired array")
    if x.dtype.kind == "c":
        raise TypeError("x may not be complex")
    if bins.size == 0:
        return _core.zeros(tuple(x.shape), "int64")
    if bins.size > 1:
        inc = bool(R.all(_core.greater_equal(bins[1:], bins[:-1])))
        dec = bool(R.all(_core.less_equal(bins[1:], bins[:-1])))
    else:
        inc, dec = True, False
    if not inc and not dec:
        raise ValueError("bins must be monotonically increasing or decreasing")
    side = "left" if right else "right"
    if inc:
        return _core.searchsorted(bins, x, side)
    return _core.subtract(bins.size, _core.searchsorted(bins[::-1], x, "left" if right else "right"))


def _hist_edges_auto(a, bins, range_):
    a = a.reshape((-1,)).astype("float64")
    if range_ is not None:
        lo, hi = range_
        a = a[_core.logical_and(_core.greater_equal(a, lo), _core.less_equal(a, hi))]
    n = a.size
    if n == 0:
        return 1
    lo_v, hi_v = float(_core.min(a)), float(_core.max(a))
    width = None
    ptp_x = hi_v - lo_v
    if bins == "sqrt":
        width = ptp_x / _math.sqrt(n)
    elif bins == "sturges":
        width = ptp_x / (_math.log2(n) + 1.0)
    elif bins == "rice":
        width = ptp_x / (2.0 * n ** (1.0 / 3))
    elif bins == "scott":
        width = (24.0 * _math.pi**0.5 / n) ** (1.0 / 3.0) * float(R.std(a))
    elif bins in ("fd", "auto"):
        iqr = float(_core.subtract(R.percentile(a, 75), R.percentile(a, 25)))
        fd = 2.0 * iqr * n ** (-1.0 / 3.0)
        if bins == "fd":
            width = fd
        else:
            sturges = ptp_x / (_math.log2(n) + 1.0)
            sqrt_w = ptp_x / _math.sqrt(n)
            width = min(max(fd, sqrt_w / 2), sturges)
    elif bins == "stone":
        if n <= 1 or ptp_x == 0:
            width = 0.0
        else:
            def jhat(nbins):
                hh = ptp_x / nbins
                p_k = _core.divide(histogram(a, bins=nbins, range=range_)[0].astype("float64"), n)
                return (2 - (n + 1) * float(_core.dot(p_k, p_k))) / hh

            upper = max(100, int(_math.sqrt(n)))
            best = min(builtins_range(1, upper + 1), key=jhat)
            if best == upper:
                warnings.warn("The number of bins estimated may be suboptimal.", RuntimeWarning, stacklevel=3)
            width = ptp_x / best
    elif bins == "doane":
        if n > 2:
            sg1 = _math.sqrt(6.0 * (n - 2) / ((n + 1.0) * (n + 3)))
            sigma = float(R.std(a))
            if sigma > 0.0:
                temp = _core.subtract(a, float(R.mean(a)))
                temp = _core.divide(temp, sigma)
                g1 = float(R.mean(_core.power(temp, 3)))
                width = ptp_x / (1.0 + _math.log2(n) + _math.log2(1.0 + abs(g1) / sg1))
            else:
                width = 0.0
        else:
            width = 0.0
    else:
        raise ValueError("'%s' is not a valid estimator for `bins`" % bins)
    if width and width > 0:
        return int(_math.ceil(ptp_x / width)) or 1
    return 1


def histogram_bin_edges(a, bins=10, range=None, weights=None):
    a = asarray(a).reshape((-1,))
    if isinstance(bins, str):
        nb = _hist_edges_auto(a, bins, range)
        return _hist_edges_from_count(a, nb, range)
    b = asarray(bins)
    if b.ndim == 0:
        if b.dtype.kind not in "iu":
            raise TypeError("`bins` must be an integer, a string, or an array")
        return _hist_edges_from_count(a, int(b), range)
    if b.ndim == 1:
        if b.size > 1 and bool(R.any(_core.less(b[1:], b[:-1]))):
            raise ValueError("`bins` must increase monotonically, when an array")
        return b
    raise ValueError("`bins` must be 1d, when an array")


def _hist_edges_from_count(a, nb, range_):
    if nb < 1:
        raise ValueError("`bins` must be positive, when an integer")
    if range_ is not None:
        lo, hi = float(range_[0]), float(range_[1])
        if lo > hi:
            raise ValueError("max must be larger than min in range parameter.")
        if not (_math.isfinite(lo) and _math.isfinite(hi)):
            raise ValueError("supplied range of [%s, %s] is not finite" % (lo, hi))
    elif a.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = float(_core.min(a)), float(_core.max(a))
        if not (_math.isfinite(lo) and _math.isfinite(hi)):
            raise ValueError("autodetected range of [%s, %s] is not finite" % (lo, hi))
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    if range_ is not None:
        r = asarray(range_)
        dt = _core.result_type(r.dtype, a.dtype) if r.dtype.kind == "f" else (a.dtype if a.dtype.kind == "f" else _core.dtype("float64"))
    else:
        dt = a.dtype if a.dtype.kind == "f" else _core.dtype("float64")
    if dt.kind not in "fc":
        dt = _core.dtype("float64")
    edges = _core.linspace(lo, hi, nb + 1, True, None).astype(dt)
    if nb > 0 and bool(_core.any(_core.greater_equal(edges[:-1], edges[1:]))):
        raise ValueError("Too many bins for data range. Cannot create %d finite-sized bins." % nb)
    return edges


def histogram(a, bins=10, range=None, density=None, weights=None):
    a = asarray(a).reshape((-1,))
    w = None if weights is None else asarray(weights).reshape((-1,))
    if w is not None and tuple(w.shape) != tuple(a.shape):
        raise ValueError("weights should have the same shape as a.")
    edges = histogram_bin_edges(a, bins, range, weights)
    if range is not None:
        lo, hi = range
        keep = _core.logical_and(_core.greater_equal(a, lo), _core.less_equal(a, hi))
        a = a[keep]
        if w is not None:
            w = w[keep]
    nb = edges.shape[0] - 1
    af = a.astype("float64") if a.dtype.kind != "f" else a
    idx = _core.subtract(_core.searchsorted(edges, af, "right"), 1)
    idx = _core.where(_core.equal(af, edges[-1]), nb - 1, idx)
    valid = _core.logical_and(_core.greater_equal(idx, 0), _core.less(idx, nb))
    idx = idx[valid]
    if w is None:
        n = _core.zeros((nb,), "int64")
        if idx.size:
            u = _core.unique_all(idx)
            n[u.values] = u.counts.astype("int64")
    else:
        w = w[valid]
        n = bincount(idx, weights=w, minlength=nb)
        if w.dtype.kind in "iu" and False:
            pass
    if density:
        db = _core.diff(edges) if False else diff(edges)
        total = float(R.sum(n))
        n = _core.divide(_core.divide(n.astype("float64"), db), total)
    return n, edges


def histogramdd(sample, bins=10, range=None, density=None, weights=None):
    sample = asarray(sample)
    if sample.ndim == 1:
        sample = sample.reshape((-1, 1)) if True else sample
    N, D = sample.shape
    if isinstance(bins, int) or asarray(bins).ndim == 0:
        bins = [int(bins)] * D
    bins = list(bins)
    if len(bins) != D:
        raise ValueError("The dimension of bins must be equal to the dimension of the sample x.")
    rng = [None] * D if range is None else list(range)
    edges = []
    for d in builtins_range(D):
        col = sample[:, d]
        b = bins[d]
        if isinstance(b, int) or asarray(b).ndim == 0:
            edges.append(histogram_bin_edges(col, int(b), rng[d]))
        else:
            edges.append(asarray(b))
    nbins = [e.shape[0] - 1 for e in edges]
    flat = _core.zeros((N,), "int64")
    valid = _core.ones((N,), "bool")
    stride = 1
    idxs = []
    for d in builtins_range(D):
        col = sample[:, d].astype("float64")
        e = edges[d]
        i = _core.subtract(_core.searchsorted(e, col, "right"), 1)
        i = _core.where(_core.equal(col, e[-1]), nbins[d] - 1, i)
        valid = _core.logical_and(valid, _core.logical_and(_core.greater_equal(i, 0), _core.less(i, nbins[d])))
        idxs.append(i)
    total = 1
    for nb in nbins:
        total *= nb
    for d in builtins_range(D):
        flat = _core.add(_core.multiply(flat, nbins[d]), _core.where(valid, idxs[d], 0))
    flat = flat[valid]
    if weights is None:
        hist = bincount(flat, minlength=total).astype("float64")
    else:
        hist = bincount(flat, weights=asarray(weights)[valid], minlength=total)
    hist = hist.reshape(tuple(nbins))
    if density:
        db = None
        s = float(R.sum(hist))
        vol = 1.0
        widths = [diff(e) for e in edges]
        volume = widths[0].reshape((-1,) + (1,) * (D - 1))
        for d in builtins_range(1, D):
            shp = [1] * D
            shp[d] = -1
            volume = _core.multiply(volume, widths[d].reshape(tuple(shp)))
        hist = _core.divide(_core.divide(hist.astype("float64"), volume), s)
    return hist, edges


def histogram2d(x, y, bins=10, range=None, density=None, weights=None):
    x = asarray(x)
    y = asarray(y)
    if tuple(x.shape) != tuple(y.shape) or x.ndim != 1:
        raise ValueError("x and y must be 1-d arrays of equal length")
    try:
        n_bins = len(bins)
    except TypeError:
        n_bins = 1
    if n_bins == 1 or n_bins == 2 and not (isinstance(bins, ndarray)):
        pass
    if n_bins != 2 and not isinstance(bins, (int,)):
        xedges = yedges = asarray(bins)
        bins_l = [xedges, yedges]
    elif isinstance(bins, int):
        bins_l = [bins, bins]
    else:
        bins_l = list(bins)
    sample = _core.stack([x, y], axis=1)
    h, edges = histogramdd(sample, bins_l, range, density, weights)
    return h, edges[0], edges[1]


builtins_range = __builtins__["range"] if isinstance(__builtins__, dict) else __builtins__.range


def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, *, dtype=None):
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be integer")
    m = asarray(m)
    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")
    if y is not None:
        y = asarray(y)
        if y.ndim > 2:
            raise ValueError("y has more than 2 dimensions")
    if dtype is None:
        dtype = _core.result_type(m.dtype, "float64") if y is None else _core.result_type(m.dtype, y.dtype, "float64")
    X = m.astype(dtype)
    if X.ndim == 0:
        X = X.reshape((1, 1))
    elif X.ndim == 1:
        X = X.reshape((1, -1))
    if not rowvar and X.shape[0] != 1:
        X = _core.transpose(X)
    if X.shape[0] == 0:
        return _core.zeros((0, 0), dtype)
    if y is not None:
        Y = y.astype(dtype)
        if Y.ndim < 2:
            Y = Y.reshape((1, -1))
        if not rowvar and Y.shape[0] != 1:
            Y = _core.transpose(Y)
        X = _core.concatenate([X, Y], 0)
    if ddof is None:
        ddof = 0 if bias else 1
    w = None
    if fweights is not None:
        fw = asarray(fweights).astype("float64")
        if fw.ndim > 1:
            raise RuntimeError("cannot handle multidimensional fweights")
        if fw.shape[0] != X.shape[1]:
            raise RuntimeError("incompatible numbers of samples and fweights")
        if fw.dtype.kind not in "iu" and bool(R.any(_core.not_equal(fw, _core.rint(fw)))) or asarray(fweights).dtype.kind not in "iub":
            raise TypeError("fweights must be integer")
        if bool(R.any(_core.less(fw, 0))):
            raise ValueError("fweights cannot be negative")
        w = fw
    if aweights is not None:
        aw = asarray(aweights).astype("float64")
        if aw.ndim > 1:
            raise RuntimeError("cannot handle multidimensional aweights")
        if aw.shape[0] != X.shape[1]:
            raise RuntimeError("incompatible numbers of samples and aweights")
        if bool(R.any(_core.less(aw, 0))):
            raise ValueError("aweights cannot be negative")
        w = aw if w is None else _core.multiply(w, aw)
    avg, w_sum = R.average(X, axis=1, weights=w, returned=True)
    w_sum = w_sum[0] if isinstance(w_sum, ndarray) and w_sum.ndim else w_sum
    if w is None:
        fact = X.shape[1] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * float(R.sum(_core.multiply(w, asarray(aweights).astype("float64")))) / float(w_sum)
    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)
        fact = 0.0
    X = _core.subtract(X, asarray(avg).reshape((-1, 1)))
    X_T = _core.transpose(X) if w is None else _core.transpose(_core.multiply(X, w))
    c = _core.dot(X, _core.conjugate(X_T))
    c = _core.multiply(c, _core.divide(1.0, fact) if fact != 0 else float("inf"))
    return _core.squeeze(c)


def corrcoef(x, y=None, rowvar=True, *, dtype=None):
    c = cov(x, y, rowvar, dtype=dtype)
    c = asarray(c)
    if c.ndim == 0:
        return _core.divide(c, c)
    d = _core.diagonal(c)
    stddev = _core.sqrt(_core.real(d)).astype(c.dtype)
    c = _core.divide(c, stddev.reshape((-1, 1)))
    c = _core.divide(c, stddev.reshape((1, -1)))
    if c.dtype.kind == "c":
        re = _core.clip(_core.real(c), -1, 1)
        im = _core.clip(_core.imag(c), -1, 1)
        return _core.add(re, _core.multiply(im, 1j)).astype(c.dtype)
    return _core.clip(c, -1, 1)


def vdot(a, b):
    a = asarray(a).reshape((-1,))
    b = asarray(b).reshape((-1,))
    if a.shape != b.shape:
        raise ValueError("vectors have different lengths")
    return _core.dot(_core.conjugate(a), b)


def inner(a, b):
    a, b = asarray(a), asarray(b)
    if a.ndim == 0 or b.ndim == 0:
        return _core.multiply(a, b)
    if a.shape[-1] != b.shape[-1]:
        raise ValueError("shapes %s and %s not aligned: %d (dim %d) != %d (dim %d)" % (tuple(a.shape), tuple(b.shape), a.shape[-1], a.ndim - 1, b.shape[-1], b.ndim - 1))
    return _core.tensordot(a, b, axes=([a.ndim - 1], [b.ndim - 1]))


def tensordot(a, b, axes=2):
    a, b = asarray(a), asarray(b)
    if isinstance(axes, int):
        return _core.tensordot(a, b, axes)
    return _core.tensordot(a, b, axes)


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    a, b = asarray(a), asarray(b)
    if axis is not None:
        axisa = axisb = axisc = axis
    a = _core.moveaxis(a, axisa, -1)
    b = _core.moveaxis(b, axisb, -1)
    if a.shape[-1] not in (2, 3) or b.shape[-1] not in (2, 3):
        raise ValueError("incompatible dimensions for cross product\n(dimension must be 2 or 3)")
    if a.shape[-1] == 2 or b.shape[-1] == 2:
        raise ValueError("2-dimensional cross product is not supported")
    res = _core.cross(a, b)
    return _core.moveaxis(res, -1, axisc)


def outer(a, b, out=None):
    a, b = asarray(a).reshape((-1,)), asarray(b).reshape((-1,))
    res = _core.outer(a, b)
    return _write_out(res, out, True, "same_kind", "outer") if out is not None else res


def vecdot(x1, x2, /, *, axis=-1):
    x1, x2 = asarray(x1), asarray(x2)
    if axis != -1:
        x1, x2 = _core.moveaxis(x1, axis, -1), _core.moveaxis(x2, axis, -1)
    return _core.vecdot(x1, x2)


def bitwise_count(x):
    x = asarray(x)
    if x.dtype.kind not in "iub":
        raise TypeError("ufunc 'bitwise_count' not supported for the input types")
    if x.dtype.kind == "i":
        ux = _core.absolute(x.astype("int64")).astype("uint64") if x.dtype.itemsize < 8 else _core.absolute(x).astype("uint64")
        ux = _core.where(_core.less(x, 0), _core.absolute(x.astype("int64")).astype("uint64"), x.astype("uint64"))
    else:
        ux = x.astype("uint64")
    count = _core.zeros(tuple(ux.shape), "uint8")
    for bit in builtins_range(64):
        count = _core.add(count, _core.bitwise_and(_core.right_shift(ux, bit), 1).astype("uint8"))
    return count


def packbits(a, /, axis=None, bitorder="big"):
    a = asarray(a)
    if a.dtype.kind not in "biu":
        raise TypeError("Expected an input array of integer or boolean data type")
    if bitorder not in ("big", "little"):
        raise ValueError("'order' must be either 'little' or 'big'")
    bits = _core.not_equal(a, 0).astype("uint8")
    if axis is None:
        bits = bits.reshape((-1,))
        axis = 0
    else:
        axis = _norm_axis(axis, a.ndim)
    n = bits.shape[axis]
    pad = (-n) % 8
    if pad:
        shp = list(bits.shape)
        shp[axis] = pad
        bits = _core.concatenate([bits, _core.zeros(tuple(shp), "uint8")], axis)
    shp = list(bits.shape)
    shp[axis : axis + 1] = [shp[axis] // 8, 8]
    bits = bits.reshape(tuple(shp))
    weights = [128, 64, 32, 16, 8, 4, 2, 1] if bitorder == "big" else [1, 2, 4, 8, 16, 32, 64, 128]
    wshape = [1] * bits.ndim
    wshape[axis + 1] = 8
    w = _core.array(weights, "uint8").reshape(tuple(wshape))
    return R.sum(_core.multiply(bits, w), axis=axis + 1, dtype="uint8")


def unpackbits(a, /, axis=None, count=None, bitorder="big"):
    a = asarray(a)
    if a.dtype.name != "uint8":
        raise TypeError("Expected an input array of unsigned byte data type")
    if bitorder not in ("big", "little"):
        raise ValueError("'order' must be either 'little' or 'big'")
    if axis is None:
        a = a.reshape((-1,))
        axis = 0
    else:
        axis = _norm_axis(axis, a.ndim)
    shifts = list(builtins_range(7, -1, -1)) if bitorder == "big" else list(builtins_range(8))
    parts = [_core.bitwise_and(_core.right_shift(a, s), 1).astype("uint8") for s in shifts]
    st = _core.stack(parts, axis=axis + 1)
    shp = list(a.shape)
    shp[axis] *= 8
    res = st.reshape(tuple(shp))
    if count is not None:
        total = res.shape[axis]
        if count < 0:
            count = total + count
        sl = [slice(None)] * res.ndim
        if count <= total:
            sl[axis] = slice(0, count)
            res = res[tuple(sl)]
        else:
            padshape = list(res.shape)
            padshape[axis] = count - total
            res = _core.concatenate([res, _core.zeros(tuple(padshape), "uint8")], axis)
    return res


def hanning(M):
    if M < 1:
        return _core.array([], "float64")
    if M == 1:
        return _core.ones((1,), "float64")
    n = _core.arange(1 - M, M, 2)
    return _core.add(0.5, _core.multiply(0.5, _core.cos(_core.divide(_core.multiply(_math.pi, n), M - 1))))


def hamming(M):
    if M < 1:
        return _core.array([], "float64")
    if M == 1:
        return _core.ones((1,), "float64")
    n = _core.arange(1 - M, M, 2)
    return _core.add(0.54, _core.multiply(0.46, _core.cos(_core.divide(_core.multiply(_math.pi, n), M - 1))))


def blackman(M):
    if M < 1:
        return _core.array([], "float64")
    if M == 1:
        return _core.ones((1,), "float64")
    n = _core.arange(1 - M, M, 2)
    t = _core.divide(_core.multiply(_math.pi, n), M - 1)
    return _core.add(_core.add(0.42, _core.multiply(0.5, _core.cos(t))), _core.multiply(0.08, _core.cos(_core.multiply(2.0, t))))


def bartlett(M):
    if M < 1:
        return _core.array([], "float64")
    if M == 1:
        return _core.ones((1,), "float64")
    n = _core.arange(1 - M, M, 2)
    return _core.where(_core.less_equal(n, 0), _core.add(1, _core.divide(n, M - 1)), _core.subtract(1, _core.divide(n, M - 1)))


def i0(x):
    x = asarray(x)
    if x.dtype.kind == "c":
        raise TypeError("ufunc 'i0' not supported for the input types")
    shape = tuple(x.shape)
    flat = asarray(_core.absolute(x.astype("float64").reshape((-1,))))
    term = _core.ones(tuple(flat.shape), "float64")
    total = _core.ones(tuple(flat.shape), "float64")
    q = _core.divide(_core.multiply(flat, flat), 4.0)
    for k in builtins_range(1, 300):
        term = _core.divide(_core.multiply(term, q), k * k)
        total = _core.add(total, term)
        if flat.size == 0 or float(_core.max(_core.divide(term, total))) < 1e-17:
            break
    return asarray(total).reshape(shape)


def kaiser(M, beta):
    if M == 1:
        return _core.ones((1,), "float64")
    if M < 1:
        return _core.array([], "float64")
    n = _core.arange(0, M)
    alpha = (M - 1) / 2.0
    return _core.divide(i0(_core.multiply(beta, _core.sqrt(_core.subtract(1, _core.power(_core.divide(_core.subtract(n, alpha), alpha), 2.0))))), float(i0(beta)))


class _Finfo:
    _cache = {}

    def __init__(self, dtype):
        dt = asarray(_core.zeros((), dtype)).dtype if not isinstance(dtype, _core.dtype) else dtype
        if dt.kind == "c":
            dt = _core.dtype("float32" if dt.name == "complex64" else "float64")
        if dt.kind != "f":
            raise ValueError("data type %r not inexact" % (dtype,))
        self.dtype = dt
        info = {
            "float16": (10, 5, 11, 15, 4, -14, 16, 5.960464477539063e-08, 0.0009765625, 65504.0, 6.103515625e-05, 3, 5.960464477539063e-08),
            "float32": (23, 8, 24, 127, 7, -126, 128, 1.401298464324817e-45, 1.1920928955078125e-07, 3.4028234663852886e38, 1.1754943508222875e-38, 6, 1.1920928955078125e-07),
            "float64": (52, 11, 53, 1023, 15, -1022, 1024, 5e-324, 2.220446049250313e-16, 1.7976931348623157e308, 2.2250738585072014e-308, 15, 2.220446049250313e-16),
        }[dt.name]
        (self.nmant, self.nexp, self.machep_bits, self.maxexp_bias, self.precision, self.minexp, self.maxexp, subnormal, eps, mx, tiny, self.dig, _) = info
        typed = lambda v: _core.array(v, dt)
        self.eps = typed(eps)
        self.max = typed(mx)
        self.tiny = typed(tiny)
        self.smallest_subnormal = typed(subnormal)
        self.bits = dt.itemsize * 8
        self.iexp = self.nexp
        self.min = typed(-mx)
        self.smallest_normal = self.tiny
        self.epsneg = typed(eps / 2)
        self.resolution = typed({"float16": 0.001, "float32": 1e-6, "float64": 1e-15}[dt.name])
        self.machep = -self.nmant
        self.negep = -self.nmant - 1
        self.dtype = dt

    def __repr__(self):
        from ._print import _shortest

        shown = lambda v: repr(float(_shortest(v, self.dtype.name)))
        return "finfo(resolution=%s, min=%s, max=%s, dtype=%s)" % (shown(self.resolution), shown(self.min), shown(self.max), self.dtype.name)


def finfo(dtype):
    if not isinstance(dtype, _core.dtype):
        try:
            dtype = _core.dtype(dtype)
        except Exception:
            dtype = asarray(dtype).dtype
    return _Finfo(dtype)


class iinfo:
    def __init__(self, int_type):
        try:
            dt = int_type.dtype if hasattr(int_type, "dtype") and not isinstance(int_type, _core.dtype) else _core.dtype(int_type)
        except Exception:
            dt = asarray(int_type).dtype
        if dt.kind not in "iu":
            raise ValueError("Invalid integer data type %r." % (dt.kind,))
        self.dtype = dt
        self.kind = dt.kind
        self.bits = dt.itemsize * 8
        if dt.kind == "u":
            self.min = 0
            self.max = 2**self.bits - 1
        else:
            self.min = -(2 ** (self.bits - 1))
            self.max = 2 ** (self.bits - 1) - 1

    def __repr__(self):
        return "iinfo(min=%d, max=%d, dtype=%s)" % (self.min, self.max, self.dtype.name)
