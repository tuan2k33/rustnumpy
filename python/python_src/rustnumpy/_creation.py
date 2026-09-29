import math as _math

from . import _core
from ._core import asarray as _asarray_native
asarray_ = _asarray_native
from ._core import ndarray
from ._ufunc import _NoValue

pi = _math.pi
e = _math.e
euler_gamma = 0.5772156649015329
inf = float("inf")
nan = float("nan")
newaxis = None
PINF = inf
NINF = -inf
PZERO = 0.0
NZERO = -0.0
NAN = nan


def _shape_tuple(shape):
    if isinstance(shape, (int,)) and not isinstance(shape, bool):
        return (int(shape),)
    if hasattr(shape, "__index__") and not isinstance(shape, (tuple, list)):
        return (shape.__index__(),)
    try:
        return tuple(int(s.__index__()) if hasattr(s, "__index__") else int(s) for s in shape)
    except TypeError:
        raise TypeError("'%s' object cannot be interpreted as an integer" % type(shape).__name__) from None


def _c_order_copy_to_f(x):
    return _core.transpose(_core.transpose(x).copy())


def _apply_order(x, order):
    if order in ("F", "f") and x.ndim > 1:
        return _c_order_copy_to_f(x)
    return x


def array(object, dtype=None, *, copy=True, order="K", subok=False, ndmin=0, like=None):
    if isinstance(object, ndarray):
        same = dtype is None or _core.dtype(dtype) == object.dtype
        if same:
            res = object.copy() if copy is True else object
        else:
            if copy is False:
                raise ValueError("Unable to avoid copy while creating an array as requested.")
            res = object.astype(dtype)
    else:
        if copy is False and isinstance(object, (list, tuple)):
            raise ValueError("Unable to avoid copy while creating an array as requested.")
        res = _core.array(object, dtype, True)
    if ndmin and res.ndim < ndmin:
        res = res.reshape((1,) * (ndmin - res.ndim) + tuple(res.shape))
    return _apply_order(res, order)


def asarray(a, dtype=None, order=None, *, device=None, copy=None, like=None):
    if copy is True:
        return array(a, dtype, copy=True, order=order or "K")
    if copy is False and not isinstance(a, ndarray):
        raise ValueError("Unable to avoid copy while creating an array as requested.")
    res = _asarray_native(a, dtype)
    if copy is False and dtype is not None and isinstance(a, ndarray) and res is not a:
        raise ValueError("Unable to avoid copy while creating an array as requested.")
    if order in ("F", "f") and res.ndim > 1 and not res.flags.f_contiguous:
        return _c_order_copy_to_f(res)
    if order in ("C", "c") and not res.flags.c_contiguous:
        return res.copy()
    return res


asanyarray = asarray


def ascontiguousarray(a, dtype=None, *, like=None):
    res = _asarray_native(a, dtype)
    if res.ndim == 0:
        return res.reshape((1,))
    return res if res.flags.c_contiguous else res.copy()


def asfortranarray(a, dtype=None, *, like=None):
    res = _asarray_native(a, dtype)
    if res.ndim == 0:
        return res.reshape((1,))
    if res.flags.f_contiguous:
        return res
    return _c_order_copy_to_f(res)


def require(a, dtype=None, requirements=None, *, like=None):
    reqs = {r.upper()[0] if r.upper() not in ("C_CONTIGUOUS", "F_CONTIGUOUS") else r.upper() for r in (requirements or ())}
    res = _asarray_native(a, dtype)
    for r in reqs:
        if r in ("C", "C_CONTIGUOUS"):
            res = ascontiguousarray(res)
        elif r in ("F", "F_CONTIGUOUS"):
            res = asfortranarray(res)
    return res


def copy(a, order="K", subok=False):
    return array(a, copy=True, order=order)


def zeros(shape, dtype=float, order="C", *, device=None, like=None):
    return _apply_order(_core.zeros(_shape_tuple(shape), dtype), order)


def ones(shape, dtype=None, order="C", *, device=None, like=None):
    return _apply_order(_core.ones(_shape_tuple(shape), float if dtype is None else dtype), order)


def empty(shape, dtype=float, order="C", *, device=None, like=None):
    return _apply_order(_core.empty(_shape_tuple(shape), dtype), order)


def full(shape, fill_value, dtype=None, order="C", *, device=None, like=None):
    if dtype is None:
        dtype = _core.result_type(fill_value) if not isinstance(fill_value, ndarray) else fill_value.dtype
    return _apply_order(_core.full(_shape_tuple(shape), fill_value, dtype), order)


def _like(fn, a, dtype, order, shape, extra=()):
    a = _asarray_native(a)
    if shape is None:
        shape = tuple(a.shape)
    else:
        shape = _shape_tuple(shape)
    dt = a.dtype if dtype is None else dtype
    res = fn(shape, *extra, dt)
    if order in ("F", "f") or (order == "K" and a.ndim > 1 and a.flags.f_contiguous and not a.flags.c_contiguous):
        return _apply_order(res, "F")
    return res


def zeros_like(a, dtype=None, order="K", subok=True, shape=None, *, device=None):
    return _like(lambda s, d: _core.zeros(s, d), a, dtype, order, shape)


def ones_like(a, dtype=None, order="K", subok=True, shape=None, *, device=None):
    return _like(lambda s, d: _core.ones(s, d), a, dtype, order, shape)


def empty_like(prototype, dtype=None, order="K", subok=True, shape=None, *, device=None):
    return _like(lambda s, d: _core.empty(s, d), prototype, dtype, order, shape)


def full_like(a, fill_value, dtype=None, order="K", subok=True, shape=None, *, device=None):
    return _like(lambda s, d: _core.full(s, fill_value, d), a, dtype, order, shape)


def arange(start, /, stop=None, step=None, dtype=None, *, device=None, like=None):
    if step is None:
        step = 1
    return _core.arange(start, stop, step, dtype)


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, *, device=None):
    num = num.__index__() if hasattr(num, "__index__") else int(num)
    if num < 0:
        raise ValueError("Number of samples, %s, must be non-negative." % num)
    scalar = not isinstance(start, ndarray) and not isinstance(stop, ndarray) and _asarray_native(start).ndim == 0 and _asarray_native(stop).ndim == 0
    if scalar and not (isinstance(start, complex) or isinstance(stop, complex)):
        a0, a1 = _asarray_native(start), _asarray_native(stop)
        if a0.dtype.kind != "c" and a1.dtype.kind != "c":
            res = _core.linspace(float(a0), float(a1), num, endpoint, None)
            div = (num - 1) if endpoint else num
            step = (float(a1) - float(a0)) / div if div > 0 else float("nan")
            if dtype is not None and _core.dtype(dtype).kind in "iu":
                res = _core.floor(res)
            if dtype is not None:
                res = res.astype(dtype)
            return (res, step) if retstep else res
    a0 = _core.multiply(_asarray_native(start), 1.0)
    a1 = _core.multiply(_asarray_native(stop), 1.0)
    dt = _core.result_type(a0.dtype, a1.dtype, "float64")
    a0, a1 = a0.astype(dt), a1.astype(dt)
    div = (num - 1) if endpoint else num
    delta = _core.subtract(a1, a0)
    y = _core.arange(0, num, 1, dt).reshape((-1,) + (1,) * delta.ndim)
    if div > 0:
        step = _core.divide(delta, div)
        if bool(_core.any(_core.equal(step, 0))) and False:
            y = _core.multiply(_core.divide(y, div), delta)
        else:
            y = _core.multiply(y, step)
    else:
        step = float("nan")
        y = _core.multiply(y, delta)
    y = _core.add(y, a0)
    if endpoint and num > 1:
        y[-1, ...] = a1
    if axis != 0:
        y = _core.moveaxis(y, 0, axis)
    if dtype is not None:
        if _core.dtype(dtype).kind in "iu":
            y = _core.floor(y)
        y = y.astype(dtype)
    return (y, step) if retstep else y


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    y = linspace(start, stop, num=num, endpoint=endpoint, axis=axis)
    res = _core.power(_asarray_native(base) if not isinstance(base, ndarray) else base, y) if not isinstance(base, (int, float)) else _core.power(float(base), y)
    if dtype is None:
        return res
    return res.astype(dtype)


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    s0, s1 = _asarray_native(start), _asarray_native(stop)
    if bool(_core.any(_core.equal(s0, 0))) or bool(_core.any(_core.equal(s1, 0))):
        raise ValueError("Geometric sequence cannot include zero")
    dt = _core.result_type(s0.dtype, s1.dtype, "float64")
    s0, s1 = s0.astype(dt), s1.astype(dt)
    out_sign = _core.ones(tuple(asarray_(_core.add(s0, s1)).shape), dt)
    if dt.kind == "c":
        pass
    else:
        neg0 = _core.less(s0, 0)
        neg1 = _core.less(s1, 0)
        both_neg = _core.logical_and(neg0, neg1)
        if bool(_core.any(both_neg)):
            out_sign = _core.where(both_neg, -1.0, out_sign)
            s0 = _core.where(both_neg, _core.negative(s0), s0)
            s1 = _core.where(both_neg, _core.negative(s1), s1)
    log_start = _core.log10(s0)
    log_stop = _core.log10(s1)
    res = logspace(log_start, log_stop, num=num, endpoint=endpoint, base=10.0, dtype=dt, axis=0)
    if num > 0:
        if endpoint and num > 1:
            res[0, ...] = s0
            res[-1, ...] = s1
        elif num > 0:
            res[0, ...] = s0
    res = _core.multiply(res, out_sign)
    if axis != 0:
        res = _core.moveaxis(res, 0, axis)
    return res if dtype is None else res.astype(dtype)


def eye(N, M=None, k=0, dtype=float, order="C", *, device=None, like=None):
    return _apply_order(_core.eye(N, M, k, dtype), order)


def identity(n, dtype=None, *, like=None):
    return _core.identity(n, float if dtype is None else dtype)


def diag(v, k=0):
    v = _asarray_native(v)
    if v.ndim == 1:
        n = v.shape[0] + abs(k)
        res = _core.zeros((n, n), v.dtype)
        idx = _core.arange(v.shape[0])
        rows = idx + (-k if k < 0 else 0)
        cols = idx + (k if k > 0 else 0)
        res[rows, cols] = v
        return res
    if v.ndim == 2:
        return _core.diagonal(v, k).copy()
    raise ValueError("Input must be 1- or 2-d.")


def diagflat(v, k=0):
    v = _asarray_native(v).reshape((-1,))
    return diag(v, k)


def tri(N, M=None, k=0, dtype=float, *, like=None):
    M = N if M is None else M
    i = _core.arange(N).reshape((-1, 1))
    j = _core.arange(M).reshape((1, -1))
    return _core.less_equal(_core.subtract(j, i), k).astype(dtype)


def tril(m, k=0):
    m = _asarray_native(m)
    if m.ndim < 2:
        raise ValueError("input array must be at least 2-d")
    mask = tri(m.shape[-2], m.shape[-1], k=k, dtype="bool")
    return _core.where(mask, m, _core.zeros((), m.dtype))


def triu(m, k=0):
    m = _asarray_native(m)
    if m.ndim < 2:
        raise ValueError("input array must be at least 2-d")
    mask = _core.logical_not(tri(m.shape[-2], m.shape[-1], k=k - 1, dtype="bool"))
    return _core.where(mask, m, _core.zeros((), m.dtype))


def vander(x, N=None, increasing=False):
    x = _asarray_native(x)
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array or sequence.")
    if N is None:
        N = x.shape[0]
    dt = _core.result_type(x.dtype, "int64") if x.dtype.kind == "b" else x.dtype
    powers = _core.arange(N).reshape((1, -1))
    if not increasing:
        powers = powers[:, ::-1]
    return _core.power(x.astype(dt).reshape((-1, 1)), powers.astype(dt if dt.kind in "iu" else "int64"))


def meshgrid(*xi, copy=True, sparse=False, indexing="xy"):
    if indexing not in ("xy", "ij"):
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")
    arrs = [_asarray_native(x) for x in xi]
    ndim = len(arrs)
    s0 = (1,) * ndim
    output = [a.reshape(s0[:i] + (-1,) + s0[i + 1:]) for i, a in enumerate(arrs)]
    if indexing == "xy" and ndim > 1:
        output[0] = output[0].reshape((1, -1) + s0[2:])
        output[1] = output[1].reshape((-1, 1) + s0[2:])
    if not sparse:
        output = _core.broadcast_arrays(*output)
    if copy:
        output = [x.copy() for x in output]
    return tuple(output)


def indices(dimensions, dtype=int, sparse=False):
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,) * N
    if sparse:
        return tuple(_core.arange(dim, dtype=dtype).reshape(shape[:i] + (dim,) + shape[i + 1:]) for i, dim in enumerate(dimensions))
    res = _core.empty((N,) + dimensions, dtype)
    for i, dim in enumerate(dimensions):
        res[i] = _core.arange(dim, dtype=dtype).reshape(shape[:i] + (dim,) + shape[i + 1:])
    return res


def fromfunction(function, shape, *, dtype=float, like=None, **kwargs):
    args = indices(shape, dtype=dtype)
    return function(*args, **kwargs)


def frombuffer(buffer, dtype=float, count=-1, offset=0, *, like=None):
    dt = _core.dtype(dtype)
    mv = memoryview(buffer).cast("B")
    data = mv[offset:]
    if count < 0:
        if len(data) % dt.itemsize:
            raise ValueError("buffer size must be a multiple of element size")
        count = len(data) // dt.itemsize
    elif count * dt.itemsize > len(data):
        raise ValueError("buffer is smaller than requested size")
    chunk = data[: count * dt.itemsize]
    return _core.array(_BytesView(chunk, dt), dt, True)


class _BytesView:
    def __init__(self, chunk, dt):
        self._chunk = chunk
        self.__array_interface__ = {
            "shape": (len(chunk) // dt.itemsize,),
            "typestr": dt.str,
            "data": bytes(chunk),
            "version": 3,
        }


def fromiter(iter, dtype, count=-1, *, like=None):
    items = []
    for i, v in enumerate(iter):
        if count >= 0 and i >= count:
            break
        items.append(v)
    if 0 <= count > len(items):
        raise ValueError("iterator too short")
    return _core.array(items, dtype)


class _IndexGrid:
    def __init__(self, sparse):
        self.sparse = sparse

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        parts = []
        for k in key:
            if isinstance(k, slice):
                start = 0 if k.start is None else k.start
                step = 1 if k.step is None else k.step
                if isinstance(step, complex):
                    n = int(abs(step))
                    parts.append(linspace(start, k.stop, n))
                else:
                    parts.append(arange(start, k.stop, step))
            else:
                raise TypeError("index must be a slice")
        nd = len(parts)
        if nd == 1 and len(key) == 1 and isinstance(key[0], slice) and not self.sparse:
            return parts[0]
        shaped = [p.reshape((1,) * i + (-1,) + (1,) * (nd - i - 1)) for i, p in enumerate(parts)]
        if self.sparse:
            return tuple(shaped)
        return _core.stack(_core.broadcast_arrays(*shaped), axis=0)


mgrid = _IndexGrid(False)
ogrid = _IndexGrid(True)


class _RClass:
    def __init__(self, axis):
        self.axis = axis

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        items = []
        for k in key:
            if isinstance(k, slice):
                start = 0 if k.start is None else k.start
                step = 1 if k.step is None else k.step
                if isinstance(step, complex):
                    items.append(linspace(start, k.stop, int(abs(step))))
                else:
                    items.append(arange(start, k.stop, step))
            elif isinstance(k, str):
                continue
            else:
                a = _asarray_native(k)
                if self.axis == -1 and a.ndim == 0:
                    a = a.reshape((1,))
                elif self.axis == -1 and a.ndim == 1:
                    a = a.reshape((-1, 1))
                elif a.ndim == 1 and self.axis == 0:
                    pass
                items.append(a)
        if self.axis == -1:
            items = [a.reshape((-1, 1)) if a.ndim == 1 else a for a in items]
            return _core.concatenate(items, axis=1)
        return _core.concatenate([a.reshape((1,)) if a.ndim == 0 else a for a in items], axis=0)


r_ = _RClass(0)
c_ = _RClass(-1)


class _SClass:
    def __getitem__(self, item):
        return item


s_ = _SClass()
index_exp = _SClass()


def diag_indices(n, ndim=2):
    idx = _core.arange(n)
    return (idx,) * ndim


def diag_indices_from(arr):
    arr = _asarray_native(arr)
    if arr.ndim < 2:
        raise ValueError("input array must be at least 2-d")
    if len(set(arr.shape)) != 1:
        raise ValueError("All dimensions of input must be of equal length")
    return diag_indices(arr.shape[0], arr.ndim)


def tril_indices(n, k=0, m=None):
    from ._indexing import nonzero

    return nonzero(tri(n, m, k=k, dtype="bool"))


def triu_indices(n, k=0, m=None):
    from ._indexing import nonzero

    return nonzero(_core.logical_not(tri(n, m, k=k - 1, dtype="bool")))


def tril_indices_from(arr, k=0):
    arr = _asarray_native(arr)
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    return tril_indices(arr.shape[-2], k=k, m=arr.shape[-1])


def triu_indices_from(arr, k=0):
    arr = _asarray_native(arr)
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    return triu_indices(arr.shape[-2], k=k, m=arr.shape[-1])


def mask_indices(n, mask_func, k=0):
    from ._indexing import nonzero

    m = ones((n, n), dtype=int)
    a = mask_func(m, k)
    return nonzero(a != 0)


def ndindex(*shape):
    import itertools

    if len(shape) == 1 and isinstance(shape[0], tuple):
        shape = shape[0]
    return itertools.product(*[range(s) for s in shape])


class ndenumerate:
    def __init__(self, arr):
        self._arr = _asarray_native(arr)

    def __iter__(self):
        for idx in ndindex(tuple(self._arr.shape)):
            yield idx, self._arr[idx]
