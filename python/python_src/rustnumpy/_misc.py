import contextlib
import functools

from . import _core
from ._core import asarray, ndarray
from ._indexing import _norm_axis

_err = {"all": None, "divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}


def geterr():
    return {k: v for k, v in _err.items() if k != "all"}


def seterr(all=None, divide=None, over=None, under=None, invalid=None):
    old = geterr()
    for key, value in (("divide", divide), ("over", over), ("under", under), ("invalid", invalid)):
        v = all if value is None else value
        if v is not None:
            _err[key] = v
    return old


def geterrcall():
    return None


def seterrcall(func):
    return None


class errstate(contextlib.ContextDecorator):
    def __init__(self, *, call=None, all=None, divide=None, over=None, under=None, invalid=None):
        self._kwargs = dict(all=all, divide=divide, over=over, under=under, invalid=invalid)
        self._old = None

    def __enter__(self):
        self._old = seterr(**self._kwargs)
        return self

    def __exit__(self, *exc):
        seterr(**self._old)


class Flags:
    def __init__(self, c, f, owndata, writeable=True):
        self.c_contiguous = c
        self.f_contiguous = f
        self.owndata = owndata
        self.writeable = writeable
        self.aligned = True
        self.writebackifcopy = False

    contiguous = property(lambda self: self.c_contiguous)
    fortran = property(lambda self: self.f_contiguous and not self.c_contiguous)
    behaved = property(lambda self: self.aligned and self.writeable)
    carray = property(lambda self: self.behaved and self.c_contiguous)
    farray = property(lambda self: self.behaved and self.f_contiguous and not self.c_contiguous)
    forc = property(lambda self: self.f_contiguous or self.c_contiguous)

    _names = {
        "C_CONTIGUOUS": "c_contiguous", "C": "c_contiguous", "F_CONTIGUOUS": "f_contiguous", "F": "f_contiguous",
        "OWNDATA": "owndata", "O": "owndata", "WRITEABLE": "writeable", "W": "writeable", "ALIGNED": "aligned", "A": "aligned",
        "WRITEBACKIFCOPY": "writebackifcopy", "X": "writebackifcopy", "CONTIGUOUS": "contiguous", "FORTRAN": "fortran",
        "BEHAVED": "behaved", "B": "behaved", "CARRAY": "carray", "CA": "carray", "FARRAY": "farray", "FA": "farray", "FNC": "fortran", "FORC": "forc",
    }

    def __getitem__(self, key):
        try:
            return getattr(self, self._names[key.upper()])
        except (KeyError, AttributeError):
            raise KeyError("Unknown flag") from None

    def __repr__(self):
        return "  C_CONTIGUOUS : %s\n  F_CONTIGUOUS : %s\n  OWNDATA : %s\n  WRITEABLE : %s\n  ALIGNED : %s\n  WRITEBACKIFCOPY : %s\n" % (
            self.c_contiguous, self.f_contiguous, self.owndata, self.writeable, self.aligned, self.writebackifcopy)


class generic:
    _kinds = "biufc"


class number(generic):
    _kinds = "iufc"


class integer(number):
    _kinds = "iu"


class signedinteger(integer):
    _kinds = "i"


class unsignedinteger(integer):
    _kinds = "u"


class inexact(number):
    _kinds = "fc"


class floating(inexact):
    _kinds = "f"


class complexfloating(inexact):
    _kinds = "c"


def issubdtype(arg1, arg2):
    def resolve(x):
        if isinstance(x, type) and issubclass(x, generic):
            return x
        if x is int:
            return signedinteger if False else _core.dtype("int64")
        if x is float:
            return _core.dtype("float64")
        if x is complex:
            return _core.dtype("complex128")
        if x is bool:
            return _core.dtype("bool")
        return _core.dtype(x)

    a1 = resolve(arg1)
    a2 = resolve(arg2)
    if isinstance(a1, type):
        return isinstance(a2, type) and issubclass(a1, a2)
    if isinstance(a2, type):
        return a1.kind in a2._kinds
    return a1 == a2


def isdtype(dtype, kind):
    dt = dtype if isinstance(dtype, _core.dtype) else _core.dtype(dtype)
    if isinstance(kind, tuple):
        return any(isdtype(dt, k) for k in kind)
    if isinstance(kind, str):
        table = {
            "bool": "b", "signed integer": "i", "unsigned integer": "u", "integral": "iu", "real floating": "f",
            "complex floating": "c", "numeric": "iufc",
        }
        if kind not in table:
            raise ValueError("kind argument is a string, but %r is not a known kind name." % kind)
        return dt.kind in table[kind]
    return dt == _core.dtype(kind)


def common_type(*arrays):
    is_complex = False
    precision = 0
    for a in arrays:
        t = asarray(a).dtype
        if t.kind in "iub":
            p = 2
        elif t.kind == "f" or t.kind == "c":
            base = {"float16": 0, "float32": 1, "float64": 2, "complex64": 1, "complex128": 2}[t.name]
            p = base
            if t.kind == "c":
                is_complex = True
        else:
            raise TypeError("can't get common type for non-numeric array")
        precision = max(precision, p)
    if is_complex:
        return _core.dtype(["complex64", "complex64", "complex128"][precision])
    return _core.dtype(["float16", "float32", "float64"][precision])


def mintypecode(typechars, typeset="GDFgdf", default="d"):
    typecodes = [(t if isinstance(t, str) else asarray(t).dtype.char) for t in typechars]
    intersection = [t for t in typecodes if t in typeset]
    if not intersection:
        return default
    if "F" in intersection and "d" in intersection:
        return "D"
    return min(intersection, key=lambda t: typeset.index(t))


def typename(char):
    names = {"?": "bool", "b": "signed char", "B": "unsigned char", "h": "short", "H": "unsigned short", "i": "integer", "I": "unsigned integer",
             "l": "long integer", "L": "unsigned long integer", "q": "long long integer", "Q": "unsigned long long integer", "f": "single precision",
             "d": "double precision", "g": "long precision", "F": "complex single precision", "D": "complex double precision"}
    return names[char]


def shares_memory(a, b, max_work=None):
    return may_share_memory(a, b)


def may_share_memory(a, b, max_work=None):
    if not isinstance(a, ndarray) or not isinstance(b, ndarray):
        return False
    return a.__array_interface__["data"][0] // 1 == b.__array_interface__["data"][0] // 1 or _overlap(a, b)


def _overlap(a, b):
    def extent(x):
        ptr = x.__array_interface__["data"][0]
        lo = hi = ptr
        for n, s in zip(x.shape, x.strides):
            if n == 0:
                return None
            if s > 0:
                hi += (n - 1) * s
            else:
                lo += (n - 1) * s
        return lo, hi + x.itemsize

    ea, eb = extent(a), extent(b)
    if ea is None or eb is None:
        return False
    return ea[0] < eb[1] and eb[0] < ea[1]


class vectorize:
    def __init__(self, pyfunc=None, otypes=None, doc=None, excluded=None, cache=False, signature=None):
        self.pyfunc = pyfunc
        self.otypes = otypes
        self.excluded = set(excluded or ())
        self.__doc__ = doc or getattr(pyfunc, "__doc__", None)
        self.signature = signature
        if pyfunc is not None:
            functools.update_wrapper(self, pyfunc, updated=())

    def __call__(self, *args, **kwargs):
        pyfunc = self.pyfunc
        if pyfunc is None:
            raise TypeError("vectorize needs a function")
        vargs = []
        for i, a in enumerate(args):
            vargs.append(a if i in self.excluded else asarray(a))
        pos = [i for i, a in enumerate(args) if i not in self.excluded]
        kw_keys = [k for k in kwargs if k not in self.excluded]
        vals = [vargs[i] for i in pos] + [asarray(kwargs[k]) for k in kw_keys]
        if not vals:
            return asarray(pyfunc(*args, **kwargs))
        bcast = _core.broadcast_arrays(*vals)
        shape = tuple(bcast[0].shape)
        flat = [b.reshape((-1,)).tolist() for b in bcast]
        n = len(flat[0]) if flat else 0
        results = []
        for k in range(n):
            call_args = list(args)
            for slot, i in enumerate(pos):
                call_args[i] = flat[slot][k]
            call_kw = dict(kwargs)
            for j, key in enumerate(kw_keys):
                call_kw[key] = flat[len(pos) + j][k]
            results.append(pyfunc(*call_args, **call_kw))
        if n == 0:
            if self.otypes is None:
                raise ValueError("cannot call `vectorize` on size 0 inputs unless `otypes` is set")
            return _core.zeros(shape, self.otypes[0] if isinstance(self.otypes, (list, tuple)) else self.otypes)
        first = results[0]
        if isinstance(first, tuple):
            cols = list(zip(*results))
            return tuple(_core.array(list(c), self.otypes[i] if self.otypes else None).reshape(shape) for i, c in enumerate(cols))
        otype = None
        if self.otypes is not None:
            otype = self.otypes[0] if isinstance(self.otypes, (list, tuple)) else self.otypes
        return _core.array(results, otype).reshape(shape + tuple(_core.array(results, otype).shape[1:]))


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    arr = asarray(arr)
    nd = arr.ndim
    axis = _norm_axis(axis, nd)
    moved = _core.moveaxis(arr, axis, -1)
    lead = tuple(moved.shape[:-1])
    n_lanes = 1
    for s in lead:
        n_lanes *= s
    flat = moved.reshape((n_lanes, moved.shape[-1]))
    if n_lanes == 0:
        raise ValueError("Cannot apply_along_axis when any iteration dimensions are 0")
    outs = [asarray(func1d(flat[i], *args, **kwargs)) for i in range(n_lanes)]
    first_shape = tuple(outs[0].shape)
    stacked = _core.stack(outs, axis=0)
    res = stacked.reshape(lead + first_shape)
    if first_shape:
        k = len(first_shape)
        res = _core.moveaxis(res, tuple(range(len(lead), len(lead) + k)), tuple(range(axis, axis + k)))
    return res


def apply_over_axes(func, a, axes):
    val = asarray(a)
    n = val.ndim
    axes_list = [axes] if isinstance(axes, int) else list(axes)
    for axis in axes_list:
        if axis < 0:
            axis = n + axis
        res = func(val, axis)
        if res.ndim == val.ndim:
            val = res
        else:
            res = _core.expand_dims(res, axis)
            if res.ndim == val.ndim:
                val = res
            else:
                raise ValueError("function is not returning an array of the correct shape")
    return val


def piecewise(x, condlist, funclist, *args, **kw):
    x = asarray(x)
    n2 = len(funclist)
    if isinstance(condlist, ndarray) and (condlist.dtype.name == "bool" and condlist.ndim == x.ndim or condlist.ndim == 0):
        condlist = [condlist]
    condlist = [asarray(c, "bool") for c in condlist]
    n = len(condlist)
    if n == n2 - 1:
        otherwise = _core.logical_not(functools.reduce(_core.logical_or, condlist)) if condlist else _core.ones(tuple(x.shape), "bool")
        condlist = condlist + [otherwise]
        n += 1
    elif n != n2:
        raise ValueError("with %d condition(s), either %d or %d functions are expected" % (n, n, n + 1))
    y = _core.zeros(tuple(x.shape), x.dtype)
    for cond, func in zip(condlist, funclist):
        if callable(func):
            mask = _core.broadcast_to(cond, tuple(x.shape))
            if bool(_core.any(mask)):
                vals = func(x[mask], *args, **kw)
                y[mask] = vals
        else:
            mask = _core.broadcast_to(cond, tuple(x.shape))
            y[mask] = func
    return y


def frompyfunc(func, /, nin, nout, *, identity=None):
    from ._ufunc import ufunc

    v = vectorize(func)
    return ufunc(lambda *a: v(*a), getattr(func, "__name__", "frompyfunc"), nin, nout, identity)


class flatiter:
    def __init__(self, arr):
        self._arr = arr
        self.base = arr

    @property
    def index(self):
        return getattr(self, "_i", 0)

    @property
    def coords(self):
        return _unravel(self.index, tuple(self._arr.shape))

    def __len__(self):
        return self._arr.size

    def __iter__(self):
        return iter(self._arr.reshape((-1,)).tolist() if self._arr.dtype.kind == "c" else (self._arr.reshape((-1,))[i] for i in range(self._arr.size)))

    def __getitem__(self, key):
        flat = self._arr.reshape((-1,)) if self._arr.flags.c_contiguous else self._arr.copy().reshape((-1,))
        return flat[key]

    def __setitem__(self, key, value):
        from ._indexing import _unravel_flat

        idx = asarray(_core.arange(self._arr.size)[key] if not isinstance(key, int) else key)
        if idx.ndim == 0:
            self._arr[_unravel(int(idx), tuple(self._arr.shape))] = value
            return
        self._arr[_unravel_flat(idx, tuple(self._arr.shape))] = value

    def copy(self):
        return self._arr.reshape((-1,)).copy() if self._arr.flags.c_contiguous else self._arr.copy().reshape((-1,))

    def __next__(self):
        i = getattr(self, "_i", 0)
        if i >= self._arr.size:
            raise StopIteration
        self._i = i + 1
        return self._arr.reshape((-1,))[i]


def _unravel(flat, shape):
    idx = []
    for s in reversed(shape):
        idx.append(flat % s if s else 0)
        flat //= s if s else 1
    return tuple(reversed(idx))
