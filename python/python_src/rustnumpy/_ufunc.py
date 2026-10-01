from . import _core
from ._core import asarray, ndarray

_NoValue = type("_NoValue", (), {"__repr__": lambda self: "<no value>"})()


class UFuncTypeError(TypeError):
    pass


def _dtype_of(x):
    return x.dtype if isinstance(x, ndarray) else _core.asarray(x).dtype


def _as_tuple(axis, ndim, name):
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, tuple):
        raw = axis
    else:
        raw = (axis,)
    out = []
    for a in raw:
        if not isinstance(a, int) or isinstance(a, bool):
            raise TypeError("'%s' object cannot be interpreted as an integer" % type(a).__name__)
        if a < -ndim or a >= ndim:
            raise _core_axis_error(a, ndim)
        out.append(a % ndim if ndim else 0)
    if len(set(out)) != len(out):
        raise ValueError("duplicate value in 'axis'")
    return tuple(out)


def _core_axis_error(axis, ndim):
    return ValueError("axis %d is out of bounds for array of dimension %d" % (axis, ndim))


def _write_out(res, out, where, casting, name):
    if isinstance(out, tuple):
        if len(out) != 1:
            raise ValueError("The 'out' tuple must have exactly one entry per ufunc output")
        out = out[0]
    if out is None:
        return res
    if not isinstance(out, ndarray):
        raise TypeError("return arrays must be of ArrayType")
    r = asarray(res)
    if not _core.can_cast(r.dtype, out.dtype, casting):
        raise UFuncTypeError(
            "Cannot cast ufunc '%s' output from dtype('%s') to dtype('%s') with casting rule '%s'"
            % (name, r.dtype.name, out.dtype.name, casting)
        )
    target = tuple(out.shape)
    try:
        r = _core.broadcast_to(r, target)
    except ValueError:
        raise ValueError(
            "non-broadcastable output operand with shape %s doesn't match the broadcast shape %s" % (target, tuple(r.shape))
        ) from None
    if where is True:
        out[...] = r
    else:
        mask = _core.broadcast_to(asarray(where, "bool"), target)
        out[mask] = r[mask]
    return out


class ufunc:
    def __init__(self, fn, name, nin, nout=1, identity=None, reorderable=True, bool_result=False):
        self._fn = fn
        self.__name__ = name
        self.__qualname__ = name
        self.__doc__ = "%s(*args, out=None, where=True, casting='same_kind', dtype=None)\n\nA rustnumpy ufunc." % name
        self.nin = nin
        self.nout = nout
        self.nargs = nin + nout
        self.identity = identity
        self.signature = None
        self._reorderable = reorderable
        self._bool_result = bool_result

    def __repr__(self):
        return "<ufunc '%s'>" % self.__name__

    def __call__(self, *args, out=None, where=True, casting="same_kind", dtype=None, order="K", subok=True, signature=None):
        if not args:
            raise TypeError("%s() takes from %d to %d positional arguments but 0 were given" % (self.__name__, self.nin, self.nargs))
        if len(args) > self.nargs:
            raise TypeError("%s() takes from %d to %d positional arguments but %d were given" % (self.__name__, self.nin, self.nargs, len(args)))
        if len(args) < self.nin:
            raise TypeError("%s() takes from %d to %d positional arguments but %d were given" % (self.__name__, self.nin, self.nargs, len(args)))
        if len(args) > self.nin:
            if out is not None:
                raise TypeError("cannot specify 'out' as both a positional and keyword argument")
            out = args[self.nin:] if self.nout > 1 else args[self.nin]
            args = args[: self.nin]
        if signature is not None and dtype is None:
            dtype = signature if isinstance(signature, (str, type)) else None
        if out is None and where is True and dtype is None and casting == "same_kind":
            return self._fn(*args)
        if casting not in ("no", "equiv", "safe", "same_kind", "unsafe"):
            raise ValueError("casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'")
        if dtype is not None:
            args = self._cast_inputs(args, dtype, casting)
        res = self._fn(*args)
        if self.nout == 1:
            return _write_out(res, out, where, casting, self.__name__)
        if out is None:
            return res
        if not isinstance(out, tuple) or len(out) != self.nout:
            raise ValueError("The 'out' tuple must have exactly one entry per ufunc output")
        return tuple(_write_out(r, o, where, casting, self.__name__) for r, o in zip(res, out))

    def _cast_inputs(self, args, dtype, casting):
        dt = _core.dtype(dtype)
        cast = []
        for a in args:
            arr = asarray(a)
            if isinstance(a, ndarray) and not _core.can_cast(arr.dtype, dt, casting):
                raise UFuncTypeError(
                    "Cannot cast ufunc '%s' input from dtype('%s') to dtype('%s') with casting rule '%s'"
                    % (self.__name__, arr.dtype.name, dt.name, casting)
                )
            cast.append(arr.astype(dt, copy=False))
        return tuple(cast)

    def outer(self, A, B, /, **kwargs):
        if self.nin != 2:
            raise ValueError("outer product only supported for binary functions")
        a, b = asarray(A), asarray(B)
        return self(a.reshape(tuple(a.shape) + (1,) * b.ndim), b, **kwargs)

    def _prepare(self, a):
        name = self.__name__
        if name in ("logical_and", "logical_or", "logical_xor"):
            return a.astype("bool")
        if self._bool_result and a.dtype.name != "bool":
            raise UFuncTypeError("No loop matching the specified signature and casting was found for ufunc %s" % name)
        return a

    def _widen(self, a):
        if self.__name__ in ("add", "multiply") and (a.dtype.name == "bool" or (a.dtype.kind in "iu" and a.dtype.itemsize < 8)):
            return a.astype("uint64" if a.dtype.kind == "u" else "int64")
        return a

    def _identity_for(self, dtype):
        ident = self.identity
        if ident is None:
            return None
        if self.__name__ == "bitwise_and" and dtype.kind == "u":
            return int(2 ** (8 * dtype.itemsize) - 1)
        return ident

    def reduce(self, array, axis=0, dtype=None, out=None, keepdims=False, initial=_NoValue, where=True):
        if self.nin != 2 or self.nout != 1:
            raise ValueError("reduce only supported for binary functions")
        a = asarray(array)
        nd = a.ndim
        axes = _as_tuple(axis, nd, "axis")
        if len(axes) > 1 and not self._reorderable:
            raise ValueError("reduction operation '%s' is not reorderable, so at most one axis may be specified" % self.__name__)
        if dtype is not None:
            a = a.astype(dtype)
        else:
            a = self._widen(self._prepare(a))
        if where is not True:
            ident = self._identity_for(a.dtype) if initial is _NoValue else initial
            if ident is None:
                raise ValueError(
                    "reduction operation '%s' does not have an identity, so to use a where mask one has to specify 'initial'" % self.__name__
                )
            mask = _core.broadcast_to(asarray(where, "bool"), tuple(a.shape))
            a = _core.where(mask, a, asarray(ident, a.dtype))
        res = a
        for ax in sorted(axes, reverse=True):
            res = self._reduce_axis(res, ax, initial if initial is not _NoValue and len(axes) == 1 else _NoValue)
        if initial is not _NoValue and len(axes) != 1:
            res = self._fn(res, initial)
        if keepdims:
            shape = list(a.shape)
            for ax in axes:
                shape[ax] = 1
            res = asarray(res).reshape(tuple(shape))
        return _write_out(res, out, True, "same_kind", self.__name__)

    def _reduce_axis(self, a, ax, initial):
        n = a.shape[ax]
        name = self.__name__
        if n == 0:
            ident = initial if initial is not _NoValue else self._identity_for(a.dtype)
            if ident is None:
                raise ValueError("zero-size array to reduction operation %s which has no identity" % name)
            shape = tuple(s for i, s in enumerate(a.shape) if i != ax)
            dt = _core.result_type(a.dtype, ident) if not isinstance(ident, bool) else a.dtype
            if name in ("logical_and", "logical_or"):
                dt = "bool"
            return _core.full(shape, ident, dt)
        native = _FAST.get(name)
        if native is not None:
            res = native(a, ax)
        else:
            idx = [slice(None)] * a.ndim
            idx[ax] = 0
            res = asarray(a[tuple(idx)]).copy()
            for i in range(1, n):
                idx[ax] = i
                res = asarray(self._fn(res, a[tuple(idx)]))
        if initial is not _NoValue:
            res = self._fn(res, initial)
        return res

    def accumulate(self, array, axis=0, dtype=None, out=None):
        if self.nin != 2 or self.nout != 1:
            raise ValueError("accumulate only supported for binary functions")
        a = asarray(array)
        if a.ndim == 0:
            raise ValueError("accumulate does not allow multiple axes")
        (ax,) = _as_tuple(axis, a.ndim, "axis")
        if dtype is not None:
            a = a.astype(dtype)
        else:
            a = self._widen(self._prepare(a))
        n = a.shape[ax]
        if n == 0:
            return _write_out(a, out, True, "unsafe", self.__name__)
        if self.__name__ == "add":
            res = _core.cumsum(a, ax)
        elif self.__name__ == "multiply":
            res = _core.cumprod(a, ax)
        else:
            idx = [slice(None)] * a.ndim
            parts = []
            idx[ax] = 0
            cur = asarray(a[tuple(idx)]).copy()
            parts.append(cur)
            for i in range(1, n):
                idx[ax] = i
                cur = asarray(self._fn(cur, a[tuple(idx)]))
                parts.append(cur)
            res = _core.stack(parts, axis=ax)
        return _write_out(res, out, True, "unsafe", self.__name__)

    def reduceat(self, array, indices, axis=0, dtype=None, out=None):
        if self.nin != 2 or self.nout != 1:
            raise ValueError("reduceat only supported for binary functions")
        a = asarray(array)
        (ax,) = _as_tuple(axis, a.ndim, "axis")
        if dtype is not None:
            a = a.astype(dtype)
        idx = [int(i) for i in asarray(indices).tolist()]
        n = a.shape[ax]
        parts = []
        for k, start in enumerate(idx):
            if start < 0 or start >= n:
                raise IndexError("index %d out-of-bounds in %s.reduceat [0, %d)" % (start, self.__name__, n))
            stop = idx[k + 1] if k + 1 < len(idx) else n
            sl = [slice(None)] * a.ndim
            if stop <= start:
                sl[ax] = start
                parts.append(asarray(a[tuple(sl)]).copy())
            else:
                sl[ax] = slice(start, stop)
                parts.append(asarray(self.reduce(a[tuple(sl)], axis=ax)))
        res = _core.stack(parts, axis=ax)
        return _write_out(res, out, True, "unsafe", self.__name__)

    def at(self, a, indices, b=None):
        if not isinstance(a, ndarray):
            raise TypeError("first operand must be array")
        positions = _core.arange(a.size).reshape(tuple(a.shape))[indices]
        flat = asarray(positions).ravel().tolist()
        if self.nin == 2:
            vals = _core.broadcast_to(asarray(b), tuple(asarray(positions).shape)).ravel()
        for k, p in enumerate(flat):
            idx = _unravel(p, tuple(a.shape))
            cur = a[idx]
            a[idx] = self._fn(cur, vals[k]) if self.nin == 2 else self._fn(cur)
        return None


class gufunc(ufunc):
    def __init__(self, fn, name, signature, core_in, core_out, single_core=False):
        super().__init__(fn, name, 2)
        self.__doc__ = "%s(x1, x2, /, out=None, *, axes=None, axis=None, keepdims=False, casting='same_kind', dtype=None)\n\nA rustnumpy generalized ufunc." % name
        self.signature = signature
        self._core_in = core_in
        self._core_out = core_out
        self._single_core = single_core

    def _core_dims(self, ndim, spec):
        return min(spec, ndim) if isinstance(spec, int) else spec(ndim)

    def __call__(self, *args, out=None, axes=None, axis=None, keepdims=False, where=_NoValue, **kwargs):
        name = self.__name__
        if where is not _NoValue:
            raise TypeError("%s() got an unexpected keyword argument 'where'" % name)
        if axes is None and axis is None and not keepdims:
            return super().__call__(*args, out=out, **kwargs)
        if len(args) != 2:
            return super().__call__(*args, out=out, **kwargs)
        if axes is not None and axis is not None:
            raise TypeError("cannot specify both 'axis' and 'axes'")
        if axis is not None and not self._single_core:
            raise TypeError("%s: axis can only be used with a single shared core dimension, not with the distinct dimensions in signature %s" % (name, self.signature))
        if keepdims and not self._single_core:
            raise TypeError("%s does not support keepdims: its signature %s requires every output to have the same number of core dimensions as the inputs" % (name, self.signature))
        if axis is not None:
            axes = [(axis,), (axis,)] + ([(axis,)] if keepdims else [])
        operands = [asarray(a) for a in args]
        counts = [self._core_dims(o.ndim, spec) for o, spec in zip(operands, self._core_in)]
        n_out = self._core_out(operands) if callable(self._core_out) else self._core_out
        if keepdims:
            n_out = 1
        if axes is None:
            axes = [tuple(range(-c, 0)) for c in counts] + [tuple(range(-n_out, 0))]
        else:
            axes = [tuple(a) if isinstance(a, (tuple, list)) else (a,) for a in axes]
            if len(axes) == 2 and n_out == 0:
                axes.append(())
            if len(axes) != 3:
                raise ValueError("axes should be a list with an entry for each operand")
        moved = []
        for i, (o, c) in enumerate(zip(operands, counts)):
            if len(axes[i]) != c:
                raise _extra_axis_error("%s: operand %d has %d core dimensions, but %d dimensions are specified by axes tuple." % (name, i, c, len(axes[i])))
            moved.append(_core.moveaxis(o, axes[i], tuple(range(-c, 0))) if c else o)
        if len(axes[2]) != n_out:
            raise _extra_axis_error("%s: operand 2 has %d core dimensions, but %d dimensions are specified by axes tuple." % (name, n_out, len(axes[2])))
        res = asarray(super().__call__(*moved, **kwargs))
        if n_out:
            if keepdims:
                res = _core.expand_dims(res, axes[2][0])
            else:
                res = _core.moveaxis(res, tuple(range(-n_out, 0)), axes[2])
        return _write_out(res, out, True, kwargs.get("casting", "same_kind"), name)


def _extra_axis_error(message):
    from ._extra import AxisError

    return AxisError(message)


def _matvec(x1, x2):
    x1, x2 = asarray(x1), asarray(x2)
    return _core.matmul(x1, x2[..., None])[..., 0]


def _vecmat(x1, x2):
    x1, x2 = asarray(x1), asarray(x2)
    return _core.matmul(x1[..., None, :], x2)[..., 0, :]


def _bitwise_count(x):
    x = asarray(x)
    if x.dtype.kind not in "iub":
        raise TypeError("ufunc 'bitwise_count' not supported for the input types")
    if x.dtype.kind == "i":
        ux = _core.where(_core.less(x, 0), _core.absolute(x.astype("int64")).astype("uint64"), x.astype("uint64"))
    else:
        ux = x.astype("uint64")
    count = _core.zeros(tuple(ux.shape), "uint8")
    for bit in range(64):
        count = _core.add(count, _core.bitwise_and(_core.right_shift(ux, bit), 1).astype("uint8"))
    return count


def _unravel(flat, shape):
    idx = []
    for s in reversed(shape):
        idx.append(flat % s)
        flat //= s
    return tuple(reversed(idx))


def _fast_add(a, ax):
    return _core.sum(a, ax)


def _fast_mul(a, ax):
    return _core.prod(a, ax)


def _fast_max(a, ax):
    return _core.max(a, ax)


def _fast_min(a, ax):
    return _core.min(a, ax)


def _fast_all(a, ax):
    return _core.all(a, ax)


def _fast_any(a, ax):
    return _core.any(a, ax)


_FAST = {
    "add": _fast_add,
    "multiply": _fast_mul,
    "maximum": _fast_max,
    "minimum": _fast_min,
    "logical_and": _fast_all,
    "logical_or": _fast_any,
}


def _make(name, fn, nin, nout=1, identity=None, reorderable=True, bool_result=False):
    return ufunc(fn, name, nin, nout, identity, reorderable, bool_result)


def _install(namespace):
    unary = (
        "absolute negative positive sign square sqrt cbrt reciprocal exp exp2 expm1 log log2 log10 log1p sin cos tan arcsin "
        "arccos arctan sinh cosh tanh arcsinh arccosh arctanh degrees radians floor ceil trunc rint fabs isnan isinf isfinite "
        "signbit invert logical_not conjugate spacing"
    ).split()
    for name in unary:
        namespace[name] = _make(name, getattr(_core, name), 1)
    for name in ("modf", "frexp"):
        namespace[name] = _make(name, getattr(_core, name), 1, 2)
    identities = {
        "add": 0, "multiply": 1, "logical_and": True, "logical_or": False, "logical_xor": False, "bitwise_and": -1,
        "bitwise_or": 0, "bitwise_xor": 0, "hypot": 0, "logaddexp": float("-inf"), "gcd": 0, "lcm": 1, "equal": None,
    }
    non_reorderable = {"subtract", "divide", "true_divide", "floor_divide", "remainder", "power", "float_power", "arctan2",
                       "fmod", "lcm", "left_shift", "right_shift", "less", "less_equal", "greater", "greater_equal", "not_equal",
                       "equal", "copysign", "nextafter", "heaviside", "ldexp", "logaddexp2"}
    binary = (
        "add subtract multiply divide floor_divide remainder fmod power float_power maximum minimum fmax fmin arctan2 hypot "
        "copysign nextafter logaddexp logaddexp2 heaviside gcd lcm logical_and logical_or logical_xor bitwise_and bitwise_or "
        "bitwise_xor left_shift right_shift equal not_equal less less_equal greater greater_equal ldexp"
    ).split()
    for name in binary:
        namespace[name] = _make(name, getattr(_core, name), 2, 1, identities.get(name), name not in non_reorderable,
                                name in ("equal", "not_equal", "less", "less_equal", "greater", "greater_equal"))
    namespace["divmod"] = _make("divmod", _core.divmod, 2, 2, None, False)
    namespace["true_divide"] = namespace["divide"]
    namespace["mod"] = namespace["remainder"]
    namespace["abs"] = namespace["absolute"]
    namespace["conj"] = namespace["conjugate"]
    namespace["bitwise_not"] = namespace["bitwise_invert"] = namespace["invert"]
    namespace["bitwise_left_shift"] = namespace["left_shift"]
    namespace["bitwise_right_shift"] = namespace["right_shift"]
    namespace["rad2deg"] = namespace["degrees"]
    namespace["deg2rad"] = namespace["radians"]
    namespace["pow"] = namespace["power"]
    for new, old in (("acos", "arccos"), ("asin", "arcsin"), ("atan", "arctan"), ("acosh", "arccosh"), ("asinh", "arcsinh"),
                     ("atanh", "arctanh"), ("atan2", "arctan2")):
        namespace[new] = namespace[old]
    namespace["matmul"] = gufunc(_core.matmul, "matmul", "(n?,k),(k,m?)->(n?,m?)", (2, 2), lambda ops: sum(o.ndim >= 2 for o in ops))
    namespace["vecdot"] = gufunc(_core.vecdot, "vecdot", "(n),(n)->()", (1, 1), 0, True)
    namespace["matvec"] = gufunc(_matvec, "matvec", "(m,n),(n)->(m)", (2, 1), 1)
    namespace["vecmat"] = gufunc(_vecmat, "vecmat", "(n),(n,m)->(m)", (1, 2), 1)
    namespace["bitwise_count"] = _make("bitwise_count", _bitwise_count, 1)
