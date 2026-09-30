import struct as _struct

from . import _core


class _ScalarBase:
    _rnp_scalar = True
    shape = ()
    ndim = 0
    size = 1

    @property
    def dtype(self):
        return _core.dtype(self._rnp_dtype)

    @property
    def itemsize(self):
        return self.dtype.itemsize

    nbytes = itemsize

    def _plain(self):
        return self._base(self)

    def _array(self):
        return _core.array(self._plain(), self._rnp_dtype)

    def __getitem__(self, index):
        if isinstance(index, tuple) and not index:
            return self
        return self._array()[index]

    def item(self, *args):
        return self._plain()

    def tolist(self):
        return self._plain()

    def astype(self, dtype, *args, **kwargs):
        return self._array().astype(dtype, *args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._array(), name)

    @property
    def __array_interface__(self):
        value = self._plain()
        typestr, data = {
            "bool": ("|b1", _struct.pack("?", value)),
            "int64": ("<i8", _struct.pack("<q", value)),
            "float64": ("<f8", _struct.pack("<d", value)),
            "complex128": ("<c16", _struct.pack("<dd", value.real, value.imag)),
        }[self._rnp_dtype]
        return {"version": 3, "shape": (), "typestr": typestr, "data": data}

    def __array_namespace__(self, *, api_version=None):
        return self._array().__array_namespace__(api_version=api_version)

    def __reduce__(self):
        return (scalar, (self._rnp_dtype, self._plain()))


_BINARY = [
    f"__{r}{name}__"
    for name in ("add", "sub", "mul", "truediv", "floordiv", "mod", "divmod", "pow", "lshift", "rshift", "and", "or", "xor", "matmul")
    for r in ("", "r")
]
_UNARY = ["__neg__", "__pos__", "__abs__", "__invert__"]
_COMPARE = ["__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"]


_CORE = {
    "add": "add", "sub": "subtract", "mul": "multiply", "truediv": "divide", "floordiv": "floor_divide",
    "mod": "remainder", "pow": "power", "lshift": "left_shift", "rshift": "right_shift",
    "and": "bitwise_and", "or": "bitwise_or", "xor": "bitwise_xor",
    "eq": "equal", "ne": "not_equal", "lt": "less", "le": "less_equal", "gt": "greater", "ge": "greater_equal",
}


def _delegate(name):
    def op(self, *args):
        return getattr(self._array(), name)(*args)

    op.__name__ = name
    return op


def _direct(name):
    reflected = name.startswith("__r") and name[3:-2] in _CORE
    fn = getattr(_core, _CORE[name[3:-2] if reflected else name[2:-2]])
    slow = _delegate(name)

    def op(self, other):
        if type(other) not in _FAST:
            return slow(self, other)
        return fn(other, self) if reflected else fn(self, other)

    op.__name__ = name
    return op


_I64_RANGE = (-(2**63), 2**63 - 1)
_ARITH_KEYS = ("add", "sub", "mul", "truediv", "pow")
_CMP_KEYS = ("eq", "ne", "lt", "le", "gt", "ge")


def _fast_float(key, reflected, slow):
    if key in _CMP_KEYS:
        method = getattr(float, f"__{key}__")

        def cmp(self, other):
            if type(other) in _FAST_FLOAT_CMP:
                return _Bool(method(self, other))
            return slow(self, other)

        return cmp
    if key not in _ARITH_KEYS:
        return None
    method = getattr(float, f"__{'r' if reflected else ''}{key}__")

    def arith(self, other):
        if type(other) in _FAST_FLOAT_ARITH:
            try:
                r = method(self, other)
            except (OverflowError, ZeroDivisionError):
                return slow(self, other)
            if type(r) is float:
                return _F64(r)
        return slow(self, other)

    return arith


def _fast_int(key, reflected, slow):
    lo, hi = _I64_RANGE
    if key in _CMP_KEYS:
        method = getattr(int, f"__{key}__")

        def cmp(self, other):
            t = type(other)
            if t is int:
                return _Bool(method(self._v, other))
            if t is _I64:
                return _Bool(method(self._v, other._v))
            return slow(self, other)

        return cmp
    if key not in ("add", "sub", "mul", "truediv", "floordiv", "mod"):
        return None
    method = getattr(int, f"__{'r' if reflected else ''}{key}__")

    def arith(self, other):
        t = type(other)
        if t is _I64:
            other = other._v
        elif t is not int or not lo <= other <= hi:
            return slow(self, other)
        try:
            r = method(self._v, other)
        except ZeroDivisionError:
            return slow(self, other)
        if key == "truediv":
            a, b = (other, self._v) if reflected else (self._v, other)
            return _F64(float(a) / float(b))
        return _I64(r) if lo <= r <= hi else slow(self, other)

    return arith


def _install(cls):
    for name in _BINARY + _UNARY + _COMPARE:
        reflected = name.startswith("__r") and name[3:-2] in _CORE
        key = name[3:-2] if reflected else name[2:-2]
        if key not in _CORE:
            setattr(cls, name, _delegate(name))
            continue
        op = _direct(name)
        fast = {"float64": _fast_float, "int64": _fast_int}.get(cls._rnp_dtype)
        setattr(cls, name, (fast(key, reflected, op) if fast else None) or op)
    cls.__hash__ = lambda self: hash(self._plain())


class _F64(_ScalarBase, float):
    __slots__ = ()


class _C128(_ScalarBase, complex):
    __slots__ = ()


class _Plain(_ScalarBase):
    __slots__ = ("_v",)

    def __init__(self, value=0):
        object.__setattr__(self, "_v", self._base(value))

    def _plain(self):
        return self._v

    def __bool__(self):
        return bool(self._v)

    def __int__(self):
        return int(self._v)

    __index__ = __int__

    def __float__(self):
        return float(self._v)

    def __complex__(self):
        return complex(self._v)

    def __repr__(self):
        return repr(self._v)

    __str__ = __repr__

    def __format__(self, spec):
        return format(self._v, spec)

    def __round__(self, ndigits=None):
        return round(self._v, ndigits)

    __trunc__ = __floor__ = __ceil__ = __int__


class _I64(_Plain):
    __slots__ = ()
    _base = int


class _Bool(_Plain):
    __slots__ = ()
    _base = bool


_FAST = {int, float, complex, bool, _F64, _C128, _I64, _Bool}
_FAST_FLOAT_ARITH = {float, _F64, int, bool}
_FAST_FLOAT_CMP = {float, _F64}

for _cls, _base, _name in ((_F64, float, "float64"), (_C128, complex, "complex128"), (_I64, int, "int64"), (_Bool, bool, "bool")):
    _cls._base = _base
    _cls._rnp_dtype = _name
    _install(_cls)
    _cls.__name__ = _cls.__qualname__ = _name
    _cls.__module__ = "rustnumpy"


def scalar(dtype, value):
    if dtype == "float64":
        return _F64(value)
    if dtype == "int64":
        return _I64(value)
    if dtype == "bool":
        return _Bool(value)
    return _C128(value)
