import struct as _struct

from . import _core

_INT64 = (-(2**63), 2**63 - 1)


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
        return (type(self), (self._plain(),))


def _wrap_result(cls_for, value):
    cls = cls_for.get(type(value))
    return cls(value) if cls is not None else value


def _install(cls, base, dtype, ops, unary):
    cls._base = base
    cls._rnp_dtype = dtype
    table = _RESULT[dtype]

    def binary(name):
        method = getattr(base, name)

        fallback = _IEEE.get(name)

        def op(self, other):
            try:
                result = method(self, other)
            except (ZeroDivisionError, OverflowError):
                if fallback is None:
                    raise
                x, y = (other, self) if name.startswith("__r") else (self, other)
                result = getattr(_core, fallback)(x, y)
                return _F64(result) if type(result) is float else result
            if result is NotImplemented:
                return result
            return _checked(table, result)

        op.__name__ = name
        return op

    def unary_op(name):
        method = getattr(base, name)

        def op(self):
            return _checked(table, method(self))

        op.__name__ = name
        return op

    def comparison(name):
        method = getattr(base, name)

        def op(self, other):
            result = method(self, other)
            return result if result is NotImplemented else _Bool(result)

        op.__name__ = name
        return op

    for name in ("__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"):
        if hasattr(base, name) and (base is not complex or name in ("__eq__", "__ne__")):
            setattr(cls, name, comparison(name))
    cls.__hash__ = base.__hash__
    for name in ops:
        setattr(cls, name, binary(name))
    for name in unary:
        setattr(cls, name, unary_op(name))


def _checked(table, value):
    cls = table.get(type(value))
    if cls is None:
        return value
    if cls is _I64 and not (_INT64[0] <= value <= _INT64[1]):
        return value
    return cls(value)


class _F64(_ScalarBase, float):
    __slots__ = ()


class _C128(_ScalarBase, complex):
    __slots__ = ()


class _I64(_ScalarBase, int):
    __slots__ = ()


class _Bool(_ScalarBase):
    __slots__ = ("_v",)
    _base = bool

    def __init__(self, value=False):
        object.__setattr__(self, "_v", bool(value))

    def _plain(self):
        return self._v

    def _array(self):
        return _core.array(self._v, "bool")

    def __bool__(self):
        return self._v

    def __int__(self):
        return int(self._v)

    __index__ = __int__

    def __float__(self):
        return float(self._v)

    def __complex__(self):
        return complex(self._v)

    def __hash__(self):
        return hash(self._v)

    def __repr__(self):
        return "True" if self._v else "False"

    __str__ = __repr__

    def __invert__(self):
        return _Bool(not self._v)

    def __eq__(self, other):
        return self._v == (other._v if isinstance(other, _Bool) else other)

    def __ne__(self, other):
        return self._v != (other._v if isinstance(other, _Bool) else other)

    def __lt__(self, other):
        return self._v < (other._v if isinstance(other, _Bool) else other)

    def __le__(self, other):
        return self._v <= (other._v if isinstance(other, _Bool) else other)

    def __gt__(self, other):
        return self._v > (other._v if isinstance(other, _Bool) else other)

    def __ge__(self, other):
        return self._v >= (other._v if isinstance(other, _Bool) else other)

    def __and__(self, other):
        return _Bool(self._v & bool(other)) if isinstance(other, (bool, int, _Bool)) else NotImplemented

    def __or__(self, other):
        return _Bool(self._v | bool(other)) if isinstance(other, (bool, int, _Bool)) else NotImplemented

    def __xor__(self, other):
        return _Bool(self._v ^ bool(other)) if isinstance(other, (bool, int, _Bool)) else NotImplemented

    __rand__, __ror__, __rxor__ = __and__, __or__, __xor__


def _int_delegate(name):
    method = getattr(int, name)

    def op(self, other):
        return method(int(self._v), other._v if isinstance(other, _Bool) else other)

    op.__name__ = name
    return op


_IEEE = {
    "__truediv__": "divide", "__rtruediv__": "divide", "__floordiv__": "floor_divide", "__rfloordiv__": "floor_divide",
    "__mod__": "remainder", "__rmod__": "remainder", "__pow__": "power", "__rpow__": "power",
}
_RESULT = {
    "float64": {float: _F64},
    "complex128": {complex: _C128},
    "int64": {int: _I64},
    "bool": {bool: _Bool},
}
_RESULT["float64"][_F64] = _F64
_ARITH = ["__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__truediv__", "__rtruediv__", "__pow__", "__rpow__"]
_INT_ARITH = _ARITH + ["__floordiv__", "__rfloordiv__", "__mod__", "__rmod__", "__lshift__", "__rshift__", "__and__", "__rand__", "__or__", "__ror__", "__xor__", "__rxor__"]
_install(_F64, float, "float64", _ARITH + ["__floordiv__", "__rfloordiv__", "__mod__", "__rmod__"], ["__neg__", "__pos__", "__abs__"])
_install(_C128, complex, "complex128", _ARITH, ["__neg__", "__pos__"])
_install(_I64, int, "int64", _INT_ARITH, ["__neg__", "__pos__", "__abs__", "__invert__"])
_Bool._rnp_dtype = "bool"
for _n in ("__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__neg__"):
    setattr(_Bool, _n, _int_delegate(_n) if _n != "__neg__" else (lambda self: -int(self._v)))
_RESULT["float64"] = {float: _F64}
_RESULT["complex128"] = {complex: _C128}
_RESULT["int64"] = {int: _I64}
for _cls, _name in ((_F64, "float64"), (_C128, "complex128"), (_I64, "int64"), (_Bool, "bool")):
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
