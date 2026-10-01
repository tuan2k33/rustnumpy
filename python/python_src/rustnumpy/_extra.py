from . import _core
from ._core import asarray, ndarray
from ._misc import flatiter as flatiter
from ._axiserror import AxisError as AxisError


class ComplexWarning(RuntimeWarning):
    pass


class RankWarning(RuntimeWarning):
    pass


class VisibleDeprecationWarning(UserWarning):
    pass


class TooHardError(RuntimeError):
    pass


class DTypePromotionError(TypeError):
    pass


def isposinf(x, out=None):
    x = asarray(x)
    if x.dtype.kind == "c":
        raise TypeError("ufunc 'isposinf' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''")
    res = _core.logical_and(_core.isinf(x), _core.greater(x, 0))
    if out is not None:
        out[...] = res
        return out
    return res


def isneginf(x, out=None):
    x = asarray(x)
    if x.dtype.kind == "c":
        raise TypeError("ufunc 'isneginf' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''")
    res = _core.logical_and(_core.isinf(x), _core.less(x, 0))
    if out is not None:
        out[...] = res
        return out
    return res


def ix_(*args):
    out = []
    nd = len(args)
    for k, new in enumerate(args):
        new = asarray(new)
        if new.ndim != 1:
            raise ValueError("Cross index must be 1 dimensional")
        if new.size == 0:
            new = new.astype("intp")
        if new.dtype.name == "bool":
            from ._indexing import nonzero

            (new,) = nonzero(new)
        out.append(new.reshape((1,) * k + (new.size,) + (1,) * (nd - k - 1)))
    return tuple(out)


def isfortran(a):
    a = asarray(a)
    return bool(a.flags.f_contiguous and not a.flags.c_contiguous)


def asarray_chkfinite(a, dtype=None, order=None):
    a = asarray(a, dtype)
    if a.dtype.kind in "fc" and not bool(_core.all(_core.isfinite(a))):
        raise ValueError("array must not contain infs or NaNs")
    return a


def astype(x, dtype, /, *, copy=True, device=None):
    x = asarray(x)
    return x.astype(dtype, copy=copy)


def min_scalar_type(a):
    a = asarray(a)
    if a.ndim == 0 and a.dtype.kind in "iu":
        v = int(a)
        if v >= 0:
            for n in ("uint8", "uint16", "uint32", "uint64"):
                if v < 2 ** (8 * _core.dtype(n).itemsize):
                    return _core.dtype(n)
        for n in ("int8", "int16", "int32", "int64"):
            b = 8 * _core.dtype(n).itemsize
            if -(2 ** (b - 1)) <= v < 2 ** (b - 1):
                return _core.dtype(n)
    if a.ndim == 0 and a.dtype.kind in "fc":
        parts = [float(a)] if a.dtype.kind == "f" else [a.item().real, a.item().imag]
        if a.dtype.kind == "f":
            v = parts[0]
            small = "float16" if v != v or abs(v) == float("inf") or abs(v) < 65000 else "float32" if abs(v) < 3.4e38 else "float64"
        else:
            small = "complex64" if all(abs(v) < 3.4e38 for v in parts) else "complex128"
        smaller = _core.dtype(small)
        return smaller if smaller.itemsize < a.dtype.itemsize else a.dtype
    return a.dtype


def binary_repr(num, width=None):
    num = int(num)
    if num >= 0:
        s = format(num, "b")
        return s.zfill(width) if width else s
    if width is None:
        return "-" + format(-num, "b")
    twocomp = (1 << width) + num
    if twocomp < 0:
        raise ValueError("Insufficient bit `width` given (%d) for binary representation of number (%d)" % (width, num))
    return format(twocomp, "b").zfill(width)


def base_repr(number, base=2, padding=0):
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if base > len(digits):
        raise ValueError("Bases greater than 36 not handled in base_repr.")
    if base < 2:
        raise ValueError("Bases less than 2 not handled in base_repr.")
    num = abs(int(number))
    res = []
    while num:
        res.append(digits[num % base])
        num //= base
    if padding:
        res.append("0" * padding)
    if not res:
        res.append("0")
    if number < 0:
        res.append("-")
    return "".join(reversed(res))


def format_float_positional(x, precision=None, unique=True, fractional=True, trim="k", sign=False, pad_left=None, pad_right=None, min_digits=None):
    from ._print import _positional

    dtype = "float32" if asarray(x).dtype.name == "float32" else ("float16" if asarray(x).dtype.name == "float16" else "float64")
    return _positional(float(x), dtype, precision, unique, trim, sign, pad_left, pad_right, min_digits or 0)


def format_float_scientific(x, precision=None, unique=True, trim="k", sign=False, pad_left=None, exp_digits=None, min_digits=None):
    from ._print import _scientific

    dtype = "float32" if asarray(x).dtype.name == "float32" else ("float16" if asarray(x).dtype.name == "float16" else "float64")
    return _scientific(float(x), dtype, precision, unique, trim, sign, pad_left, exp_digits, min_digits or 0)


typecodes = {"Character": "c", "Integer": "bhilqnp", "UnsignedInteger": "BHILQNP", "Float": "efdg", "Complex": "FDG", "AllInteger": "bBhHiIlLqQnNpP", "AllFloat": "efdgFDG", "Datetime": "Mm", "All": "?bhilqnpBHILQNPefdgFDGSUVOMm"}
little_endian = True


def getbufsize():
    return 8192


def setbufsize(size):
    return 8192


def show_config(mode="stdout"):
    info = {"build": "rustnumpy (Rust core, PyO3)"}
    if mode == "dicts":
        return info
    print(info)


class _Emath:
    def _needs_complex(self, x):
        return x.dtype.kind != "c"

    def _lift(self, x, cond, fn):
        x = asarray(x)
        if x.dtype.kind in "biu":
            x = x.astype("float64")
        if x.dtype.kind != "c" and bool(_core.any(cond(x))):
            x = x.astype("complex64" if x.dtype.name in ("float32", "float16") else "complex128")
        return fn(x)

    def sqrt(self, x):
        return self._lift(x, lambda v: _core.less(v, 0), _core.sqrt)

    def log(self, x):
        return self._lift(x, lambda v: _core.less(v, 0), _core.log)

    def log2(self, x):
        return self._lift(x, lambda v: _core.less(v, 0), _core.log2)

    def log10(self, x):
        return self._lift(x, lambda v: _core.less(v, 0), _core.log10)

    def logn(self, n, x):
        return _core.divide(self.log(x), self.log(n))

    def power(self, x, p):
        return self._lift(x, lambda v: _core.less(v, 0), lambda v: _core.power(v, p))

    def arccos(self, x):
        return self._lift(x, lambda v: _core.greater(_core.absolute(v), 1), _core.arccos)

    def arcsin(self, x):
        return self._lift(x, lambda v: _core.greater(_core.absolute(v), 1), _core.arcsin)

    def arctanh(self, x):
        return self._lift(x, lambda v: _core.greater(_core.absolute(v), 1), _core.arctanh)


emath = _Emath()
