import builtins as _b

from . import _core
from ._core import fft, hfft, ifft, ihfft, irfft, rfft


def _prod_check(out):
    if out is not None:
        raise NotImplementedError("out= is not supported by the fft functions")


def _cook_nd_args(a, s, axes, invreal=False):
    a = _core.asarray(a)
    shapeless = s is None
    if s is None:
        s = list(a.shape) if axes is None else [a.shape[ax] for ax in axes]
    s = [int(v) for v in s]
    if axes is None:
        axes = list(range(a.ndim)) if shapeless else list(range(-len(s), 0))
    axes = [int(ax) for ax in axes]
    if len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")
    if invreal and shapeless and axes:
        s[-1] = (a.shape[axes[-1]] - 1) * 2
    return a, s, axes


def _nd_c2c(one, a, s, axes, norm, out):
    _prod_check(out)
    a, s, axes = _cook_nd_args(a, s, axes)
    for ii in reversed(range(len(axes))):
        a = one(a, n=s[ii], axis=axes[ii], norm=norm)
    return a


def fftn(a, s=None, axes=None, norm=None, out=None):
    return _nd_c2c(fft, a, s, axes, norm, out)


def ifftn(a, s=None, axes=None, norm=None, out=None):
    return _nd_c2c(ifft, a, s, axes, norm, out)


def fft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    return _nd_c2c(fft, a, s, axes, norm, out)


def ifft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    return _nd_c2c(ifft, a, s, axes, norm, out)


def rfftn(a, s=None, axes=None, norm=None, out=None):
    _prod_check(out)
    a, s, axes = _cook_nd_args(a, s, axes)
    a = rfft(a, s[-1], axes[-1], norm)
    for ii in reversed(range(len(axes) - 1)):
        a = fft(a, s[ii], axes[ii], norm)
    return a


def irfftn(a, s=None, axes=None, norm=None, out=None):
    _prod_check(out)
    a, s, axes = _cook_nd_args(a, s, axes, invreal=True)
    for ii in range(len(axes) - 1):
        a = ifft(a, s[ii], axes[ii], norm)
    return irfft(a, s[-1], axes[-1], norm)


def rfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    return rfftn(a, s, axes, norm, out)


def irfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    return irfftn(a, s, axes, norm, out)


def _shift(x, axes, sign):
    x = _core.asarray(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [sign * (dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = sign * (x.shape[axes] // 2)
    else:
        shift = [sign * (x.shape[ax] // 2) for ax in axes]
    return _core.roll(x, shift, axes)


def fftshift(x, axes=None):
    return _shift(x, axes, 1)


def ifftshift(x, axes=None):
    return _shift(x, axes, -1)


def _freq_result(r, dtype, device):
    if device not in (None, "cpu"):
        raise ValueError('Device not understood. Only "cpu" is allowed, but received: %s' % (device,))
    return r if dtype is None else r.astype(dtype)


def fftfreq(n, d=1.0, *, xp=None, dtype=None, device=None):
    if not isinstance(n, _b.int):
        try:
            n = n.__index__()
        except AttributeError:
            raise ValueError("n should be an integer") from None
    if n == 0:
        raise ZeroDivisionError("float division by zero")
    val = 1.0 / (n * d)
    half = (n - 1) // 2 + 1
    ints = _core.concatenate([_core.arange(0, half), _core.arange(-(n // 2), 0)])
    return _freq_result(ints * val, dtype, device)


def rfftfreq(n, d=1.0, *, xp=None, dtype=None, device=None):
    if not isinstance(n, _b.int):
        try:
            n = n.__index__()
        except AttributeError:
            raise ValueError("n should be an integer") from None
    if n == 0:
        raise ZeroDivisionError("float division by zero")
    val = 1.0 / (n * d)
    return _freq_result(_core.arange(0, n // 2 + 1) * val, dtype, device)
