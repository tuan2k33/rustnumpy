import pickle

from . import _core
from ._core import asarray


def _m_sort(a, axis=-1, kind=None, order=None, *, stable=None, descending=False):
    if a.ndim == 0:
        raise ValueError("Cannot sort a 0-d array")
    from ._manip import sort

    a[...] = sort(a, axis, descending=descending)
    return None


def _m_partition(a, kth, axis=-1, kind="introselect", order=None):
    from ._manip import partition

    a[...] = partition(a, kth, axis)
    return None


def _m_resize(a, *new_shape, refcheck=True):
    from ._manip import resize

    if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
        new_shape = tuple(new_shape[0])
    if not a.flags.owndata:
        raise ValueError("cannot resize this array: it does not own its data")
    raise NotImplementedError("in-place ndarray.resize is not supported; use rustnumpy.resize")


def _m_view(a, dtype=None, type=None):
    if dtype is None or _core.dtype(dtype) == a.dtype:
        return a[...]
    dt = _core.dtype(dtype)
    if not a.flags.c_contiguous:
        raise ValueError("To change to a dtype of a different size, the last axis must be contiguous")
    raw = a.tobytes()
    from ._creation import frombuffer

    shape = list(a.shape)
    if a.ndim == 0:
        shape = [1]
    total = len(raw)
    last_bytes = (shape[-1] if shape else 1) * a.itemsize
    if last_bytes % dt.itemsize:
        raise ValueError("When changing to a smaller dtype, its size must be a divisor of the size of original dtype")
    shape[-1] = last_bytes // dt.itemsize
    return frombuffer(raw, dt).reshape(tuple(shape)) if total else _core.zeros(tuple(shape), dt)


def _m_byteswap(a, inplace=False):
    raw = bytearray(a.tobytes())
    sz = a.itemsize
    for i in range(0, len(raw), sz):
        raw[i : i + sz] = raw[i : i + sz][::-1]
    from ._creation import frombuffer

    res = frombuffer(bytes(raw), a.dtype).reshape(tuple(a.shape)) if raw else a.copy()
    if inplace:
        a[...] = res
        return a
    return res


def _m_tofile(a, fid, sep="", format="%s"):
    if hasattr(fid, "write"):
        fp, close = fid, False
    else:
        fp, close = open(fid, "wb" if not sep else "w"), True
    try:
        if sep:
            fp.write(sep.join(format % v for v in a.reshape((-1,)).tolist()))
        else:
            fp.write(a.tobytes())
    finally:
        if close:
            fp.close()


def _m_dump(a, file):
    if hasattr(file, "write"):
        pickle.dump(a, file)
    else:
        with open(file, "wb") as fp:
            pickle.dump(a, fp)


def _m_dumps(a):
    return pickle.dumps(a)


def _m_to_device(a, device, /, *, stream=None):
    if device != "cpu":
        raise ValueError("Unsupported device %r" % (device,))
    return a


def _m_setflags(a, write=None, align=None, uic=None):
    return None


def _m_unsupported(*args, **kwargs):
    raise _core.Unsupported("this ndarray method is not supported")


def _m_tobytes_alias(a, order="C"):
    return a.tobytes()


def _m_compress(a, condition, axis=None, out=None):
    from ._indexing import compress

    return compress(condition, a, axis=axis, out=out)
