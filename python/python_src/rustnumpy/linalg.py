from . import _core
from ._core import LinAlgError, asarray
from . import _core as _c
from ._core import matmul
from ._numeric import cross, outer, vecdot
from ._manip import matrix_transpose, swapaxes, transpose as _transpose, moveaxis
from ._numeric import tensordot
import collections as _collections
from . import _manip


def diagonal(x, /, *, offset=0):
    return _manip.diagonal(x, offset, -2, -1)


def trace(x, /, *, offset=0, dtype=None):
    return _manip.trace(x, offset, -2, -1, dtype)

def _count(shape):
    n = 1
    for d in shape:
        n *= d
    return n


_single = {
    "inv": _c.inv, "cholesky": _c.cholesky, "det": _c.det, "solve": _c.solve, "pinv": _c.pinv, "matrix_power": _c.matrix_power,
}


def _stack(fn, a, *rest, core_dims=2, **kw):
    a = asarray(a)
    if a.ndim < core_dims:
        raise LinAlgError("%d-dimensional array given. Array must be at least two-dimensional" % a.ndim)
    if a.ndim == core_dims:
        return fn(a, *rest, **kw)
    lead = tuple(a.shape[:-core_dims])
    count = _count(lead)
    flat = a.reshape((count,) + tuple(a.shape[-core_dims:]))
    if count == 0:
        res = asarray(fn(_probe(a, core_dims), *rest, **kw))
        return _core.zeros(lead + tuple(res.shape), res.dtype)
    outs = [asarray(fn(flat[i], *rest, **kw)) for i in range(count)]
    return _c.stack(outs, axis=0).reshape(lead + tuple(outs[0].shape))


def _probe(a, core_dims):
    """A well-conditioned stand-in matrix, used only to learn the output shapes/dtypes for empty stacks."""
    shape = tuple(a.shape[-core_dims:])
    if core_dims == 2:
        return _core.eye(shape[0], shape[1], 0, a.dtype)
    return _core.zeros(shape, a.dtype)


def _rdt(m):
    return {"complex128": "float64", "complex64": "float32", "float32": "float32"}.get(m.dtype.name, "float64")


def _cdt(m):
    return m.dtype.name if m.dtype.kind in "fc" else "float64"


def _cxdt(m):
    return "complex64" if m.dtype.name in ("float32", "complex64") else "complex128"


def _guard(fn, empty):
    """Matrices with a zero dimension are answered from their shapes; the numeric kernels never see them."""
    return lambda m: empty(m) if m.size == 0 else fn(m)


def _empty_svd(m, full, uv):
    rows, cols = m.shape
    k = min(rows, cols)
    s = _core.zeros((k,), _rdt(m))
    if not uv:
        return s
    if full:
        return (_core.eye(rows, rows, 0, _cdt(m)), s, _core.eye(cols, cols, 0, _cdt(m)))
    return (_core.zeros((rows, k), _cdt(m)), s, _core.zeros((k, cols), _cdt(m)))


def _empty_qr(m, mode):
    rows, cols = m.shape
    k = min(rows, cols)
    dt = _cdt(m)
    if mode == "complete":
        return (_core.eye(rows, rows, 0, dt), _core.zeros((rows, cols), dt))
    return (_core.zeros((rows, k), dt), _core.zeros((k, cols), dt))


def _square(a):
    if a.shape[-1] != a.shape[-2]:
        raise LinAlgError("Last 2 dimensions of the array must be square")


def _cx(a):
    return a.dtype.kind == "c"


def inv(a):
    a = asarray(a)
    _square(a) if a.ndim >= 2 else None
    return _stack(_c.c_inv if _cx(a) else _c.inv, a)


def cholesky(a, /, *, upper=False):
    a = asarray(a)
    _square(a) if a.ndim >= 2 else None
    res = _stack(_guard(_c.c_cholesky if _cx(a) else _c.cholesky, lambda m: _core.zeros((0, 0), _cdt(m))), a)
    return _c.conjugate(matrix_transpose(res)) if upper else res


def det(a):
    a = asarray(a)
    _square(a) if a.ndim >= 2 else None
    return _stack(_c.c_det if _cx(a) else _c.det, a)


SlogdetResult = _collections.namedtuple("SlogdetResult", ["sign", "logabsdet"])


def slogdet(a):
    a = asarray(a)
    _square(a) if a.ndim >= 2 else None
    if _cx(a):
        one = lambda m: _c.c_slogdet(m)
    else:
        one = lambda m: tuple(_c.slogdet(m))
    if a.ndim == 2:
        sgn, lg = one(a)
        return SlogdetResult(sgn, lg)
    signs = _stack(lambda m: asarray(one(m)[0]), a)
    logs = _stack(lambda m: asarray(one(m)[1]), a)
    return SlogdetResult(signs, logs)


def solve(a, b):
    from ._manip import broadcast_shapes

    a, b = asarray(a), asarray(b)
    _square(a)
    solver = _c.c_solve if (_cx(a) or _cx(b)) else _c.solve
    vec = b.ndim == 1
    bb = b[:, None] if vec else b
    lead = broadcast_shapes(tuple(a.shape[:-2]), tuple(bb.shape[:-2]))
    dtype = _c.result_type(a.dtype, b.dtype, "float32" if a.dtype.kind in "iub" and b.dtype.kind in "iub" else a.dtype)
    if a.size == 0 or bb.size == 0 or _count(lead) == 0:
        res = _core.zeros(lead + (a.shape[-1], bb.shape[-1]), dtype if dtype.kind in "fc" else "float64")
    elif a.ndim == 2 and bb.ndim == 2:
        res = asarray(solver(a, bb))
    else:
        aa = _c.broadcast_to(a, lead + tuple(a.shape[-2:])).reshape((_count(lead),) + tuple(a.shape[-2:]))
        bb = _c.broadcast_to(bb, lead + tuple(bb.shape[-2:])).reshape((_count(lead),) + tuple(bb.shape[-2:]))
        outs = [asarray(solver(aa[i], bb[i])) for i in range(aa.shape[0])]
        res = _c.stack(outs, axis=0).reshape(lead + tuple(outs[0].shape))
    return res[..., 0] if vec else res


QRResult = _collections.namedtuple("QRResult", ["Q", "R"])
EighResult = _collections.namedtuple("EighResult", ["eigenvalues", "eigenvectors"])
EigResult = _collections.namedtuple("EigResult", ["eigenvalues", "eigenvectors"])
SVDResult = _collections.namedtuple("SVDResult", ["U", "S", "Vh"])


def _tuple_stack(fn, a, wrap):
    a = asarray(a)
    if a.ndim < 2:
        raise LinAlgError("%d-dimensional array given. Array must be at least two-dimensional" % a.ndim)
    if a.ndim == 2:
        return wrap(*fn(a))
    lead = tuple(a.shape[:-2])
    flat = a.reshape((_count(lead),) + tuple(a.shape[-2:]))
    parts = [fn(flat[i]) for i in range(flat.shape[0])]
    if not parts:
        sample = [asarray(x) for x in fn(_probe(a, 2))]
        return wrap(*[_core.zeros(lead + tuple(x.shape), x.dtype) for x in sample])
    cols = []
    for k in range(len(parts[0])):
        cs = [asarray(p[k]) for p in parts]
        cols.append(_c.stack(cs, axis=0).reshape(lead + tuple(cs[0].shape)))
    return wrap(*cols)


def qr(a, mode="reduced"):
    if mode not in ("reduced", "complete", "r", "raw"):
        raise ValueError("Unrecognized mode '%s'" % mode)
    a = asarray(a)
    qr_ = _c.c_qr if _cx(a) else _c.qr
    if mode == "reduced":
        return _tuple_stack(_guard(lambda m: tuple(qr_(m)), lambda m: _empty_qr(m, mode)), a, QRResult)
    if mode == "r":
        return _tuple_stack(_guard(lambda m: (qr_(m)[1],), lambda m: _empty_qr(m, mode)[1:]), a, lambda r: r)
    if mode == "complete":
        full = _c.c_qr_complete if _cx(a) else _c.qr_complete
        return _tuple_stack(_guard(lambda m: tuple(full(m)), lambda m: _empty_qr(m, mode)), a, QRResult)
    raise _core.Unsupported("qr mode %r is not supported" % mode)


def eigh(a, UPLO="L"):
    if UPLO not in ("L", "U", "l", "u"):
        raise ValueError("UPLO argument must be 'L' or 'U'")
    a = asarray(a)
    _square(a) if a.ndim >= 2 else None
    empty = lambda m: (_core.zeros((0,), _rdt(m)), _core.zeros((0, 0), _cdt(m)))
    return _tuple_stack(_guard(lambda m: tuple((_c.c_eigh if _cx(a) else _c.eigh)(m, UPLO.upper())), empty), a, EighResult)


def eigvalsh(a, UPLO="L"):
    a = asarray(a)
    _square(a) if a.ndim >= 2 else None
    return _stack(_guard(lambda m: (_c.c_eigh(m, UPLO.upper())[0] if _cx(a) else _c.eigvalsh(m, UPLO.upper())), lambda m: _core.zeros((0,), _rdt(m))), a)


def eig(a):
    a = asarray(a)
    _square(a) if a.ndim >= 2 else None
    empty = lambda m: (_core.zeros((0,), _cxdt(m)), _core.zeros((0, 0), _cxdt(m)))
    return _tuple_stack(_guard(lambda m: tuple((_c.c_eig if _cx(a) else _c.eig)(m)), empty), a, EigResult)


def eigvals(a):
    a = asarray(a)
    _square(a) if a.ndim >= 2 else None
    return _stack(_guard(lambda m: (_c.c_eig(m)[0] if _cx(a) else _c.eigvals(m)), lambda m: _core.zeros((0,), _cxdt(m))), a)


def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    a = asarray(a)
    if _cx(a):
        one = _c.c_svd_full if full_matrices else _c.c_svd
        if not compute_uv:
            return _stack(_guard(lambda m: _c.c_svd(m)[1], lambda m: _empty_svd(m, False, False)), a)
        return _tuple_stack(_guard(lambda m: tuple(one(m)), lambda m: _empty_svd(m, full_matrices, True)), a, SVDResult)
    if not compute_uv:
        return _stack(_guard(lambda m: _c.svd(m, full_matrices, False), lambda m: _empty_svd(m, False, False)), a)
    return _tuple_stack(_guard(lambda m: tuple(_c.svd(m, full_matrices, True)), lambda m: _empty_svd(m, full_matrices, True)), a, SVDResult)


def svdvals(x, /):
    x = asarray(x)
    return _stack(_guard((lambda m: _c.c_svd(m)[1]) if _cx(x) else _c.svdvals, lambda m: _empty_svd(m, False, False)), x)


def _cx_pinv(m, rcond):
    if m.size == 0:
        return _core.zeros((m.shape[1], m.shape[0]), m.dtype)
    u, s, vh = _c.c_svd(m)
    if rcond is None:
        rcond = max(m.shape) * 2.220446049250313e-16
    cut = float(rcond) * float(s.max()) if s.size else 0.0
    inv_s = _c.where(_c.greater(s, cut), _c.divide(1.0, _c.where(_c.greater(s, cut), s, 1.0)), 0.0)
    return _c.matmul(_c.multiply(_c.conjugate(_c.transpose(vh)), inv_s.reshape((1, -1))), _c.conjugate(_c.transpose(u)))


def pinv(a, rcond=None, hermitian=False, *, rtol=None):
    if rtol is not None:
        rcond = rtol
    a = asarray(a)
    one = (lambda m, rc: _cx_pinv(m, rc)) if _cx(a) else (lambda m, rc: _c.pinv(m, rc))
    if rcond is not None and asarray(rcond).ndim > 0 and a.ndim > 2:
        lead = tuple(a.shape[:-2])
        if _count(lead) == 0:
            return _stack(lambda m: one(m, None), a)
        rc = _c.broadcast_to(asarray(rcond), lead).reshape((-1,))
        flat = a.reshape((_count(lead),) + tuple(a.shape[-2:]))
        outs = [asarray(one(flat[i], float(rc[i]))) for i in range(flat.shape[0])]
        return _c.stack(outs, axis=0).reshape(lead + tuple(outs[0].shape))
    if rcond is not None and asarray(rcond).ndim > 0:
        rcond = float(asarray(rcond).reshape(()))
    return _stack(lambda m: one(m, rcond), a)


def matrix_rank(A, tol=None, hermitian=False, *, rtol=None):
    from . import _reductions as R

    A = asarray(A)
    if A.ndim < 2:
        return int(bool(R.any(_c.not_equal(A, 0))))
    if rtol is not None and tol is None:
        s = svdvals(A)
        cutoff = _c.multiply(R.max(s, axis=-1, keepdims=True), rtol)
        return R.sum(_c.greater(s, cutoff), axis=-1, dtype="int64")
    if tol is not None:
        s = svdvals(A)
        return R.sum(_c.greater(s, tol), axis=-1, dtype="int64")
    if _cx(A):
        def rank(m):
            if m.size == 0:
                return _c.array(0, "int64").reshape(())
            sv = _c.c_svd(m)[1]
            tol_ = float(sv.max()) * max(m.shape) * 2.220446049250313e-16 if sv.size else 0.0
            return _c.array(int(R.sum(_c.greater(sv, tol_))), "int64").reshape(())
        return _stack(rank, A)
    return _stack(lambda m: _c.array(_c.matrix_rank(m, None), "int64").reshape(()), A)


def matrix_power(a, n):
    a = asarray(a)
    _square(a)
    if _cx(a):
        return _stack(lambda m: _cx_power(m, n), a)
    return _stack(lambda m: _c.matrix_power(m, n), a)


def _cx_power(m, n):
    if n < 0:
        m, n = _c.c_inv(m), -n
    result = _c.eye(m.shape[0], None, 0, m.dtype)
    base = m
    while n:
        if n & 1:
            result = _c.matmul(result, base)
        n >>= 1
        if n:
            base = _c.matmul(base, base)
    return result


def lstsq(a, b, rcond=None):
    a, b = asarray(a), asarray(b)
    if _cx(a) or _cx(b):
        m, n = a.shape
        eps = 2.220446049250313e-16
        rc = eps * max(m, n) if rcond is None else (eps / 2 if rcond < 0 else rcond)
        u, sv, vh = _c.c_svd(a)
        cut = float(sv.max()) * rc if sv.size else 0.0
        rank = int(_c.sum(_c.greater(sv, cut)))
        x = _c.matmul(_cx_pinv(a, rc), b)
        if rank == n and m > n:
            r = _c.subtract(b, _c.matmul(a, x))
            res = _c.sum(_c.real(_c.multiply(r, _c.conjugate(r))), 0) if b.ndim > 1 else asarray(_c.sum(_c.real(_c.multiply(r, _c.conjugate(r))))).reshape((1,))
        else:
            res = _c.zeros((0,), sv.dtype)
        return x, asarray(res), _c.array(rank, "int32"), sv
    return _c.lstsq(a, b, rcond)


def _cond_by_inverse(m, p, real):
    try:
        return _c.multiply(norm(m, p), norm(inv(m), p))
    except LinAlgError:
        return _c.array(float("inf"), real)


def cond(x, p=None):
    x = asarray(x)
    if _cx(x):
        sv = svdvals(x)
        hi, lo = _c.max(sv, -1), _c.min(sv, -1)
        if p is None or p == 2:
            return _c.divide(hi, lo)
        if p == -2:
            return _c.divide(lo, hi)
        return _stack(lambda m: _cond_by_inverse(m, p, sv.dtype), x)
    if x.ndim == 2:
        return _c.cond(x, p)
    return _stack(lambda m: _c.cond(m, p), x)


def norm(x, ord=None, axis=None, keepdims=False):
    x = asarray(x)
    if x.dtype.kind in "biu":
        x = x.astype("float64")
    if axis is None:
        if ord is None:
            res = asarray(vector_norm(x.reshape((-1,)), ord=2))
            return res.reshape((1,) * x.ndim) if keepdims else res
        if x.ndim == 1:
            axis = 0
        elif x.ndim == 2:
            axis = (0, 1)
        else:
            raise ValueError("Improper number of dimensions to norm.")
    if isinstance(axis, int):
        return vector_norm(x, axis=axis, keepdims=keepdims, ord=2 if ord is None else ord)
    if len(axis) == 2:
        return _matrix_norm(x, keepdims, "fro" if ord is None else ord, axis)
    raise ValueError("Improper number of dimensions to norm.")


def vector_norm(x, /, *, axis=None, keepdims=False, ord=2):
    from . import _reductions as R

    x = asarray(x)
    if x.dtype.kind in "biu":
        x = x.astype("float64")
    ax = _c.absolute(x)
    if axis is None and not isinstance(axis, tuple):
        ax_flat = ax.reshape((-1,))
        res = _vec_norm(ax_flat, 0, ord)
        res = asarray(res)
        return res.reshape((1,) * x.ndim) if keepdims else res
    return _vec_norm(ax, axis, ord, keepdims)


def _vec_norm(ax, axis, ord, keepdims=False):
    from . import _reductions as R

    if ord == float("inf"):
        return R.max(ax, axis=axis, keepdims=keepdims)
    if ord == float("-inf"):
        return R.min(ax, axis=axis, keepdims=keepdims)
    if ord == 0:
        return R.sum(_c.astype_(_c.not_equal(ax, 0), ax.dtype.name), axis=axis, keepdims=keepdims)
    if ord == 1:
        return R.sum(ax, axis=axis, keepdims=keepdims)
    if ord == 2:
        return _c.sqrt(R.sum(_c.multiply(ax, ax), axis=axis, keepdims=keepdims))
    return _c.power(R.sum(_c.power(ax, ord), axis=axis, keepdims=keepdims), 1.0 / ord)


def matrix_norm(x, /, *, keepdims=False, ord="fro"):
    return _matrix_norm(x, keepdims, ord, (-2, -1))


def _matrix_norm(x, keepdims, ord, axes):
    from . import _reductions as R

    x = asarray(x)
    if x.ndim < 2:
        raise ValueError("Improper number of dimensions to norm.")
    if x.dtype.kind in "biu":
        x = x.astype("float64")
    r, c = (a % x.ndim for a in axes)
    if ord == "fro" or ord is None:
        res = _c.sqrt(R.sum(_c.real(_c.multiply(x, _c.conjugate(x))), axis=(r, c), keepdims=keepdims))
        return res
    moved = moveaxis(x, (r, c), (-2, -1))
    if ord == "nuc":
        res = R.sum(svdvals(moved), axis=-1)
    elif ord in (2, -2):
        s = svdvals(moved)
        res = R.max(s, axis=-1) if ord == 2 else R.min(s, axis=-1)
    elif ord in (1, -1):
        colsum = R.sum(_c.absolute(moved), axis=-2)
        res = R.max(colsum, axis=-1) if ord == 1 else R.min(colsum, axis=-1)
    elif ord in (float("inf"), float("-inf")):
        rowsum = R.sum(_c.absolute(moved), axis=-1)
        res = R.max(rowsum, axis=-1) if ord > 0 else R.min(rowsum, axis=-1)
    else:
        raise ValueError("Invalid norm order for matrices.")
    res = asarray(res)
    if keepdims:
        shape = list(x.shape)
        shape[r] = 1
        shape[c] = 1
        res = res.reshape(tuple(shape))
    return res


def multi_dot(arrays, *, out=None):
    arrays = [asarray(a) for a in arrays]
    n = len(arrays)
    if n < 2:
        raise ValueError("Expecting at least two arrays.")
    if n == 2:
        return _c.dot(arrays[0], arrays[1])
    first_1d = arrays[0].ndim == 1
    last_1d = arrays[-1].ndim == 1
    if first_1d:
        arrays[0] = arrays[0].reshape((1, -1))
    if last_1d:
        arrays[-1] = arrays[-1].reshape((-1, 1))
    res = arrays[0]
    for m in arrays[1:]:
        res = _c.dot(res, m)
    if first_1d and last_1d:
        return asarray(res).reshape(())
    if first_1d:
        return asarray(res).reshape((-1,))
    if last_1d:
        return asarray(res).reshape((-1,))
    return res


def tensorsolve(a, b, axes=None):
    a, b = asarray(a), asarray(b)
    an = a.ndim
    if axes is not None:
        allaxes = list(range(an))
        for k in axes:
            allaxes.remove(k)
            allaxes.insert(an, k)
        a = _transpose(a, tuple(allaxes))
    oldshape = tuple(a.shape[-(an - b.ndim):])
    prod = 1
    for d in oldshape:
        prod *= d
    a = a.reshape((-1, prod))
    b = b.reshape((-1,))
    res = solve(a, b)
    return res.reshape(oldshape)


def tensorinv(a, ind=2):
    a = asarray(a)
    if ind <= 0:
        raise ValueError("Invalid ind argument.")
    oldshape = tuple(a.shape)
    invshape = oldshape[ind:] + oldshape[:ind]
    prod = 1
    for d in oldshape[ind:]:
        prod *= d
    a = a.reshape((prod, -1))
    return inv(a).reshape(invshape)
