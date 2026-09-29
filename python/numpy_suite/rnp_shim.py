"""pytest plugin that routes NumPy function calls through rustnumpy.

Two modes (env RNP_SHIM_MODE):

* ``shadow`` (default): every eligible call is computed by BOTH NumPy and
  rustnumpy; NumPy's answer is returned (so NumPy's own tests behave
  normally) and any difference is appended to a JSONL log. Every call
  made by NumPy's test-suite therefore becomes a differential test case.
* ``serve``: rustnumpy's answer is returned to the test, so a NumPy test
  that passes on plain NumPy but fails here points at a divergence.

Calls rustnumpy cannot handle raise ``rustnumpy.Unsupported`` and
transparently fall back to NumPy (counted as ``fallback``).

Usage:  pytest -p rnp_shim <numpy test files>   (with numpy_suite on PYTHONPATH)
"""

import functools
import json
import os
import threading

import numpy as np

import rustnumpy as rnp

MODE = os.environ.get("RNP_SHIM_MODE", "shadow")
UFUNC_PROXIES = os.environ.get("RNP_SHIM_UFUNCS", "1") == "1"
LOG_DIR = os.environ.get("RNP_SHIM_LOG", "/tmp/rnp_shim_log")
_lock = threading.Lock()
_log = None
_STATS = {}
_IN_SHIM = threading.local()

PLAIN_SCALARS = (bool, int, float, complex)


def _log_file():
    global _log
    if _log is None:
        os.makedirs(LOG_DIR, exist_ok=True)
        _log = open(os.path.join(LOG_DIR, f"{os.getpid()}.jsonl"), "a", buffering=1)
    return _log


def _record(name, outcome, **extra):
    key = (name, outcome)
    _STATS[key] = _STATS.get(key, 0) + 1
    if outcome in ("served", "fallback", "match", "both_error", "skipped", "errstate_or_warning"):
        return
    with _lock:
        _log_file().write(json.dumps({"func": name, "outcome": outcome, **extra}, default=str) + "\n")


def _describe(x):
    if isinstance(x, np.ndarray):
        return f"ndarray[{x.dtype},{x.shape}]"
    if isinstance(x, np.generic):
        return f"np.{type(x).__name__}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[{len(x)}]"
    return type(x).__name__ + (f"({x!r})" if isinstance(x, PLAIN_SCALARS) else "")


def _to_np(x):
    """rustnumpy results back to NumPy objects (arrays share memory; Python scalars become NumPy scalars)."""
    if isinstance(x, rnp.ndarray):
        a = np.asarray(x)
        return a[()] if a.ndim == 0 else a
    if isinstance(x, (bool, int, float, complex)):
        return np.asarray(x)[()]
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        return type(x)(*[_to_np(i) for i in x])
    if isinstance(x, (tuple, list)):
        return type(x)(_to_np(i) for i in x)
    return x


def _short(x, limit=160):
    try:
        text = repr(np.asarray(x).tolist() if isinstance(x, (np.ndarray, np.generic)) and np.asarray(x).size <= 16 else x)
    except Exception:  # noqa: BLE001
        text = "<unprintable>"
    return text if len(text) <= limit else text[:limit] + "..."


def _plain(x):
    if type(x) is np.ndarray:
        return x.dtype.kind in "biufc" and x.dtype.isnative and x.dtype.itemsize in (1, 2, 4, 8, 16) and x.dtype != np.float16
    if isinstance(x, np.generic):
        return type(x).__module__ == "numpy" and x.dtype.kind in "biufc" and x.dtype != np.float16 and x.dtype.itemsize <= 16
    if type(x) in PLAIN_SCALARS:
        return True
    if type(x) in (list, tuple):
        return all(_plain(i) for i in x) and len(x) > 0
    return False


def _tolerance(dtype):
    if dtype in (np.float32, np.complex64):
        return 2e-5, 1e-6
    return 1e-9, 1e-12


def _same(got, want, argsort=False):
    """Return None if equal enough, else a short reason."""
    if isinstance(want, tuple) and hasattr(want, "_fields") or isinstance(want, (tuple, list)):
        if not isinstance(got, (tuple, list)) or len(got) != len(want):
            return "structure"
        for i, (g, w) in enumerate(zip(got, want)):
            r = _same(g, w)
            if r:
                return f"[{i}] {r}"
        return None
    g, w = np.asarray(got), np.asarray(want)
    if g.dtype != w.dtype:
        return f"dtype {g.dtype} != {w.dtype}"
    if g.shape != w.shape:
        return f"shape {g.shape} != {w.shape}"
    if w.dtype.kind in "fc":
        rtol, scale = _tolerance(w.dtype.type)
        fin = np.isfinite(w) if w.size else np.array([], bool)
        magnitude = float(np.abs(w[fin]).max()) if fin.any() else 1.0
        atol = scale * max(magnitude, 1.0) * (1e3 if rtol > 1e-6 else 1e2)
        if not np.allclose(g, w, rtol=rtol, atol=atol, equal_nan=True):
            return "values"
        return None
    if not np.array_equal(g, w):
        return "values"
    return None


def _make(name, orig, fn, max_pos=None, allowed_kw=(), comparator=None):
    @functools.wraps(orig)
    def wrapper(*args, **kw):
        if getattr(_IN_SHIM, "active", False):
            return orig(*args, **kw)
        eligible = (
            all(_plain(a) for a in args)
            and (max_pos is None or len(args) <= max_pos)
            and all(k in allowed_kw for k in kw)
            and not any(k in kw and kw[k] is not None and k in ("out", "where", "dtype", "casting", "order", "subok") for k in kw)
        )
        if not eligible:
            _record(name, "skipped")
            return orig(*args, **kw)
        _IN_SHIM.active = True
        try:
            try:
                ours = fn(*args, **kw)
                ours_err = None
            except (rnp.Unsupported, TypeError):
                _record(name, "fallback")
                return orig(*args, **kw)
            except BaseException as e:  # noqa: BLE001  (includes pyo3_runtime.PanicException)
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                ours, ours_err = None, e
                if type(e).__name__ == "PanicException":
                    _record(name, "PANIC", args=[_describe(a) for a in args], detail=str(e))
            try:
                theirs = orig(*args, **kw)
                theirs_err = None
            except Exception as e:  # noqa: BLE001
                theirs, theirs_err = None, e
            sig = [_describe(a) for a in args] + [f"{k}={_describe(v)}" for k, v in kw.items()]
            if ours_err is not None or theirs_err is not None:
                if isinstance(theirs_err, (FloatingPointError, Warning)) and ours_err is None:
                    _record(name, "errstate_or_warning")
                elif ours_err is not None and theirs_err is not None:
                    _record(name, "both_error")
                elif ours_err is not None:
                    _record(name, "we_error_numpy_ok", args=sig, detail=f"{type(ours_err).__name__}: {ours_err}")
                else:
                    _record(name, "numpy_error_we_ok", args=sig, detail=f"{type(theirs_err).__name__}: {theirs_err}")
                if theirs_err is not None:
                    raise theirs_err
                if MODE == "serve" and ours_err is not None:
                    raise ours_err
                return theirs
            ours_np = _to_np(ours)
            reason = comparator(ours_np, theirs, args, kw) if comparator else _same(ours_np, theirs)
        finally:
            _IN_SHIM.active = False
        if reason:
            _record(name, "mismatch", args=sig, detail=reason, ours=_short(ours), theirs=_short(theirs),
                    inputs=[_short(a) for a in args[:3]])
        else:
            _record(name, "match")
        _record(name, "served")
        return _to_np(ours) if MODE == "serve" else theirs

    wrapper._rnp_wrapped = True
    return wrapper


class _UfuncProxy:
    """Delegates everything (reduce, accumulate, outer, nin, ...) to the real ufunc."""

    def __init__(self, orig, call):
        object.__setattr__(self, "_orig", orig)
        object.__setattr__(self, "_call", call)

    def __call__(self, *a, **k):
        return self._call(*a, **k)

    def __getattr__(self, item):
        return getattr(self._orig, item)

    def __repr__(self):
        return repr(self._orig)


def _argsort_equiv(ours, theirs, args, kw):
    a = np.asarray(args[0])
    axis = kw.get("axis", args[1] if len(args) > 1 else -1)
    if axis is None:
        return "unsupported-axis"
    if ours.dtype != theirs.dtype or ours.shape != theirs.shape:
        return f"dtype/shape {ours.dtype}{ours.shape} != {theirs.dtype}{theirs.shape}"
    if np.array_equal(ours, theirs):
        return None
    taken_o = np.take_along_axis(a, ours, axis)
    taken_t = np.take_along_axis(a, theirs, axis)
    return None if _same(taken_o, taken_t) is None else "values"


def _close(a, b, rtol=1e-8, atol=1e-8):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))


def _tol_for(x):
    return (2e-3, 2e-3) if np.asarray(x).dtype in (np.float32, np.complex64) else (1e-7, 1e-8)


def _sorted_eigs(w):
    w = np.asarray(w)
    order = np.lexsort((np.round(w.imag, 6), np.round(w.real, 6)))
    return w[order]


def _eig_cmp(ours, theirs, args, kw):
    a = np.asarray(args[0])
    rtol, atol = _tol_for(a)
    if ours[0].dtype != theirs[0].dtype or ours[1].dtype != theirs[1].dtype:
        return f"dtype {ours[0].dtype}/{ours[1].dtype} != {theirs[0].dtype}/{theirs[1].dtype}"
    if not _close(_sorted_eigs(ours[0]), _sorted_eigs(theirs[0]), rtol * 5, atol * 5):
        return "eigenvalues"
    if not _close(a @ ours[1], ours[1] * ours[0], rtol * 5, atol * 5 * max(1.0, float(np.abs(a).max(initial=0)))):
        return "A v != lambda v"
    return None


def _eigh_cmp(ours, theirs, args, kw):
    a = np.asarray(args[0])
    rtol, atol = _tol_for(a)
    if ours[0].dtype != theirs[0].dtype or ours[1].dtype != theirs[1].dtype:
        return "dtype"
    if not _close(ours[0], theirs[0], rtol * 5, atol * 5 * max(1.0, float(np.abs(a).max(initial=0)))):
        return "eigenvalues"
    sym = np.tril(a) + np.tril(a, -1).T
    if not _close(sym @ ours[1], ours[1] * ours[0], rtol * 5, atol * 5 * max(1.0, float(np.abs(a).max(initial=0)))):
        return "A v != lambda v"
    return None


def _svd_cmp(ours, theirs, args, kw):
    if not isinstance(ours, tuple) and not hasattr(ours, "_fields"):
        return _same(ours, theirs)
    a = np.asarray(args[0])
    rtol, atol = _tol_for(a)
    u, s, vh = ours
    if s.dtype != theirs[1].dtype or u.dtype != theirs[0].dtype or u.shape != theirs[0].shape or vh.shape != theirs[2].shape:
        return "dtype/shape"
    scale = max(1.0, float(np.abs(a).max(initial=0)))
    if not _close(s, theirs[1], rtol * 5, atol * 5 * scale):
        return "singular values"
    if not _close((u * s) @ vh, a, rtol * 20, atol * 20 * scale):
        return "U S Vh != A"
    return None


def _qr_cmp(ours, theirs, args, kw):
    a = np.asarray(args[0])
    q, r = ours
    if q.shape != theirs[0].shape or r.shape != theirs[1].shape or q.dtype != theirs[0].dtype:
        return "shape/dtype"
    rtol, atol = _tol_for(a)
    return None if _close(q @ r, a, rtol * 20, atol * 20 * max(1.0, float(np.abs(a).max(initial=0)))) else "Q R != A"


LINALG_COMPARATORS = {"eig": _eig_cmp, "eigh": _eigh_cmp, "svd": _svd_cmp, "qr": _qr_cmp}


BINARY_UFUNCS = {
    "add": rnp.add, "subtract": rnp.subtract, "multiply": rnp.multiply, "maximum": rnp.maximum, "minimum": rnp.minimum,
    "fmax": rnp.fmax, "fmin": rnp.fmin, "floor_divide": rnp.floor_divide, "remainder": rnp.remainder, "mod": rnp.remainder,
    "power": rnp.power, "arctan2": rnp.arctan2, "hypot": rnp.hypot, "copysign": rnp.copysign, "fmod": rnp.fmod,
}
UNARY_UFUNCS = {n: getattr(rnp, n) for n in (
    "sqrt cbrt exp exp2 expm1 log log2 log10 log1p sin cos tan arcsin arccos arctan sinh cosh tanh arcsinh arccosh arctanh "
    "floor ceil trunc rint reciprocal degrees radians absolute negative square sign").split()}
UNARY_UFUNCS["abs"] = rnp.absolute

REDUCTIONS = {
    "sum": (rnp.sum, ("axis", "keepdims")), "prod": (rnp.prod, ("axis", "keepdims")), "max": (rnp.max, ("axis", "keepdims")),
    "min": (rnp.min, ("axis", "keepdims")), "amax": (rnp.amax, ("axis", "keepdims")), "amin": (rnp.amin, ("axis", "keepdims")),
    "mean": (rnp.mean, ("axis",)), "var": (rnp.var, ("axis", "ddof")), "std": (rnp.std, ("axis", "ddof")),
    "median": (rnp.median, ("axis",)), "percentile": (rnp.percentile, ("axis",)), "nansum": (rnp.nansum, ("axis", "keepdims")),
    "nanmean": (rnp.nanmean, ("axis",)), "nanvar": (rnp.nanvar, ("axis", "ddof")), "nanstd": (rnp.nanstd, ("axis", "ddof")),
    "nanmin": (rnp.nanmin, ("axis",)), "nanmax": (rnp.nanmax, ("axis",)), "nanmedian": (rnp.nanmedian, ("axis",)),
    "cov": (rnp.cov, ("ddof",)), "corrcoef": (rnp.corrcoef, ()), "histogram": (rnp.histogram, ("bins", "range")),
    "sort": (rnp.sort, ("axis",)), "searchsorted": (rnp.searchsorted, ("side",)),
}

SHAPE_FUNCS = {
    "reshape": (rnp.reshape, ()), "ravel": (rnp.ravel, ()), "transpose": (rnp.transpose, ("axes",)), "swapaxes": (rnp.swapaxes, ()),
    "moveaxis": (rnp.moveaxis, ()), "expand_dims": (rnp.expand_dims, ("axis",)), "squeeze": (rnp.squeeze, ("axis",)),
    "flip": (rnp.flip, ("axis",)), "roll": (rnp.roll, ("shift", "axis")), "repeat": (rnp.repeat, ("repeats", "axis")),
    "broadcast_to": (rnp.broadcast_to, ()), "diagonal": (rnp.diagonal, ("offset",)), "trace": (rnp.trace, ("offset",)),
    "kron": (rnp.kron, ()), "outer": (rnp.outer, ()), "matmul": (rnp.matmul, ()), "dot": (rnp.dot, ()),
    "tensordot": (rnp.tensordot, ("axes",)), "vecdot": (rnp.vecdot, ()), "where": (rnp.where, ()),
    "select": (rnp.select, ("default",)), "array_split": (rnp.array_split, ("axis",)),
    "permute_dims": (rnp.permute_dims, ()), "unstack": (rnp.unstack, ("axis",)),
    "unique_all": (rnp.unique_all, ()), "unique_counts": (rnp.unique_counts, ()), "unique_inverse": (rnp.unique_inverse, ()),
    "unique_values": (rnp.unique_values, ()),
}

LINALG = {n: getattr(rnp, n) for n in (
    "inv det slogdet solve qr cholesky eigh eigvalsh eig eigvals svd svdvals pinv matrix_rank lstsq cond norm matrix_power").split()}
FFT = {n: getattr(rnp, n) for n in (
    "fft ifft rfft irfft hfft ihfft fftn ifftn fft2 ifft2 rfftn irfftn fftfreq rfftfreq fftshift ifftshift").split()}


def _unique_sort_cmp(ours, theirs, args, kw):
    return _same(np.sort(ours), np.sort(theirs))


def install():
    patched = []

    def put(module, name, wrapper):
        if hasattr(module, name):
            setattr(module, name, wrapper)
            patched.append(f"{module.__name__}.{name}")

    for name, fn in ({**BINARY_UFUNCS} if UFUNC_PROXIES else {}).items():
        orig = getattr(np, name, None)
        if orig is not None:
            put(np, name, _UfuncProxy(orig, _make(name, orig, fn, max_pos=2)))
    for name, fn in (UNARY_UFUNCS if UFUNC_PROXIES else {}).items():
        orig = getattr(np, name, None)
        if orig is not None:
            put(np, name, _UfuncProxy(orig, _make(name, orig, fn, max_pos=1)))
    if UFUNC_PROXIES:
        orig = np.matmul
        put(np, "matmul", _UfuncProxy(orig, _make("matmul", orig, rnp.matmul, max_pos=2)))
    for name, (fn, kws) in {**REDUCTIONS, **SHAPE_FUNCS}.items():
        if name in ("matmul",):
            continue
        orig = getattr(np, name, None)
        if orig is None:
            continue
        cmp = _unique_sort_cmp if name == "unique_values" else None
        put(np, name, _make(name, orig, fn, allowed_kw=kws, comparator=cmp))
    orig = np.argsort
    put(np, "argsort", _make("argsort", orig, rnp.argsort, allowed_kw=("axis",), comparator=_argsort_equiv))
    for name, fn in LINALG.items():
        orig = getattr(np.linalg, name, None)
        if orig is not None:
            put(np.linalg, name, _make("linalg." + name, orig, fn, allowed_kw=("full_matrices", "compute_uv", "rcond", "tol", "UPLO", "p", "ord"),
                                       comparator=LINALG_COMPARATORS.get(name)))
    for name, fn in FFT.items():
        orig = getattr(np.fft, name, None)
        if orig is not None:
            put(np.fft, name, _make("fft." + name, orig, fn, allowed_kw=("n", "axis", "norm", "s", "axes", "d")))
    return patched


def pytest_configure(config):
    install()


def pytest_unconfigure(config):
    if _STATS:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, f"stats-{os.getpid()}.json"), "w") as f:
            json.dump([[k[0], k[1], v] for k, v in _STATS.items()], f)
