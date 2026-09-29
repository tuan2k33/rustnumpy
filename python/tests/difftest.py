import numpy as np

import rustnumpy as rnp


def run(mod, fn, cast):
    try:
        return fn(mod, cast), None
    except Exception as e:
        return None, e


def flatten(x):
    if isinstance(x, (tuple, list)) and not (len(x) and isinstance(x[0], (int, float)) and False):
        return [flatten(i) for i in x]
    return x


def compare(got, want, path="result", rtol=1e-9, atol=1e-12):
    if isinstance(want, (tuple, list)) and not isinstance(want, np.ndarray):
        assert isinstance(got, (tuple, list)), f"{path}: expected sequence, got {type(got).__name__}"
        assert len(got) == len(want), f"{path}: length {len(got)} != {len(want)}"
        for i, (g, w) in enumerate(zip(got, want)):
            compare(g, w, f"{path}[{i}]", rtol, atol)
        return
    if want is None:
        assert got is None, f"{path}: expected None, got {got!r}"
        return
    if isinstance(want, (str, bytes)):
        assert got == want, f"{path}: {got!r} != {want!r}"
        return
    if hasattr(want, "_fields") and False:
        pass
    g = np.asarray(got)
    w = np.asarray(want)
    assert g.shape == w.shape, f"{path}: shape {g.shape} != {w.shape}"
    if isinstance(want, (bool, np.bool_)) or w.dtype.kind == "b":
        assert g.dtype.kind == "b", f"{path}: dtype {g.dtype} vs bool"
        np.testing.assert_array_equal(g, w, err_msg=path)
        return
    if w.dtype.kind in "iu":
        assert g.dtype.kind in "iu", f"{path}: dtype {g.dtype} vs {w.dtype}"
        np.testing.assert_array_equal(g, w, err_msg=path)
        return
    assert g.dtype.kind == w.dtype.kind, f"{path}: dtype {g.dtype} vs {w.dtype}"
    if g.dtype != w.dtype and not isinstance(want, (float, complex)):
        raise AssertionError(f"{path}: dtype {g.dtype} != {w.dtype}")
    np.testing.assert_allclose(g, w, rtol=rtol, atol=atol, equal_nan=True, err_msg=path)


def differential(fn, rtol=1e-9, atol=1e-12):
    want, werr = run(np, fn, np.array)
    got, gerr = run(rnp, fn, rnp.array)
    if werr is not None:
        assert gerr is not None, f"NumPy raised {type(werr).__name__}: {werr}; rustnumpy returned {got!r}"
        assert not isinstance(gerr, NotImplementedError), f"unsupported: {gerr}"
        assert isinstance(gerr, type(werr)) or (isinstance(gerr, (TypeError, ValueError, IndexError)) and isinstance(werr, (TypeError, ValueError, IndexError))), (
            f"NumPy raised {type(werr).__name__}: {werr}; rustnumpy raised {type(gerr).__name__}: {gerr}"
        )
        return
    assert gerr is None, f"rustnumpy raised {type(gerr).__name__}: {gerr}"
    compare(got, want, rtol=rtol, atol=atol)
