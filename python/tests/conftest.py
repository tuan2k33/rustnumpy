import numpy as np
import pytest

import rustnumpy as rnp

DTYPES = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
          "float32", "float64", "complex64", "complex128"]
INT_DTYPES = [d for d in DTYPES if d.startswith(("int", "uint"))]
FLOAT_DTYPES = ["float32", "float64"]
REAL_DTYPES = INT_DTYPES + FLOAT_DTYPES


@pytest.fixture(autouse=True)
def quiet_numpy_warnings():
    with np.errstate(all="ignore"):
        yield


def sample(dtype, shape=(4, 5), seed=0):
    """Deterministic values for `dtype`, including its edge cases."""
    rng = np.random.default_rng(seed)
    dt = np.dtype(dtype)
    n = int(np.prod(shape))
    if dt.kind == "b":
        return rng.integers(0, 2, n).astype(bool).reshape(shape)
    if dt.kind in "iu":
        info = np.iinfo(dt)
        edges = np.array([info.min, info.max, 0, 1, info.max // 2], dtype=dt)
        body = rng.integers(info.min, info.max, n, dtype=dt, endpoint=True)
        body[: len(edges)] = edges[: min(len(edges), n)]
        return body.reshape(shape)
    if dt.kind == "f":
        edges = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0, np.finfo(dt).tiny / 4, np.finfo(dt).max], dtype=dt)
        body = (rng.standard_normal(n) * 10).astype(dt)
        body[: min(len(edges), n)] = edges[: min(len(edges), n)]
        return body.reshape(shape)
    real_dt = "float32" if dt == np.complex64 else "float64"
    re, im = sample(real_dt, shape, seed), sample(real_dt, shape, seed + 1)
    finite_max = np.finfo(real_dt).max
    re[re == finite_max] = 1.0
    im[im == finite_max] = 1.0
    return (re + 1j * im).astype(dt)


def assert_same(actual, expected, rtol=0.0, atol=0.0):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype, f"dtype {actual.dtype} != {expected.dtype}"
    assert actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}"
    if rtol == 0.0 and atol == 0.0 and actual.dtype.kind != "c":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
