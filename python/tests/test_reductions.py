import numpy as np
import pytest

import rustnumpy as rnp
from conftest import DTYPES, FLOAT_DTYPES, INT_DTYPES, REAL_DTYPES, assert_same, sample

AXES = [None, 0, 1, -1]


def float_tol(dtype):
    return {"float32": (1e-5, 1e-30), "complex64": (1e-5, 1e-30), "complex128": (1e-10, 1e-300)}.get(str(dtype), (1e-11, 1e-300))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("name", ["sum", "prod"])
def test_sum_and_prod_match_numpy_including_accumulator_dtype(name, dtype, axis, keepdims):
    x = sample(dtype, (4, 5), seed=11)
    if name == "prod" and np.dtype(dtype).kind in "fc":
        x = np.where(np.isfinite(x), np.clip(x.real if x.dtype.kind == "c" else x, -1.5, 1.5), 1.0).astype(dtype)
    expected = getattr(np, name)(x, axis=axis, keepdims=keepdims)
    rtol, atol = float_tol(np.asarray(expected).dtype)
    assert_same(getattr(rnp, name)(x, axis=axis, keepdims=keepdims), expected, rtol, atol)


@pytest.mark.parametrize("name", ["sum", "prod"])
def test_empty_axis_uses_the_identity(name):
    x = np.zeros((0, 3))
    assert_same(getattr(rnp, name)(x, axis=0), getattr(np, name)(x, axis=0))
    assert_same(getattr(rnp, name)(x), getattr(np, name)(x))


def test_integer_sum_wraps_like_numpy():
    x = np.array([2**62, 2**62, 2**62], dtype=np.int64)
    assert_same(rnp.sum(x), np.sum(x))
    y = np.array([200, 100], dtype=np.uint8)
    assert_same(rnp.sum(y), np.sum(y))


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("name", ["max", "min", "amax", "amin"])
def test_min_max(name, dtype, axis):
    x = sample(dtype, (4, 5), seed=12)
    assert_same(getattr(rnp, name)(x, axis=axis), getattr(np, name)(x, axis=axis))
    assert_same(getattr(rnp, name)(x, axis=axis, keepdims=True), getattr(np, name)(x, axis=axis, keepdims=True))


def test_min_max_of_empty_raise_value_error():
    with pytest.raises(ValueError):
        rnp.max(np.zeros(0))
    with pytest.raises(ValueError):
        rnp.min(np.zeros(0))


def clean(dtype, shape=(4, 6), seed=13):
    x = sample(dtype, shape, seed)
    if np.dtype(dtype).kind == "f":
        x = np.nan_to_num(x, nan=0.5, posinf=1e3, neginf=-1e3)
        x = np.clip(x, -1e6, 1e6).astype(dtype)
    return x


@pytest.mark.parametrize("dtype", ["bool"] + REAL_DTYPES)
@pytest.mark.parametrize("name", ["mean", "median"])
def test_mean_and_median(name, dtype):
    x = clean(dtype)
    if np.dtype(dtype).kind in "iu":
        x = (x % 100).astype(dtype)
    expected = getattr(np, name)(x)
    rtol, atol = float_tol(np.asarray(expected).dtype)
    assert_same(getattr(rnp, name)(x), expected, rtol, atol)


@pytest.mark.parametrize("dtype", ["bool"] + REAL_DTYPES)
@pytest.mark.parametrize("ddof", [0, 1, 2])
@pytest.mark.parametrize("name", ["var", "std"])
def test_var_and_std(name, dtype, ddof):
    x = clean(dtype)
    if np.dtype(dtype).kind in "iu":
        x = (x % 100).astype(dtype)
    expected = getattr(np, name)(x, ddof=ddof)
    rtol, atol = float_tol(np.asarray(expected).dtype)
    assert_same(getattr(rnp, name)(x, ddof=ddof), expected, max(rtol, 1e-5), atol)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("q", [0, 10, 25, 50, 75, 99, 100])
def test_percentile(dtype, q):
    x = clean(dtype)
    expected = np.percentile(x, q)
    rtol, atol = float_tol(np.asarray(expected).dtype)
    assert_same(rnp.percentile(x, q), expected, max(rtol, 1e-5), atol)


def with_nans(dtype):
    x = sample(dtype, (30,), seed=14)
    x = np.clip(np.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3), -1e6, 1e6).astype(dtype)
    x[[2, 9, 17]] = np.nan
    return x


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("name", ["nansum", "nanmean", "nanmedian", "nanmin", "nanmax", "nanvar", "nanstd"])
def test_nan_functions(name, dtype):
    x = with_nans(dtype)
    expected = getattr(np, name)(x)
    rtol, atol = float_tol(np.asarray(expected).dtype)
    assert_same(getattr(rnp, name)(x), expected, max(rtol, 1e-5), atol)


def test_axis_not_yet_bound_falls_back_cleanly():
    with pytest.raises(NotImplementedError):
        rnp.mean(np.ones((2, 2)), axis=0)


@pytest.mark.parametrize("ddof", [0, 1])
def test_cov_and_corrcoef(ddof):
    x = np.random.default_rng(0).standard_normal((3, 20))
    np.testing.assert_allclose(rnp.cov(x, ddof=ddof), np.cov(x, ddof=ddof), rtol=1e-12)
    np.testing.assert_allclose(rnp.corrcoef(x), np.corrcoef(x), rtol=1e-12)
    v = x[0]
    np.testing.assert_allclose(rnp.cov(v, ddof=ddof), np.cov(v, ddof=ddof), rtol=1e-12)
    assert np.asarray(rnp.cov(v)).shape == ()


@pytest.mark.parametrize("bins", [1, 4, 10])
@pytest.mark.parametrize("rng", [None, (-1.0, 2.0)])
def test_histogram(bins, rng):
    x = np.random.default_rng(1).standard_normal(200)
    counts, edges = rnp.histogram(x, bins=bins, range=rng)
    ecounts, eedges = np.histogram(x, bins=bins, range=rng)
    assert_same(counts, ecounts)
    np.testing.assert_allclose(edges, eedges, rtol=1e-12)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_sort_and_argsort(dtype, axis):
    x = sample(dtype, (4, 6), seed=15)
    assert_same(rnp.sort(x, axis=axis), np.sort(x, axis=axis))
    assert_same(rnp.argsort(x, axis=axis), np.argsort(x, axis=axis, kind="stable"))


def test_sort_of_0d_and_bad_axis():
    with pytest.raises(ValueError):
        rnp.sort(np.float64(1.0))
    with pytest.raises(ValueError):
        rnp.sort(np.zeros((2, 2)), axis=2)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("dtype", ["int32", "float64", "uint8"])
def test_searchsorted(side, dtype):
    a = np.sort(sample(dtype, (12,), seed=16))
    v = sample(dtype, (3, 4), seed=17)
    assert_same(rnp.searchsorted(a, v, side=side), np.searchsorted(a, v, side=side))
    assert_same(rnp.searchsorted(a, a, side=side), np.searchsorted(a, a, side=side))


def test_searchsorted_mixed_dtypes_and_nan():
    a = np.array([1.0, 2.0, np.nan, np.nan])
    v = np.array([np.nan, 2, 3], dtype=np.float32)
    assert_same(rnp.searchsorted(a, v), np.searchsorted(a, v))
    assert_same(rnp.searchsorted(a, v, side="right"), np.searchsorted(a, v, side="right"))
