import itertools

import numpy as np
import pytest

import rustnumpy as rnp
from conftest import DTYPES, FLOAT_DTYPES, INT_DTYPES, REAL_DTYPES, assert_same, sample

PAIRS = list(itertools.product(DTYPES, DTYPES))


def tol(dtype):
    dt = np.dtype(dtype)
    if dt.kind != "c":
        return (0.0, 0.0)
    return (1e-5, 1e-30) if dt.itemsize == 8 else (1e-12, 1e-300)


@pytest.mark.parametrize("da,db", PAIRS)
@pytest.mark.parametrize("name", ["add", "multiply", "subtract"])
def test_binary_arithmetic_promotes_and_matches_numpy(name, da, db):
    a, b = sample(da, seed=1), sample(db, seed=2)
    if name == "subtract" and da == db == "bool":
        with pytest.raises(TypeError):
            rnp.subtract(a, b)
        return
    expected = getattr(np, name)(a, b)
    rtol, atol = tol(expected.dtype)
    assert_same(getattr(rnp, name)(a, b), expected, rtol, atol)


@pytest.mark.parametrize("name", ["add", "multiply", "subtract"])
def test_broadcasting(name):
    a, b = sample("int32", (3, 1, 4)), sample("float64", (2, 1))
    assert_same(getattr(rnp, name)(a, b), getattr(np, name)(a, b))
    with pytest.raises(ValueError):
        getattr(rnp, name)(sample("int8", (3,)), sample("int8", (4,)))


@pytest.mark.parametrize("dtype", [d for d in DTYPES if d != "bool"])
@pytest.mark.parametrize("scalar", [3, -2, 0, 100, 2.5, -0.0, 1e30, float("inf"), float("nan")])
@pytest.mark.parametrize("name", ["add", "multiply", "subtract"])
def test_python_scalars_are_weak_like_nep_50(name, dtype, scalar):
    a = sample(dtype, seed=3)
    for lhs, rhs in ((a, scalar), (scalar, a)):
        try:
            expected = getattr(np, name)(lhs, rhs)
        except OverflowError:
            with pytest.raises(OverflowError):
                getattr(rnp, name)(lhs, rhs)
            continue
        rtol, atol = tol(expected.dtype)
        assert_same(getattr(rnp, name)(lhs, rhs), expected, rtol, atol)


def test_bool_array_with_python_scalars():
    a = sample("bool")
    assert_same(rnp.add(a, 3), np.add(a, 3))
    assert_same(rnp.multiply(a, 2.5), np.multiply(a, 2.5))


def test_zero_dim_results_are_python_scalars_or_0d_arrays():
    r = rnp.add(np.float64(1.5), np.float64(2.0))
    assert type(r) is float and r == 3.5
    r = rnp.add(np.int64(1), np.int64(2))
    assert type(r) is int and r == 3
    assert type(rnp.add(True, True)) is bool
    r = rnp.add(np.float32(1.5), np.float32(2.0))
    assert isinstance(r, rnp.ndarray) and r.shape == () and r.dtype == rnp.float32 and float(r) == 3.5
    r = rnp.add(np.int8(1), np.int8(2))
    assert isinstance(r, rnp.ndarray) and r.dtype == "int8" and int(r) == 3


def test_unsupported_dtypes_raise_not_implemented():
    for arr in (np.zeros(2, np.float16), np.array(["a"]), np.array([1], dtype=object)):
        with pytest.raises((NotImplementedError, TypeError)):
            rnp.add(arr, arr)


UNARY_FLOAT = ["sqrt", "cbrt", "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "sin", "cos", "tan",
               "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "floor",
               "ceil", "trunc", "rint", "reciprocal", "degrees", "radians"]


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("name", UNARY_FLOAT)
def test_unary_float_functions(name, dtype):
    x = sample(dtype, (40,), seed=4)
    x = np.concatenate([x, np.linspace(-3, 3, 25).astype(dtype), np.linspace(0, 1, 15).astype(dtype)])
    rtol = 1e-5 if dtype == "float32" else 1e-12
    assert_same(getattr(rnp, name)(x), getattr(np, name)(x), rtol, 1e-300 if dtype == "float64" else 1e-37)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("name", ["absolute", "negative", "square", "sign"])
def test_unary_arith(name, dtype):
    x = sample(dtype, (6, 5), seed=5)
    assert_same(getattr(rnp, name)(x), getattr(np, name)(x))


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("name", ["maximum", "minimum", "fmax", "fmin", "floor_divide", "remainder", "power"])
def test_binary_same_type_functions(name, dtype):
    a, b = sample(dtype, seed=6), sample(dtype, seed=7)
    if name == "power" and np.dtype(dtype).kind in "iu":
        b = np.abs(b) % 6 if np.dtype(dtype).kind == "i" else b % 6
        b = b.astype(dtype)
    elif name in ("floor_divide", "remainder") and np.dtype(dtype).kind == "f":
        pass
    expected = getattr(np, name)(a, b)
    rtol = (1e-5 if dtype == "float32" else 1e-12) if name == "power" else 0.0
    assert_same(getattr(rnp, name)(a, b), expected, rtol, 0.0)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("name", ["arctan2", "hypot", "copysign", "fmod"])
def test_binary_float_functions(name, dtype):
    a, b = sample(dtype, seed=8), sample(dtype, seed=9)
    rtol = 1e-5 if dtype == "float32" else 1e-12
    assert_same(getattr(rnp, name)(a, b), getattr(np, name)(a, b), rtol, 0.0)


def test_negative_integer_power_raises_like_numpy():
    a = np.array([2, 3], dtype=np.int64)
    with pytest.raises(ValueError):
        np.power(a, np.array([-1, 1]))
    with pytest.raises(ValueError):
        rnp.power(a, np.array([-1, 1]))


@pytest.mark.parametrize("src,dst", PAIRS)
def test_astype_matches_numpy_for_in_range_values(src, dst):
    x = sample(src, seed=10)
    if np.dtype(src).kind in "fc":
        x = np.nan_to_num(x, nan=1.0, posinf=1e4, neginf=-1e4)
    if np.dtype(src).kind in "fiuc" and np.dtype(dst).kind in "iu":
        x = np.abs(np.clip(x.real if x.dtype.kind == "c" else x, -100, 100)).astype(src)
    expected = np.asarray(x).astype(dst)
    assert_same(rnp.astype_(x, dst), expected)
