import numpy as np
import pytest

import rustnumpy as rnp
from conftest import DTYPES, sample


def same_text(x, **opts):
    if opts:
        with np.printoptions(**opts), rnp.printoptions(**opts):
            return repr(rnp.array(x)), repr(x), str(rnp.array(x)), str(x)
    return repr(rnp.array(x)), repr(x), str(rnp.array(x)), str(x)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(), (1,), (5,), (2, 3), (2, 2, 3), (0,), (0, 3), (1, 1, 1)])
def test_repr_and_str_of_every_dtype_and_shape(dtype, shape):
    with np.errstate(all="ignore"):
        x = sample(dtype, shape, seed=9) if shape and 0 not in shape else np.zeros(shape, dtype)
        if x.dtype.kind in "fc" and x.size:
            tiny = np.finfo(x.real.dtype).tiny
            x = np.where(np.abs(x) < tiny, 0.5, x).astype(x.dtype)
    r, wr, s, ws = same_text(x)
    assert r == wr
    assert s == ws


@pytest.mark.parametrize("values", [
    [1.5, 2.25, -3.0], [1e-5, 1.0, 100.0], [1e10, 1.0], [0.1, 0.2, 0.30000000000000004], [1 / 3, 2 / 3], [123456.789, 0.001],
    [float("nan"), 1.0, float("inf"), -float("inf")], [1e-10, 5e-11], [1e100, 1e-100], [-0.0, 0.0], [1e8, 2e8], [99999999.0, 1.0],
    [0.5, 0.25, 0.125, 0.0625], [12345678.9, 1.5],
])
def test_float_formats(values):
    for dt in ("float64", "float32", "float16"):
        x = np.array(values, dtype=dt)
        assert same_text(x)[:2][0] == same_text(x)[1]
        r, wr, s, ws = same_text(x)
        assert r == wr and s == ws


@pytest.mark.parametrize("opts", [
    {"precision": 3}, {"precision": 0}, {"suppress": True}, {"floatmode": "fixed"}, {"floatmode": "unique"},
    {"floatmode": "maxprec_equal"}, {"sign": "+"}, {"sign": " "}, {"linewidth": 30}, {"threshold": 5}, {"edgeitems": 1, "threshold": 4},
    {"nanstr": "NAN", "infstr": "INF"},
])
def test_print_options(opts):
    x = np.array([[1.5, -2.0, 1e-3, float("nan")], [float("inf"), 3.14159265, 0.0, 100.0]])
    r, wr, s, ws = same_text(x, **opts)
    assert r == wr and s == ws


def test_long_arrays_wrap_and_summarise():
    x = np.arange(200) * 1.5
    assert same_text(x)[:2] == (repr(x), repr(x))
    r, wr, s, ws = same_text(x)
    assert r == wr and s == ws
    y = np.arange(3000).reshape(30, 100)
    r, wr, s, ws = same_text(y)
    assert r == wr and s == ws
    z = np.arange(2000).reshape(10, 10, 20) * 0.5
    r, wr, s, ws = same_text(z)
    assert r == wr and s == ws
    w = np.arange(50).astype("int8")
    r, wr, s, ws = same_text(w)
    assert r == wr and s == ws


def test_complex_and_bool_and_mixed_widths():
    x = np.array([1 + 2j, -3.5 - 1e-3j, 0j, 1e5 + 1j])
    r, wr, s, ws = same_text(x)
    assert r == wr and s == ws
    b = np.array([[True, False], [False, True]])
    r, wr, s, ws = same_text(b)
    assert r == wr and s == ws
    i = np.array([-1, 10, 1000])
    r, wr, s, ws = same_text(i)
    assert r == wr and s == ws
    u = np.array([1, 2, 3], dtype="uint8")
    r, wr, s, ws = same_text(u)
    assert r == wr and s == ws


def test_array2string_and_helpers():
    x = np.array([[1.0, 2.5], [3.0, 4.0]])
    assert rnp.array2string(rnp.array(x), separator=", ") == np.array2string(x, separator=", ")
    assert rnp.array_repr(rnp.array(x)) == np.array_repr(x)
    assert rnp.array_str(rnp.array(x)) == np.array_str(x)
    assert rnp.array2string(rnp.array(x), formatter={"float_kind": lambda v: "%.1f!" % v}) == np.array2string(x, formatter={"float_kind": lambda v: "%.1f!" % v})
    assert rnp.get_printoptions()["precision"] == np.get_printoptions()["precision"]


@pytest.mark.parametrize("dtype", ["float16", "float32", "complex64"])
@pytest.mark.parametrize("value", [0.0, -0.0, 1.0, 1e-4, 1e-6, 0.1, 123456.7, 1e6, 999999.9, 1e15, 1000.0, 999.0, 65504.0, 5e-5, -2.5, float("nan"), float("inf")])
def test_str_of_zero_d_low_precision_arrays_uses_the_scalar_format(dtype, value):
    arg = complex(value, 1.5) if dtype == "complex64" else value
    with np.errstate(all="ignore"):
        assert str(rnp.array(arg, dtype)) == str(np.array(arg, dtype))
