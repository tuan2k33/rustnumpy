import copy
import math
import operator
import pickle
import warnings

import numpy as np
import pytest

import rustnumpy as rnp

FLOATS = [0.0, -0.0, 1.5, -2.5, float("inf"), float("-inf"), float("nan"), 1e308, 5e-324, 3.0]
INTS = [0, 1, -1, 7, -7, 2**31, 2**53 + 1, 2**62, -(2**63), 2**63 - 1]
PY_INTS = INTS + [2**63, -(2**63) - 1, 2**70, 10**400]
COMPLEXES = [0j, 1 + 2j, -0.5 + 0j, complex(float("inf"), 1.0), complex(float("nan"), 0.0)]
BINARY = ["add", "sub", "mul", "truediv", "floordiv", "mod", "pow", "lt", "le", "gt", "ge", "eq", "ne"]
BITWISE = ["and_", "or_", "xor", "lshift", "rshift"]


def np_scalars(dtype, values):
    return [np.array([v], dtype)[0] for v in values]


def rn_scalars(dtype, values):
    return [rnp.asarray(np.array([v], dtype))[0] for v in values]


def outcome(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return fn(), None
        except Exception as e:
            return None, {"UFuncTypeError": "TypeError"}.get(type(e).__name__, type(e).__name__)


def same_value(a, b):
    if str(getattr(a, "dtype", type(a).__name__)) != str(getattr(b, "dtype", type(b).__name__)):
        return False
    x, y = np.asarray(a), np.asarray(b)
    if x.dtype.kind == "c":
        return all(same_value(np.asarray(p), np.asarray(q)) for p, q in ((x.real, y.real), (x.imag, y.imag)))
    if x.dtype.kind == "f":
        return bool(np.isnan(x) and np.isnan(y)) or x.tobytes() == y.tobytes()
    return bool(x == y)


def compare_grid(dtype, left, others_py, others_np_dtypes, ops, tolerance_ulps=0):
    bad = []
    nps, rns = np_scalars(dtype, left), rn_scalars(dtype, left)
    operands = [(o, o) for o in others_py]
    for d, vals in others_np_dtypes:
        operands += list(zip(np_scalars(d, vals), rn_scalars(d, vals)))
    for name in ops:
        f = getattr(operator, name)
        for sn, sr in zip(nps, rns):
            for on, orr in operands:
                for reflected in (False, True):
                    call = (lambda a, b: f(b, a)) if reflected else f
                    w, we = outcome(lambda: call(sn, on))
                    g, ge = outcome(lambda: call(sr, orr))
                    if we or ge:
                        ok = we == ge
                    else:
                        ok = same_value(w, g)
                        if not ok and tolerance_ulps and np.asarray(w).dtype.kind in "fc":
                            ok = bool(np.allclose(np.asarray(w), np.asarray(g), rtol=1e-12, atol=0, equal_nan=True))
                    if not ok:
                        bad.append((name, reflected, repr(sn), repr(on), we or repr(w), ge or repr(g)))
    return bad


def test_float64_scalar_operators_match_numpy():
    bad = compare_grid("float64", FLOATS, FLOATS + PY_INTS + [True, False], [("int64", INTS), ("float64", FLOATS)], BINARY, 0)
    assert not bad, bad[:10]


def test_float64_scalar_with_complex_operands_matches_numpy():
    bad = compare_grid("float64", FLOATS, COMPLEXES, [("complex128", COMPLEXES)], ["add", "sub", "mul", "eq", "ne", "truediv", "pow"], 1)
    assert not bad, bad[:10]


def test_int64_scalar_operators_match_numpy():
    bad = compare_grid("int64", INTS, PY_INTS + FLOATS + [True], [("int64", INTS), ("float64", FLOATS)], BINARY + BITWISE, 0)
    assert not bad, bad[:10]


def test_complex128_scalar_operators_match_numpy():
    bad = compare_grid("complex128", COMPLEXES, COMPLEXES + FLOATS[:6] + [1, -3, True], [("float64", FLOATS[:6]), ("complex128", COMPLEXES)], ["add", "sub", "mul", "truediv", "pow", "eq", "ne"], 1)
    assert not bad, bad[:10]


def test_bool_scalar_operators_match_numpy():
    bad = compare_grid("bool", [True, False], [True, False, 0, 1, 2.5, 1j], [("bool", [True, False]), ("int64", INTS[:4])], ["add", "mul", "and_", "or_", "xor", "eq", "ne", "lt"], 0)
    assert not bad, bad[:10]


@pytest.mark.parametrize("dtype,values", [("float64", FLOATS), ("int64", INTS), ("complex128", COMPLEXES), ("bool", [True, False])])
def test_unary_operators_match_numpy(dtype, values):
    for name in ("neg", "pos", "abs", "invert"):
        f = getattr(operator, name)
        for sn, sr in zip(np_scalars(dtype, values), rn_scalars(dtype, values)):
            w, we = outcome(lambda: f(sn))
            g, ge = outcome(lambda: f(sr))
            assert (we is None) == (ge is None), (name, repr(sn), we, ge)
            if we is None:
                assert same_value(w, g), (name, repr(sn), w, g)


@pytest.mark.parametrize("dtype,value", [("float64", 1.5), ("int64", -7), ("complex128", 1 + 2j), ("bool", True)])
def test_scalar_protocol(dtype, value):
    x = rn_scalars(dtype, [value])[0]
    assert x.dtype == dtype and x.shape == () and x.ndim == 0 and x.size == 1
    assert x.item() == value and x.tolist() == value
    assert hash(x) == hash(value) and x == value
    assert pickle.loads(pickle.dumps(x)) == x and type(pickle.loads(pickle.dumps(x))) is type(x)
    assert copy.deepcopy(x) == x and copy.copy(x) == x
    assert np.asarray(x).dtype == np.dtype(dtype)
    assert str(x) == str(value) and repr(x) == repr(value)
    assert x[()] is x and x.astype("float32").dtype == "float32"
    assert {x: 1}[value] == 1
    assert x.reshape(1).shape == (1,)
    with pytest.raises(AttributeError):
        x.__nonexistent__


def test_python_number_conversions():
    i, f, b = rn_scalars("int64", [7])[0], rn_scalars("float64", [2.5])[0], rn_scalars("bool", [True])[0]
    assert int(i) == 7 and float(i) == 7.0 and complex(i) == 7 + 0j and bool(i) and operator.index(i) == 7
    assert [10, 20, 30, 40, 50, 60, 70, 80][i] == 80
    assert round(i) == 7 and round(i, -1) == 10 and math.floor(i) == 7 and f"{i:04d}" == "0007"
    assert int(f) == 2 and round(f) == 2 and math.sqrt(f) == math.sqrt(2.5) and f"{f:.2f}" == "2.50" and isinstance(f, float)
    assert int(b) == 1 and bool(b) is True and operator.index(b) == 1 and b is rn_scalars("bool", [True])[0]


def test_float_and_complex_scalars_are_python_numbers_and_ints_are_not():
    assert isinstance(rn_scalars("float64", [1.0])[0], float)
    assert isinstance(rn_scalars("complex128", [1j])[0], complex)
    assert not isinstance(rn_scalars("int64", [1])[0], int) and not isinstance(rn_scalars("bool", [True])[0], bool)


def test_scalar_results_stay_strong_under_nep_50():
    f32 = rnp.asarray([1.0], dtype="float32")
    assert (rn_scalars("float64", [2.0])[0] * f32).dtype == "float64"
    assert (2.0 * f32).dtype == "float32"
    assert (rn_scalars("int64", [2])[0] + rnp.asarray([1], dtype="int8")).dtype == "int64"


def test_scalars_interoperate_with_arrays_and_python_numbers_on_the_left():
    x = rn_scalars("float64", [2.0])[0]
    a = rnp.asarray([1.0, 2.0])
    assert np.array_equal(np.asarray(x * a), [2.0, 4.0]) and np.array_equal(np.asarray(a * x), [2.0, 4.0])
    assert np.array_equal(np.asarray(3 - a), [2.0, 1.0]) and 3.0 - x == 1.0 and 2 ** x == 4.0 and 1 / x == 0.5
    assert type(2.0 * x).__name__ == "float64" and type(2 * rn_scalars("int64", [3])[0]).__name__ == "int64"
    assert (x + "a" if False else True)
    with pytest.raises(TypeError):
        x + "a"
