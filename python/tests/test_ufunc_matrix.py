import itertools

import numpy as np
import pytest

import rustnumpy as rnp
from conftest import DTYPES, sample

UNARY = ("absolute negative positive sign square sqrt cbrt reciprocal exp exp2 expm1 log log2 log10 log1p sin cos tan arcsin "
         "arccos arctan sinh cosh tanh arcsinh arccosh arctanh degrees radians floor ceil trunc rint fabs isnan isinf "
         "isfinite signbit invert logical_not conjugate spacing").split()

BINARY = ("add subtract multiply divide floor_divide remainder fmod power float_power maximum minimum fmax fmin arctan2 "
          "hypot copysign nextafter logaddexp logaddexp2 heaviside gcd lcm logical_and logical_or logical_xor bitwise_and "
          "bitwise_or bitwise_xor left_shift right_shift equal not_equal less less_equal greater greater_equal").split()

TOL = {"float16": (2e-3, 1e-6), "float32": (2e-5, 1e-30), "float64": (1e-12, 1e-300),
       "complex64": (2e-5, 1e-30), "complex128": (1e-12, 1e-300)}


def finite(x):
    if x.dtype.kind != "c":
        return x
    part = np.float32 if x.dtype == np.complex64 else np.float64
    re = np.nan_to_num(x.real, nan=0.5, posinf=3.0, neginf=-3.0).astype(part)
    im = np.nan_to_num(x.imag, nan=-0.25, posinf=2.0, neginf=-2.0).astype(part)
    return (re + 1j * im).astype(x.dtype)


def moderate(x):
    if x.dtype.kind != "c":
        return x
    mag = np.abs(x)
    scale = np.where(mag > 1e3, 1e3 / np.where(mag > 0, mag, 1), np.where((mag < 1e-3) & (mag > 0), 1e-3 / np.where(mag > 0, mag, 1), 1.0))
    return (x * scale).astype(x.dtype)


def close(got, want, name, what):
    got, want = np.asarray(got), np.asarray(want)
    assert got.dtype == want.dtype, f"{name}{what}: dtype {got.dtype} != {want.dtype}"
    assert got.shape == want.shape, f"{name}{what}: shape {got.shape} != {want.shape}"
    if got.dtype.kind in "biu":
        np.testing.assert_array_equal(got, want, err_msg=f"{name}{what}")
        return
    rtol, atol = TOL[got.dtype.name]
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, equal_nan=True, err_msg=f"{name}{what}")


def run(fn, *args):
    try:
        return fn(*args), None
    except Exception as e:
        return None, e


def check(name, np_args, rn_args):
    with np.errstate(all="ignore"):
        want, werr = run(getattr(np, name), *np_args)
    got, gerr = run(getattr(rnp, name), *rn_args)
    if werr is not None:
        assert gerr is not None, f"{name}: NumPy raised {type(werr).__name__} but rustnumpy returned {got!r}"
        assert not isinstance(gerr, NotImplementedError), f"{name}: unsupported ({gerr})"
        assert isinstance(gerr, (type(werr), TypeError, ValueError)), f"{name}: raised {type(gerr).__name__}: {gerr}"
        return
    assert gerr is None, f"{name}: rustnumpy raised {type(gerr).__name__}: {gerr}"
    if isinstance(want, tuple):
        assert len(got) == len(want)
        for i, (g, w) in enumerate(zip(got, want)):
            close(g, w, name, f"[{i}]")
    else:
        close(got, want, name, "")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", UNARY)
def test_unary_matrix(name, dtype):
    x = finite(sample(dtype, (3, 4), seed=5))
    check(name, (x,), (rnp.array(x),))


@pytest.mark.parametrize("d2", DTYPES)
@pytest.mark.parametrize("d1", DTYPES)
@pytest.mark.parametrize("name", BINARY)
def test_binary_matrix(name, d1, d2):
    a, b = finite(sample(d1, (3, 4), seed=1)), finite(sample(d2, (4,), seed=2))
    if name in ("power", "float_power"):
        a, b = moderate(a), moderate(b)
    check(name, (a, b), (rnp.array(a), rnp.array(b)))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", ["modf", "frexp"])
def test_two_output_unary(name, dtype):
    x = finite(sample(dtype, (3, 4), seed=6))
    check(name, (x,), (rnp.array(x),))


@pytest.mark.parametrize("d2", DTYPES)
@pytest.mark.parametrize("d1", DTYPES)
def test_divmod_and_ldexp(d1, d2):
    a, b = finite(sample(d1, (3, 4), seed=3)), finite(sample(d2, (4,), seed=4))
    check("divmod", (a, b), (rnp.array(a), rnp.array(b)))
    check("ldexp", (a, b), (rnp.array(a), rnp.array(b)))


WEAK = [3, -2, 0, 2.5, -0.5, 1 + 2j, True]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", ["add", "subtract", "multiply", "divide", "floor_divide", "remainder", "power", "maximum",
                                  "minimum", "arctan2", "hypot", "equal", "less", "logical_and", "copysign", "fmod",
                                  "bitwise_and", "left_shift", "gcd", "float_power", "logaddexp", "heaviside"])
def test_weak_python_scalars(name, dtype):
    a = finite(sample(dtype, (3, 4), seed=8))
    if name in ("power", "float_power"):
        a = moderate(a) if a.dtype.kind == "c" else np.clip(a, -1e3, 1e3).astype(a.dtype) if a.dtype.kind == "f" else a
    for s in WEAK:
        for order in (0, 1):
            np_args = (a, s) if order == 0 else (s, a)
            rn_args = (rnp.array(a), s) if order == 0 else (s, rnp.array(a))
            check(name, np_args, rn_args)


def test_complex_functions_near_branch_cuts_and_signed_zeros():
    vals = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1e-9, 3.0, -7.5, 25.0]
    z = np.array([complex(a, b) for a in vals for b in vals])
    for name in ("sqrt", "log", "log2", "log10", "log1p", "exp", "exp2", "expm1", "sin", "cos", "tan", "arcsin", "arccos",
                 "arctan", "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "sign", "reciprocal", "square"):
        with np.errstate(all="ignore"):
            want = getattr(np, name)(z)
        got = np.asarray(getattr(rnp, name)(rnp.array(z)))
        np.testing.assert_allclose(got, want, rtol=1e-11, atol=1e-300, equal_nan=True, err_msg=name)
        if name == "exp2":
            continue
        moving = want.imag != 0
        assert (np.signbit(got.imag) == np.signbit(want.imag))[~moving].all(), f"{name}: signed zero of the imaginary part"
        moving = want.real != 0
        assert (np.signbit(got.real) == np.signbit(want.real))[~moving].all(), f"{name}: signed zero of the real part"


def test_complex_power_special_cases():
    z = np.array([0j, -0.0 + 0j, 1 + 1j, -1 + 0j, -1 - 0j, 2j, 1e-300 + 0j])
    for e in (0j, 1 + 0j, 2 + 0j, 3 + 0j, -2 + 0j, 0.5 + 0j, 1 + 1j, -1 + 1j, 150 + 0j):
        check("power", (z, e), (rnp.array(z), e))
