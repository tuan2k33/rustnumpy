import numpy as np
import pytest

import rustnumpy_python as rnp
from conftest import DTYPES, FLOAT_DTYPES, INT_DTYPES, REAL_DTYPES, assert_same, sample


def x3(dtype, seed=20):
    return sample(dtype, (2, 3, 4), seed)


@pytest.mark.parametrize("dtype", DTYPES)
def test_view_style_functions_on_every_dtype(dtype):
    a = x3(dtype)
    for shape in [(6, 4), (-1,), (4, -1, 2), (2, 12)]:
        assert_same(rnp.reshape(a, shape), np.reshape(a, shape))
    assert_same(rnp.ravel(a), np.ravel(a))
    assert_same(rnp.transpose(a), np.transpose(a))
    for axes in [(2, 0, 1), (1, 2, 0), (-1, 0, 1)]:
        assert_same(rnp.transpose(a, axes), np.transpose(a, axes))
        assert_same(rnp.permute_dims(a, axes), np.transpose(a, axes))
    assert_same(rnp.swapaxes(a, 0, 2), np.swapaxes(a, 0, 2))
    for src, dst in [(0, -1), ([0, 1], [-1, -2]), (2, 0), (0, 0)]:
        assert_same(rnp.moveaxis(a, src, dst), np.moveaxis(a, src, dst))
    for axis in [0, 1, -1, (0, 4), (1, -2)]:
        assert_same(rnp.expand_dims(a, axis), np.expand_dims(a, axis))
    for axis in [None, 0, 2, (0, 1), (0, 1, 2), -1]:
        assert_same(rnp.flip(a, axis), np.flip(a, axis))
    assert_same(rnp.squeeze(a[:, :1]), np.squeeze(a[:, :1]))
    assert_same(rnp.squeeze(a[:, :1], axis=1), np.squeeze(a[:, :1], axis=1))
    assert_same(rnp.diagonal(a[0]), np.diagonal(a[0]))
    for off in (-1, 1, 3, -5):
        assert_same(rnp.diagonal(a[0], off), np.diagonal(a[0], off))


@pytest.mark.parametrize("dtype", ["int8", "float64", "complex64", "bool"])
def test_roll_and_repeat(dtype):
    a = x3(dtype)
    for shift, axis in [(1, None), (-2, 0), (5, 2), ((1, 2), (0, 2)), (1, (0, 1)), ((1, 2), None), (0, 1)]:
        assert_same(rnp.roll(a, shift, axis), np.roll(a, shift, axis))
    for reps, axis in [(2, None), (3, 0), ([1, 2, 0], 1), (1, -1), ([2], 2), ([1, 0, 3, 2], 2), (0, 1)]:
        assert_same(rnp.repeat(a, reps, axis), np.repeat(a, reps, axis))
    with pytest.raises(ValueError):
        rnp.repeat(a, [1, 2], axis=1)
    with pytest.raises(ValueError):
        rnp.repeat(a, -1)


def test_shape_errors_are_value_errors():
    a = x3("int32")
    for call in (lambda: rnp.reshape(a, (5, -1)), lambda: rnp.reshape(a, (-1, -1)), lambda: rnp.transpose(a, (0, 1)),
                 lambda: rnp.transpose(a, (0, 0, 1)), lambda: rnp.expand_dims(a, 9), lambda: rnp.squeeze(a, axis=1),
                 lambda: rnp.moveaxis(a, [0, 0], [1, 2]), lambda: rnp.flip(a, 3), lambda: rnp.diagonal(a)):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_broadcast_unstack_and_split(dtype):
    a, b, c = sample(dtype, (3, 1)), sample(dtype, (4,)), sample(dtype, (2, 1, 1))
    assert isinstance(rnp.broadcast_arrays(a, b, c), tuple) and isinstance(rnp.unstack(x3(dtype)), tuple)
    for got, want in zip(rnp.broadcast_arrays(a, b, c), np.broadcast_arrays(a, b, c)):
        assert_same(got, want)
    assert_same(rnp.broadcast_to(a, (2, 3, 5)), np.broadcast_to(a, (2, 3, 5)))
    with pytest.raises(ValueError):
        rnp.broadcast_arrays(np.ones(3), np.ones(4))
    x = x3(dtype)
    for axis in (0, 1, -1):
        for got, want in zip(rnp.unstack(x, axis=axis), np.unstack(x, axis=axis)):
            assert_same(got, want)
    for sections in (1, 2, 3, 5):
        for axis in (0, 1, 2):
            got, want = rnp.array_split(x, sections, axis), np.array_split(x, sections, axis)
            assert len(got) == len(want)
            for g, w in zip(got, want):
                assert_same(g, w)
    for idx in ([1, 2], [0, 5, 5, 20], []):
        for g, w in zip(rnp.array_split(x, idx, 2), np.array_split(x, idx, 2)):
            assert_same(g, w)
    with pytest.raises(ValueError):
        rnp.array_split(x, 0)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_unique_family(dtype):
    x = sample(dtype, (5, 6), seed=21)
    x = np.concatenate([x.ravel(), x.ravel()[:7]]).reshape(-1, 1)[:37].reshape(37)
    for name in ("unique_all", "unique_counts", "unique_inverse"):
        got, want = getattr(rnp, name)(x), getattr(np, name)(x)
        assert got._fields == want._fields
        for g, w in zip(got, want):
            assert_same(g, w)
    assert_same(np.sort(rnp.unique_values(x)), np.sort(np.unique_values(x)))
    m = x[:36].reshape(6, 6)
    got, want = rnp.unique_all(m), np.unique_all(m)
    for g, w in zip(got, want):
        assert_same(g, w)


def test_unique_keeps_nans_separate_and_merges_signed_zeros():
    x = np.array([3.0, np.nan, 1.0, np.nan, 0.0, -0.0])
    for g, w in zip(rnp.unique_all(x), np.unique_all(x)):
        assert_same(g, w)


@pytest.mark.parametrize("da,db", [("int8", "int8"), ("int32", "float64"), ("float32", "float32"), ("uint8", "int16"),
                                   ("complex64", "complex128"), ("bool", "bool"), ("int64", "bool")])
def test_matmul_dot_outer_kron(da, db):
    a, b = sample(da, (3, 4), 22), sample(db, (4, 2), 23)
    for name in ("matmul", "dot"):
        assert_same(getattr(rnp, name)(a, b), getattr(np, name)(a, b), *tol(np.result_type(da, db)))
    assert_same(rnp.outer(a[0], b[:, 0]), np.outer(a[0], b[:, 0]), *tol(np.result_type(da, db)))
    assert_same(rnp.kron(a, b), np.kron(a, b), *tol(np.result_type(da, db)))
    assert_same(rnp.matmul(a[0], b), np.matmul(a[0], b), *tol(np.result_type(da, db)))
    assert_same(rnp.matmul(a, b[:, 0]), np.matmul(a, b[:, 0]), *tol(np.result_type(da, db)))


def tol(dtype):
    dt = np.dtype(dtype)
    if dt.kind == "f" or dt.kind == "c":
        return (1e-4, 1e-4) if dt.itemsize in (4, 8) and dt.kind != "f" or dt == np.float32 else (1e-10, 1e-10)
    return (0.0, 0.0)


def finite(dtype, shape, seed):
    x = sample(dtype, shape, seed)
    if np.dtype(dtype).kind == "f":
        x = np.clip(np.nan_to_num(x, nan=0.5, posinf=3.0, neginf=-3.0), -100, 100).astype(dtype)
    elif np.dtype(dtype).kind == "c":
        clean = lambda v: np.clip(np.nan_to_num(v, nan=0.5, posinf=3.0, neginf=-3.0), -100, 100)
        x = (clean(x.real) + 1j * clean(x.imag)).astype(dtype)
    return x


@pytest.mark.parametrize("dtype", ["int32", "uint16", "float32", "float64", "complex128"])
def test_contraction_on_finite_values_is_close_to_numpy(dtype):
    a, b = finite(dtype, (2, 3, 4), 24), finite(dtype, (4, 5), 25)
    r, t = tol(dtype)
    assert_same(rnp.matmul(a, b), np.matmul(a, b), r, t)
    a3 = finite(dtype, (2, 3, 4), 26)
    b3 = finite(dtype, (2, 4, 3), 27)
    assert_same(rnp.matmul(a3, b3), np.matmul(a3, b3), r, t)
    assert_same(rnp.dot(a, b), np.dot(a, b), r, t)
    assert_same(rnp.tensordot(a, b, axes=1), np.tensordot(a, b, axes=1), r, t)
    assert_same(rnp.tensordot(a3, b3, axes=([1, 2], [2, 1])), np.tensordot(a3, b3, axes=([1, 2], [2, 1])), r, t)
    b34 = finite(dtype, (3, 4, 5), 38)
    assert_same(rnp.tensordot(a3, b34, axes=2), np.tensordot(a3, b34, axes=2), r, t)
    for sub, ops in [("ij,jk->ik", (a[0], b)), ("bij,bjk->bik", (a3, b3)), ("ii", (finite(dtype, (4, 4), 28),)),
                     ("ij->ji", (a[0],)), ("ij->", (a[0],)), ("i,i", (a[0, 0], a[1, 0])), ("...ij,...jk", (a3, b3)),
                     ("ij,ij->i", (a[0], a[1])), ("ijk->kji", (a,))]:
        assert_same(rnp.einsum(sub, *ops), np.einsum(sub, *ops), r, t)


@pytest.mark.parametrize("dtype", ["int8", "int64", "float32", "float64", "complex128"])
def test_cross_and_trace(dtype):
    a, b = finite(dtype, (5, 3), 29), finite(dtype, (5, 3), 30)
    assert_same(rnp.cross(a, b), np.cross(a, b), *tol(dtype))
    assert_same(rnp.cross(a[0], b), np.cross(a[0], b), *tol(dtype))
    m = finite(dtype, (4, 6), 31)
    for off in (0, 1, -2, 7):
        assert_same(rnp.trace(m, off), np.trace(m, off), *tol(dtype))


def test_contraction_errors():
    with pytest.raises(ValueError):
        rnp.matmul(np.ones((2, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        rnp.matmul(np.float64(3), np.ones((2, 3)))
    with pytest.raises(ValueError):
        rnp.einsum("ij,jk", np.ones((2, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        rnp.cross(np.ones(2), np.ones(2))


@pytest.mark.parametrize("dtype", ["int8", "float64", "complex64", "bool"])
def test_where_select_choose(dtype):
    x, y = sample(dtype, (3, 4), 32), sample(dtype, (4,), 33)
    cond = sample("float64", (3, 1), 34) > 0
    assert_same(rnp.where(cond, x, y), np.where(cond, x, y))
    assert_same(rnp.where(sample("int8", (3, 4), 35), x, y), np.where(sample("int8", (3, 4), 35), x, y))
    conds = [sample("float64", (3, 4), 36) > 0, sample("float64", (3, 4), 37) > 1]
    choices = [x, y]
    assert_same(rnp.select(conds, choices), np.select(conds, choices))
    assert_same(rnp.select(conds, choices, default=5), np.select(conds, choices, default=5))
    idx = np.array([[0, 1, 1, 0]] * 3)
    assert_same(rnp.choose(idx, [x, y]), np.choose(idx, [x, y]))
    wild = np.array([[-1, 3, 5, 1]] * 3)
    for mode in ("wrap", "clip"):
        assert_same(rnp.choose(wild, [x, y], mode=mode), np.choose(wild, [x, y], mode=mode))
    with pytest.raises(ValueError):
        rnp.choose(wild, [x, y])
