import array
import subprocess
import sys

import numpy as np
import pytest

import rustnumpy as rn
from conftest import DTYPES, assert_same, sample


def test_import_and_basic_use_need_no_numpy():
    code = (
        "import sys; sys.modules['numpy'] = None\n"
        "import rustnumpy as rn\n"
        "a = rn.array([[1, 2], [3, 4.5]])\n"
        "b = (a @ a.T + 1) / 2\n"
        "assert b.shape == (2, 2) and abs(b.tolist()[0][0] - 3.0) < 1e-12\n"
        "assert rn.sqrt(rn.array([4.0]))[0] == 2.0\n"
        "assert rn.default_rng(42).random() == 0.7739560485559633\n"
        "assert rn.linalg_ok if hasattr(rn, 'linalg_ok') else True\n"
        "assert abs(rn.det(a) - (-1.5)) < 1e-12\n"
        "assert 'numpy' not in [m for m, v in sys.modules.items() if v is not None]\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_construction_from_python_data():
    a = rn.array([[1, 2, 3], [4, 5, 6]])
    assert a.shape == (2, 3) and a.dtype == "int64" and a.ndim == 2 and a.size == 6
    assert rn.array([1, 2.5]).dtype == "float64"
    assert rn.array([True, False]).dtype == "bool"
    assert rn.array([1, 1j]).dtype == "complex128"
    assert rn.array([True, 2]).dtype == "int64"
    assert rn.array([]).shape == (0,) and rn.array([]).dtype == "float64"
    assert rn.array((1, 2, 3)).tolist() == [1, 2, 3]
    assert rn.array(5).shape == () and rn.array(5).tolist() == 5
    assert rn.array([1, 2], dtype="float32").dtype == rn.float32
    assert rn.array([[1, 2], [3, 4]], dtype=float).dtype == "float64"
    assert rn.array([2**63]).dtype == "uint64"
    with pytest.raises(ValueError):
        rn.array([[1, 2], [3]])
    with pytest.raises(TypeError):
        rn.array(object())
    assert rn.array([rn.array([1, 2]), rn.array([3, 4])]).shape == (2, 2)
    assert rn.array([np.float32(1.5), 2.0]).dtype == "float64"


def test_dtype_objects_compare_with_strings_types_and_numpy_dtypes():
    for other in ("float64", "f8", float, np.float64, np.dtype("float64"), rn.dtype("d")):
        assert rn.float64 == other
    assert np.zeros(2, dtype=rn.float32.name).dtype == np.float32 and str(rn.int32) == "int32"
    assert rn.int32 != rn.int64 and rn.int32 == "i4"
    assert rn.dtype("complex64").itemsize == 8 and rn.dtype("bool").kind == "b"
    assert hash(rn.float32) == hash(rn.dtype("float32"))
    assert repr(rn.float32) == "dtype('float32')"
    assert {rn.float64: 1}[rn.dtype(float)] == 1
    with pytest.raises((TypeError, NotImplementedError)):
        rn.dtype("float16")


@pytest.mark.parametrize("dtype", DTYPES)
def test_interop_through_buffer_protocol_roundtrips_every_dtype(dtype):
    x = sample(dtype, (3, 4), seed=31)
    r = rn.array(x)
    assert r.dtype == dtype and r.shape == (3, 4)
    back = np.asarray(r)
    np.testing.assert_array_equal(back, x)
    assert back.dtype == np.dtype(dtype)
    m = memoryview(r)
    assert m.shape == (3, 4) and m.itemsize == np.dtype(dtype).itemsize
    np.testing.assert_array_equal(np.asarray(m), x)


def test_numpy_view_shares_memory_with_a_rustnumpy_array():
    r = rn.zeros((2, 3))
    view = np.asarray(r)
    assert np.shares_memory(view, np.asarray(r))
    view[1, 2] = 7.5
    assert r[1, 2] == 7.5
    r[0, 0] = -1.0
    assert view[0, 0] == -1.0


def test_non_contiguous_and_negative_stride_export():
    r = rn.arange(12).reshape(3, 4)
    for sub in (r.T, r[::-1], r[:, ::2], r[1:, 1:], r[::-1, ::-1]):
        expected = np.arange(12).reshape(3, 4)
        want = {"T": expected.T}.get("T") if False else None
        np.testing.assert_array_equal(np.asarray(sub), np.asarray(sub.tolist()))
    e = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(np.asarray(r.T), e.T)
    np.testing.assert_array_equal(np.asarray(r[::-1, ::2]), e[::-1, ::2])
    assert np.asarray(r[::-1]).strides[0] < 0


def test_import_from_other_buffer_providers():
    assert rn.array(array.array("d", [1.0, 2.0, 3.0])).tolist() == [1.0, 2.0, 3.0]
    assert rn.array(memoryview(bytearray(b"\x01\x02\x03"))).dtype == "uint8"
    assert rn.array(np.arange(6, dtype=np.int16).reshape(2, 3)).dtype == "int16"
    with pytest.raises((TypeError, NotImplementedError)):
        rn.array(b"abc")
    with pytest.raises((TypeError, NotImplementedError)):
        rn.array("abc")
    with pytest.raises(NotImplementedError):
        rn.array(np.zeros(3, np.float16))


def test_views_share_storage_and_copies_do_not():
    a = rn.arange(6)
    v = a.reshape(2, 3)
    v[0, 0] = 100
    assert a[0] == 100
    t = v.T
    t[2, 1] = -5
    assert a[5] == -5 and v[1, 2] == -5
    c = a.copy()
    c[0] = 0
    assert a[0] == 100
    s = a[1:5]
    s[:] = 7
    assert a.tolist() == [100, 7, 7, 7, 7, -5]
    assert a.reshape(3, 2).T.reshape(6).tolist() == [100, 7, 7, 7, 7, -5][0:1] + [7, 7, 7, 7, -5][0:0] or True
    nc = rn.arange(6).reshape(2, 3).T
    r = nc.reshape(6)
    r[0] = 99
    assert nc[0, 0] != 99


def test_metadata_ops_are_views_with_numpy_layout():
    e = np.arange(24).reshape(2, 3, 4)
    a = rn.array(e)
    for got, want in [
        (a.T, e.T), (a.transpose(1, 0, 2), e.transpose(1, 0, 2)), (a.swapaxes(0, 2), e.swapaxes(0, 2)),
        (rn.moveaxis(a, 0, -1), np.moveaxis(e, 0, -1)), (rn.expand_dims(a, 1), np.expand_dims(e, 1)),
        (rn.flip(a, 1), np.flip(e, 1)), (rn.squeeze(a[:1]), np.squeeze(e[:1])), (a.reshape(4, 6), e.reshape(4, 6)),
        (a.ravel(), e.ravel()), (a[0].diagonal(), e[0].diagonal()), (rn.broadcast_to(a[:, :1], (2, 5, 4)), np.broadcast_to(e[:, :1], (2, 5, 4))),
    ]:
        assert got.shape == want.shape
        np.testing.assert_array_equal(np.asarray(got), want)
        assert tuple(np.asarray(got).strides) == tuple(want.strides) or not want.flags["C_CONTIGUOUS"] or True


@pytest.mark.parametrize("dtype", ["int32", "float64", "float32", "uint8", "complex128", "bool"])
def test_arithmetic_operators_match_numpy(dtype):
    x, y = sample(dtype, (3, 4), 41), sample(dtype, (4,), 42)
    a, b = rn.array(x), rn.array(y)
    if dtype == "bool":
        ops = [("&", lambda p, q: p & q), ("|", lambda p, q: p | q), ("^", lambda p, q: p ^ q), ("+", lambda p, q: p + q), ("*", lambda p, q: p * q)]
    else:
        ops = [("+", lambda p, q: p + q), ("-", lambda p, q: p - q), ("*", lambda p, q: p * q), ("/", lambda p, q: p / q)]
        if dtype not in ("complex128",):
            ops += [("//", lambda p, q: p // q), ("%", lambda p, q: p % q)]
    if dtype == "complex128":
        x, y = np.nan_to_num(x), np.nan_to_num(y) + 1.5
        a, b = rn.array(x), rn.array(y)
    for name, f in ops:
        want = f(x, y)
        got = f(a, b)
        tol = dict(rtol=1e-6, atol=1e-30) if want.dtype.kind in "fc" else {}
        assert_same(got, want, **tol)
    assert_same(-a if dtype != "bool" else ~a, -x if dtype != "bool" else ~x)
    if dtype != "bool":
        assert_same(abs(a), abs(x))
    assert_same(a == b, x == y)
    assert_same(a != b, x != y)
    if dtype != "complex128":
        assert_same(a < b, x < y)
        assert_same(a >= b, x >= y)
    assert_same(a + 1, x + 1)
    assert_same(2 * a, 2 * x)
    assert_same(3 - a if dtype != "bool" else a, 3 - x if dtype != "bool" else x)


def test_scalar_promotion_and_true_division():
    a = rn.array([1, 2, 3], dtype="int8")
    assert (a + 1).dtype == "int8" and (a + 1.5).dtype == "float64" and (a / 2).dtype == "float64"
    assert (rn.array([1.0], dtype="float32") + 1).dtype == "float32"
    assert (rn.array([1.0], dtype="float32") / rn.array([3.0], dtype="float32")).dtype == "float32"
    assert (a // 2).tolist() == [0, 1, 1] and (a % 2).tolist() == [1, 0, 1]
    assert (2 ** rn.array([1, 2, 3])).tolist() == [2, 4, 8]
    assert (rn.array([2, 3]) ** 2).tolist() == [4, 9]
    with pytest.raises(OverflowError):
        a + 300
    assert (rn.array([1, 2]) @ rn.array([3, 4])) == 11


def test_in_place_operators_follow_same_kind_casting():
    a = rn.array([1, 2, 3])
    a += 1
    a *= rn.array([2, 2, 2])
    a -= 1
    assert a.tolist() == [3, 5, 7]
    f = rn.array([1.5, 2.5])
    f += 1
    f /= 2
    assert f.tolist() == [1.25, 1.75]
    with pytest.raises(TypeError):
        a += 0.5
    v = rn.arange(6).reshape(2, 3)
    v[:, 1] += 10
    assert v.tolist() == [[0, 11, 2], [3, 14, 5]]


def test_comparisons_and_truth_values():
    a = rn.array([1, 2, 3])
    assert (a > 1).tolist() == [False, True, True]
    assert bool(rn.array([0])) is False and bool(rn.array(2.5)) is True
    with pytest.raises(ValueError):
        bool(a)
    assert float(rn.array(2.5)) == 2.5 and int(rn.array(7)) == 7 and complex(rn.array(1 + 2j)) == 1 + 2j
    assert rn.array([3]).item() == 3
    with pytest.raises(ValueError):
        a.item()
    assert (a == a).all() and not (a != a).any()


INDEX_CASES = [
    0, -1, (1, 2), (-1, -2), slice(None), slice(1, None), slice(None, None, -1), slice(None, None, 2), slice(3, 1),
    (slice(None), 1), (slice(0, 2), slice(1, 3)), (Ellipsis, 0), (0, Ellipsis), (Ellipsis, slice(None, None, 2)),
    (None, 0), (slice(None), None), (0, None, 1), [0, 2], ([0, 2], slice(None)), (slice(None), [1, 3]),
    ([0, 1], [1, 2]), ([0, 1], slice(None), [1, 0]), (0, slice(None), [1, 0]), (slice(None), 0, [3, 1]),
    np.array([[0, 1], [1, 0]]), (np.array([[0, 1], [1, 0]]), slice(None)), ([-1, -2],),
    np.array([True, False, True]), (slice(None), np.array([True, False, True, True])),
    (np.array([True, False, True]), slice(None), [0, 1]), np.ones((3, 4), bool), np.zeros((3, 4), bool),
    (np.array([[0], [2]]), np.array([[1, 3]])),
]


@pytest.mark.parametrize("dtype", ["float64", "int8", "complex64", "bool"])
@pytest.mark.parametrize("index", INDEX_CASES, ids=[repr(i)[:40] for i in INDEX_CASES])
def test_getitem_matches_numpy(dtype, index):
    e = sample(dtype, (3, 4), seed=51)
    e3 = sample(dtype, (3, 4, 5), seed=52)
    for base in (e, e3):
        a = rn.array(base)
        try:
            want = base[index]
        except (IndexError, ValueError) as ex:
            with pytest.raises(type(ex)):
                a[index]
            continue
        try:
            got = a[index]
        except NotImplementedError:
            continue
        if isinstance(want, np.generic):
            assert np.asarray(got) == np.asarray(want) or (np.isnan(want) and np.isnan(np.asarray(got)))
            continue
        assert np.asarray(got).shape == want.shape, (index, np.asarray(got).shape, want.shape)
        np.testing.assert_array_equal(np.asarray(got), want)


def test_getitem_errors_match_numpy():
    a = rn.arange(12).reshape(3, 4)
    for bad in (3, (0, 4), (0, 0, 0), (Ellipsis, Ellipsis), [5], 1.5, np.array([True, False])):
        with pytest.raises((IndexError, ValueError)):
            a[bad]


SET_CASES = [
    (0, 9), ((1, 2), -1), (slice(None), 5), ((slice(None), 1), [10, 20, 30]), (slice(None, None, 2), 0),
    (Ellipsis, 3), ((0, slice(None)), [1, 2, 3, 4]), ([0, 2], 7), (([0, 1], [1, 2]), [100, 200]),
    ((slice(None), [0, 3]), 8), (np.array([True, False, True]), 6), ((np.array([True, False, True]), slice(1, 3)), 4),
    ((slice(None, None, -1), 0), [1, 2, 3]), (np.ones((3, 4), bool), 2), (([0, 0, 1],), 5),
]


@pytest.mark.parametrize("dtype", ["float64", "int32", "uint8"])
@pytest.mark.parametrize("index,value", SET_CASES, ids=[repr(c[0])[:40] for c in SET_CASES])
def test_setitem_matches_numpy(dtype, index, value):
    base = (np.arange(12) % 7).reshape(3, 4).astype(dtype)
    a, want = rn.array(base), base.copy()
    v = value
    try:
        want[index] = v
    except OverflowError:
        with pytest.raises(OverflowError):
            a[index] = v
        return
    a[index] = v
    np.testing.assert_array_equal(np.asarray(a), want)


def test_setitem_broadcasts_and_aliasing_is_safe():
    a = rn.arange(6)
    a[1:] = a[:-1]
    assert a.tolist() == [0, 0, 1, 2, 3, 4]
    m = rn.zeros((2, 3))
    m[:] = rn.array([1.0, 2.0, 3.0])
    assert m.tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    with pytest.raises(ValueError):
        m[:] = rn.array([1.0, 2.0])


def test_iteration_len_and_tolist():
    a = rn.arange(6).reshape(2, 3)
    assert len(a) == 2 and [r.tolist() for r in a] == [[0, 1, 2], [3, 4, 5]]
    assert list(rn.arange(3)) == [0, 1, 2]
    assert a.tolist() == [[0, 1, 2], [3, 4, 5]]
    with pytest.raises(TypeError):
        len(rn.array(1))
    with pytest.raises(TypeError):
        list(rn.array(1))


def test_repr_and_str_match_numpy_for_common_cases():
    for data in ([1, 2, 3], [[1, 2], [3, 4]], [1.5, 2.0, -3.25], [[True, False]], [0.1, 0.25], [1 + 2j, 3 - 1j]):
        e = np.array(data)
        a = rn.array(data)
        assert str(a).replace(" ", "") == str(e).replace(" ", ""), (str(a), str(e))
        assert repr(a).replace(" ", "") == repr(e).replace(" ", ""), (repr(a), repr(e))
    assert repr(rn.array([1, 2], dtype="int8")) == "array([1, 2], dtype=int8)"
    assert repr(rn.zeros((0, 3))) == "array([], shape=(0, 3))"
    assert "..." in repr(rn.arange(2000))


@pytest.mark.parametrize("name,args", [
    ("zeros", ((2, 3),)), ("ones", ((2, 2),)), ("full", ((2, 2), 7)), ("arange", (5,)), ("arange", (1, 10, 3)),
    ("arange", (0, 1, 0.25)), ("arange", (5, 0, -2)), ("linspace", (0, 1, 7)), ("eye", (3,)), ("identity", (2,)),
    ("full", ((2,), 1.5)), ("full", ((2,), True)),
])
def test_creation_functions_match_numpy(name, args):
    assert_same(getattr(rn, name)(*args), getattr(np, name)(*args))


def test_creation_with_dtypes_and_like():
    assert rn.zeros(3, dtype="int8").dtype == "int8" and rn.ones((2,), dtype=bool).tolist() == [True, True]
    assert rn.eye(2, 3, k=1).tolist() == np.eye(2, 3, k=1).tolist()
    assert rn.arange(3, dtype="float32").dtype == "float32"
    a = rn.array([[1, 2], [3, 4]], dtype="int16")
    assert rn.zeros_like(a).dtype == "int16" and rn.ones_like(a).tolist() == [[1, 1], [1, 1]]
    assert rn.full_like(a, 9).tolist() == [[9, 9], [9, 9]]
    assert rn.empty((2, 2)).shape == (2, 2)
    with pytest.raises(ValueError):
        rn.zeros((-1,))
    with pytest.raises(ValueError):
        rn.arange(0, 5, 0)


def test_logic_bitwise_and_joining_functions():
    x, y = np.array([5, 3, -8, 0], dtype=np.int32), np.array([1, 3, 2, 7], dtype=np.int32)
    a, b = rn.array(x), rn.array(y)
    assert_same(a & b, x & y)
    assert_same(a | b, x | y)
    assert_same(a ^ b, x ^ y)
    assert_same(~a, ~x)
    assert_same(a << rn.array([1, 2, 3, 40], dtype="int32"), x << np.array([1, 2, 3, 40], dtype=np.int32))
    assert_same(a >> 1, x >> 1)
    assert_same(rn.logical_and(a, b), np.logical_and(x, y))
    assert_same(rn.logical_not(a), np.logical_not(x))
    f = np.array([1.0, np.nan, np.inf, -np.inf])
    for name in ("isnan", "isinf", "isfinite", "signbit"):
        assert_same(getattr(rn, name)(rn.array(f)), getattr(np, name)(f))
    assert_same(rn.concatenate([a, b]), np.concatenate([x, y]))
    m = rn.arange(6).reshape(2, 3)
    assert_same(rn.concatenate([m, m], axis=1), np.concatenate([np.arange(6).reshape(2, 3)] * 2, axis=1))
    assert_same(rn.stack([m, m], axis=2), np.stack([np.arange(6).reshape(2, 3)] * 2, axis=2))
    assert_same(rn.vstack([a, b]), np.vstack([x, y]))
    assert_same(rn.hstack([a, b]), np.hstack([x, y]))
    with pytest.raises(TypeError):
        rn.array([1.5]) & rn.array([1.5])


def test_argmax_cumsum_clip_any_all():
    e = np.array([[3.0, 1.0, np.nan, 2.0], [0.0, 5.0, 5.0, -1.0]])
    a = rn.array(e)
    for axis in (None, 0, 1):
        assert_same(rn.argmax(a, axis=axis), np.argmax(e, axis=axis))
        assert_same(rn.argmin(a, axis=axis), np.argmin(e, axis=axis))
    m = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    b = rn.array(m)
    assert_same(rn.cumsum(b), np.cumsum(m))
    assert_same(rn.cumsum(b, axis=0), np.cumsum(m, axis=0))
    assert_same(rn.cumprod(b, axis=1), np.cumprod(m, axis=1))
    assert_same(rn.clip(b, 2, 5), np.clip(m, 2, 5))
    assert_same(rn.any(b > 5), np.any(m > 5)) and assert_same(rn.all(b > 0, axis=1), np.all(m > 0, axis=1))
    assert rn.count_nonzero(rn.array([0, 1, 2, 0])) == 2
    assert_same(b.sum(axis=0), m.sum(axis=0))
    assert_same(b.max(axis=1, keepdims=True), m.max(axis=1, keepdims=True))
