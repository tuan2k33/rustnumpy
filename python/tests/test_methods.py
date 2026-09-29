import pickle
import copy

import numpy as np
import pytest

import rustnumpy as rnp
from conftest import DTYPES, sample
from difftest import compare, differential


def a_of(m, shape=(3, 4), dtype="float64"):
    return m.array(np.random.default_rng(0).standard_normal(shape).astype(dtype) * 4)


METHOD_CASES = {
    "sum": lambda m, A: a_of(m).sum(axis=1, keepdims=True),
    "prod": lambda m, A: a_of(m).prod(axis=0),
    "max": lambda m, A: a_of(m).max(axis=(0, 1)),
    "min": lambda m, A: a_of(m).min(initial=-10, axis=1),
    "mean": lambda m, A: a_of(m).mean(axis=0, dtype="float32"),
    "var": lambda m, A: a_of(m).var(axis=1, ddof=1),
    "std": lambda m, A: a_of(m).std(),
    "argmax": lambda m, A: a_of(m).argmax(axis=1),
    "argmin": lambda m, A: a_of(m).argmin(),
    "any": lambda m, A: (a_of(m) > 3).any(axis=0),
    "all": lambda m, A: (a_of(m) > -30).all(),
    "cumsum": lambda m, A: a_of(m).cumsum(axis=1),
    "cumprod": lambda m, A: a_of(m).cumprod(),
    "clip": lambda m, A: a_of(m).clip(-1, 1),
    "clip_kw": lambda m, A: a_of(m).clip(min=0),
    "argsort": lambda m, A: a_of(m).argsort(axis=0),
    "dot": lambda m, A: a_of(m).dot(a_of(m, (4, 2))),
    "trace": lambda m, A: a_of(m, (3, 3)).trace(),
    "diagonal": lambda m, A: a_of(m, (3, 4)).diagonal(1),
    "repeat": lambda m, A: a_of(m).repeat(2, axis=0),
    "take": lambda m, A: a_of(m).take([0, 2], axis=1),
    "nonzero": lambda m, A: (a_of(m) > 0).nonzero(),
    "compress": lambda m, A: a_of(m).compress([True, False, True], axis=0),
    "round": lambda m, A: a_of(m).round(1),
    "conj": lambda m, A: A([1 + 2j, 3 - 1j]).conj(),
    "conjugate": lambda m, A: A([1 + 2j, 3 - 1j]).conjugate(),
    "searchsorted": lambda m, A: A([1.0, 2.0, 3.0]).searchsorted(A([0.5, 2.5])),
    "ptp": lambda m, A: a_of(m).ptp(axis=1) if hasattr(a_of(m), "ptp") else None,
    "T": lambda m, A: a_of(m).T,
    "mT": lambda m, A: a_of(m, (2, 3, 4)).mT,
    "reshape": lambda m, A: a_of(m).reshape(2, -1),
    "reshape_tuple": lambda m, A: a_of(m).reshape((4, 3)),
    "astype": lambda m, A: a_of(m).astype("int8"),
    "astype_casting": lambda m, A: a_of(m).astype("float32", casting="safe"),
    "astype_casting_bad": lambda m, A: a_of(m).astype("int8", casting="safe"),
    "flatten": lambda m, A: a_of(m).flatten(),
    "ravel": lambda m, A: a_of(m).T.ravel(),
    "squeeze": lambda m, A: a_of(m, (1, 3, 1)).squeeze(),
    "swapaxes": lambda m, A: a_of(m, (2, 3, 4)).swapaxes(0, 2),
    "transpose": lambda m, A: a_of(m, (2, 3, 4)).transpose(2, 0, 1),
    "tolist": lambda m, A: a_of(m, (2, 2)).tolist(),
    "item": lambda m, A: a_of(m, (1,)).item(),
    "copy": lambda m, A: a_of(m).copy(),
    "fill": lambda m, A: _fill(m),
    "sort_inplace": lambda m, A: _sort_inplace(m),
    "itemsize": lambda m, A: (a_of(m, dtype="float32").itemsize, a_of(m).nbytes, a_of(m).ndim, a_of(m).size),
    "strides": lambda m, A: (a_of(m).strides, a_of(m).T.strides, a_of(m, dtype="int8").strides),
    "flags": lambda m, A: (a_of(m).flags.c_contiguous, a_of(m).T.flags.c_contiguous, a_of(m).T.flags.f_contiguous, a_of(m).flags["OWNDATA"], a_of(m)[:, ::2].flags.c_contiguous),
    "flat": lambda m, A: (a_of(m).flat[3], a_of(m).flat[[1, 5]], list(a_of(m, (2, 2)).flat)),
    "contains": lambda m, A: (1.0 in A([1.0, 2.0]), 5.0 in A([1.0, 2.0])),
    "divmod": lambda m, A: divmod(A([7, -7]), 3),
    "rdivmod": lambda m, A: divmod(7, A([2, 3])),
    "view_same": lambda m, A: a_of(m).view().shape,
    "tobytes": lambda m, A: A([1, 2], dtype="int16").tobytes(),
    "bool_scalar": lambda m, A: (bool(A([1.0])), bool(A([[0]]))),
    "int_float": lambda m, A: (int(A([3.7])), float(A([2])), complex(A([1 + 2j]))),
    "len_iter": lambda m, A: (len(a_of(m)), [tuple(r.tolist()) for r in a_of(m, (2, 2))]),
    "hash_unhashable": lambda m, A: _unhashable(m),
    "neg_pos_abs_inv": lambda m, A: (-A([1, -2]), +A([1, -2]), abs(A([1, -2])), ~A([1, -2])),
    "pow_mod": lambda m, A: (A([2, 3]) ** 2, 2 ** A([2, 3]), A([7, 8]) % 3, A([7, 8]) // 3),
    "matmul_inplace": lambda m, A: _imatmul(m),
    "setitem_bool": lambda m, A: _setitem(m),
    "ellipsis_newaxis": lambda m, A: a_of(m, (2, 3, 4))[..., None, 1:3],
    "fancy_2d": lambda m, A: a_of(m)[[0, 2], 1:3],
    "shape_assign_error": lambda m, A: a_of(m).reshape(5),
}


def _fill(m):
    x = m.zeros((2, 2))
    x.fill(3.5)
    return x


def _sort_inplace(m):
    x = m.array([[3.0, 1.0, 2.0], [9.0, 8.0, 7.0]])
    r = x.sort(axis=1)
    return x, r


def _unhashable(m):
    try:
        hash(m.array([1]))
    except TypeError:
        return "unhashable"
    return "hashable"


def _imatmul(m):
    x = m.array([[1.0, 2.0], [3.0, 4.0]])
    x @= m.array([[0.0, 1.0], [1.0, 0.0]])
    return x


def _setitem(m):
    x = m.arange(10)
    x[x > 6] = -1
    x[[0, 1]] = [10, 20]
    x[2:4] += 5
    return x


@pytest.mark.parametrize("name", sorted(METHOD_CASES))
def test_method(name):
    if name == "ptp":
        pytest.skip("ndarray.ptp was removed in NumPy 2")
    differential(lambda mod, cast: METHOD_CASES[name](mod, cast))


def test_pickle_copy_and_deepcopy():
    x = rnp.array(np.arange(12, dtype="float32").reshape(3, 4))
    for clone in (pickle.loads(pickle.dumps(x)), copy.copy(x), copy.deepcopy(x)):
        assert clone.dtype == x.dtype and clone.shape == x.shape
        np.testing.assert_array_equal(np.asarray(clone), np.asarray(x))
    view = x[:, ::2]
    assert np.asarray(pickle.loads(pickle.dumps(view))).tolist() == np.asarray(view).tolist()
    c = copy.copy(x)
    c[0, 0] = 99
    assert x[0, 0] == 0


def test_base_and_owndata():
    x = rnp.arange(6.0)
    assert x.base is None and x.flags.owndata
    v = x.reshape(2, 3)
    assert v.base is not None and not v.flags.owndata
    v[0, 0] = 42.0
    assert x[0] == 42.0


@pytest.mark.parametrize("dtype", DTYPES)
def test_dtype_objects_are_callable_and_have_numpy_attributes(dtype):
    dt = rnp.dtype(dtype)
    n = np.dtype(dtype)
    assert dt.name == n.name and dt.kind == n.kind and dt.itemsize == n.itemsize and dt.char == n.char and dt.str == n.str.replace("=", "<") or dt.str in (n.str, n.str.replace("=", "<"))
    value = dt.type(3) if hasattr(dt, "type") else None
    assert value is not None
    assert rnp.asarray(value).dtype == dt


def test_array_namespace_and_device():
    x = rnp.arange(3)
    assert x.__array_namespace__() is rnp
    assert x.device == "cpu" and x.to_device("cpu") is x


def test_view_reinterpretation_and_byteswap():
    x = np.arange(4, dtype="int32")
    got = np.asarray(rnp.array(x).view("uint8"))
    np.testing.assert_array_equal(got, x.view("uint8"))
    np.testing.assert_array_equal(np.asarray(rnp.array(x).byteswap()), x.byteswap())


@pytest.mark.parametrize("dtype", DTYPES)
def test_dlpack_round_trips_with_numpy(dtype):
    x = sample(dtype, (3, 4), seed=2)
    ours = rnp.array(x)
    back = np.from_dlpack(ours)
    assert back.dtype == x.dtype
    np.testing.assert_array_equal(back, x)
    strided = ours[:, ::2].T
    np.testing.assert_array_equal(np.from_dlpack(strided), x[:, ::2].T)
    theirs = rnp.from_dlpack(x)
    assert theirs.dtype == dtype
    np.testing.assert_array_equal(np.asarray(theirs), x)
    np.testing.assert_array_equal(np.asarray(rnp.from_dlpack(np.asfortranarray(x))), x)
    assert ours.__dlpack_device__() == (1, 0)


def test_dlpack_shares_memory_and_keeps_the_owner_alive():
    ours = rnp.arange(6.0)
    view = np.from_dlpack(ours)
    assert not view.flags.writeable
    ours[0] = 99.0
    assert view[0] == 99.0
    del ours
    assert view[0] == 99.0
