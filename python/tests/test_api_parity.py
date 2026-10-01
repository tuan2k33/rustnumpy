import io
import warnings

import numpy as np
import pytest

import rustnumpy as rnp

A = [[1.0, 2.0], [3.0, 4.0]]
V3 = [1.0, 2.0, 3.0]


def outcome(m, fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = fn(m)
        except Exception as e:
            return type(e).__name__.replace("UFuncTypeError", "TypeError"), None
        r = np.asarray(r)
        return None, (str(r.dtype), r.shape, r.tolist())


def same(a, b):
    (ea, va), (eb, vb) = a, b
    if ea or eb:
        return ea == eb
    if va[:2] != vb[:2]:
        return False
    return np.allclose(np.asarray(va[2]), np.asarray(vb[2]), rtol=1e-12, equal_nan=True)


CASES = {
    "percentile_interpolation_removed": lambda m: m.percentile(m.asarray(V3), 50, interpolation="lower"),
    "nanpercentile_interpolation_removed": lambda m: m.nanpercentile(m.asarray(V3), 50, interpolation="lower"),
    "reshape_newshape_removed": lambda m: m.reshape(m.asarray(V3), newshape=(3, 1)),
    "reshape_needs_shape": lambda m: m.reshape(m.asarray(V3)),
    "array_ndmax_1": lambda m: m.array([[1, 2], [3, 4]], ndmax=1),
    "array_ndmax_2": lambda m: m.array([[1, 2], [3, 4]], ndmax=2),
    "array_ndmax_0": lambda m: m.array([[1, 2], [3, 4]], ndmax=0),
    "array_ndmax_scalar": lambda m: m.array(1, ndmax=0),
    "array_ndmax_ndarray": lambda m: m.array(m.ones((2, 2)), ndmax=1),
    "array_ndmax_negative": lambda m: m.array([1], ndmax=-1),
    "array_ndmin_over_ndmax": lambda m: m.array([[1, 2]], ndmin=3, ndmax=2),
    "einsum_out": lambda m: m.einsum("ij,jk->ik", m.asarray(A), m.asarray(A), out=m.zeros((2, 2))),
    "einsum_optimize_true": lambda m: m.einsum("ij,jk->ik", m.asarray(A), m.asarray(A), optimize=True),
    "einsum_dtype": lambda m: m.einsum("ij,jk->ik", m.asarray(A), m.asarray(A), dtype="float32"),
    "einsum_dtype_not_safe": lambda m: m.einsum("ij,jk->ik", m.asarray(A), m.asarray(A), dtype="int32"),
    "einsum_dtype_unsafe": lambda m: m.einsum("ij,jk->ik", m.asarray(A), m.asarray(A), dtype="int32", casting="unsafe"),
    "einsum_unknown_kwarg": lambda m: m.einsum("ij,jk->ik", m.asarray(A), m.asarray(A), foo=1),
    "einsum_sublist": lambda m: m.einsum(m.asarray(A), [0, 1], m.asarray(A), [1, 2]),
    "einsum_sublist_out": lambda m: m.einsum(m.asarray(A), [0, 1], m.asarray(A), [1, 2], [0, 2]),
    "einsum_three_greedy": lambda m: m.einsum("ij,jk,kl->il", m.ones((3, 4)), m.ones((4, 5)), m.ones((5, 2)), optimize="greedy"),
    "einsum_three_optimal": lambda m: m.einsum("ij,jk,kl->il", m.ones((3, 4)), m.ones((4, 5)), m.ones((5, 2)), optimize="optimal"),
    "dot_out": lambda m: m.dot(m.asarray(A), m.asarray(A), out=m.zeros((2, 2))),
    "dot_out_wrong_dtype": lambda m: m.dot(m.asarray(A), m.asarray(A), out=m.zeros((2, 2), dtype="float32")),
    "dot_out_0d": lambda m: m.dot(m.asarray(V3), m.asarray(V3), out=m.zeros(())),
    "fft_out": lambda m: m.fft.fft(m.asarray([1.0, 2.0, 3.0, 4.0]), out=m.zeros(4, dtype="complex128")),
    "fft_out_wrong_shape": lambda m: m.fft.fft(m.asarray(V3), out=m.zeros(4, dtype="complex128")),
    "fft_out_wrong_dtype": lambda m: m.fft.fft(m.asarray(V3), out=m.zeros(3)),
    "fft_out_list": lambda m: m.fft.fft(m.asarray(V3), out=[0, 0, 0]),
    "irfft_out": lambda m: m.fft.irfft(m.asarray([1, 2, 3], dtype="complex128"), out=m.zeros(4)),
    "fft2_out": lambda m: m.fft.fft2(m.asarray(A), out=m.zeros((2, 2), dtype="complex128")),
    "rfftn_out": lambda m: m.fft.rfftn(m.asarray(A), out=m.zeros((2, 2), dtype="complex128")),
    "vecdot_out": lambda m: m.vecdot(m.asarray(A), m.asarray(A), out=m.zeros(2)),
    "vecdot_keepdims": lambda m: m.vecdot(m.asarray(A), m.asarray(A), keepdims=True),
    "vecdot_axis0": lambda m: m.vecdot(m.asarray(A), m.asarray(A), axis=0),
    "vecdot_axis0_keepdims": lambda m: m.vecdot(m.asarray(A), m.asarray(A), axis=0, keepdims=True),
    "vecdot_axes": lambda m: m.vecdot(m.asarray(A), m.asarray(A), axes=[(0,), (0,), ()]),
    "vecdot_axes_and_axis": lambda m: m.vecdot(m.asarray(A), m.asarray(A), axis=0, axes=[(0,), (0,), ()]),
    "vecdot_where": lambda m: m.vecdot(m.asarray(A), m.asarray(A), where=True),
    "vecdot_complex_conjugates_first": lambda m: m.vecdot(m.asarray([1j, 2]), m.asarray([3, 4])),
    "vecdot_dtype": lambda m: m.vecdot(m.asarray(A), m.asarray(A), dtype="float32"),
    "vecdot_mismatch": lambda m: m.vecdot(m.ones((2, 3)), m.ones((2, 4))),
    "matmul_keepdims": lambda m: m.matmul(m.asarray(A), m.asarray(A), keepdims=True),
    "matmul_axis": lambda m: m.matmul(m.asarray(A), m.asarray(A), axis=0),
    "matmul_axes": lambda m: m.matmul(m.ones((4, 2, 3)), m.ones((4, 5, 3)), axes=[(-2, -1), (-1, -2), (-2, -1)]),
    "matmul_axes_1d": lambda m: m.matmul(m.ones((2, 3)), m.ones(3), axes=[(-2, -1), (0,), (0,)]),
    "matmul_axes_wrong_output_len": lambda m: m.matmul(m.ones((2, 3)), m.ones(3), axes=[(-2, -1), (0,), (-2, -1)]),
    "matvec_out": lambda m: m.matvec(m.asarray(A), m.asarray([1.0, 1.0]), out=m.zeros(2)),
    "matvec_keepdims": lambda m: m.matvec(m.asarray(A), m.asarray([1.0, 1.0]), keepdims=True),
    "matvec_axis": lambda m: m.matvec(m.asarray(A), m.asarray([1.0, 1.0]), axis=0),
    "vecmat_axes": lambda m: m.vecmat(m.asarray([1.0, 1.0]), m.asarray(A), axes=[(-1,), (-2, -1), (-1,)]),
    "bitwise_count_out": lambda m: m.bitwise_count(m.asarray([7, 8]), out=m.zeros(2, dtype="uint8")),
    "bitwise_count_where": lambda m: m.bitwise_count(m.asarray([7, 8]), where=m.asarray([True, False]), out=m.zeros(2, dtype="uint8")),
    "bitwise_count_negative": lambda m: m.bitwise_count(m.asarray([-1, -128], dtype="int8")),
    "real_keyword": lambda m: m.real(val=m.asarray([1 + 2j])),
    "imag_keyword": lambda m: m.imag(val=m.asarray([1 + 2j])),
    "linalg_matmul": lambda m: m.linalg.matmul(m.asarray(A), m.asarray(A)),
    "linalg_vecdot_axis": lambda m: m.linalg.vecdot(m.asarray(A), m.asarray(A), axis=0),
    "array2string_style_removed": lambda m: m.array2string(m.ones(2), style=repr),
    "save_fix_imports_removed": lambda m: m.save(io.BytesIO(), m.ones(2), fix_imports=True),
    "loadtxt_lines": lambda m: m.loadtxt(["1 2", "3 4"]),
}


@pytest.mark.parametrize("name", CASES)
def test_case(name):
    fn = CASES[name]
    assert same(outcome(np, fn), outcome(rnp, fn)), (outcome(np, fn), outcome(rnp, fn))


def test_gufunc_attributes():
    assert rnp.matmul.signature == np.matmul.signature
    assert rnp.vecdot.signature == np.vecdot.signature
    assert rnp.matvec.signature == np.matvec.signature
    assert rnp.vecmat.signature == np.vecmat.signature


def test_generator_dir_lists_distributions():
    assert {"normal", "uniform", "shuffle", "permutation"} <= set(dir(rnp.random.default_rng(0)))


def test_einsum_path_matches_numpy():
    rng = np.random.default_rng(0)
    ops = [rng.random((3, 4)), rng.random((4, 5)), rng.random((5, 2))]
    for optimize in ("greedy", "optimal", False):
        want = np.einsum_path("ij,jk,kl->il", *ops, optimize=optimize)
        got = rnp.einsum_path("ij,jk,kl->il", *[rnp.asarray(o) for o in ops], optimize=optimize)
        assert want[0] == got[0] and want[1] == got[1]


def test_einsum_optimized_matches_unoptimized():
    rng = np.random.default_rng(1)
    ops = [rng.random((3, 4)), rng.random((4, 5)), rng.random((5, 2))]
    plain = np.einsum("ij,jk,kl->il", *ops)
    for optimize in (True, "greedy", "optimal"):
        got = rnp.einsum("ij,jk,kl->il", *[rnp.asarray(o) for o in ops], optimize=optimize)
        assert np.allclose(np.asarray(got), plain)


GEN = [
    (["1,2", "3,4"], dict(delimiter=",")),
    (["1,,3", "4,5,"], dict(delimiter=",")),
    (["1,,3", "4,5,"], dict(delimiter=",", dtype=int)),
    (["1,,3"], dict(delimiter=",", dtype=bool)),
    (["1,,3"], dict(delimiter=",", filling_values=-9)),
    (["1,N/A,3"], dict(delimiter=",", missing_values="N/A")),
    (["1,N/A,-,3"], dict(delimiter=",", missing_values=["N/A", "-"])),
    (["1,x", "2,y"], dict(delimiter=",", missing_values={1: "x"})),
    (["1,", "2,y"], dict(delimiter=",", filling_values={1: 7})),
    (["1 2", "3 4"], dict(dtype=None)),
    (["True False"], dict(dtype=None)),
    (["1,,3"], dict(delimiter=",", dtype=None)),
    (["1.5 2.5"], dict(dtype=None)),
    (["1.5 2"], dict(dtype=int)),
    (["1 abc"], {}),
    (["1 abc"], dict(dtype=int)),
    (["1 abc"], dict(loose=False)),
    (["True false 1 0"], dict(dtype=bool)),
    (["1+2j 3"], dict(dtype=complex)),
    (["1,2", "3,4,5"], dict(delimiter=",")),
    (["h", "1 2", "3 4"], dict(skip_header=1)),
    (["1 2", "3 4", "f"], dict(skip_footer=1)),
    (["1 2 3", "4 5 6"], dict(usecols=(0, 2))),
    (["1 2 3", "4 5 6"], dict(usecols=1)),
    (["1 2", "3 4"], dict(unpack=True)),
    (["1 2 3"], {}),
    (["1", "2", "3"], {}),
    (["5"], {}),
    (["1 2", "3 4"], dict(converters={0: lambda s: float(s) * 10})),
    (["1 2 # c", "# full", "3 4"], {}),
    (["1 2", "3 4", "5 6"], dict(max_rows=2)),
    (["12345678"], dict(delimiter=3, dtype=int)),
    ([b"1 2", b"3 4"], {}),
    (["1 2"], dict(ndmin=2)),
    (["1 2", "3 4"], dict(ndmin=1)),
    (["7"], dict(ndmin=2)),
    (["1\t2"], dict(delimiter="\t")),
    (["nan inf -inf"], {}),
    (["1 2", "", "3 4"], {}),
    (["1 2"], dict(dtype="float32")),
    ([], {}),
    (["# only a comment"], {}),
]


@pytest.mark.parametrize("lines,kwargs", GEN, ids=[str(i) for i in range(len(GEN))])
def test_genfromtxt_matches_numpy(lines, kwargs):
    run = lambda m: m.genfromtxt(lines, **kwargs)
    assert same(outcome(np, run), outcome(rnp, run)), (outcome(np, run), outcome(rnp, run))


def test_genfromtxt_reads_paths_and_handles(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("1 2\n3 4\n")
    want = np.genfromtxt(path)
    assert np.array_equal(np.asarray(rnp.genfromtxt(path)), want)
    assert np.array_equal(np.asarray(rnp.genfromtxt(str(path))), want)
    assert np.array_equal(np.asarray(rnp.genfromtxt(io.StringIO("1 2\n3 4\n"))), want)


def test_genfromtxt_invalid_raise_false_warns_and_skips():
    with pytest.warns(UserWarning, match="Line #2 \\(got 3 columns instead of 2\\)"):
        got = rnp.genfromtxt(["1,2", "3,4,5", "6,7"], delimiter=",", invalid_raise=False)
    assert np.array_equal(np.asarray(got), [[1.0, 2.0], [6.0, 7.0]])


@pytest.mark.parametrize("kwargs", [dict(names=True), dict(usemask=True)])
def test_genfromtxt_unsupported_options_say_so(kwargs):
    with pytest.raises(NotImplementedError):
        rnp.genfromtxt(["a b", "1 2"], **kwargs)


Q_FORMS = {
    "py": lambda m: 0.5,
    "py_int": lambda m: 1,
    "np_float64": lambda m: np.float64(0.5),
    "np_int64": lambda m: np.int64(1),
    "zero_d": lambda m: m.asarray(0.5),
    "list": lambda m: [0.25, 0.5],
    "int_list": lambda m: [0, 1],
    "f64": lambda m: m.asarray([0.25, 0.5]),
    "f32": lambda m: m.asarray([0.25, 0.5], dtype="float32"),
    "f16": lambda m: m.asarray([0.25, 0.5], dtype="float16"),
    "i8": lambda m: m.asarray([0, 1], dtype="int8"),
}


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int8", "int64", "uint8"])
@pytest.mark.parametrize("qname", list(Q_FORMS))
@pytest.mark.parametrize("fname", ["quantile", "nanquantile", "percentile", "nanpercentile"])
def test_quantile_result_dtype_follows_q(dtype, qname, fname):
    def run(m):
        q = Q_FORMS[qname](m)
        if "percentile" in fname and not (isinstance(q, np.generic) and q.dtype.kind in "iu"):
            if isinstance(q, list):
                q = [v * 100 for v in q]
            elif isinstance(q, (int, float)) or q.dtype.kind == "f":
                q = q * 100
        return getattr(m, fname)(m.asarray([[1, 2, 3], [4, 5, 7]], dtype=dtype), q, axis=1)

    want, got = outcome(np, run), outcome(rnp, run)
    assert same(want, got), (want, got)


def test_quantile_is_float64_for_array_q_whatever_the_slice_order():
    a = rnp.asarray([[1.0, 2.0], [float("nan"), float("nan")]], dtype="float32")
    q = rnp.asarray([0.3])
    assert rnp.nanquantile(a, q, axis=1).dtype == "float64"
    assert rnp.nanquantile(a[::-1], q, axis=1).dtype == "float64"


NAN = float("nan")


@pytest.mark.parametrize(
    "value",
    [0.0, 0.1, 1.5, 64999.0, 65000.0, 65504.0, -65000.0, 1e5, 3.39e38, 3.4e38, 3.4028234663852886e38, 1e39, 1.7976931348623157e308,
     float("inf"), float("-inf"), NAN, 5e-324, 1 + 2j, 65503 + 0j, 3.4e38 + 0j, 1e39 + 0j, complex(NAN, 0), complex(0, float("inf")), 3, -129, 300, 2**63, -(2**63)],
)
def test_min_scalar_type_matches_numpy(value):
    assert str(rnp.min_scalar_type(value)) == str(np.min_scalar_type(value))


def test_min_scalar_type_of_sized_scalars_and_arrays():
    assert str(rnp.min_scalar_type(rnp.asarray([3.5], dtype="float32")[0])) == str(np.min_scalar_type(np.float32(3.5)))
    assert str(rnp.min_scalar_type(rnp.asarray([1.0]))) == "float64"


def test_int_and_index_of_zero_d_uint64_above_int64_max():
    big = 2**63 + 5
    a = rnp.asarray(big, dtype="uint64")
    assert int(a) == big and a.__index__() == big and a.item() == big


NAN_EXTREME_DATA = {"plain": [1.0, 5.0, 2.0], "nan": [1.0, NAN, 3.0], "all_nan": [NAN, NAN], "empty": [], "one": [7.0], "two_d": [[1.0, NAN], [NAN, NAN]]}


@pytest.mark.parametrize("data", list(NAN_EXTREME_DATA))
@pytest.mark.parametrize("initial", [None, NAN, 0.0, 10.0, -5.0, float("inf")])
@pytest.mark.parametrize("fname", ["nanmax", "nanmin"])
@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_nanmax_nanmin_initial(data, initial, fname, axis, dtype):
    arr = np.array(NAN_EXTREME_DATA[data], dtype=dtype)
    if axis is not None and arr.ndim <= axis:
        pytest.skip("axis out of range")
    kwargs = {} if initial is None else {"initial": initial}
    run = lambda m: getattr(m, fname)(m.asarray(arr), axis=axis, **kwargs)
    assert same(outcome(np, run), outcome(rnp, run))


@pytest.mark.parametrize("shape", [(0,), (3,), (0, 3), (3, 0), (0, 0), (2, 3), (2, 2, 2), (2, 0, 2)])
@pytest.mark.parametrize("ord_", [None, "fro", "nuc", 0, 1, -1, 2, -2, 3, 0.5, float("inf"), float("-inf")])
@pytest.mark.parametrize("axis", [None, 0, -1, (0, 1), (-2, -1), (0, 1, 2), 5, (0, 5), (0, 0)])
def test_norm_family_matches_numpy_including_empty_and_axis_errors(shape, ord_, axis):
    def make(m):
        return m.asarray(np.arange(int(np.prod(shape)), dtype="float64").reshape(shape) + 1)

    for run in (
        lambda m: m.linalg.norm(make(m), ord=ord_, axis=axis),
        lambda m: m.linalg.vector_norm(make(m), ord=2 if ord_ is None else ord_, axis=axis),
    ):
        assert same(outcome(np, run), outcome(rnp, run)), (outcome(np, run), outcome(rnp, run))
    run = lambda m: m.linalg.matrix_norm(make(m), ord="fro" if ord_ is None else ord_)
    assert same(outcome(np, run), outcome(rnp, run))


def exact_unique_outcome(m, data, dtype, equal_nan):
    a = m.asarray(np.array(data, dtype=dtype))
    r = m.unique(a, return_index=True, return_inverse=True, return_counts=True, equal_nan=equal_nan)
    return [np.asarray(x) for x in r]


UNIQUE_DATA = [
    ("float64", [-0.0, 2.0, 0.0, 2.0, float("-inf"), NAN, 1.0, float("-inf"), NAN, 0.0]),
    ("float32", [NAN, NAN]),
    ("float64", []),
    ("complex128", [1 + 1j, NAN, complex(0, NAN), complex(1, NAN)]),
    ("complex128", [complex(NAN, NAN), complex(NAN, 0), complex(0, NAN), 5, complex(NAN, 1), complex(NAN, 0)]),
    ("complex128", [complex(1, NAN), complex(0, NAN), complex(1, NAN)]),
    ("complex64", [2j, 1, 2j, complex(NAN, 0), complex(NAN, 0)]),
]


@pytest.mark.parametrize("dtype,data", UNIQUE_DATA)
@pytest.mark.parametrize("equal_nan", [True, False])
def test_unique_nan_handling_matches_numpy(dtype, data, equal_nan):
    want, got = exact_unique_outcome(np, data, dtype, equal_nan), exact_unique_outcome(rnp, data, dtype, equal_nan)
    for w, g in zip(want, got):
        assert w.dtype == g.dtype and w.shape == g.shape and np.array_equal(w, g, equal_nan=True), (want, got)


@pytest.mark.parametrize("dtype", ["uint8", "int8", "uint16", "int32", "uint64", "int64"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_searchsorted_python_int_is_exact(dtype, side):
    info = np.iinfo(dtype)
    base = [info.min, info.min + 1, info.max - 1, info.max, 0, 1] if dtype != "uint64" else [0, 1, 2**62, 2**62 + 9, 2**63, info.max]
    a = np.array(sorted(set(base)), dtype=dtype)
    for v in [info.min, info.max, int(info.max) + 1, int(info.min) - 1, 0, -1, 2**63, 2**64, 10**30, 2**62 + 25, -(2**63), 2**62]:
        want = sum((int(x) < v) if side == "left" else (int(x) <= v) for x in a)
        assert int(rnp.searchsorted(rnp.asarray(a), v, side=side)) == want, (dtype, v, side)


AXIS_ERROR_CALLS = {
    "sum": lambda m: m.sum(m.ones((2, 3)), axis=5),
    "mean": lambda m: m.mean(m.ones((2, 3)), axis=5),
    "max": lambda m: m.max(m.ones((2, 3)), axis=5),
    "argmax": lambda m: m.argmax(m.ones((2, 3)), axis=5),
    "cumsum": lambda m: m.cumsum(m.ones((2, 3)), axis=5),
    "sort": lambda m: m.sort(m.ones((2, 3)), axis=5),
    "median": lambda m: m.median(m.ones((2, 3)), axis=5),
    "diff": lambda m: m.diff(m.ones((2, 3)), axis=5),
    "flip": lambda m: m.flip(m.ones((2, 3)), axis=5),
    "squeeze": lambda m: m.squeeze(m.ones((2, 3)), axis=5),
    "partition": lambda m: m.partition(m.ones((2, 3)), 1, axis=5),
    "argpartition": lambda m: m.argpartition(m.ones((2, 3)), 1, axis=5),
    "moveaxis": lambda m: m.moveaxis(m.ones((2, 3)), 5, 0),
    "add_reduce": lambda m: m.add.reduce(m.ones((2, 3)), axis=5),
    "negative_axis": lambda m: m.sum(m.ones((2, 3)), axis=-3),
}


@pytest.mark.parametrize("name", list(AXIS_ERROR_CALLS))
def test_out_of_range_axis_raises_axis_error(name):
    fn = AXIS_ERROR_CALLS[name]
    with pytest.raises(np.exceptions.AxisError) as want:
        fn(np)
    with pytest.raises(rnp.exceptions.AxisError) as got:
        fn(rnp)
    assert str(got.value) == str(want.value)
    assert isinstance(got.value, (ValueError, IndexError))


def test_percentile_empty_axis_and_known_numpy_bugs_we_do_not_copy():
    big = rnp.zeros(65521, dtype="float16")
    big[:10] = 1
    assert float(rnp.nanquantile(big, rnp.asarray([0.5], dtype="float16"))[0]) == 0.0
    a = rnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    w = rnp.asarray([0.1, 0.2, 0.3, 0.25, 0.15])
    got = rnp.nanpercentile(a, [33.3, 66.6], weights=w, axis=0, method="inverted_cdf")
    assert np.array_equal(np.asarray(got), np.asarray(rnp.percentile(a, [33.3, 66.6], weights=w, axis=0, method="inverted_cdf")))
    assert np.array_equal(np.asarray(rnp.trim_zeros([[0, 1], [0, 0]])), [[1]])
    assert np.asarray(rnp.einsum("i...->i", rnp.ones((3, 3, 3), dtype="int64"))).tolist() == [9, 9, 9]
