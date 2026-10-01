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
