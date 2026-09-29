import numpy as np
import pytest

import rustnumpy as rnp
from conftest import assert_same


def spd(n, seed, dtype="float64"):
    m = np.random.default_rng(seed).standard_normal((n, n))
    return (m @ m.T + n * np.eye(n)).astype(dtype)


def rand(shape, seed, dtype="float64"):
    return np.random.default_rng(seed).standard_normal(shape).astype(dtype)


def rtol(dtype):
    return 2e-4 if dtype == "float32" else 1e-9


DT = ["float64", "float32"]


@pytest.mark.parametrize("dtype", DT)
def test_inv_det_slogdet_solve(dtype):
    a = rand((5, 5), 1, dtype) + 3 * np.eye(5, dtype=dtype)
    b1, b2 = rand((5,), 2, dtype), rand((5, 3), 3, dtype)
    assert_same(rnp.linalg.inv(a), np.linalg.inv(a), rtol(dtype), 1e-5 if dtype == "float32" else 1e-10)
    assert_same(rnp.linalg.det(a), np.linalg.det(a), rtol(dtype), 0)
    s, l = rnp.linalg.slogdet(a), np.linalg.slogdet(a)
    assert_same(s.sign, l.sign, rtol(dtype), 0)
    assert_same(s.logabsdet, l.logabsdet, rtol(dtype), 1e-6)
    assert_same(rnp.linalg.solve(a, b1), np.linalg.solve(a, b1), rtol(dtype), 1e-5)
    assert_same(rnp.linalg.solve(a, b2), np.linalg.solve(a, b2), rtol(dtype), 1e-5)


def test_integer_input_is_promoted_to_float64():
    a = np.array([[4, 3], [6, 3]])
    assert_same(rnp.linalg.inv(a), np.linalg.inv(a), 1e-12)
    assert_same(rnp.linalg.det(a), np.linalg.det(a), 1e-12)


def test_singular_and_shape_errors_are_linalg_errors():
    with pytest.raises(rnp.linalg.LinAlgError):
        rnp.linalg.inv(np.array([[1.0, 2.0], [2.0, 4.0]]))
    with pytest.raises(ValueError):
        rnp.linalg.inv(np.ones((2, 3)))
    with pytest.raises(rnp.linalg.LinAlgError):
        rnp.linalg.cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))
    np.testing.assert_allclose(np.asarray(rnp.linalg.inv(np.eye(2, dtype=complex))), np.eye(2))
    assert issubclass(rnp.linalg.LinAlgError, ValueError)


@pytest.mark.parametrize("dtype", DT)
def test_cholesky_and_eigh(dtype):
    a = spd(5, 4, dtype)
    assert_same(rnp.linalg.cholesky(a), np.linalg.cholesky(a), rtol(dtype), 1e-5 if dtype == "float32" else 1e-10)
    w, v = rnp.linalg.eigh(a)
    ew, _ = np.linalg.eigh(a)
    assert_same(w, ew, rtol(dtype), 1e-4 if dtype == "float32" else 1e-10)
    assert w.dtype == ew.dtype and v.dtype == ew.dtype
    np.testing.assert_allclose(a @ v, v * w, rtol=rtol(dtype), atol=1e-3 if dtype == "float32" else 1e-9)
    assert_same(rnp.linalg.eigvalsh(a), np.linalg.eigvalsh(a), rtol(dtype), 1e-4 if dtype == "float32" else 1e-10)


@pytest.mark.parametrize("shape", [(5, 5), (6, 3), (3, 6)])
def test_qr_reconstructs_and_is_orthogonal(shape):
    a = rand(shape, 5)
    q, r = rnp.linalg.qr(a)
    eq, er = np.linalg.qr(a)
    assert q.shape == eq.shape and r.shape == er.shape
    np.testing.assert_allclose(q @ r, a, atol=1e-12)
    np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-12)
    np.testing.assert_allclose(np.tril(r, -1), 0, atol=1e-12)
    np.testing.assert_allclose(np.abs(np.diag(r)), np.abs(np.diag(er)), rtol=1e-10)


@pytest.mark.parametrize("dtype", DT)
def test_eig_general_matrices(dtype):
    for a in (rand((5, 5), 6, dtype), np.array([[0.0, -1.0], [1.0, 0.0]], dtype=dtype), np.array([[2.0, 0.0], [0.0, 2.0]], dtype=dtype),
              spd(4, 7, dtype)):
        w, v = rnp.linalg.eig(a)
        ew, ev = np.linalg.eig(a)
        assert w.dtype == ew.dtype and v.dtype == ev.dtype
        key = lambda z: (round(float(np.real(z)), 4), round(float(np.imag(z)), 4))
        np.testing.assert_allclose(sorted(w, key=key), sorted(ew, key=key), rtol=rtol(dtype) * 10, atol=1e-4 if dtype == "float32" else 1e-9)
        np.testing.assert_allclose(a @ v, v * w, rtol=1e-3 if dtype == "float32" else 1e-8, atol=1e-4 if dtype == "float32" else 1e-8)
        np.testing.assert_allclose(np.linalg.norm(v, axis=0), 1.0, rtol=1e-5)
        wv = rnp.linalg.eigvals(a)
        np.testing.assert_allclose(sorted(wv, key=key), sorted(ew, key=key), rtol=rtol(dtype) * 10, atol=1e-4 if dtype == "float32" else 1e-9)


@pytest.mark.parametrize("dtype", DT)
@pytest.mark.parametrize("shape", [(5, 5), (6, 3), (3, 6)])
def test_svd_family(dtype, shape):
    a = rand(shape, 8, dtype)
    tol = dict(rtol=rtol(dtype), atol=1e-4 if dtype == "float32" else 1e-10)
    s = rnp.linalg.svdvals(a)
    assert_same(s, np.linalg.svdvals(a), **tol)
    assert_same(rnp.linalg.svd(a, compute_uv=False), np.linalg.svd(a, compute_uv=False), **tol)
    u, sv, vh = rnp.linalg.svd(a, full_matrices=False)
    eu, es, evh = np.linalg.svd(a, full_matrices=False)
    assert u.shape == eu.shape and vh.shape == evh.shape
    np.testing.assert_allclose((u * sv) @ vh, a, rtol=rtol(dtype) * 10, atol=1e-4 if dtype == "float32" else 1e-10)
    if shape[0] == shape[1]:
        assert rnp.linalg.svd(a)[0].shape == (5, 5)
    else:
        with pytest.raises(NotImplementedError):
            rnp.linalg.svd(a)


@pytest.mark.parametrize("shape", [(5, 5), (6, 3), (3, 6)])
def test_pinv_rank_lstsq(shape):
    a = rand(shape, 9)
    assert_same(rnp.linalg.pinv(a), np.linalg.pinv(a), 1e-8, 1e-10)
    assert_same(rnp.linalg.pinv(a, rcond=1e-3), np.linalg.pinv(a, rcond=1e-3), 1e-8, 1e-10)
    assert int(rnp.linalg.matrix_rank(a)) == int(np.linalg.matrix_rank(a))
    low = np.outer(np.arange(1.0, 5.0), np.arange(1.0, 4.0))
    assert int(rnp.linalg.matrix_rank(low)) == int(np.linalg.matrix_rank(low)) == 1
    assert int(rnp.linalg.matrix_rank(np.zeros(4))) == 0
    assert int(rnp.linalg.matrix_rank(a, tol=100.0)) == 0
    for b in (rand((shape[0],), 10), rand((shape[0], 2), 11)):
        got, want = rnp.linalg.lstsq(a, b), np.linalg.lstsq(a, b, rcond=None)
        assert_same(got[0], want[0], 1e-8, 1e-10)
        assert_same(got[1], want[1], 1e-8, 1e-10)
        assert int(got[2]) == int(want[2])
        assert_same(got[3], want[3], 1e-9, 1e-12)
    with pytest.raises(ValueError):
        rnp.linalg.lstsq(a, np.ones(shape[0] + 1))


@pytest.mark.parametrize("p", [None, 2, -2, 1, -1, np.inf, -np.inf, "fro", "nuc"])
def test_cond_and_matrix_norms(p):
    a = rand((4, 4), 12) + 2 * np.eye(4)
    if p != "nuc":
        assert_same(rnp.linalg.cond(a, p), np.linalg.cond(a, p), 1e-9, 0)
    assert_same(rnp.linalg.norm(a, p), np.linalg.norm(a, p), 1e-9, 0) if p not in (None,) else assert_same(rnp.linalg.norm(a), np.linalg.norm(a), 1e-9, 0)
    assert np.isinf(rnp.linalg.cond(np.array([[1.0, 2.0], [2.0, 4.0]]), "fro"))


@pytest.mark.parametrize("p", [None, 1, 2, np.inf])
def test_vector_norms(p):
    v = rand((7,), 13)
    assert_same(rnp.linalg.norm(v, p), np.linalg.norm(v, p), 1e-12, 0)
    assert_same(rnp.linalg.norm(v, 3), np.linalg.norm(v, 3), 1e-12, 0)


@pytest.mark.parametrize("n", [0, 1, 2, 5, -1, -3])
def test_matrix_power(n):
    a = rand((3, 3), 14) + 3 * np.eye(3)
    assert_same(rnp.linalg.matrix_power(a, n), np.linalg.matrix_power(a, n), 1e-8, 1e-10)


def _stacked(seed, shape=(3, 4, 4)):
    return np.random.default_rng(seed).standard_normal(shape) + 4 * np.eye(shape[-1])


@pytest.mark.parametrize("name", ["inv", "det", "cholesky", "eigvals", "eigvalsh", "svdvals", "pinv", "matrix_rank"])
def test_stacked_matrices(name):
    a = _stacked(1)
    if name in ("cholesky", "eigvalsh"):
        a = a @ np.swapaxes(a, -1, -2)
    got, want = getattr(rnp.linalg, name)(rnp.array(a)), getattr(np.linalg, name)(a)
    if name == "eigvals":
        got, want = np.sort_complex(np.asarray(got).reshape(-1)), np.sort_complex(want.reshape(-1))
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-8, atol=1e-10)


def test_stacked_solve_qr_eigh_svd_slogdet_matrix_power():
    a = _stacked(2)
    b = np.random.default_rng(3).standard_normal((3, 4))
    np.testing.assert_allclose(np.asarray(rnp.linalg.solve(rnp.array(a), rnp.array(b))), np.linalg.solve(a, b[..., None])[..., 0], rtol=1e-8)
    s, l = rnp.linalg.slogdet(rnp.array(a)), np.linalg.slogdet(a)
    np.testing.assert_allclose(np.asarray(s.sign), l.sign)
    np.testing.assert_allclose(np.asarray(s.logabsdet), l.logabsdet)
    np.testing.assert_allclose(np.asarray(rnp.linalg.matrix_power(rnp.array(a), 3)), np.linalg.matrix_power(a, 3), rtol=1e-8)
    q, r = rnp.linalg.qr(rnp.array(a))
    np.testing.assert_allclose(np.asarray(q) @ np.asarray(r), a, atol=1e-10)
    sym = a + np.swapaxes(a, -1, -2)
    w, v = rnp.linalg.eigh(rnp.array(sym))
    np.testing.assert_allclose(np.asarray(v) * np.asarray(w)[..., None, :] @ np.swapaxes(np.asarray(v), -1, -2), sym, atol=1e-9)
    u, sv, vh = rnp.linalg.svd(rnp.array(a))
    np.testing.assert_allclose(np.asarray(u) * np.asarray(sv)[..., None, :] @ np.asarray(vh), a, atol=1e-9)


@pytest.mark.parametrize("ord_", [None, "fro", "nuc", 1, -1, 2, -2, np.inf, -np.inf])
def test_matrix_norm_orders(ord_):
    a = _stacked(4, (4, 4))
    np.testing.assert_allclose(np.asarray(rnp.linalg.norm(rnp.array(a), ord=ord_)), np.linalg.norm(a, ord=ord_), rtol=1e-9)
    np.testing.assert_allclose(np.asarray(rnp.linalg.matrix_norm(rnp.array(a), ord="fro" if ord_ is None else ord_)), np.linalg.matrix_norm(a, ord="fro" if ord_ is None else ord_), rtol=1e-9)


@pytest.mark.parametrize("ord_", [None, 0, 1, 2, 3, np.inf, -np.inf])
def test_vector_norm_and_axis(ord_):
    a = np.random.default_rng(5).standard_normal((3, 4))
    for axis in (None, 0, 1):
        got = rnp.linalg.vector_norm(rnp.array(a), axis=axis, ord=2 if ord_ is None else ord_)
        np.testing.assert_allclose(np.asarray(got), np.linalg.vector_norm(a, axis=axis, ord=2 if ord_ is None else ord_), rtol=1e-9)
    np.testing.assert_allclose(np.asarray(rnp.linalg.norm(rnp.array(a), ord=ord_, axis=1, keepdims=True)), np.linalg.norm(a, ord=ord_, axis=1, keepdims=True), rtol=1e-9)


def test_multi_dot_tensorsolve_tensorinv_cross():
    r = np.random.default_rng(6)
    ms = [r.standard_normal(s) for s in ((3, 4), (4, 5), (5, 2))]
    np.testing.assert_allclose(np.asarray(rnp.linalg.multi_dot([rnp.array(m) for m in ms])), np.linalg.multi_dot(ms), rtol=1e-9)
    a = np.eye(6).reshape(2, 3, 6) * 2.0
    b = r.standard_normal((2, 3))
    np.testing.assert_allclose(np.asarray(rnp.linalg.tensorsolve(rnp.array(a), rnp.array(b))), np.linalg.tensorsolve(a, b), rtol=1e-9)
    t = np.eye(4).reshape(2, 2, 2, 2) + 0.1 * r.standard_normal((2, 2, 2, 2))
    np.testing.assert_allclose(np.asarray(rnp.linalg.tensorinv(rnp.array(t))), np.linalg.tensorinv(t), rtol=1e-8)
    np.testing.assert_allclose(np.asarray(rnp.linalg.cross(rnp.array([1.0, 2, 3]), rnp.array([4.0, 5, 6]))), np.linalg.cross([1.0, 2, 3], [4.0, 5, 6]))
    with pytest.raises(rnp.linalg.LinAlgError):
        rnp.linalg.inv(rnp.array([[1.0, 2], [2, 4]]))


def _cplx(seed, shape, dtype="complex128"):
    r = np.random.default_rng(seed)
    return (r.standard_normal(shape) + 1j * r.standard_normal(shape)).astype(dtype)


@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
@pytest.mark.parametrize("shape", [(4, 4), (3, 4, 4)])
def test_complex_inverse_det_solve_and_friends(dtype, shape):
    a = _cplx(1, shape, dtype) + 3 * np.eye(4, dtype=dtype)
    tol = 1e-3 if dtype == "complex64" else 1e-9
    ra = rnp.array(a)
    for name in ("inv", "det", "pinv", "matrix_rank", "svdvals", "eigvals"):
        got, want = np.asarray(getattr(rnp.linalg, name)(ra)), getattr(np.linalg, name)(a)
        if name == "eigvals":
            got, want = np.sort_complex(got.reshape(-1)), np.sort_complex(want.reshape(-1))
        assert got.dtype == want.dtype, name
        np.testing.assert_allclose(got, want, rtol=tol, atol=tol)
    s, l = rnp.linalg.slogdet(ra), np.linalg.slogdet(a)
    np.testing.assert_allclose(np.asarray(s.sign), l.sign, rtol=tol, atol=tol)
    np.testing.assert_allclose(np.asarray(s.logabsdet), l.logabsdet, rtol=tol, atol=tol)
    b = _cplx(2, shape[:-1], dtype)
    np.testing.assert_allclose(np.asarray(rnp.linalg.solve(ra, rnp.array(b))), np.linalg.solve(a, b[..., None])[..., 0], rtol=tol, atol=tol)
    np.testing.assert_allclose(np.asarray(rnp.linalg.matrix_power(ra, 3)), np.linalg.matrix_power(a, 3), rtol=tol, atol=tol)
    np.testing.assert_allclose(np.asarray(rnp.linalg.matrix_power(ra, -2)), np.linalg.matrix_power(a, -2), rtol=tol, atol=tol)
    q, r = rnp.linalg.qr(ra)
    np.testing.assert_allclose(np.asarray(q) @ np.asarray(r), a, rtol=tol, atol=tol)
    herm = a @ np.conj(np.swapaxes(a, -1, -2))
    w, v = rnp.linalg.eigh(rnp.array(herm))
    np.testing.assert_allclose(np.asarray(w), np.linalg.eigvalsh(herm), rtol=tol, atol=tol * 10)
    np.testing.assert_allclose((np.asarray(v) * np.asarray(w)[..., None, :]) @ np.conj(np.swapaxes(np.asarray(v), -1, -2)), herm, rtol=tol, atol=tol * 10)
    L = np.asarray(rnp.linalg.cholesky(rnp.array(herm)))
    np.testing.assert_allclose(L @ np.conj(np.swapaxes(L, -1, -2)), herm, rtol=tol, atol=tol * 10)
    u, sv, vh = rnp.linalg.svd(ra, full_matrices=False)
    np.testing.assert_allclose((np.asarray(u) * np.asarray(sv)[..., None, :]) @ np.asarray(vh), a, rtol=tol, atol=tol)
    w2, v2 = rnp.linalg.eig(ra)
    np.testing.assert_allclose(a @ np.asarray(v2), np.asarray(v2) * np.asarray(w2)[..., None, :], rtol=tol * 10, atol=tol * 10)


def test_complex_lstsq_cond_norm_and_singular():
    a = _cplx(3, (6, 3))
    b = _cplx(4, (6,))
    x, res, rank, sv = rnp.linalg.lstsq(rnp.array(a), rnp.array(b))
    wx, wres, wrank, wsv = np.linalg.lstsq(a, b)
    np.testing.assert_allclose(np.asarray(x), wx, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(res), wres, rtol=1e-9)
    assert int(rank) == wrank
    sq = _cplx(5, (4, 4))
    for p in (None, 2, "fro", 1, np.inf):
        np.testing.assert_allclose(np.asarray(rnp.linalg.cond(rnp.array(sq), p)), np.linalg.cond(sq, p), rtol=1e-8)
    for ord_ in (None, "fro", "nuc", 1, np.inf, 2):
        np.testing.assert_allclose(np.asarray(rnp.linalg.norm(rnp.array(sq), ord=ord_)), np.linalg.norm(sq, ord=ord_), rtol=1e-9)
    with pytest.raises(rnp.linalg.LinAlgError):
        rnp.linalg.inv(rnp.array(np.array([[1j, 2j], [2j, 4j]])))
    assert complex(rnp.linalg.det(rnp.array(np.array([[1j, 2j], [2j, 4j]])))) == 0
