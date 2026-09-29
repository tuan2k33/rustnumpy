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
    assert_same(rnp.inv(a), np.linalg.inv(a), rtol(dtype), 1e-5 if dtype == "float32" else 1e-10)
    assert_same(rnp.det(a), np.linalg.det(a), rtol(dtype), 0)
    s, l = rnp.slogdet(a), np.linalg.slogdet(a)
    assert_same(s.sign, l.sign, rtol(dtype), 0)
    assert_same(s.logabsdet, l.logabsdet, rtol(dtype), 1e-6)
    assert_same(rnp.solve(a, b1), np.linalg.solve(a, b1), rtol(dtype), 1e-5)
    assert_same(rnp.solve(a, b2), np.linalg.solve(a, b2), rtol(dtype), 1e-5)


def test_integer_input_is_promoted_to_float64():
    a = np.array([[4, 3], [6, 3]])
    assert_same(rnp.inv(a), np.linalg.inv(a), 1e-12)
    assert_same(rnp.det(a), np.linalg.det(a), 1e-12)


def test_singular_and_shape_errors_are_linalg_errors():
    with pytest.raises(rnp.LinAlgError):
        rnp.inv(np.array([[1.0, 2.0], [2.0, 4.0]]))
    with pytest.raises(ValueError):
        rnp.inv(np.ones((2, 3)))
    with pytest.raises(rnp.LinAlgError):
        rnp.cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(NotImplementedError):
        rnp.inv(np.eye(2, dtype=complex))
    assert issubclass(rnp.LinAlgError, ValueError)


@pytest.mark.parametrize("dtype", DT)
def test_cholesky_and_eigh(dtype):
    a = spd(5, 4, dtype)
    assert_same(rnp.cholesky(a), np.linalg.cholesky(a), rtol(dtype), 1e-5 if dtype == "float32" else 1e-10)
    w, v = rnp.eigh(a)
    ew, _ = np.linalg.eigh(a)
    assert_same(w, ew, rtol(dtype), 1e-4 if dtype == "float32" else 1e-10)
    assert w.dtype == ew.dtype and v.dtype == ew.dtype
    np.testing.assert_allclose(a @ v, v * w, rtol=rtol(dtype), atol=1e-3 if dtype == "float32" else 1e-9)
    assert_same(rnp.eigvalsh(a), np.linalg.eigvalsh(a), rtol(dtype), 1e-4 if dtype == "float32" else 1e-10)


@pytest.mark.parametrize("shape", [(5, 5), (6, 3), (3, 6)])
def test_qr_reconstructs_and_is_orthogonal(shape):
    a = rand(shape, 5)
    q, r = rnp.qr(a)
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
        w, v = rnp.eig(a)
        ew, ev = np.linalg.eig(a)
        assert w.dtype == ew.dtype and v.dtype == ev.dtype
        key = lambda z: (round(float(np.real(z)), 4), round(float(np.imag(z)), 4))
        np.testing.assert_allclose(sorted(w, key=key), sorted(ew, key=key), rtol=rtol(dtype) * 10, atol=1e-4 if dtype == "float32" else 1e-9)
        np.testing.assert_allclose(a @ v, v * w, rtol=1e-3 if dtype == "float32" else 1e-8, atol=1e-4 if dtype == "float32" else 1e-8)
        np.testing.assert_allclose(np.linalg.norm(v, axis=0), 1.0, rtol=1e-5)
        wv = rnp.eigvals(a)
        np.testing.assert_allclose(sorted(wv, key=key), sorted(ew, key=key), rtol=rtol(dtype) * 10, atol=1e-4 if dtype == "float32" else 1e-9)


@pytest.mark.parametrize("dtype", DT)
@pytest.mark.parametrize("shape", [(5, 5), (6, 3), (3, 6)])
def test_svd_family(dtype, shape):
    a = rand(shape, 8, dtype)
    tol = dict(rtol=rtol(dtype), atol=1e-4 if dtype == "float32" else 1e-10)
    s = rnp.svdvals(a)
    assert_same(s, np.linalg.svdvals(a), **tol)
    assert_same(rnp.svd(a, compute_uv=False), np.linalg.svd(a, compute_uv=False), **tol)
    u, sv, vh = rnp.svd(a, full_matrices=False)
    eu, es, evh = np.linalg.svd(a, full_matrices=False)
    assert u.shape == eu.shape and vh.shape == evh.shape
    np.testing.assert_allclose((u * sv) @ vh, a, rtol=rtol(dtype) * 10, atol=1e-4 if dtype == "float32" else 1e-10)
    if shape[0] == shape[1]:
        assert rnp.svd(a)[0].shape == (5, 5)
    else:
        with pytest.raises(NotImplementedError):
            rnp.svd(a)


@pytest.mark.parametrize("shape", [(5, 5), (6, 3), (3, 6)])
def test_pinv_rank_lstsq(shape):
    a = rand(shape, 9)
    assert_same(rnp.pinv(a), np.linalg.pinv(a), 1e-8, 1e-10)
    assert_same(rnp.pinv(a, rcond=1e-3), np.linalg.pinv(a, rcond=1e-3), 1e-8, 1e-10)
    assert int(rnp.matrix_rank(a)) == int(np.linalg.matrix_rank(a))
    low = np.outer(np.arange(1.0, 5.0), np.arange(1.0, 4.0))
    assert int(rnp.matrix_rank(low)) == int(np.linalg.matrix_rank(low)) == 1
    assert int(rnp.matrix_rank(np.zeros(4))) == 0
    assert int(rnp.matrix_rank(a, tol=100.0)) == 0
    for b in (rand((shape[0],), 10), rand((shape[0], 2), 11)):
        got, want = rnp.lstsq(a, b), np.linalg.lstsq(a, b, rcond=None)
        assert_same(got[0], want[0], 1e-8, 1e-10)
        assert_same(got[1], want[1], 1e-8, 1e-10)
        assert int(got[2]) == int(want[2])
        assert_same(got[3], want[3], 1e-9, 1e-12)
    with pytest.raises(ValueError):
        rnp.lstsq(a, np.ones(shape[0] + 1))


@pytest.mark.parametrize("p", [None, 2, -2, 1, -1, np.inf, -np.inf, "fro", "nuc"])
def test_cond_and_matrix_norms(p):
    a = rand((4, 4), 12) + 2 * np.eye(4)
    if p != "nuc":
        assert_same(rnp.cond(a, p), np.linalg.cond(a, p), 1e-9, 0)
    assert_same(rnp.norm(a, p), np.linalg.norm(a, p), 1e-9, 0) if p not in (None,) else assert_same(rnp.norm(a), np.linalg.norm(a), 1e-9, 0)
    assert np.isinf(rnp.cond(np.array([[1.0, 2.0], [2.0, 4.0]]), "fro"))


@pytest.mark.parametrize("p", [None, 1, 2, np.inf])
def test_vector_norms(p):
    v = rand((7,), 13)
    assert_same(rnp.norm(v, p), np.linalg.norm(v, p), 1e-12, 0)
    with pytest.raises(NotImplementedError):
        rnp.norm(v, 3)


@pytest.mark.parametrize("n", [0, 1, 2, 5, -1, -3])
def test_matrix_power(n):
    a = rand((3, 3), 14) + 3 * np.eye(3)
    assert_same(rnp.matrix_power(a, n), np.linalg.matrix_power(a, n), 1e-8, 1e-10)
