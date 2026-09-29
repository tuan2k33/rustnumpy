import numpy as np
import pytest

import rustnumpy as rnp

SEEDS = [0, 1, 42, 12345, 2**32 + 5, 2**63]


@pytest.mark.parametrize("entropy", SEEDS)
def test_seed_sequence_matches_numpy(entropy):
    a, b = rnp.random.SeedSequence(entropy), np.random.SeedSequence(entropy)
    for dtype in ("uint32", "uint64"):
        np.testing.assert_array_equal(a.generate_state(6, dtype), b.generate_state(6, getattr(np, dtype)))
    ka, kb = a.spawn(3), b.spawn(3)
    for x, y in zip(ka, kb):
        assert x.spawn_key == list(y.spawn_key)
        np.testing.assert_array_equal(x.generate_state(4, "uint64"), y.generate_state(4, np.uint64))
    assert [c.spawn_key for c in a.spawn(2)] == [list(c.spawn_key) for c in b.spawn(2)]
    grand_a, grand_b = ka[1].spawn(2), kb[1].spawn(2)
    for x, y in zip(grand_a, grand_b):
        np.testing.assert_array_equal(x.generate_state(2, "uint64"), y.generate_state(2, np.uint64))


@pytest.mark.parametrize("seed", SEEDS)
def test_raw_stream_and_basic_draws_are_bit_identical(seed):
    g, n = rnp.random.default_rng(seed), np.random.default_rng(seed)
    bg = np.random.PCG64(seed)
    assert [rnp.random.default_rng(seed).random_raw()] == [np.random.PCG64(seed).random_raw()]
    np.testing.assert_array_equal(g.random(7), n.random(7))
    np.testing.assert_array_equal(g.random((2, 3)), n.random((2, 3)))
    assert g.random() == n.random()
    np.testing.assert_array_equal(g.uniform(-2.0, 5.5, (4,)), n.uniform(-2.0, 5.5, (4,)))
    np.testing.assert_array_equal(g.integers(0, 10, 9), n.integers(0, 10, 9))
    np.testing.assert_array_equal(g.integers(-5, 5, (2, 3)), n.integers(-5, 5, (2, 3)))
    np.testing.assert_array_equal(g.integers(0, 2**40, 4), n.integers(0, 2**40, 4))
    np.testing.assert_array_equal(g.integers(0, 2**32, 4), n.integers(0, 2**32, 4))
    np.testing.assert_array_equal(g.integers(7, size=5), n.integers(7, size=5))
    np.testing.assert_array_equal(g.integers(0, 4, 6, endpoint=True), n.integers(0, 4, 6, endpoint=True))
    assert g.integers(0, 100) == n.integers(0, 100)
    np.testing.assert_array_equal(g.random(3), n.random(3))


@pytest.mark.parametrize("seed", SEEDS)
def test_shuffle_permutation_choice_are_bit_identical(seed):
    g, n = rnp.random.default_rng(seed), np.random.default_rng(seed)
    x, y = rnp.arange(20), np.arange(20)
    g.shuffle(x)
    n.shuffle(y)
    np.testing.assert_array_equal(x, y)
    m1, m2 = rnp.arange(24).reshape(6, 4), np.arange(24).reshape(6, 4)
    g.shuffle(m1)
    n.shuffle(m2)
    np.testing.assert_array_equal(m1, m2)
    with pytest.raises(TypeError):
        g.shuffle(np.arange(4))
    np.testing.assert_array_equal(g.permutation(11), n.permutation(11))
    np.testing.assert_array_equal(g.permutation(np.arange(12.0).reshape(4, 3)), n.permutation(np.arange(12.0).reshape(4, 3)))
    np.testing.assert_array_equal(g.choice(10, 5), n.choice(10, 5))
    np.testing.assert_array_equal(g.choice(9, 4, replace=False), n.choice(9, 4, replace=False))
    p = [0.1, 0.2, 0.3, 0.4]
    np.testing.assert_array_equal(g.choice(4, 6, p=p), n.choice(4, 6, p=p))
    np.testing.assert_array_equal(g.choice(4, 3, replace=False, p=p), n.choice(4, 3, replace=False, p=p))
    np.testing.assert_array_equal(g.choice(20000, 5, replace=False), n.choice(20000, 5, replace=False))
    np.testing.assert_array_equal(g.choice(20000, 3000, replace=False)[:50], n.choice(20000, 3000, replace=False)[:50])
    vals = np.array([1.5, -2.0, 7.0, 0.25])
    np.testing.assert_array_equal(g.choice(vals, (2, 3)), n.choice(vals, (2, 3)))
    assert g.choice(6) == n.choice(6)


def test_choice_errors_match_numpy():
    g = rnp.random.default_rng(1)
    for call in (lambda: g.choice(3, 5, replace=False), lambda: g.choice(3, 2, p=[0.5, 0.6, 0.1]), lambda: g.choice(0, 2),
                 lambda: g.choice(3, 2, p=[-0.1, 0.6, 0.5]), lambda: g.choice(3, 3, replace=False, p=[0.5, 0.5, 0.0]),
                 lambda: g.choice(3, 2, p=[0.5, 0.5])):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("seed", [0, 42])
def test_spawned_generators_match_numpy(seed):
    g, n = rnp.random.default_rng(seed), np.random.default_rng(seed)
    for a, b in zip(g.spawn(3), n.spawn(3)):
        np.testing.assert_array_equal(a.random(4), b.random(4))
    for a, b in zip(g.spawn(2), n.spawn(2)):
        np.testing.assert_array_equal(a.random(2), b.random(2))
    gs = rnp.random.Generator.from_seed_sequence(rnp.random.SeedSequence(7))
    np.testing.assert_array_equal(gs.random(3), np.random.default_rng(np.random.SeedSequence(7)).random(3))


def test_normal_family_is_statistically_correct_not_bit_identical():
    g = rnp.random.default_rng(5)
    x = np.asarray(g.standard_normal(200_000))
    assert abs(x.mean()) < 0.01 and abs(x.std() - 1) < 0.01
    y = np.asarray(g.normal(3.0, 2.0, 200_000))
    assert abs(y.mean() - 3) < 0.03 and abs(y.std() - 2) < 0.03
    assert abs(np.asarray(g.exponential(2.0, 200_000)).mean() - 2.0) < 0.03
    assert abs(np.asarray(g.gamma(3.0, 2.0, 200_000)).mean() - 6.0) < 0.06
    assert abs(np.asarray(g.beta(2.0, 3.0, 200_000)).mean() - 0.4) < 0.01
    assert abs(np.asarray(g.binomial(20, 0.3, 100_000)).mean() - 6.0) < 0.05
    assert abs(np.asarray(g.poisson(4.5, 100_000)).mean() - 4.5) < 0.05
    d = np.asarray(g.dirichlet([1.0, 2.0, 3.0], 50_000))
    np.testing.assert_allclose(d.mean(0), [1 / 6, 2 / 6, 3 / 6], atol=0.01)
    assert g.binomial(5, 0.5, 4).dtype == np.int64 and g.poisson(2.0, 3).dtype == np.int64


@pytest.mark.parametrize("method", ["svd", "eigh", "cholesky"])
def test_multivariate_normal_moments(method):
    g = rnp.random.default_rng(3)
    cov = np.array([[2.0, 1.0], [1.0, 2.0]])
    x = np.asarray(g.multivariate_normal([1.0, -1.0], cov, 100_000, method=method))
    np.testing.assert_allclose(x.mean(0), [1, -1], atol=0.03)
    np.testing.assert_allclose(np.cov(x.T), cov, atol=0.06)
    with pytest.raises(ValueError):
        g.multivariate_normal([0, 0], [[1, 2], [2, 1]], check_valid="raise")


DISTRIBUTIONS = {
    "standard_exponential": ((), 1.0, 1.0), "standard_cauchy": None, "standard_t": ((5.0,), 0.0, 1.6667), "chisquare": ((4.0,), 4.0, 8.0),
    "f": ((6.0, 12.0), 1.2, None), "gumbel": ((1.0, 2.0), 1.0 + 0.5772156649 * 2.0, (3.14159265**2 / 6) * 4), "laplace": ((0.5, 2.0), 0.5, 8.0),
    "logistic": ((1.0, 2.0), 1.0, (3.14159265**2 / 3) * 4), "lognormal": ((0.0, 0.5), 1.1331484530668263, None), "pareto": ((5.0,), 0.25, None),
    "power": ((3.0,), 0.75, None), "rayleigh": ((2.0,), 2.5066282746310002, None), "weibull": ((2.0,), 0.8862269254527579, None),
    "wald": ((2.0, 3.0), 2.0, 2.0**3 / 3.0), "triangular": ((0.0, 1.0, 4.0), 5.0 / 3.0, None), "geometric": ((0.25,), 4.0, None),
    "negative_binomial": ((5, 0.4), 7.5, None), "zipf": ((3.5,), None, None), "logseries": ((0.6,), None, None),
    "noncentral_chisquare": ((3.0, 2.0), 5.0, None), "vonmises": ((0.5, 3.0), None, None), "standard_gamma": ((3.0,), 3.0, 3.0),
    "hypergeometric": ((10, 15, 8), 3.2, None), "noncentral_f": ((5.0, 10.0, 2.0), None, None),
}


@pytest.mark.parametrize("name", sorted(DISTRIBUTIONS))
def test_more_distributions_match_theoretical_moments(name):
    params, mean, var = DISTRIBUTIONS[name] if DISTRIBUTIONS[name] is not None else ((), None, None)
    g = rnp.random.default_rng(123)
    x = np.asarray(getattr(g, name)(*params, size=40000))
    assert x.shape == (40000,) and np.isfinite(x).all()
    ref = np.asarray(getattr(np.random.default_rng(123), name)(*params, size=40000))
    if mean is not None and name != "standard_cauchy":
        assert abs(x.mean() - mean) < 5 * ref.std() / np.sqrt(40000) + 0.02 * abs(mean)
    assert abs(np.median(x) - np.median(ref)) < 0.08 * max(1.0, np.abs(ref).std())
    if name == "standard_cauchy":
        assert abs(np.subtract(*np.percentile(x, [75, 25])) - 2.0) < 0.1
        return
    assert abs(x.std() - ref.std()) < 0.1 * max(ref.std(), 1e-9) + 0.05


def test_multinomial_permuted_bytes_integers_dtypes():
    g = rnp.random.default_rng(5)
    m = np.asarray(g.multinomial(20, [0.2, 0.3, 0.5], size=1000))
    assert (m.sum(axis=1) == 20).all() and abs(m.mean(axis=0)[2] - 10) < 0.5
    p = np.asarray(g.permuted(rnp.arange(12).reshape(3, 4), axis=1))
    assert (np.sort(p, axis=1) == np.arange(12).reshape(3, 4)).all()
    assert len(g.bytes(13)) == 13
    for dt in ("int8", "uint8", "int16", "uint32", "bool"):
        v = np.asarray(g.integers(0, 2 if dt == "bool" else 100, size=500, dtype=dt))
        assert v.dtype == np.dtype(dt)
    v = np.asarray(g.integers([0, 10], [5, 20], size=(4, 2)))
    assert (v[:, 0] < 5).all() and (v[:, 1] >= 10).all()
    h = np.asarray(g.multivariate_hypergeometric([5, 6, 4], 7, size=200))
    assert (h.sum(axis=1) == 7).all()
    assert rnp.random.default_rng() is not None and rnp.random.default_rng(rnp.random.SeedSequence(3)) is not None
