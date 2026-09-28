//! `numpy.random`'s modern (NEP 19) `Generator` API — binomial, poisson,
//! gamma, beta, dirichlet, exponential, plus the basic uniform/normal
//! every distribution here is built on top of.
//!
//! Maps NEP 19's BitGenerator/Generator split directly onto the `rand`
//! ecosystem, per `NumPy.md`'s own note: `rand_pcg::Pcg64` is the
//! BitGenerator (the raw bit stream, matching NumPy's own new default
//! algorithm), and each `rand_distr` type is a Generator's distribution
//! logic (`Distribution<T>` where NumPy has e.g. `Generator.gamma`).
//! `RandomState`/the old `np.random.seed()` API is deliberately not
//! ported — NumPy itself has permanently frozen it for backward
//! compatibility only (see `NumPy.md`'s "Parts Worth Dropping" section).
//!
//! **Known, accepted deviation from NumPy**: this `Generator` is only
//! statistically equivalent to NumPy's, not bit-for-bit stream-compatible
//! -- given the same seed, it will *not* produce the same sequence NumPy's
//! `Generator(PCG64(seed))` would (matching that exactly would mean
//! reimplementing NumPy's specific seeding/squashing scheme and its exact
//! per-distribution sampling algorithms, which `NumPy.md`'s own RNG
//! Architecture note treats as optional: "you just need to ensure PCG64
//! produces results identical to NumPy's *if stream compatibility is
//! required*" -- it isn't, here). What's verified against real NumPy
//! instead: each distribution's documented parameterization (e.g.
//! `exponential(scale)`'s mean is `scale`, not a rate) and, statistically,
//! that a large sample's mean/variance land where the distribution's own
//! formula says they should -- using this project's own [`crate::reductions`]
//! module, so no running NumPy is needed even for that check.
//!
//! All distributions here return an [`NdArray`] of the requested `shape`
//! -- `NdArray` is `f64`-only (see `lib.rs`'s doc comment), so integer-
//! valued distributions ([`Generator::binomial`], [`Generator::poisson`])
//! come back as whole-number `f64`s rather than NumPy's `int64` dtype.

use rand::distr::{Distribution, Uniform};
use rand::{RngExt, SeedableRng};
use rand_distr::multi::Dirichlet;
use rand_distr::{Beta, Binomial, Exp, Gamma, Normal, Poisson, StandardNormal};
use rand_pcg::Pcg64;

use crate::ndarray::NdArray;

/// Any error from constructing a distribution with invalid parameters
/// (e.g. a negative `scale`). Real NumPy raises `ValueError`; each
/// `rand_distr` constructor's own error is wrapped here as a message
/// rather than exposing `rand_distr`'s types directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RandomError(pub String);

impl std::fmt::Display for RandomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for RandomError {}

fn wrap<E: std::fmt::Display>(e: E) -> RandomError {
    RandomError(e.to_string())
}

/// `numpy.random.Generator` (NEP 19's modern, non-legacy RNG). Wraps a
/// `Pcg64` bit generator -- construct with [`Generator::seed`] for a
/// reproducible sequence (this project's own tests always do this; see
/// this module's doc comment for why that sequence won't match NumPy's).
pub struct Generator {
    rng: Pcg64,
}

impl Generator {
    /// A reproducible generator: the same seed always produces the same
    /// sequence of draws (from this generator itself -- see this module's
    /// doc comment on stream compatibility with NumPy).
    pub fn seed(seed: u64) -> Self {
        Self { rng: Pcg64::seed_from_u64(seed) }
    }

    fn fill<D: Distribution<f64>>(&mut self, dist: D, shape: &[usize]) -> NdArray {
        let len: usize = shape.iter().product();
        let data: Vec<f64> = (0..len).map(|_| dist.sample(&mut self.rng)).collect();
        NdArray::from_vec(data, shape).expect("data.len() == shape.iter().product() by construction")
    }

    /// `Generator.random(size)`: uniform floats in the half-open interval `[0, 1)`.
    pub fn random(&mut self, shape: &[usize]) -> NdArray {
        let len: usize = shape.iter().product();
        let data: Vec<f64> = (0..len).map(|_| self.rng.random::<f64>()).collect();
        NdArray::from_vec(data, shape).expect("data.len() == shape.iter().product() by construction")
    }

    /// `Generator.uniform(low, high, size)`.
    pub fn uniform(&mut self, low: f64, high: f64, shape: &[usize]) -> Result<NdArray, RandomError> {
        let dist = Uniform::new(low, high).map_err(wrap)?;
        Ok(self.fill(dist, shape))
    }

    /// `Generator.integers(low, high, size)` (NumPy's default
    /// `endpoint=False`: `high` is exclusive). Returned as whole-number
    /// `f64`s -- see this module's doc comment.
    pub fn integers(&mut self, low: i64, high: i64, shape: &[usize]) -> Result<NdArray, RandomError> {
        let dist = Uniform::new(low, high).map_err(wrap)?;
        let len: usize = shape.iter().product();
        let data: Vec<f64> = (0..len).map(|_| dist.sample(&mut self.rng) as f64).collect();
        Ok(NdArray::from_vec(data, shape).expect("data.len() == shape.iter().product() by construction"))
    }

    /// `Generator.standard_normal(size)`: mean 0, standard deviation 1.
    pub fn standard_normal(&mut self, shape: &[usize]) -> NdArray {
        self.fill(StandardNormal, shape)
    }

    /// `Generator.normal(loc, scale, size)`. NumPy requires `scale >= 0`
    /// (a `ValueError` otherwise); `rand_distr::Normal` itself accepts a
    /// negative `std_dev` (it just flips the distribution's sign, which
    /// has no meaningful effect since it's symmetric) -- checked here
    /// explicitly so this matches NumPy's stricter contract instead of
    /// silently accepting what NumPy would reject.
    pub fn normal(&mut self, loc: f64, scale: f64, shape: &[usize]) -> Result<NdArray, RandomError> {
        if scale < 0.0 {
            return Err(RandomError("scale must be non-negative".to_string()));
        }
        let dist = Normal::new(loc, scale).map_err(wrap)?;
        Ok(self.fill(dist, shape))
    }

    /// `Generator.exponential(scale, size)` -- NumPy parameterizes by
    /// `scale` (the mean); `rand_distr::Exp` parameterizes by the rate
    /// `lambda = 1 / scale`.
    pub fn exponential(&mut self, scale: f64, shape: &[usize]) -> Result<NdArray, RandomError> {
        let dist = Exp::new(1.0 / scale).map_err(wrap)?;
        Ok(self.fill(dist, shape))
    }

    /// `Generator.gamma(shape, scale, size)` (`shape` here is the
    /// distribution's shape parameter `k`, unrelated to the array `shape`
    /// argument -- named to match NumPy's own parameter name).
    pub fn gamma(&mut self, shape_param: f64, scale: f64, shape: &[usize]) -> Result<NdArray, RandomError> {
        let dist = Gamma::new(shape_param, scale).map_err(wrap)?;
        Ok(self.fill(dist, shape))
    }

    /// `Generator.beta(a, b, size)`.
    pub fn beta(&mut self, a: f64, b: f64, shape: &[usize]) -> Result<NdArray, RandomError> {
        let dist = Beta::new(a, b).map_err(wrap)?;
        Ok(self.fill(dist, shape))
    }

    /// `Generator.binomial(n, p, size)`. Returned as whole-number `f64`s
    /// -- see this module's doc comment.
    pub fn binomial(&mut self, n: u64, p: f64, shape: &[usize]) -> Result<NdArray, RandomError> {
        let dist = Binomial::new(n, p).map_err(wrap)?;
        let len: usize = shape.iter().product();
        let data: Vec<f64> = (0..len).map(|_| dist.sample(&mut self.rng) as f64).collect();
        Ok(NdArray::from_vec(data, shape).expect("data.len() == shape.iter().product() by construction"))
    }

    /// `Generator.poisson(lam, size)`. Returned as whole-number `f64`s --
    /// see this module's doc comment.
    pub fn poisson(&mut self, lam: f64, shape: &[usize]) -> Result<NdArray, RandomError> {
        let dist = Poisson::new(lam).map_err(wrap)?;
        Ok(self.fill(dist, shape))
    }

    /// `Generator.dirichlet(alpha, size)`: `size` independent draws from
    /// `Dirichlet(alpha)`, returned as a 2-D `NdArray` of shape
    /// `(size, alpha.len())` (matching NumPy's own shape for `size` as a
    /// plain integer).
    pub fn dirichlet(&mut self, alpha: &[f64], size: usize) -> Result<NdArray, RandomError> {
        let dist = Dirichlet::new(alpha).map_err(wrap)?;
        let mut data = Vec::with_capacity(size * alpha.len());
        for _ in 0..size {
            data.extend(dist.sample(&mut self.rng));
        }
        Ok(NdArray::from_vec(data, &[size, alpha.len()])
            .expect("data.len() == size * alpha.len() by construction"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reductions::{mean, var_default};

    #[test]
    fn same_seed_reproduces_the_same_sequence() {
        let mut a = Generator::seed(42);
        let mut b = Generator::seed(42);
        assert_eq!(a.random(&[10]).as_slice(), b.random(&[10]).as_slice());
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Generator::seed(1);
        let mut b = Generator::seed(2);
        assert_ne!(a.random(&[10]).as_slice(), b.random(&[10]).as_slice());
    }

    #[test]
    fn random_is_bounded_to_zero_one() {
        let mut rng_gen = Generator::seed(0);
        let a = rng_gen.random(&[1000]);
        assert!(a.as_slice().iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn uniform_matches_its_own_mean_formula() {
        // E[Uniform(low, high)] == (low + high) / 2.
        let mut rng_gen = Generator::seed(1);
        let a = rng_gen.uniform(2.0, 10.0, &[20_000]).unwrap();
        assert!((mean(&a.view()) - 6.0).abs() < 0.1);
    }

    #[test]
    fn standard_normal_matches_mean_zero_var_one() {
        let mut rng_gen = Generator::seed(2);
        let a = rng_gen.standard_normal(&[50_000]);
        assert!(mean(&a.view()).abs() < 0.05);
        assert!((var_default(&a.view()) - 1.0).abs() < 0.1);
    }

    #[test]
    fn normal_matches_its_own_mean_and_variance_formula() {
        let mut rng_gen = Generator::seed(3);
        let a = rng_gen.normal(5.0, 2.0, &[50_000]).unwrap();
        assert!((mean(&a.view()) - 5.0).abs() < 0.1);
        assert!((var_default(&a.view()) - 4.0).abs() < 0.3);
    }

    #[test]
    fn exponential_mean_equals_scale() {
        // NumPy parameterizes by scale (the mean), not rate.
        let mut rng_gen = Generator::seed(4);
        let a = rng_gen.exponential(3.0, &[50_000]).unwrap();
        assert!((mean(&a.view()) - 3.0).abs() < 0.1);
    }

    #[test]
    fn gamma_mean_equals_shape_times_scale() {
        // E[Gamma(k, theta)] == k * theta.
        let mut rng_gen = Generator::seed(5);
        let a = rng_gen.gamma(2.0, 3.0, &[50_000]).unwrap();
        assert!((mean(&a.view()) - 6.0).abs() < 0.2);
    }

    #[test]
    fn beta_mean_matches_its_own_formula() {
        // E[Beta(a, b)] == a / (a + b).
        let mut rng_gen = Generator::seed(6);
        let a = rng_gen.beta(2.0, 3.0, &[50_000]).unwrap();
        assert!((mean(&a.view()) - 0.4).abs() < 0.02);
    }

    #[test]
    fn binomial_mean_equals_n_times_p() {
        let mut rng_gen = Generator::seed(7);
        let a = rng_gen.binomial(20, 0.3, &[50_000]).unwrap();
        assert!((mean(&a.view()) - 6.0).abs() < 0.1);
        assert!(a.as_slice().iter().all(|&x| x.fract() == 0.0));
    }

    #[test]
    fn poisson_mean_equals_lambda() {
        let mut rng_gen = Generator::seed(8);
        let a = rng_gen.poisson(4.0, &[50_000]).unwrap();
        assert!((mean(&a.view()) - 4.0).abs() < 0.1);
        assert!(a.as_slice().iter().all(|&x| x.fract() == 0.0));
    }

    #[test]
    fn dirichlet_rows_sum_to_one() {
        let mut rng_gen = Generator::seed(9);
        let a = rng_gen.dirichlet(&[1.0, 2.0, 3.0], 100).unwrap();
        assert_eq!(a.shape(), &[100, 3]);
        for row in a.as_slice().chunks(3) {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn integers_are_within_half_open_range() {
        let mut rng_gen = Generator::seed(10);
        let a = rng_gen.integers(0, 5, &[1000]).unwrap();
        assert!(a.as_slice().iter().all(|&x| (0.0..5.0).contains(&x) && x.fract() == 0.0));
    }

    #[test]
    fn invalid_parameters_err_instead_of_panicking() {
        let mut rng_gen = Generator::seed(0);
        assert!(rng_gen.normal(0.0, -1.0, &[1]).is_err());
        assert!(rng_gen.beta(-1.0, 1.0, &[1]).is_err());
    }
}
