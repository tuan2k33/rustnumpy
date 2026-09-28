use rand::distr::{Distribution, Uniform};
use rand::{RngExt, SeedableRng};
use rand_distr::multi::Dirichlet;
use rand_distr::{Beta, Binomial, Exp, Gamma, Normal, Poisson, StandardNormal};
use rand_pcg::Pcg64;

use crate::ndarray::NdArray;

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

pub struct Generator {
    rng: Pcg64,
}

impl Generator {

    pub fn seed(seed: u64) -> Self {
        Self { rng: Pcg64::seed_from_u64(seed) }
    }

    fn fill<D: Distribution<f64>>(&mut self, dist: D, shape: &[usize]) -> NdArray {
        let len: usize = shape.iter().product();
        let data: Vec<f64> = (0..len).map(|_| dist.sample(&mut self.rng)).collect();
        NdArray::from_vec(data, shape).expect("data.len() == shape.iter().product() by construction")
    }

    pub fn random(&mut self, shape: &[usize]) -> NdArray {
        let len: usize = shape.iter().product();
        let data: Vec<f64> = (0..len).map(|_| self.rng.random::<f64>()).collect();
        NdArray::from_vec(data, shape).expect("data.len() == shape.iter().product() by construction")
    }

    pub fn uniform(&mut self, low: f64, high: f64, shape: &[usize]) -> Result<NdArray, RandomError> {
        let dist = Uniform::new(low, high).map_err(wrap)?;
        Ok(self.fill(dist, shape))
    }

    pub fn integers(&mut self, low: i64, high: i64, shape: &[usize]) -> Result<NdArray, RandomError> {
        let dist = Uniform::new(low, high).map_err(wrap)?;
        let len: usize = shape.iter().product();
        let data: Vec<f64> = (0..len).map(|_| dist.sample(&mut self.rng) as f64).collect();
        Ok(NdArray::from_vec(data, shape).expect("data.len() == shape.iter().product() by construction"))
    }

    pub fn standard_normal(&mut self, shape: &[usize]) -> NdArray {
        self.fill(StandardNormal, shape)
    }

    pub fn normal(&mut self, loc: f64, scale: f64, shape: &[usize]) -> Result<NdArray, RandomError> {
        if scale < 0.0 {
            return Err(RandomError("scale must be non-negative".to_string()));
        }
        let dist = Normal::new(loc, scale).map_err(wrap)?;
        Ok(self.fill(dist, shape))
    }

    pub fn exponential(&mut self, scale: f64, shape: &[usize]) -> Result<NdArray, RandomError> {
        let dist = Exp::new(1.0 / scale).map_err(wrap)?;
        Ok(self.fill(dist, shape))
    }

    pub fn gamma(&mut self, shape_param: f64, scale: f64, shape: &[usize]) -> Result<NdArray, RandomError> {
        let dist = Gamma::new(shape_param, scale).map_err(wrap)?;
        Ok(self.fill(dist, shape))
    }

    pub fn beta(&mut self, a: f64, b: f64, shape: &[usize]) -> Result<NdArray, RandomError> {
        let dist = Beta::new(a, b).map_err(wrap)?;
        Ok(self.fill(dist, shape))
    }

    pub fn binomial(&mut self, n: u64, p: f64, shape: &[usize]) -> Result<NdArray, RandomError> {
        let dist = Binomial::new(n, p).map_err(wrap)?;
        let len: usize = shape.iter().product();
        let data: Vec<f64> = (0..len).map(|_| dist.sample(&mut self.rng) as f64).collect();
        Ok(NdArray::from_vec(data, shape).expect("data.len() == shape.iter().product() by construction"))
    }

    pub fn poisson(&mut self, lam: f64, shape: &[usize]) -> Result<NdArray, RandomError> {
        let dist = Poisson::new(lam).map_err(wrap)?;
        Ok(self.fill(dist, shape))
    }

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

        let mut rng_gen = Generator::seed(4);
        let a = rng_gen.exponential(3.0, &[50_000]).unwrap();
        assert!((mean(&a.view()) - 3.0).abs() < 0.1);
    }

    #[test]
    fn gamma_mean_equals_shape_times_scale() {

        let mut rng_gen = Generator::seed(5);
        let a = rng_gen.gamma(2.0, 3.0, &[50_000]).unwrap();
        assert!((mean(&a.view()) - 6.0).abs() < 0.2);
    }

    #[test]
    fn beta_mean_matches_its_own_formula() {

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
