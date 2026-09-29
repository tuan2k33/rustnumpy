use rand::distr::Distribution;
use rand::Rng;
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

const INIT_A: u32 = 0x43b0d7e5;
const MULT_A: u32 = 0x931e8875;
const INIT_B: u32 = 0x8b51f9dd;
const MULT_B: u32 = 0x58f38ded;
const MIX_MULT_L: u32 = 0xca01f9dd;
const MIX_MULT_R: u32 = 0x4973f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^ (v >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> XSHIFT)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedSequence {
    entropy: Vec<u32>,
    spawn_key: Vec<u32>,
    pool: [u32; POOL_SIZE],
    n_children_spawned: u32,
}

impl SeedSequence {
    pub fn new(entropy: u64) -> Self {
        let mut words = vec![entropy as u32];
        if entropy >> 32 != 0 {
            words.push((entropy >> 32) as u32);
        }
        Self::from_words(&words, &[])
    }

    pub fn from_words(entropy: &[u32], spawn_key: &[u32]) -> Self {
        let mut run: Vec<u32> = entropy.to_vec();
        if !spawn_key.is_empty() && run.len() < POOL_SIZE {
            run.resize(POOL_SIZE, 0);
        }
        let entropy_array: Vec<u32> = run.iter().chain(spawn_key).copied().collect();
        let mut pool = [0u32; POOL_SIZE];
        let mut hash_const = INIT_A;
        for (i, slot) in pool.iter_mut().enumerate() {
            *slot = hashmix(entropy_array.get(i).copied().unwrap_or(0), &mut hash_const);
        }
        for i_src in 0..POOL_SIZE {
            for i_dst in 0..POOL_SIZE {
                if i_src != i_dst {
                    let h = hashmix(pool[i_src], &mut hash_const);
                    pool[i_dst] = mix(pool[i_dst], h);
                }
            }
        }
        for &word in entropy_array.iter().skip(POOL_SIZE) {
            for slot in pool.iter_mut() {
                let h = hashmix(word, &mut hash_const);
                *slot = mix(*slot, h);
            }
        }
        Self { entropy: entropy.to_vec(), spawn_key: spawn_key.to_vec(), pool, n_children_spawned: 0 }
    }

    pub fn spawn_key(&self) -> &[u32] {
        &self.spawn_key
    }

    pub fn generate_state_u32(&self, n_words: usize) -> Vec<u32> {
        let mut hash_const = INIT_B;
        (0..n_words)
            .map(|i| {
                let mut v = self.pool[i % POOL_SIZE] ^ hash_const;
                hash_const = hash_const.wrapping_mul(MULT_B);
                v = v.wrapping_mul(hash_const);
                v ^ (v >> XSHIFT)
            })
            .collect()
    }

    pub fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        self.generate_state_u32(2 * n_words)
            .chunks(2)
            .map(|w| u64::from(w[0]) | (u64::from(w[1]) << 32))
            .collect()
    }

    pub fn spawn(&mut self, n: usize) -> Vec<SeedSequence> {
        let children = (0..n)
            .map(|i| {
                let mut key = self.spawn_key.clone();
                key.push(self.n_children_spawned + i as u32);
                SeedSequence::from_words(&self.entropy, &key)
            })
            .collect();
        self.n_children_spawned += n as u32;
        children
    }
}

fn normalized_cdf(p: &[f64]) -> Vec<f64> {
    let mut acc = 0.0;
    let mut cdf: Vec<f64> = p
        .iter()
        .map(|&x| {
            acc += x;
            acc
        })
        .collect();
    let total = *cdf.last().unwrap_or(&1.0);
    cdf.iter_mut().for_each(|c| *c /= total);
    cdf
}

fn search_right(cdf: &[f64], x: f64) -> usize {
    cdf.partition_point(|&c| c <= x)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MvnMethod {
    Svd,
    Eigh,
    Cholesky,
}

pub fn mvn_factor(cov: &NdArray, method: MvnMethod, check_valid: bool) -> Result<NdArray, RandomError> {
    use crate::linalg::{cholesky, eigh, svd};
    let n = cov.shape()[0];
    let (s, vh) = match method {
        MvnMethod::Cholesky => {
            let l = cholesky(cov).map_err(wrap)?;
            let mut lt = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    lt[j * n + i] = l.as_slice()[i * n + j];
                }
            }
            return Ok(NdArray::from_vec(lt, &[n, n]).expect("n x n"));
        }
        MvnMethod::Svd => {
            let (_, s, vt) = svd(cov).map_err(wrap)?;
            (s, vt)
        }
        MvnMethod::Eigh => {
            let (w, u) = eigh(cov).map_err(wrap)?;
            let mut ut = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    ut[j * n + i] = u.as_slice()[i * n + j];
                }
            }
            (w, NdArray::from_vec(ut, &[n, n]).expect("n x n"))
        }
    };
    let mut factor = vec![0.0; n * n];
    for i in 0..n {
        let root = s[i].max(0.0).sqrt();
        for j in 0..n {
            factor[i * n + j] = root * vh.as_slice()[i * n + j];
        }
    }
    let factor = NdArray::from_vec(factor, &[n, n]).expect("n x n");
    if check_valid {
        let ft = crate::contraction::matmul(&factor.view().matrix_transpose().map_err(wrap)?, &factor.view()).map_err(wrap)?;
        let tol = 1e-8;
        let psd = ft.as_slice().iter().zip(cov.as_slice()).all(|(a, b)| (a - b).abs() <= tol + tol * b.abs());
        if !psd {
            return Err(RandomError("covariance is not symmetric positive-semidefinite.".to_string()));
        }
    }
    Ok(factor)
}

pub fn default_rng(seed: u64) -> Generator {
    Generator::from_seed_sequence(SeedSequence::new(seed))
}

pub struct Generator {
    rng: Pcg64,
    seed_seq: SeedSequence,
    has_u32: bool,
    u32_buf: u32,
}

impl Generator {
    pub fn from_seed_sequence(seed_seq: SeedSequence) -> Self {
        let w = seed_seq.generate_state_u64(4);
        let state = (u128::from(w[0]) << 64) | u128::from(w[1]);
        let stream = (u128::from(w[2]) << 64) | u128::from(w[3]);
        Self { rng: Pcg64::new(state, stream), seed_seq, has_u32: false, u32_buf: 0 }
    }

    pub fn spawn(&mut self, n: usize) -> Vec<Generator> {
        self.seed_seq.spawn(n).into_iter().map(Generator::from_seed_sequence).collect()
    }

    pub fn random_raw(&mut self) -> u64 {
        self.rng.next_u64()
    }

    fn next_u32(&mut self) -> u32 {
        if self.has_u32 {
            self.has_u32 = false;
            return self.u32_buf;
        }
        let next = self.rng.next_u64();
        self.has_u32 = true;
        self.u32_buf = (next >> 32) as u32;
        next as u32
    }

    fn next_double(&mut self) -> f64 {
        (self.rng.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    fn bounded(&mut self, max_inclusive: u64) -> u64 {
        if max_inclusive == 0 {
            return 0;
        }
        if max_inclusive <= u64::from(u32::MAX) {
            if max_inclusive == u64::from(u32::MAX) {
                return u64::from(self.next_u32());
            }
            let rng = max_inclusive as u32;
            let excl = rng + 1;
            let mut m = u64::from(self.next_u32()) * u64::from(excl);
            let mut leftover = m as u32;
            if leftover < excl {
                let threshold = (u32::MAX - rng) % excl;
                while leftover < threshold {
                    m = u64::from(self.next_u32()) * u64::from(excl);
                    leftover = m as u32;
                }
            }
            return m >> 32;
        }
        if max_inclusive == u64::MAX {
            return self.rng.next_u64();
        }
        let excl = max_inclusive + 1;
        let mut m = u128::from(self.rng.next_u64()) * u128::from(excl);
        let mut leftover = m as u64;
        if leftover < excl {
            let threshold = (u64::MAX - max_inclusive) % excl;
            while leftover < threshold {
                m = u128::from(self.rng.next_u64()) * u128::from(excl);
                leftover = m as u64;
            }
        }
        (m >> 64) as u64
    }

    fn interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        for shift in [1, 2, 4, 8, 16, 32] {
            mask |= mask >> shift;
        }
        if max <= u64::from(u32::MAX) {
            loop {
                let value = u64::from(self.next_u32()) & mask;
                if value <= max {
                    return value;
                }
            }
        }
        loop {
            let value = self.rng.next_u64() & mask;
            if value <= max {
                return value;
            }
        }
    }

    fn fill<D: Distribution<f64>>(&mut self, dist: D, shape: &[usize]) -> NdArray {
        let len: usize = shape.iter().product();
        let data: Vec<f64> = (0..len).map(|_| dist.sample(&mut self.rng)).collect();
        NdArray::from_vec(data, shape).expect("data.len() == shape.iter().product() by construction")
    }

    pub fn random(&mut self, shape: &[usize]) -> NdArray {
        let len: usize = shape.iter().product();
        let data: Vec<f64> = (0..len).map(|_| self.next_double()).collect();
        NdArray::from_vec(data, shape).expect("data.len() == shape.iter().product() by construction")
    }

    pub fn uniform(&mut self, low: f64, high: f64, shape: &[usize]) -> Result<NdArray, RandomError> {
        if !(low.is_finite() && high.is_finite()) {
            return Err(RandomError("low and high must be finite".to_string()));
        }
        let len: usize = shape.iter().product();
        let data: Vec<f64> = (0..len).map(|_| low + (high - low) * self.next_double()).collect();
        Ok(NdArray::from_vec(data, shape).expect("data.len() == shape.iter().product() by construction"))
    }

    pub fn integers(&mut self, low: i64, high: i64, shape: &[usize]) -> Result<NdArray, RandomError> {
        if low >= high {
            return Err(RandomError("low >= high".to_string()));
        }
        let span = (high as i128 - low as i128 - 1) as u64;
        let len: usize = shape.iter().product();
        let data: Vec<f64> = (0..len).map(|_| (low as i128 + self.bounded(span) as i128) as f64).collect();
        Ok(NdArray::from_vec(data, shape).expect("data.len() == shape.iter().product() by construction"))
    }

    pub fn shuffle<T>(&mut self, data: &mut [T]) {
        for i in (1..data.len()).rev() {
            let j = self.interval(i as u64) as usize;
            data.swap(i, j);
        }
    }

    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        self.shuffle(&mut v);
        v
    }

    pub fn shuffle_rows<T>(&mut self, arr: &mut NdArray<T>) {
        if arr.ndim() == 0 || arr.is_empty() {
            return;
        }
        let rows = arr.shape()[0];
        let width = arr.len() / rows;
        let data = arr.as_mut_slice();
        for i in (1..rows).rev() {
            let j = self.interval(i as u64) as usize;
            if i != j {
                let (head, tail) = data.split_at_mut(i * width);
                head[j * width..(j + 1) * width].swap_with_slice(&mut tail[..width]);
            }
        }
    }

    pub fn permutation_rows<T: Clone>(&mut self, arr: &NdArray<T>) -> NdArray<T> {
        let mut copy = arr.clone();
        self.shuffle_rows(&mut copy);
        copy
    }

    fn shuffle_from(&mut self, data: &mut [u64], first: usize) {
        for i in (first..data.len()).rev() {
            let j = self.bounded(i as u64) as usize;
            data.swap(i, j);
        }
    }

    pub fn choice_indices(
        &mut self,
        pop_size: usize,
        size: usize,
        replace: bool,
        p: Option<&[f64]>,
    ) -> Result<Vec<usize>, RandomError> {
        if pop_size == 0 && size > 0 {
            return Err(RandomError("a must be a positive integer unless no samples are taken".to_string()));
        }
        if size == 0 {
            return Ok(Vec::new());
        }
        if let Some(p) = p {
            if p.len() != pop_size {
                return Err(RandomError("a and p must have same size".to_string()));
            }
            if p.iter().any(|x| x.is_nan()) {
                return Err(RandomError("probabilities contain NaN".to_string()));
            }
            if p.iter().any(|&x| x < 0.0) {
                return Err(RandomError("Probabilities are not non-negative".to_string()));
            }
            if (p.iter().sum::<f64>() - 1.0).abs() > f64::EPSILON.sqrt() {
                return Err(RandomError("Probabilities do not sum to 1".to_string()));
            }
        }
        if replace {
            return Ok(match p {
                Some(p) => {
                    let cdf = normalized_cdf(p);
                    (0..size).map(|_| search_right(&cdf, self.next_double())).collect()
                }
                None => (0..size).map(|_| self.bounded(pop_size as u64 - 1) as usize).collect(),
            });
        }
        if size > pop_size {
            return Err(RandomError("Cannot take a larger sample than population when replace is False".to_string()));
        }
        if let Some(p) = p {
            return self.choice_without_replacement_weighted(p, size);
        }
        if pop_size > 10000 && size > pop_size / 50 {
            let mut idx: Vec<u64> = (0..pop_size as u64).collect();
            self.shuffle_from(&mut idx, (pop_size - size).max(1));
            return Ok(idx[pop_size - size..].iter().map(|&v| v as usize).collect());
        }
        let mut seen = std::collections::HashSet::new();
        let mut idx: Vec<u64> = Vec::with_capacity(size);
        for j in (pop_size - size) as u64..pop_size as u64 {
            let val = self.bounded(j);
            let chosen = if seen.insert(val) { val } else {
                seen.insert(j);
                j
            };
            idx.push(chosen);
        }
        self.shuffle_from(&mut idx, 1);
        Ok(idx.into_iter().map(|v| v as usize).collect())
    }

    fn choice_without_replacement_weighted(&mut self, p: &[f64], size: usize) -> Result<Vec<usize>, RandomError> {
        if p.iter().filter(|&&x| x != 0.0).count() < size {
            return Err(RandomError("Fewer non-zero entries in p than size".to_string()));
        }
        let mut weights = p.to_vec();
        let mut found: Vec<usize> = Vec::with_capacity(size);
        while found.len() < size {
            let draws: Vec<f64> = (0..size - found.len()).map(|_| self.next_double()).collect();
            for &f in &found {
                weights[f] = 0.0;
            }
            let cdf = normalized_cdf(&weights);
            let mut firsts: Vec<usize> = Vec::new();
            for x in draws {
                let candidate = search_right(&cdf, x);
                if !firsts.contains(&candidate) {
                    firsts.push(candidate);
                }
            }
            found.extend(firsts);
        }
        Ok(found)
    }

    pub fn choice<T: Clone>(
        &mut self,
        a: &[T],
        size: usize,
        replace: bool,
        p: Option<&[f64]>,
    ) -> Result<Vec<T>, RandomError> {
        Ok(self.choice_indices(a.len(), size, replace, p)?.into_iter().map(|i| a[i].clone()).collect())
    }

    pub fn multivariate_normal(
        &mut self,
        mean: &[f64],
        cov: &NdArray,
        size: usize,
        method: MvnMethod,
        check_valid: bool,
    ) -> Result<NdArray, RandomError> {
        let n = mean.len();
        if cov.ndim() != 2 || cov.shape()[0] != cov.shape()[1] {
            return Err(RandomError("cov must be 2 dimensional and square".to_string()));
        }
        if cov.shape()[0] != n {
            return Err(RandomError("mean and cov must have same length".to_string()));
        }
        let factor = mvn_factor(cov, method, check_valid)?;
        let z = self.standard_normal(&[size, n]);
        let mut x = crate::contraction::matmul(&z.view(), &factor.view()).map_err(wrap)?;
        for row in x.as_mut_slice().chunks_mut(n.max(1)) {
            for (v, m) in row.iter_mut().zip(mean) {
                *v += m;
            }
        }
        Ok(x)
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
        let mut a = default_rng(42);
        let mut b = default_rng(42);
        assert_eq!(a.random(&[10]).as_slice(), b.random(&[10]).as_slice());
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = default_rng(1);
        let mut b = default_rng(2);
        assert_ne!(a.random(&[10]).as_slice(), b.random(&[10]).as_slice());
    }

    #[test]
    fn random_is_bounded_to_zero_one() {
        let mut rng_gen = default_rng(0);
        let a = rng_gen.random(&[1000]);
        assert!(a.as_slice().iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn uniform_matches_its_own_mean_formula() {

        let mut rng_gen = default_rng(1);
        let a = rng_gen.uniform(2.0, 10.0, &[20_000]).unwrap();
        assert!((mean(&a.view()) - 6.0).abs() < 0.1);
    }

    #[test]
    fn standard_normal_matches_mean_zero_var_one() {
        let mut rng_gen = default_rng(2);
        let a = rng_gen.standard_normal(&[50_000]);
        assert!(mean(&a.view()).abs() < 0.05);
        assert!((var_default(&a.view()) - 1.0).abs() < 0.1);
    }

    #[test]
    fn normal_matches_its_own_mean_and_variance_formula() {
        let mut rng_gen = default_rng(3);
        let a = rng_gen.normal(5.0, 2.0, &[50_000]).unwrap();
        assert!((mean(&a.view()) - 5.0).abs() < 0.1);
        assert!((var_default(&a.view()) - 4.0).abs() < 0.3);
    }

    #[test]
    fn exponential_mean_equals_scale() {

        let mut rng_gen = default_rng(4);
        let a = rng_gen.exponential(3.0, &[50_000]).unwrap();
        assert!((mean(&a.view()) - 3.0).abs() < 0.1);
    }

    #[test]
    fn gamma_mean_equals_shape_times_scale() {

        let mut rng_gen = default_rng(5);
        let a = rng_gen.gamma(2.0, 3.0, &[50_000]).unwrap();
        assert!((mean(&a.view()) - 6.0).abs() < 0.2);
    }

    #[test]
    fn beta_mean_matches_its_own_formula() {

        let mut rng_gen = default_rng(6);
        let a = rng_gen.beta(2.0, 3.0, &[50_000]).unwrap();
        assert!((mean(&a.view()) - 0.4).abs() < 0.02);
    }

    #[test]
    fn binomial_mean_equals_n_times_p() {
        let mut rng_gen = default_rng(7);
        let a = rng_gen.binomial(20, 0.3, &[50_000]).unwrap();
        assert!((mean(&a.view()) - 6.0).abs() < 0.1);
        assert!(a.as_slice().iter().all(|&x| x.fract() == 0.0));
    }

    #[test]
    fn poisson_mean_equals_lambda() {
        let mut rng_gen = default_rng(8);
        let a = rng_gen.poisson(4.0, &[50_000]).unwrap();
        assert!((mean(&a.view()) - 4.0).abs() < 0.1);
        assert!(a.as_slice().iter().all(|&x| x.fract() == 0.0));
    }

    #[test]
    fn dirichlet_rows_sum_to_one() {
        let mut rng_gen = default_rng(9);
        let a = rng_gen.dirichlet(&[1.0, 2.0, 3.0], 100).unwrap();
        assert_eq!(a.shape(), &[100, 3]);
        for row in a.as_slice().chunks(3) {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn integers_are_within_half_open_range() {
        let mut rng_gen = default_rng(10);
        let a = rng_gen.integers(0, 5, &[1000]).unwrap();
        assert!(a.as_slice().iter().all(|&x| (0.0..5.0).contains(&x) && x.fract() == 0.0));
    }

    #[test]
    fn invalid_parameters_err_instead_of_panicking() {
        let mut rng_gen = default_rng(0);
        assert!(rng_gen.normal(0.0, -1.0, &[1]).is_err());
        assert!(rng_gen.beta(-1.0, 1.0, &[1]).is_err());
    }

    fn hex(v: &[u64]) -> Vec<String> {
        v.iter().map(|x| format!("{x:#x}")).collect()
    }

    #[test]
    fn seed_sequence_matches_numpy_bit_for_bit() {
        let ss = SeedSequence::new(42);
        assert_eq!(
            ss.generate_state_u64(4),
            vec![0x9f1e2e6dcd540ab7, 0xd57873dc79fb94b6, 0x7d282a1b64d420b7, 0x336579714692d5ff]
        );
        assert_eq!(
            SeedSequence::new(0).generate_state_u64(4),
            vec![0xdb2cd7e7b0f478be, 0xabf4641a2c71ba49, 0x20c6ed6d9d7b8d41, 0x2c4099de223c39d4]
        );
        assert_eq!(
            SeedSequence::new((1 << 32) + 5).generate_state_u64(4),
            vec![0xf9c15bd6e249d9ac, 0xb2021b4036320ed7, 0x9388b47fe6dcf739, 0x5eeba8d9ad6439e1]
        );
        assert_eq!(SeedSequence::new(u64::MAX).generate_state_u64(2), vec![0xaebca151928cad0d, 0x119c30448638dc7a]);
        assert_eq!(hex(&[ss.generate_state_u32(3)[0].into()]), vec!["0xcd540ab7"]);
        assert_eq!(ss.generate_state_u32(3), vec![0xcd540ab7, 0x9f1e2e6d, 0x79fb94b6]);
    }

    #[test]
    fn spawned_children_match_numpy_and_the_counter_advances() {
        let mut ss = SeedSequence::new(42);
        let kids = ss.spawn(3);
        let want = [
            (vec![0u32], [0xdff6ed7da001c6a4u64, 0x4323988864d84a1f, 0x9c90dafb22be369b, 0x77fd4b767416fa91]),
            (vec![1], [0x01dcb763f3e63cba, 0x6aee7dd615de6f6e, 0x3e2df752dbd3217f, 0x29d15a391cf7e717]),
            (vec![2], [0xa102097a0de6c5ed, 0x957ebe91c9f23d3f, 0x7615a8ddb1cfb7e4, 0x535c93f339b18b5c]),
        ];
        for (kid, (key, state)) in kids.iter().zip(want) {
            assert_eq!(kid.spawn_key(), key.as_slice());
            assert_eq!(kid.generate_state_u64(4), state.to_vec());
        }
        let mut first = kids[0].clone();
        let grand = first.spawn(2);
        assert_eq!(grand[0].spawn_key(), &[0, 0]);
        assert_eq!(grand[0].generate_state_u64(2), vec![0x6eef93eebb5642fa, 0x8a452ccc553c466c]);
        let again = ss.spawn(2);
        assert_eq!(again.iter().map(|c| c.spawn_key().to_vec()).collect::<Vec<_>>(), vec![vec![3], vec![4]]);
    }

    #[test]
    fn raw_stream_and_random_are_bit_identical_to_numpy() {
        let mut g = default_rng(42);
        assert_eq!(
            (0..3).map(|_| g.random_raw()).collect::<Vec<_>>(),
            vec![0xc621fbcd16d92688, 0x705a5661a791ffc1, 0xdbcd12c26eda1624]
        );
        let mut g = default_rng(42);
        assert_eq!(g.random(&[3]).as_slice(), &[0.7739560485559633, 0.4388784397520523, 0.8585979199113825]);
        let mut from_ss = Generator::from_seed_sequence(SeedSequence::new(7));
        assert_eq!(from_ss.random(&[2]).as_slice(), &[0.625095466604667, 0.8972138009695755]);
    }

    #[test]
    fn integers_and_uniform_use_numpys_bounded_algorithms() {
        let mut g = default_rng(42);
        assert_eq!(g.integers(0, 10, &[5]).unwrap().as_slice(), &[0.0, 7.0, 6.0, 4.0, 4.0]);
        assert_eq!(g.integers(-5, 5, &[4]).unwrap().as_slice(), &[3.0, -5.0, 1.0, -3.0]);
        assert_eq!(g.integers(0, 1 << 40, &[2]).unwrap().as_slice(), &[1072708119942.0, 836881952700.0]);
        assert_eq!(g.integers(0, 1 << 32, &[2]).unwrap().as_slice(), &[404488629.0, 3081541440.0]);
        assert_eq!(g.integers(0, 1, &[2]).unwrap().as_slice(), &[0.0, 0.0]);
        assert!(g.integers(3, 3, &[1]).is_err());
        let mut u = default_rng(42);
        assert_eq!(u.uniform(2.0, 5.0, &[3]).unwrap().as_slice(), &[4.3218681456678905, 3.316635319256157, 4.575793759734148]);
    }

    #[test]
    fn shuffle_and_permutation_reproduce_numpy_exactly() {
        let mut g = default_rng(42);
        let mut x: Vec<usize> = (0..10).collect();
        g.shuffle(&mut x);
        assert_eq!(x, vec![5, 6, 0, 7, 3, 2, 4, 9, 1, 8]);
        assert_eq!(default_rng(42).permutation(10), vec![5, 6, 0, 7, 3, 2, 4, 9, 1, 8]);
        assert_eq!(default_rng(3).permutation(5), vec![4, 2, 1, 3, 0]);
        assert!(default_rng(1).permutation(0).is_empty());
        assert_eq!(default_rng(1).permutation(1), vec![0]);
    }

    #[test]
    fn shuffle_rows_moves_whole_rows_like_numpy() {
        let mut m = NdArray::from_vec((0..12).collect::<Vec<i32>>(), &[4, 3]).unwrap();
        default_rng(5).shuffle_rows(&mut m);
        assert_eq!(m.as_slice(), &[9, 10, 11, 3, 4, 5, 6, 7, 8, 0, 1, 2]);
        let original = NdArray::from_vec((0..12).collect::<Vec<i32>>(), &[4, 3]).unwrap();
        let copy = default_rng(5).permutation_rows(&original);
        assert_eq!(copy, m);
        assert_eq!(original.as_slice()[0], 0);
        let mut empty: NdArray<i32> = NdArray::zeros(&[0, 3]);
        default_rng(1).shuffle_rows(&mut empty);
    }

    #[test]
    fn choice_reproduces_numpy_for_every_mode() {
        let mut g = default_rng(42);
        assert_eq!(g.choice_indices(10, 5, true, None).unwrap(), vec![0, 7, 6, 4, 4]);
        assert_eq!(g.choice_indices(5, 3, false, None).unwrap(), vec![0, 3, 2]);

        let p = [0.1, 0.2, 0.3, 0.4];
        let mut g = default_rng(42);
        assert_eq!(g.choice_indices(4, 6, true, Some(&p)).unwrap(), vec![3, 2, 3, 3, 0, 3]);
        assert_eq!(g.choice_indices(4, 3, false, Some(&p)).unwrap(), vec![3, 1, 2]);

        let mut g = default_rng(42);
        assert_eq!(g.choice_indices(20, 7, false, None).unwrap(), vec![19, 16, 7, 10, 1, 11, 17]);
        assert_eq!(g.choice_indices(20000, 5, false, None).unwrap(), vec![15719, 15219, 10264, 14347, 2562]);
        let tail = default_rng(9).choice_indices(20000, 3000, false, None).unwrap();
        assert_eq!(&tail[..6], &[9306, 4925, 10242, 7429, 8377, 19025]);
        let unique: std::collections::HashSet<_> = tail.iter().collect();
        assert_eq!(unique.len(), 3000);
    }

    #[test]
    fn choice_returns_elements_and_validates_like_numpy() {
        let names = ["a", "b", "c", "d"];
        assert_eq!(default_rng(42).choice(&names, 3, true, None).unwrap(), vec!["a", "d", "c"]);
        assert!(default_rng(1).choice(&names, 0, true, None).unwrap().is_empty());
        let err = |r: Result<Vec<usize>, RandomError>| r.unwrap_err().0;
        assert!(err(default_rng(1).choice_indices(3, 5, false, None)).contains("larger sample than population"));
        assert!(err(default_rng(1).choice_indices(3, 2, true, Some(&[0.5, 0.6, 0.1]))).contains("do not sum to 1"));
        assert!(err(default_rng(1).choice_indices(0, 2, true, None)).contains("positive integer"));
        assert!(err(default_rng(1).choice_indices(3, 2, true, Some(&[-0.1, 0.6, 0.5]))).contains("non-negative"));
        assert!(err(default_rng(1).choice_indices(3, 3, false, Some(&[0.5, 0.5, 0.0]))).contains("Fewer non-zero"));
        assert!(err(default_rng(1).choice_indices(3, 2, true, Some(&[0.5, 0.5]))).contains("same size"));
        assert!(err(default_rng(1).choice_indices(2, 1, true, Some(&[f64::NAN, 1.0]))).contains("NaN"));
        assert!(default_rng(1).choice_indices(0, 0, true, None).unwrap().is_empty());
    }

    #[test]
    fn spawned_generators_match_numpy_and_are_independent() {
        let mut g = default_rng(42);
        let mut kids = g.spawn(2);
        assert_eq!(kids[0].random(&[2]).as_slice(), &[0.9167441575549085, 0.9109866676343232]);
        assert_eq!(kids[1].random(&[2]).as_slice(), &[0.4674907799518424, 0.04644889644868733]);
        let mut more = g.spawn(1);
        assert_eq!(more[0].random(&[1]).as_slice(), &[0.07123920291270869]);

        let a = default_rng(42).spawn(2).remove(0).random(&[1000]);
        let b = default_rng(42).spawn(2).remove(1).random(&[1000]);
        assert_ne!(a.as_slice(), b.as_slice());
        let (ma, mb) = (mean(&a.view()), mean(&b.view()));
        assert!((ma - 0.5).abs() < 0.05 && (mb - 0.5).abs() < 0.05);
        let corr: f64 = a.as_slice().iter().zip(b.as_slice()).map(|(x, y)| (x - ma) * (y - mb)).sum::<f64>() / 1000.0;
        assert!(corr.abs() < 0.02, "children look correlated: {corr}");
    }

    #[test]
    fn mvn_factors_reproduce_the_covariance_and_cholesky_matches_numpy() {
        let cov = NdArray::from_vec(vec![2.0, 1.0, 1.0, 2.0], &[2, 2]).unwrap();
        for method in [MvnMethod::Svd, MvnMethod::Eigh, MvnMethod::Cholesky] {
            let t = mvn_factor(&cov, method, true).unwrap();
            let back = crate::contraction::matmul(&t.view().matrix_transpose().unwrap(), &t.view()).unwrap();
            for (got, want) in back.as_slice().iter().zip(cov.as_slice()) {
                assert!((got - want).abs() < 1e-12, "{method:?}: T^T T != cov");
            }
        }
        let chol = mvn_factor(&cov, MvnMethod::Cholesky, true).unwrap();
        let l_t = [2f64.sqrt(), 1.0 / 2f64.sqrt(), 0.0, (1.5f64).sqrt()];
        for (got, want) in chol.as_slice().iter().zip(l_t) {
            assert!((got - want).abs() < 1e-12);
        }
        let z = NdArray::from_vec(vec![0.5, -1.0, 1.5, 0.25, -0.75, 2.0], &[3, 2]).unwrap();
        let shifted = crate::contraction::matmul(&z.view(), &chol.view()).unwrap();
        let want = [[0.7071067811865475, -1.8711914807983152], [2.1213203435596424, 0.3668463896277183], [-1.0606601717798212, 0.9191596568932674]];
        for (row, w) in shifted.as_slice().chunks(2).zip(want) {
            assert!((row[0] + 1.0 - (w[0] + 1.0)).abs() < 1e-9 && (row[1] - (w[1] + 1.0)).abs() < 1e-9);
        }
    }

    #[test]
    fn multivariate_normal_has_the_requested_mean_and_covariance() {
        let cov = NdArray::from_vec(vec![2.0, 1.0, 1.0, 2.0], &[2, 2]).unwrap();
        for method in [MvnMethod::Svd, MvnMethod::Eigh, MvnMethod::Cholesky] {
            let x = default_rng(3).multivariate_normal(&[1.0, -1.0], &cov, 100_000, method, true).unwrap();
            assert_eq!(x.shape(), &[100_000, 2]);
            let n = 100_000.0;
            let m0 = x.as_slice().iter().step_by(2).sum::<f64>() / n;
            let m1 = x.as_slice().iter().skip(1).step_by(2).sum::<f64>() / n;
            assert!((m0 - 1.0).abs() < 0.03 && (m1 + 1.0).abs() < 0.03, "{method:?}: mean ({m0}, {m1})");
            let (mut c00, mut c01, mut c11) = (0.0, 0.0, 0.0);
            for row in x.as_slice().chunks(2) {
                c00 += (row[0] - m0).powi(2);
                c01 += (row[0] - m0) * (row[1] - m1);
                c11 += (row[1] - m1).powi(2);
            }
            assert!((c00 / n - 2.0).abs() < 0.06, "{method:?}: var0 {}", c00 / n);
            assert!((c01 / n - 1.0).abs() < 0.06, "{method:?}: cov01 {}", c01 / n);
            assert!((c11 / n - 2.0).abs() < 0.06, "{method:?}: var1 {}", c11 / n);
        }
    }

    #[test]
    fn multivariate_normal_rejects_bad_shapes_and_non_psd_covariance() {
        let mut g = default_rng(1);
        let bad = NdArray::from_vec(vec![1.0, 2.0, 2.0, 1.0], &[2, 2]).unwrap();
        let err = g.multivariate_normal(&[0.0, 0.0], &bad, 1, MvnMethod::Svd, true).unwrap_err();
        assert!(err.0.contains("positive-semidefinite"));
        assert!(g.multivariate_normal(&[0.0, 0.0], &bad, 1, MvnMethod::Svd, false).is_ok());
        assert!(g.multivariate_normal(&[0.0, 0.0], &bad, 1, MvnMethod::Cholesky, true).is_err());
        let id3: NdArray = NdArray::zeros(&[3, 3]);
        assert!(g.multivariate_normal(&[0.0, 0.0], &id3, 1, MvnMethod::Svd, true).unwrap_err().0.contains("same length"));
        let rect: NdArray = NdArray::zeros(&[2, 3]);
        assert!(g.multivariate_normal(&[0.0, 0.0], &rect, 1, MvnMethod::Svd, true).unwrap_err().0.contains("square"));
        let single = g.multivariate_normal(&[5.0], &NdArray::from_vec(vec![0.0], &[1, 1]).unwrap(), 3, MvnMethod::Svd, true).unwrap();
        assert_eq!(single.as_slice(), &[5.0, 5.0, 5.0]);
    }
}
