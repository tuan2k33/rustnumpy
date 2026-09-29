use crate::dispatch::WrapAdd;
use crate::view::ArrayView;

pub trait FloatIsh: Copy + PartialOrd {
    fn is_nan_ish(self) -> bool {
        false
    }
}

macro_rules! impl_floatish_for_ints {
    ($($t:ty),*) => {
        $(impl FloatIsh for $t {})*
    };
}
impl_floatish_for_ints!(i8, i16, i32, i64, u8, u16, u32, u64);

impl FloatIsh for f32 {
    fn is_nan_ish(self) -> bool {
        self.is_nan()
    }
}

impl FloatIsh for f64 {
    fn is_nan_ish(self) -> bool {
        self.is_nan()
    }
}

pub trait AsF64: Copy {
    fn as_f64(self) -> f64;
}

macro_rules! impl_as_f64_for_ints {
    ($($t:ty),*) => {
        $(impl AsF64 for $t {
            fn as_f64(self) -> f64 {
                self as f64
            }
        })*
    };
}
impl_as_f64_for_ints!(i8, i16, i32, i64, u8, u16, u32, u64);

impl AsF64 for f32 {
    fn as_f64(self) -> f64 {
        self as f64
    }
}

impl AsF64 for f64 {
    fn as_f64(self) -> f64 {
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReductionError {

    EmptyInput,

    PercentileOutOfRange { q: f64 },

    InvalidBins,

    TooManyBins { bins: usize },

    InvalidRange { low: f64, high: f64 },

    NonFiniteRange { low: f64, high: f64 },
}

impl std::fmt::Display for ReductionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReductionError::EmptyInput => write!(f, "zero-size array has no reduction identity"),
            ReductionError::PercentileOutOfRange { q } => {
                write!(f, "percentile {q} must be in the range [0, 100]")
            }
            ReductionError::InvalidBins => write!(f, "`bins` must be positive, when an integer"),
            ReductionError::TooManyBins { bins } => {
                write!(f, "Too many bins for data range. Cannot create {bins} finite-sized bins.")
            }
            ReductionError::InvalidRange { .. } => write!(f, "max must be larger than min in range parameter."),
            ReductionError::NonFiniteRange { low, high } => {
                write!(f, "autodetected range of [{low}, {high}] is not finite")
            }
        }
    }
}

impl std::error::Error for ReductionError {}

fn values<T: Copy>(view: &ArrayView<T>) -> Vec<T> {
    crate::shape::IndexIter::new(view.shape())
        .map(|idx| view.get(&idx).expect("IndexIter only yields valid indices"))
        .collect()
}

pub fn sum<T: Copy + Default + WrapAdd>(view: &ArrayView<T>) -> T {
    values(view).into_iter().fold(T::default(), |acc, x| acc.wrap_add(x))
}

pub fn mean<T: AsF64>(view: &ArrayView<T>) -> f64 {
    let v = values(view);
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().map(|x| x.as_f64()).sum::<f64>() / v.len() as f64
    }
}

pub fn var<T: AsF64>(view: &ArrayView<T>, ddof: usize) -> f64 {
    let v = values(view);
    let n = v.len();
    if n == 0 || n <= ddof {
        return f64::NAN;
    }
    let m = v.iter().map(|x| x.as_f64()).sum::<f64>() / n as f64;
    let ss: f64 = v.iter().map(|x| (x.as_f64() - m).powi(2)).sum();
    ss / (n - ddof) as f64
}

pub fn var_default<T: AsF64>(view: &ArrayView<T>) -> f64 {
    var(view, 0)
}

pub fn std<T: AsF64>(view: &ArrayView<T>, ddof: usize) -> f64 {
    var(view, ddof).sqrt()
}

pub fn std_default<T: AsF64>(view: &ArrayView<T>) -> f64 {
    std(view, 0)
}

pub fn min<T: FloatIsh>(view: &ArrayView<T>) -> Result<T, ReductionError> {
    let v = values(view);
    let mut iter = v.into_iter();
    let first = iter.next().ok_or(ReductionError::EmptyInput)?;
    Ok(iter.fold(first, |acc, x| {
        if acc.is_nan_ish() {
            acc
        } else if x.is_nan_ish() || x < acc {
            x
        } else {
            acc
        }
    }))
}

pub fn max<T: FloatIsh>(view: &ArrayView<T>) -> Result<T, ReductionError> {
    let v = values(view);
    let mut iter = v.into_iter();
    let first = iter.next().ok_or(ReductionError::EmptyInput)?;
    Ok(iter.fold(first, |acc, x| {
        if acc.is_nan_ish() {
            acc
        } else if x.is_nan_ish() || x > acc {
            x
        } else {
            acc
        }
    }))
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let diff = b - a;
    if t >= 0.5 { b - diff * (1.0 - t) } else { a + diff * t }
}

fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let index = (n - 1) as f64 * (q / 100.0);
    let lower = index.floor();
    let below = (lower as usize).min(n - 1);
    let above = (below + 1).min(n - 1);
    lerp(sorted[below], sorted[above], index - lower)
}

pub fn percentile<T: AsF64>(view: &ArrayView<T>, q: f64) -> Result<f64, ReductionError> {
    if !(0.0..=100.0).contains(&q) {
        return Err(ReductionError::PercentileOutOfRange { q });
    }
    let mut v: Vec<f64> = values(view).into_iter().map(|x| x.as_f64()).collect();
    if v.is_empty() {
        return Err(ReductionError::EmptyInput);
    }
    if v.iter().any(|x| x.is_nan()) {
        return Ok(f64::NAN);
    }
    v.sort_unstable_by(|a, b| a.total_cmp(b));
    Ok(percentile_sorted(&v, q))
}

fn median_of_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

pub fn median<T: AsF64>(view: &ArrayView<T>) -> Result<f64, ReductionError> {
    let mut v: Vec<f64> = values(view).into_iter().map(|x| x.as_f64()).collect();
    if v.is_empty() {
        return Err(ReductionError::EmptyInput);
    }
    if v.iter().any(|x| x.is_nan()) {
        return Ok(f64::NAN);
    }
    v.sort_unstable_by(|a, b| a.total_cmp(b));
    Ok(median_of_sorted(&v))
}

fn non_nan_values<T: FloatIsh>(view: &ArrayView<T>) -> Vec<T> {
    values(view).into_iter().filter(|x| !x.is_nan_ish()).collect()
}

pub fn nansum<T: FloatIsh + Default + WrapAdd>(view: &ArrayView<T>) -> T {
    non_nan_values(view).into_iter().fold(T::default(), |acc, x| acc.wrap_add(x))
}

pub fn nanmean<T: FloatIsh + AsF64>(view: &ArrayView<T>) -> f64 {
    let v = non_nan_values(view);
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().map(|x| x.as_f64()).sum::<f64>() / v.len() as f64
    }
}

pub fn nanvar<T: FloatIsh + AsF64>(view: &ArrayView<T>, ddof: usize) -> f64 {
    let v = non_nan_values(view);
    let n = v.len();
    if n == 0 || n <= ddof {
        return f64::NAN;
    }
    let m = v.iter().map(|x| x.as_f64()).sum::<f64>() / n as f64;
    let ss: f64 = v.iter().map(|x| (x.as_f64() - m).powi(2)).sum();
    ss / (n - ddof) as f64
}

pub fn nanvar_default<T: FloatIsh + AsF64>(view: &ArrayView<T>) -> f64 {
    nanvar(view, 0)
}

pub fn nanstd<T: FloatIsh + AsF64>(view: &ArrayView<T>, ddof: usize) -> f64 {
    nanvar(view, ddof).sqrt()
}

pub fn nanstd_default<T: FloatIsh + AsF64>(view: &ArrayView<T>) -> f64 {
    nanstd(view, 0)
}

pub fn nanmin<T: FloatIsh>(view: &ArrayView<T>) -> Result<T, ReductionError> {
    if view.is_empty() {
        return Err(ReductionError::EmptyInput);
    }
    let v = non_nan_values(view);
    let mut iter = v.into_iter();
    let Some(first) = iter.next() else {

        return Ok(values(view)[0]);
    };
    Ok(iter.fold(first, |acc, x| if x < acc { x } else { acc }))
}

pub fn nanmax<T: FloatIsh>(view: &ArrayView<T>) -> Result<T, ReductionError> {
    if view.is_empty() {
        return Err(ReductionError::EmptyInput);
    }
    let v = non_nan_values(view);
    let mut iter = v.into_iter();
    let Some(first) = iter.next() else {
        return Ok(values(view)[0]);
    };
    Ok(iter.fold(first, |acc, x| if x > acc { x } else { acc }))
}

pub fn nanmedian<T: FloatIsh + AsF64>(view: &ArrayView<T>) -> Result<f64, ReductionError> {
    if view.is_empty() {
        return Err(ReductionError::EmptyInput);
    }
    let mut v: Vec<f64> = non_nan_values(view).into_iter().map(|x| x.as_f64()).collect();
    if v.is_empty() {
        return Ok(f64::NAN);
    }
    v.sort_unstable_by(|a, b| a.total_cmp(b));
    Ok(median_of_sorted(&v))
}

fn linspace_edges(low: f64, high: f64, bins: usize) -> Vec<f64> {
    let step = (high - low) / bins as f64;
    let mut edges: Vec<f64> = (0..=bins).map(|i| i as f64 * step + low).collect();
    edges[bins] = high;
    edges
}

pub fn histogram(data: &[f64], bins: usize, range: Option<(f64, f64)>) -> Result<(Vec<usize>, Vec<f64>), ReductionError> {
    if bins == 0 {
        return Err(ReductionError::InvalidBins);
    }
    let (mut low, mut high) = match range {
        Some((lo, hi)) => {
            if lo > hi {
                return Err(ReductionError::InvalidRange { low: lo, high: hi });
            }
            (lo, hi)
        }
        None if data.is_empty() => (0.0, 1.0),
        None => {
            let lo = data.iter().copied().fold(f64::INFINITY, |a, x| if x < a || x.is_nan() { x } else { a });
            let hi = data.iter().copied().fold(f64::NEG_INFINITY, |a, x| if x > a || x.is_nan() { x } else { a });
            (lo, hi)
        }
    };
    if !(low.is_finite() && high.is_finite()) {
        return Err(ReductionError::NonFiniteRange { low, high });
    }
    if low == high {
        low -= 0.5;
        high += 0.5;
    }
    let edges = linspace_edges(low, high, bins);
    if edges.windows(2).any(|w| w[0] >= w[1]) {
        return Err(ReductionError::TooManyBins { bins });
    }
    let norm = bins as f64 / (high - low);
    let mut counts = vec![0usize; bins];
    for &x in data {
        if !(x >= low && x <= high) {
            continue;
        }
        let mut idx = (((x - low) * norm) as usize).min(bins);
        if idx == bins {
            idx -= 1;
        }
        if x < edges[idx] {
            idx -= 1;
        } else if x >= edges[idx + 1] && idx != bins - 1 {
            idx += 1;
        }
        counts[idx] += 1;
    }
    Ok((counts, edges))
}

pub fn cov(matrix: &ArrayView, ddof: usize) -> Result<crate::NdArray, ReductionError> {
    if matrix.ndim() != 2 {
        return Err(ReductionError::EmptyInput);
    }
    let rows = matrix.shape()[0];
    let cols = matrix.shape()[1];
    if cols == 0 || cols <= ddof {
        return Err(ReductionError::EmptyInput);
    }
    let row_data: Vec<Vec<f64>> = (0..rows)
        .map(|r| (0..cols).map(|c| matrix.get(&[r, c]).expect("in-bounds index")).collect())
        .collect();
    let means: Vec<f64> = row_data.iter().map(|row| row.iter().sum::<f64>() / cols as f64).collect();

    let mut out = vec![0.0; rows * rows];
    for i in 0..rows {
        for j in 0..rows {
            let s: f64 = (0..cols).map(|k| (row_data[i][k] - means[i]) * (row_data[j][k] - means[j])).sum();
            out[i * rows + j] = s / (cols - ddof) as f64;
        }
    }
    Ok(crate::NdArray::from_vec(out, &[rows, rows]).expect("out.len() == rows*rows by construction"))
}

pub fn cov_default(matrix: &ArrayView) -> Result<crate::NdArray, ReductionError> {
    cov(matrix, 1)
}

pub fn corrcoef(matrix: &ArrayView) -> Result<crate::NdArray, ReductionError> {
    let covariance = cov_default(matrix)?;
    let rows = covariance.shape()[0];
    let std_devs: Vec<f64> = (0..rows).map(|i| covariance.get(&[i, i]).unwrap().sqrt()).collect();
    let mut out = vec![0.0; rows * rows];
    for i in 0..rows {
        for j in 0..rows {
            out[i * rows + j] = (covariance.get(&[i, j]).unwrap() / (std_devs[i] * std_devs[j])).clamp(-1.0, 1.0);
        }
    }
    Ok(crate::NdArray::from_vec(out, &[rows, rows]).expect("out.len() == rows*rows by construction"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArray;

    fn arr(values: &[f64]) -> NdArray {
        NdArray::from_vec(values.to_vec(), &[values.len()]).unwrap()
    }

    #[test]
    fn sum_mean_match_real_numpy() {
        let a = arr(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(sum(&a.view()), 15.0);
        assert_eq!(mean(&a.view()), 3.0);
    }

    #[test]
    fn var_std_default_ddof_zero_match_real_numpy() {

        let a = arr(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(var_default(&a.view()), 2.0);
        assert!((std_default(&a.view()) - std::f64::consts::SQRT_2).abs() < 1e-12);

        assert_eq!(var(&a.view(), 1), 2.5);
    }

    #[test]
    fn var_with_ddof_at_least_n_is_nan_not_an_error() {
        let a = arr(&[1.0]);
        assert!(var(&a.view(), 1).is_nan());
    }

    #[test]
    fn min_max_propagate_nan() {
        let a = arr(&[1.0, f64::NAN, 3.0]);
        assert!(min(&a.view()).unwrap().is_nan());
        assert!(max(&a.view()).unwrap().is_nan());
    }

    #[test]
    fn min_max_on_empty_array_errs() {
        let a = arr(&[]);
        assert_eq!(min(&a.view()), Err(ReductionError::EmptyInput));
        assert_eq!(max(&a.view()), Err(ReductionError::EmptyInput));
    }

    #[test]
    fn histogram_matches_numpy_on_bin_edges_and_edge_cases() {
        let x: Vec<f64> = (0..50).map(|i| -10.0 + 20.0 * i as f64 / 49.0).collect();
        let (counts, edges) = histogram(&x, 58, Some((-10.0, 10.0))).unwrap();
        let want: Vec<usize> = vec![
            1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1,
            1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
        ];
        assert_eq!(counts, want);
        assert_eq!(edges[1], -9.655172413793103);
        assert_eq!(edges[2], -9.310344827586206);
        assert_eq!(edges[57], 9.655172413793103);
        assert_eq!(edges[58], 10.0);
        let (c, e) = histogram(&[3.0, 3.0, 3.0], 4, None).unwrap();
        assert_eq!((c, e), (vec![0, 0, 3, 0], vec![2.5, 2.75, 3.0, 3.25, 3.5]));
        assert_eq!(histogram(&[], 2, None).unwrap(), (vec![0, 0], vec![0.0, 0.5, 1.0]));
        assert_eq!(histogram(&[1.0, 2.0, f64::NAN, 5.0], 2, Some((0.0, 4.0))).unwrap().0, vec![1, 1]);
        assert_eq!(histogram(&[1.0, 2.0], 0, None), Err(ReductionError::InvalidBins));
        assert!(matches!(histogram(&[1.0, 2.0], 2, Some((3.0, 1.0))), Err(ReductionError::InvalidRange { .. })));
        assert!(matches!(histogram(&[1.0, f64::INFINITY], 2, None), Err(ReductionError::NonFiniteRange { .. })));
    }

    #[test]
    fn corrcoef_never_exceeds_one_in_magnitude() {
        let x: Vec<f64> = (0..60).map(|i| ((i * 37 % 101) as f64) * 0.1 + 0.3).collect();
        let m = NdArray::from_vec(x.iter().chain(x.iter()).copied().collect(), &[2, 60]).unwrap();
        let c = corrcoef(&m.view()).unwrap();
        assert!(c.as_slice().iter().all(|v| v.abs() <= 1.0));
        assert_eq!(c.get(&[0, 1]).unwrap(), 1.0);
    }

    #[test]
    fn median_uses_the_mean_of_the_middle_pair_so_infinities_survive() {
        let a = NdArray::from_vec(vec![f64::INFINITY, f64::INFINITY], &[2]).unwrap();
        assert_eq!(median(&a.view()).unwrap(), f64::INFINITY);
        let b = NdArray::from_vec(vec![f64::NAN, f64::NAN, f64::INFINITY, f64::INFINITY], &[4]).unwrap();
        assert_eq!(nanmedian(&b.view()).unwrap(), f64::INFINITY);
        assert!(percentile(&a.view(), 50.0).unwrap().is_nan());
    }

    #[test]
    fn median_and_percentile_propagate_nan_like_numpy() {
        let a = NdArray::from_vec(vec![0.0445, f64::NAN, 0.0463], &[3]).unwrap();
        assert!(median(&a.view()).unwrap().is_nan());
        assert!(percentile(&a.view(), 0.0).unwrap().is_nan());
        assert!(percentile(&a.view(), 100.0).unwrap().is_nan());
        let b = NdArray::from_vec(vec![f64::INFINITY, 1.0, f64::NAN], &[3]).unwrap();
        assert!(median(&b.view()).unwrap().is_nan());
        assert_eq!(nanmedian(&a.view()).unwrap(), (0.0445 + 0.0463) / 2.0);
    }

    #[test]
    fn median_matches_real_numpy_even_and_odd() {

        assert_eq!(median(&arr(&[1.0, 2.0, 3.0, 4.0, 5.0]).view()).unwrap(), 3.0);
        assert_eq!(median(&arr(&[1.0, 2.0, 3.0, 4.0]).view()).unwrap(), 2.5);
    }

    #[test]
    fn percentile_matches_real_numpy_linear_interpolation() {

        let d = arr(&(1..=10).map(|i| i as f64).collect::<Vec<_>>());
        assert!((percentile(&d.view(), 10.0).unwrap() - 1.9).abs() < 1e-12);
        assert!((percentile(&d.view(), 90.0).unwrap() - 9.1).abs() < 1e-12);

        let e = arr(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(percentile(&e.view(), 25.0).unwrap(), 2.0);
        assert_eq!(percentile(&e.view(), 50.0).unwrap(), 3.0);
        assert_eq!(percentile(&e.view(), 75.0).unwrap(), 4.0);
    }

    #[test]
    fn percentile_out_of_range_errs() {
        let a = arr(&[1.0, 2.0]);
        assert_eq!(percentile(&a.view(), 150.0), Err(ReductionError::PercentileOutOfRange { q: 150.0 }));
    }

    #[test]
    fn nan_variants_skip_nan_and_match_real_numpy() {

        let b = arr(&[1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
        assert_eq!(nansum(&b.view()), 9.0);
        assert_eq!(nanmean(&b.view()), 3.0);
        assert!((nanvar_default(&b.view()) - 2.6666666666666665).abs() < 1e-12);
        assert!((nanstd_default(&b.view()) - 1.632993161855452).abs() < 1e-12);
        assert_eq!(nanmedian(&b.view()).unwrap(), 3.0);
        assert_eq!(nanmin(&b.view()).unwrap(), 1.0);
        assert_eq!(nanmax(&b.view()).unwrap(), 5.0);
    }

    #[test]
    fn nansum_of_all_nan_is_zero_not_nan() {

        let a = arr(&[f64::NAN, f64::NAN]);
        assert_eq!(nansum(&a.view()), 0.0);
    }

    #[test]
    fn nanmean_and_friends_of_all_nan_is_nan() {

        let a = arr(&[f64::NAN, f64::NAN]);
        assert!(nanmean(&a.view()).is_nan());
        assert!(nanvar_default(&a.view()).is_nan());
        assert!(nanstd_default(&a.view()).is_nan());
        assert!(nanmedian(&a.view()).unwrap().is_nan());
        assert!(nanmin(&a.view()).unwrap().is_nan());
        assert!(nanmax(&a.view()).unwrap().is_nan());
    }

    #[test]
    fn nanmin_on_genuinely_empty_array_still_errs() {
        let a = arr(&[]);
        assert_eq!(nanmin(&a.view()), Err(ReductionError::EmptyInput));
    }

    #[test]
    fn histogram_default_range_matches_real_numpy() {

        let data = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0];
        let (counts, edges) = histogram(&data, 4, None).unwrap();
        assert_eq!(counts, vec![1, 1, 2, 2]);
        for (got, want) in edges.iter().zip([1.0, 1.75, 2.5, 3.25, 4.0]) {
            assert!((got - want).abs() < 1e-12);
        }
    }

    #[test]
    fn histogram_explicit_range_matches_real_numpy() {

        let data = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0];
        let (counts, edges) = histogram(&data, 4, Some((0.0, 4.0))).unwrap();
        assert_eq!(counts, vec![0, 1, 2, 3]);
        assert_eq!(edges, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn cov_and_corrcoef_match_real_numpy() {

        let m = NdArray::from_vec(vec![0.0, 2.0, 1.0, 3.0, 2.0, 1.0, 0.0, 3.0], &[2, 4]).unwrap();
        let c = cov_default(&m.view()).unwrap();
        assert!((c.get(&[0, 0]).unwrap() - 1.6666666666666667).abs() < 1e-9);
        assert!((c.get(&[0, 1]).unwrap() - 0.6666666666666667).abs() < 1e-9);
        assert!((c.get(&[1, 1]).unwrap() - 1.6666666666666667).abs() < 1e-9);

        let r = corrcoef(&m.view()).unwrap();
        assert!((r.get(&[0, 0]).unwrap() - 1.0).abs() < 1e-9);
        assert!((r.get(&[0, 1]).unwrap() - 0.4).abs() < 1e-9);
        assert!((r.get(&[1, 1]).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn generic_numeric_sum_min_max_preserve_integer_dtype_but_mean_promotes_to_f64() {

        let a = NdArray::from_vec(vec![1i32, 2, 3, 4, 5], &[5]).unwrap();
        let s: i32 = sum(&a.view());
        assert_eq!(s, 15);
        assert_eq!(mean(&a.view()), 3.0);
        assert_eq!(min(&a.view()).unwrap(), 1);
        assert_eq!(max(&a.view()).unwrap(), 5);
        assert_eq!(median(&a.view()).unwrap(), 3.0);

        let u = NdArray::from_vec(vec![10u64, 20, 30], &[3]).unwrap();
        let nu: u64 = nansum(&u.view());
        assert_eq!(nu, 60);
        assert_eq!(nanmean(&u.view()), 20.0);
        assert_eq!(nanmin(&u.view()).unwrap(), 10);
        assert_eq!(nanmax(&u.view()).unwrap(), 30);
    }
}
