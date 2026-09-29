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
}

impl std::fmt::Display for ReductionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReductionError::EmptyInput => write!(f, "zero-size array has no reduction identity"),
            ReductionError::PercentileOutOfRange { q } => {
                write!(f, "percentile {q} must be in the range [0, 100]")
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

fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let index = (n - 1) as f64 * q / 100.0;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }
    let frac = index - lower as f64;
    sorted[lower] + frac * (sorted[upper] - sorted[lower])
}

pub fn percentile<T: AsF64>(view: &ArrayView<T>, q: f64) -> Result<f64, ReductionError> {
    if !(0.0..=100.0).contains(&q) {
        return Err(ReductionError::PercentileOutOfRange { q });
    }
    let mut v: Vec<f64> = values(view).into_iter().map(|x| x.as_f64()).collect();
    if v.is_empty() {
        return Err(ReductionError::EmptyInput);
    }
    v.sort_unstable_by(|a, b| a.total_cmp(b));
    Ok(percentile_sorted(&v, q))
}

pub fn median<T: AsF64>(view: &ArrayView<T>) -> Result<f64, ReductionError> {
    percentile(view, 50.0)
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
    Ok(percentile_sorted(&v, 50.0))
}

pub fn histogram(data: &[f64], bins: usize, range: Option<(f64, f64)>) -> Result<(Vec<usize>, Vec<f64>), ReductionError> {
    let (low, high) = match range {
        Some(r) => r,
        None => {
            if data.is_empty() {
                return Err(ReductionError::EmptyInput);
            }
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &x in data {
                lo = lo.min(x);
                hi = hi.max(x);
            }
            (lo, hi)
        }
    };
    let edges: Vec<f64> = (0..=bins).map(|i| low + (high - low) * i as f64 / bins as f64).collect();
    let mut counts = vec![0usize; bins];
    for &x in data {
        if x < low || x > high {
            continue;
        }
        let bin = if x == high {
            bins - 1
        } else {

            (((x - low) / (high - low)) * bins as f64).floor() as usize
        };
        counts[bin.min(bins - 1)] += 1;
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
            out[i * rows + j] = covariance.get(&[i, j]).unwrap() / (std_devs[i] * std_devs[j]);
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
