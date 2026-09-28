//! Step 10: reductions/statistics — `sum`/`mean`/`var`/`std`/`median`/
//! `percentile`, their `nan*` variants, `histogram`, and `cov`/`corrcoef`.
//!
//! Scope: **whole-array reductions only** — no `axis=` parameter yet.
//! Real NumPy's reductions all take an optional `axis` that collapses one
//! dimension instead of the whole array down to a scalar; that requires
//! computing a lower-rank output shape and iterating the remaining axes,
//! which is real additional machinery this step doesn't build yet (the
//! ufunc engine itself is still whole-array/elementwise-only — see
//! `ufunc.rs`). Every function here matches NumPy's *no-`axis`* behavior
//! (`arr.sum()`, not `arr.sum(axis=0)`) exactly.
//!
//! No runtime warnings anywhere, on purpose (consistent with `NumPy.md`'s
//! masked-array design note): real NumPy emits a `RuntimeWarning` for
//! things like an empty mean or `ddof >= n`, then still returns a value
//! (`NaN`). This module just returns that same value — the "something
//! unusual happened" signal lives in the value itself (`NaN`) or in an
//! `Err` for genuinely-can't-compute-anything cases (an empty array has
//! no minimum), never in a side-channel print/log.
//!
//! Every formula and edge case below — including the exact `nan*` result
//! for an all-`NaN` input, `percentile`'s linear-interpolation formula,
//! and `histogram`'s bin-edge/last-bin-inclusive behavior — was checked
//! against real NumPy 2.5.3 first, not assumed.
//!
//! Generic over `T` as of this crate's full-numeric-support work (extending step 4/10), in two different ways depending on
//! what real NumPy itself does to the *output* dtype:
//! - `sum`/`min`/`max` (and their `nan*` counterparts) **preserve** `T` —
//!   `np.array([1,2,3], dtype=np.int32).sum()` stays an `int32`, and so
//!   does this crate's `sum::<i32>`.
//! - `mean`/`var`/`std`/`median`/`percentile` (and `nan*`) always
//!   **promote to `f64`**, even for integer input — matching real NumPy's
//!   own "the default reduction dtype for these is always a float" rule
//!   (`np.array([1,2,3], dtype=np.int32).mean()` is a `float64`, not
//!   rounded/truncated `int32` division).
//!
//! `cov`/`corrcoef`/`histogram` stay `f64`-only for now (real NumPy
//! promotes their input to float64 internally too, so genericizing their
//! *signature* over `T` wouldn't change their actual arithmetic — not
//! done here since nothing yet calls them with a non-`f64` array).

use crate::view::ArrayView;

/// Whether a value is the float "not-a-number" sentinel — `false` for
/// every integer type (which have no such value), the real `is_nan()`
/// check for `f32`/`f64`. Lets [`min`]/[`max`]/the `nan*` reductions share
/// one generic implementation instead of one f64-only copy plus a
/// separate "just do a plain `PartialOrd` compare" copy for integers.
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

/// Lossless-where-possible (lossy for `i64`/`u64` magnitudes past 2^53,
/// exactly like real NumPy's own `int64 -> float64` conversion) numeric
/// widening, for the reductions that always promote their result to
/// `f64` regardless of the input's own type.
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

/// Errors from a reduction that genuinely cannot produce a value (as
/// opposed to producing `NaN`, which real NumPy does for most "unusual
/// input" cases — see the module doc comment for why those aren't `Err`
/// here either).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReductionError {
    /// `min`/`max`/`median`/`percentile`/histogram-without-a-range on an
    /// empty array — matches real NumPy's `ValueError: zero-size array to
    /// reduction operation ... which has no identity`.
    EmptyInput,
    /// `percentile`'s `q` was outside `[0, 100]` — matches real NumPy's
    /// `ValueError: Percentiles must be in the range [0, 100]`.
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

/// `arr.sum()`. Preserves `T` exactly — an `i32` array sums to an `i32`.
///
/// **Known, deliberate deviation from real NumPy**: verified real NumPy
/// 2.5.3 does *not* preserve narrow integer widths through `sum()` —
/// `np.array([1,2,3], dtype=np.int32).sum().dtype` is `int64`, not
/// `int32` (it upcasts every integer narrower than the platform's default
/// integer width to avoid the reduction silently overflowing, the same
/// "reduction default dtype" idea that makes `.mean()` always promote to
/// `float64`). Reproducing that exactly would mean `sum`'s *return type*
/// depending on its *input* type in a way plain Rust generics can't
/// express (`sum::<i32>` would need to return `i64`, a different concrete
/// type) — not just a formula difference like the `int64->float64`
/// `can_cast` quirk documented in `dtype.rs`. This project instead keeps
/// the simpler, Rust-idiomatic rule "the output type is always exactly
/// the input type", and documents the difference here rather than
/// silently diverging.
///
/// Empty input sums to `T::default()` (`0`/`0.0`,
/// the additive identity) — except for `f64`, where Rust's own `Sum for
/// f64` picks `-0.0` as that identity (IEEE-754's more precisely-signed
/// zero: `-0.0 + -0.0 == -0.0`, whereas `0.0` would flip the sign of an
/// all-`-0.0` sum), so an empty `f64` array here sums to `-0.0` where real
/// NumPy prints `0.0`. Deliberately left as-is rather than special-cased:
/// `-0.0 == 0.0` is `true` under IEEE-754, so nothing downstream can
/// observe a difference unless it specifically inspects the sign bit —
/// not worth the extra branch to paper over.
pub fn sum<T: Copy + Default + std::ops::Add<Output = T>>(view: &ArrayView<T>) -> T {
    values(view).into_iter().fold(T::default(), |acc, x| acc + x)
}

/// `arr.mean()`. Always promotes to `f64` (matches real NumPy: the mean
/// of an integer array is a `float64`, never truncated back to the
/// input's own type). Empty input is `NaN` (matches real NumPy: `0.0 / 0`).
pub fn mean<T: AsF64>(view: &ArrayView<T>) -> f64 {
    let v = values(view);
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().map(|x| x.as_f64()).sum::<f64>() / v.len() as f64
    }
}

/// `arr.var(ddof=ddof)`. `n <= ddof` (including the empty-array case,
/// `n = 0`) is `NaN` (a `0.0 / 0` division), matching real NumPy rather
/// than erroring.
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

/// [`var`] with real NumPy's own default (`ddof=0`, the *population*
/// variance).
pub fn var_default<T: AsF64>(view: &ArrayView<T>) -> f64 {
    var(view, 0)
}

/// `arr.std(ddof=ddof)`.
pub fn std<T: AsF64>(view: &ArrayView<T>, ddof: usize) -> f64 {
    var(view, ddof).sqrt()
}

/// [`std`] with `ddof=0`.
pub fn std_default<T: AsF64>(view: &ArrayView<T>) -> f64 {
    std(view, 0)
}

/// `arr.min()`. Preserves `T`. `NaN` propagates for float `T` (any `NaN`
/// present makes the result `NaN`, matching real NumPy's plain,
/// non-`nan`-prefixed `min`); integer `T` has no such value, so this is
/// just an ordinary fold by [`FloatIsh::is_nan_ish`]'s always-`false`
/// default.
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

/// `arr.max()`.
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

/// The shared engine behind [`percentile`]/[`median`]: real NumPy's exact
/// `'linear'`-method formula (its default) on an already-sorted slice —
/// `index = (n-1) * q/100`, then linearly interpolate between the values
/// at the floor and ceiling of that index.
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

/// `np.percentile(arr, q)`. `q` must be in `[0, 100]`. Always promotes to
/// `f64`, same as [`mean`]/[`var`]/[`std`].
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

/// `np.median(arr)` — exactly `percentile(arr, 50)`, the same relationship
/// real NumPy's implementation has.
pub fn median<T: AsF64>(view: &ArrayView<T>) -> Result<f64, ReductionError> {
    percentile(view, 50.0)
}

fn non_nan_values<T: FloatIsh>(view: &ArrayView<T>) -> Vec<T> {
    values(view).into_iter().filter(|x| !x.is_nan_ish()).collect()
}

/// `np.nansum(arr)` — `NaN`s are skipped entirely; an all-`NaN` (or
/// empty) input sums to `T::default()` (Rust's own `-0.0` for `f64`, in
/// practice — see [`sum`]'s comment), matching real NumPy (verified: NOT
/// `NaN`, unlike [`nanmean`]). Preserves `T`, same as [`sum`]; for
/// integer `T` this is identical to [`sum`] (there's no `NaN` to skip).
pub fn nansum<T: FloatIsh + Default + std::ops::Add<Output = T>>(view: &ArrayView<T>) -> T {
    non_nan_values(view).into_iter().fold(T::default(), |acc, x| acc + x)
}

/// `np.nanmean(arr)` — an all-`NaN` (or empty) input is `NaN`.
pub fn nanmean<T: FloatIsh + AsF64>(view: &ArrayView<T>) -> f64 {
    let v = non_nan_values(view);
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().map(|x| x.as_f64()).sum::<f64>() / v.len() as f64
    }
}

/// `np.nanvar(arr, ddof=ddof)`.
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

/// [`nanvar`] with `ddof=0`.
pub fn nanvar_default<T: FloatIsh + AsF64>(view: &ArrayView<T>) -> f64 {
    nanvar(view, 0)
}

/// `np.nanstd(arr, ddof=ddof)`.
pub fn nanstd<T: FloatIsh + AsF64>(view: &ArrayView<T>, ddof: usize) -> f64 {
    nanvar(view, ddof).sqrt()
}

/// [`nanstd`] with `ddof=0`.
pub fn nanstd_default<T: FloatIsh + AsF64>(view: &ArrayView<T>) -> f64 {
    nanstd(view, 0)
}

/// `np.nanmin(arr)`. Only the array being genuinely zero-length is an
/// `Err`; an array that's non-empty but *entirely* `NaN` returns `Ok(NaN)`
/// instead (verified against real NumPy: these are two different cases
/// with two different real-NumPy behaviors — an empty array raises, an
/// all-`NaN` array just warns and returns `NaN`). Preserves `T`.
pub fn nanmin<T: FloatIsh>(view: &ArrayView<T>) -> Result<T, ReductionError> {
    if view.is_empty() {
        return Err(ReductionError::EmptyInput);
    }
    let v = non_nan_values(view);
    let mut iter = v.into_iter();
    let Some(first) = iter.next() else {
        // Non-empty overall but every value was NaN -- NumPy's answer here
        // is NaN, not an error; the caller already established T's own
        // NaN-shaped value exists (is_nan_ish() can only ever be true for
        // a float T), so recovering one from the original view is safe.
        return Ok(values(view)[0]);
    };
    Ok(iter.fold(first, |acc, x| if x < acc { x } else { acc }))
}

/// `np.nanmax(arr)`.
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

/// `np.nanmedian(arr)`. Same empty-vs-all-`NaN` distinction as
/// [`nanmin`]/[`nanmax`]. Always promotes to `f64`.
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

/// `np.histogram(data, bins, range)`. `range` defaults to
/// `(data.min(), data.max())` when `None`, matching real NumPy. Bin edges
/// are evenly spaced (`bins + 1` of them); every bin is half-open
/// `[edge[i], edge[i+1])` **except the last, which is closed on both
/// ends** (`[edge[bins-1], edge[bins]]`) — verified against real NumPy,
/// this is why a value exactly at the overall maximum still gets counted.
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
            // Position within [low, high) scaled to [0, bins).
            (((x - low) / (high - low)) * bins as f64).floor() as usize
        };
        counts[bin.min(bins - 1)] += 1;
    }
    Ok((counts, edges))
}

/// `np.cov(matrix, ddof=ddof)` with `rowvar=True` (real NumPy's default):
/// each **row** of `matrix` is one variable, each column one observation.
/// Returns the covariance matrix as a new `NdArray` of shape
/// `(rows, rows)`.
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

/// [`cov`] with real NumPy's own default (`ddof=1`, the unbiased/sample
/// covariance).
pub fn cov_default(matrix: &ArrayView) -> Result<crate::NdArray, ReductionError> {
    cov(matrix, 1)
}

/// `np.corrcoef(matrix)` — the covariance matrix normalized so every
/// diagonal entry is `1.0`. The `ddof` used internally is irrelevant to
/// the result (it appears in, and cancels out of, both the numerator and
/// denominator of the ratio) — `cov_default`'s `ddof=1` is used purely
/// for consistency with `cov`, not because it changes the answer.
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
        // np.array([1,2,3,4,5]).var() == 2.0, .std() == 1.4142135623730951
        let a = arr(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(var_default(&a.view()), 2.0);
        assert!((std_default(&a.view()) - std::f64::consts::SQRT_2).abs() < 1e-12);
        // np.var(..., ddof=1) == 2.5
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
        // np.median([1,2,3,4,5]) == 3.0; np.median([1,2,3,4]) == 2.5
        assert_eq!(median(&arr(&[1.0, 2.0, 3.0, 4.0, 5.0]).view()).unwrap(), 3.0);
        assert_eq!(median(&arr(&[1.0, 2.0, 3.0, 4.0]).view()).unwrap(), 2.5);
    }

    #[test]
    fn percentile_matches_real_numpy_linear_interpolation() {
        // np.percentile(np.arange(1,11), 10) == 1.9; ..., 90) == 9.1
        let d = arr(&(1..=10).map(|i| i as f64).collect::<Vec<_>>());
        assert!((percentile(&d.view(), 10.0).unwrap() - 1.9).abs() < 1e-12);
        assert!((percentile(&d.view(), 90.0).unwrap() - 9.1).abs() < 1e-12);
        // np.percentile([1,2,3,4,5], 25/50/75) == 2.0/3.0/4.0
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
        // np.nansum/[nanmean/nanvar/nanstd/nanmedian]([1,NaN,3,NaN,5])
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
        // np.nansum([NaN, NaN]) == 0.0 -- verified distinct from nanmean.
        // (Rust's own Sum for f64 actually produces -0.0 here, not +0.0 --
        // see `nansum`'s doc comment -- but `-0.0 == 0.0` under IEEE-754,
        // so a plain equality check is the right test, not a sign check.)
        let a = arr(&[f64::NAN, f64::NAN]);
        assert_eq!(nansum(&a.view()), 0.0);
    }

    #[test]
    fn nanmean_and_friends_of_all_nan_is_nan() {
        // np.nanmean/nanvar/nanstd/nanmedian/nanmin/nanmax([NaN, NaN]) == NaN
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
        // np.histogram([1,2,2.5,3,3.5,4], bins=4)
        // -> counts [1,1,2,2], edges [1.,1.75,2.5,3.25,4.]
        let data = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0];
        let (counts, edges) = histogram(&data, 4, None).unwrap();
        assert_eq!(counts, vec![1, 1, 2, 2]);
        for (got, want) in edges.iter().zip([1.0, 1.75, 2.5, 3.25, 4.0]) {
            assert!((got - want).abs() < 1e-12);
        }
    }

    #[test]
    fn histogram_explicit_range_matches_real_numpy() {
        // np.histogram([1,2,2.5,3,3.5,4], bins=4, range=(0,4))
        // -> counts [0,1,2,3], edges [0,1,2,3,4]
        let data = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0];
        let (counts, edges) = histogram(&data, 4, Some((0.0, 4.0))).unwrap();
        assert_eq!(counts, vec![0, 1, 2, 3]);
        assert_eq!(edges, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn cov_and_corrcoef_match_real_numpy() {
        // np.cov([[0,2,1,3],[2,1,0,3]]) ->
        //   [[1.66666667, 0.66666667], [0.66666667, 1.66666667]]
        // np.corrcoef(...) -> [[1., 0.4], [0.4, 1.]]
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
        // np.array([1,2,3,4,5], dtype=np.int32).sum() -> 15 (still int32)
        // np.array([1,2,3,4,5], dtype=np.int32).mean() -> 3.0 (float64)
        let a = NdArray::from_vec(vec![1i32, 2, 3, 4, 5], &[5]).unwrap();
        let s: i32 = sum(&a.view());
        assert_eq!(s, 15);
        assert_eq!(mean(&a.view()), 3.0);
        assert_eq!(min(&a.view()).unwrap(), 1);
        assert_eq!(max(&a.view()).unwrap(), 5);
        assert_eq!(median(&a.view()).unwrap(), 3.0);

        // Same for unsigned and a wider width, and nan* variants (no NaN
        // to skip for an integer T, so they must agree with the plain
        // ones exactly).
        let u = NdArray::from_vec(vec![10u64, 20, 30], &[3]).unwrap();
        let nu: u64 = nansum(&u.view());
        assert_eq!(nu, 60);
        assert_eq!(nanmean(&u.view()), 20.0);
        assert_eq!(nanmin(&u.view()).unwrap(), 10);
        assert_eq!(nanmax(&u.view()).unwrap(), 30);
    }
}
