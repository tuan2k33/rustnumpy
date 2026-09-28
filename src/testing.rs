//! Step 11: a `numpy.testing` equivalent — enough of `assert_array_equal`/
//! `assert_allclose`/`assert_array_almost_equal` to write this project's
//! own tests without any of them depending on a running NumPy.
//!
//! Scope: a representative subset, not the whole namespace. Left out on
//! purpose:
//! - `assert_warns`/`suppress_warnings` — already tracked in `NumPy.md`'s
//!   deprecation list (NumPy >= 2.5 itself deprecates these in favor of
//!   the standard `warnings` module), and Rust has no runtime warning
//!   system to mirror in the first place.
//! - `assert_raises`/`assert_string_equal`/etc. — Rust already has
//!   idiomatic equivalents for these (`#[should_panic]`, `assert_eq!` on
//!   `&str`) that don't need an array-specific helper.
//! - Custom comparison functions (`assert_array_compare`'s general form)
//!   — this module hardcodes the comparison NumPy itself hardcodes for
//!   each named function, rather than exposing the generic machinery
//!   underneath.
//!
//! Every default and piece of message-formatting behavior below was
//! checked against real NumPy 2.5.3 first, not assumed from memory —
//! including two easy-to-get-wrong details: `assert_array_equal` treats
//! `NaN == NaN` as **equal** (it's a testing tool, not `==`), and
//! `assert_allclose`'s default tolerance (`rtol=1e-7`, `atol=0`) is
//! **not** the same as `np.isclose`'s default (`rtol=1e-5`, `atol=1e-8`)
//! — a distinction worth being explicit about since mixing them up
//! silently makes tests either too strict or too loose.

use crate::view::ArrayView;

/// A failed array assertion — `Display` renders a message shaped like
/// (not byte-for-byte identical to, but structurally the same
/// information as) real NumPy's own `AssertionError` text: what failed,
/// how many elements mismatched, and the worst offender.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrayAssertionError {
    message: String,
}

impl std::fmt::Display for ArrayAssertionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ArrayAssertionError {}

fn shape_mismatch(what: &str, actual: &ArrayView, desired: &ArrayView) -> ArrayAssertionError {
    ArrayAssertionError {
        message: format!(
            "{what}\n\n(shapes {:?}, {:?} mismatch)\n ACTUAL: {:?}\n DESIRED: {:?}",
            actual.shape(),
            desired.shape(),
            actual.to_owned().as_slice(),
            desired.to_owned().as_slice(),
        ),
    }
}

/// Walk both arrays in lock-step (already shape-checked by the caller)
/// and collect every index where `is_mismatch` says the pair fails,
/// alongside the worst absolute/relative difference seen — the shared
/// engine behind every assertion below, since they only differ in what
/// counts as "close enough" and how the failure message is worded.
struct Mismatch {
    count: usize,
    total: usize,
    max_abs_diff: f64,
    max_rel_diff: f64,
    first_index: Vec<usize>,
    first_actual: f64,
    first_desired: f64,
}

fn scan(
    actual: &ArrayView,
    desired: &ArrayView,
    is_mismatch: impl Fn(f64, f64) -> bool,
) -> Option<Mismatch> {
    let mut count = 0;
    let mut total = 0;
    let mut max_abs_diff = 0.0f64;
    let mut max_rel_diff = 0.0f64;
    let mut first: Option<(Vec<usize>, f64, f64)> = None;

    for index in crate::shape::IndexIter::new(actual.shape()) {
        total += 1;
        let a = actual.get(&index).expect("IndexIter only yields valid indices");
        let d = desired.get(&index).expect("same shape as actual, already checked by the caller");
        if is_mismatch(a, d) {
            count += 1;
            let abs_diff = (a - d).abs();
            let rel_diff = if d != 0.0 { abs_diff / d.abs() } else { f64::INFINITY };
            max_abs_diff = max_abs_diff.max(abs_diff);
            max_rel_diff = max_rel_diff.max(rel_diff);
            if first.is_none() {
                first = Some((index, a, d));
            }
        }
    }

    if count == 0 {
        return None;
    }
    let (first_index, first_actual, first_desired) = first.expect("count > 0 implies first is Some");
    Some(Mismatch { count, total, max_abs_diff, max_rel_diff, first_index, first_actual, first_desired })
}

fn mismatch_message(header: &str, m: &Mismatch, actual: &ArrayView, desired: &ArrayView) -> String {
    let pct = 100.0 * m.count as f64 / m.total as f64;
    format!(
        "{header}\n\nMismatched elements: {} / {} ({pct:.3}%)\nMismatch at index:\n {:?}: {} (ACTUAL), {} (DESIRED)\nMax absolute difference among violations: {:e}\nMax relative difference among violations: {:e}\n ACTUAL: {:?}\n DESIRED: {:?}",
        m.count,
        m.total,
        m.first_index,
        m.first_actual,
        m.first_desired,
        m.max_abs_diff,
        m.max_rel_diff,
        actual.to_owned().as_slice(),
        desired.to_owned().as_slice(),
    )
}

/// `np.testing.assert_array_equal(actual, desired)`: every element must
/// be bit-for-bit equal, **except** that `NaN == NaN` counts as a match
/// (verified against real NumPy: this is a testing tool checking "are
/// these the values you expected", not IEEE-754 `==`, so two arrays that
/// both put `NaN` in the same place are considered equal here).
pub fn assert_array_equal(actual: &ArrayView, desired: &ArrayView) -> Result<(), ArrayAssertionError> {
    if actual.shape() != desired.shape() {
        return Err(shape_mismatch("Arrays are not equal", actual, desired));
    }
    let is_mismatch = |a: f64, d: f64| !(a == d || (a.is_nan() && d.is_nan()));
    match scan(actual, desired, is_mismatch) {
        None => Ok(()),
        Some(m) => Err(ArrayAssertionError { message: mismatch_message("Arrays are not equal", &m, actual, desired) }),
    }
}

/// `np.testing.assert_allclose(actual, desired, rtol, atol)`: every
/// element must satisfy `|actual - desired| <= atol + rtol * |desired|`
/// (the same formula real NumPy uses), with `NaN == NaN` also counting
/// as a match (matches `equal_nan=True`, `assert_allclose`'s own
/// default).
pub fn assert_allclose(
    actual: &ArrayView,
    desired: &ArrayView,
    rtol: f64,
    atol: f64,
) -> Result<(), ArrayAssertionError> {
    if actual.shape() != desired.shape() {
        return Err(shape_mismatch("Not equal to tolerance", actual, desired));
    }
    let is_mismatch = |a: f64, d: f64| {
        if a.is_nan() && d.is_nan() {
            false
        } else {
            (a - d).abs() > atol + rtol * d.abs()
        }
    };
    match scan(actual, desired, is_mismatch) {
        None => Ok(()),
        Some(m) => Err(ArrayAssertionError {
            message: mismatch_message(&format!("Not equal to tolerance rtol={rtol:e}, atol={atol:e}"), &m, actual, desired),
        }),
    }
}

/// [`assert_allclose`] with real NumPy's own default tolerance
/// (`rtol=1e-7`, `atol=0`) — **not** the same default as `np.isclose`
/// (`rtol=1e-5`, `atol=1e-8`); real NumPy's two functions genuinely
/// disagree on this, so this module names the default explicitly rather
/// than silently picking one.
pub fn assert_allclose_default(actual: &ArrayView, desired: &ArrayView) -> Result<(), ArrayAssertionError> {
    assert_allclose(actual, desired, 1e-7, 0.0)
}

/// `np.testing.assert_array_almost_equal(actual, desired, decimal)`:
/// every element must satisfy `|actual - desired| < 1.5 * 10^(-decimal)`
/// (real NumPy's exact formula, including the `1.5` and the strict `<`
/// — verified at the boundary: a `1.4e-6` difference passes at
/// `decimal=6`, a `1.6e-6` difference fails).
pub fn assert_array_almost_equal(
    actual: &ArrayView,
    desired: &ArrayView,
    decimal: i32,
) -> Result<(), ArrayAssertionError> {
    if actual.shape() != desired.shape() {
        return Err(shape_mismatch("Arrays are not almost equal", actual, desired));
    }
    let threshold = 1.5 * 10f64.powi(-decimal);
    let is_mismatch = |a: f64, d: f64| !(a.is_nan() && d.is_nan()) && (a - d).abs() >= threshold;
    match scan(actual, desired, is_mismatch) {
        None => Ok(()),
        Some(m) => Err(ArrayAssertionError {
            message: mismatch_message(
                &format!("Arrays are not almost equal to {decimal} decimals"),
                &m,
                actual,
                desired,
            ),
        }),
    }
}

/// [`assert_array_almost_equal`] with real NumPy's own default
/// (`decimal=6`).
pub fn assert_array_almost_equal_default(actual: &ArrayView, desired: &ArrayView) -> Result<(), ArrayAssertionError> {
    assert_array_almost_equal(actual, desired, 6)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArray;

    #[test]
    fn assert_array_equal_passes_on_identical_arrays() {
        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        assert_array_equal(&a.view(), &b.view()).unwrap();
    }

    #[test]
    fn assert_array_equal_treats_nan_as_equal_to_nan() {
        // np.testing.assert_array_equal([1,NaN,3],[1,NaN,3]) passes
        let a = NdArray::from_vec(vec![1.0, f64::NAN, 3.0], &[3]).unwrap();
        let b = NdArray::from_vec(vec![1.0, f64::NAN, 3.0], &[3]).unwrap();
        assert_array_equal(&a.view(), &b.view()).unwrap();
    }

    #[test]
    fn assert_array_equal_fails_on_shape_mismatch() {
        let a = NdArray::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let b = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let err = assert_array_equal(&a.view(), &b.view()).unwrap_err();
        assert!(err.to_string().contains("shapes [2], [3] mismatch"));
    }

    #[test]
    fn assert_array_equal_fails_on_value_mismatch() {
        // np.testing.assert_array_equal([1,2,3],[1,5,3]) -> 1/3 mismatched
        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = NdArray::from_vec(vec![1.0, 5.0, 3.0], &[3]).unwrap();
        let err = assert_array_equal(&a.view(), &b.view()).unwrap_err();
        assert!(err.to_string().contains("Mismatched elements: 1 / 3"));
    }

    #[test]
    fn assert_allclose_default_matches_real_numpy_boundary() {
        // np.testing.assert_allclose(1.0, 1.0 + 1e-6) fails (default
        // rtol=1e-7 is too tight); np.testing.assert_allclose(1.0, 1.0 + 1e-8) passes.
        let a = NdArray::from_vec(vec![1.0], &[1]).unwrap();
        let b_close = NdArray::from_vec(vec![1.0 + 1e-8], &[1]).unwrap();
        let b_far = NdArray::from_vec(vec![1.0 + 1e-6], &[1]).unwrap();
        assert_allclose_default(&a.view(), &b_close.view()).unwrap();
        assert!(assert_allclose_default(&a.view(), &b_far.view()).is_err());
    }

    #[test]
    fn assert_allclose_treats_nan_as_equal() {
        let a = NdArray::from_vec(vec![f64::NAN], &[1]).unwrap();
        let b = NdArray::from_vec(vec![f64::NAN], &[1]).unwrap();
        assert_allclose_default(&a.view(), &b.view()).unwrap();
    }

    #[test]
    fn assert_allclose_custom_tolerance() {
        let a = NdArray::from_vec(vec![100.0], &[1]).unwrap();
        let b = NdArray::from_vec(vec![101.0], &[1]).unwrap();
        // |100-101| = 1 <= atol(0) + rtol(0.02)*|101| = 2.02 -> passes
        assert_allclose(&a.view(), &b.view(), 0.02, 0.0).unwrap();
        // |100-101| = 1 > atol(0) + rtol(0.001)*|101| = 0.101 -> fails
        assert!(assert_allclose(&a.view(), &b.view(), 0.001, 0.0).is_err());
    }

    #[test]
    fn assert_array_almost_equal_matches_real_numpy_boundary() {
        // np.testing.assert_array_almost_equal(1.0, 1.0+1.4e-6, decimal=6) passes
        // np.testing.assert_array_almost_equal(1.0, 1.0+1.6e-6, decimal=6) fails
        let a = NdArray::from_vec(vec![1.0], &[1]).unwrap();
        let close = NdArray::from_vec(vec![1.0 + 1.4e-6], &[1]).unwrap();
        let far = NdArray::from_vec(vec![1.0 + 1.6e-6], &[1]).unwrap();
        assert_array_almost_equal(&a.view(), &close.view(), 6).unwrap();
        assert!(assert_array_almost_equal(&a.view(), &far.view(), 6).is_err());
    }

    #[test]
    fn assert_array_almost_equal_default_uses_decimal_6() {
        let a = NdArray::from_vec(vec![1.0], &[1]).unwrap();
        let close = NdArray::from_vec(vec![1.0 + 1.4e-6], &[1]).unwrap();
        assert_array_almost_equal_default(&a.view(), &close.view()).unwrap();
    }
}
