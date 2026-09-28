use crate::view::ArrayView;

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

pub fn assert_allclose_default(actual: &ArrayView, desired: &ArrayView) -> Result<(), ArrayAssertionError> {
    assert_allclose(actual, desired, 1e-7, 0.0)
}

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

        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = NdArray::from_vec(vec![1.0, 5.0, 3.0], &[3]).unwrap();
        let err = assert_array_equal(&a.view(), &b.view()).unwrap_err();
        assert!(err.to_string().contains("Mismatched elements: 1 / 3"));
    }

    #[test]
    fn assert_allclose_default_matches_real_numpy_boundary() {

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

        assert_allclose(&a.view(), &b.view(), 0.02, 0.0).unwrap();

        assert!(assert_allclose(&a.view(), &b.view(), 0.001, 0.0).is_err());
    }

    #[test]
    fn assert_array_almost_equal_matches_real_numpy_boundary() {

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
