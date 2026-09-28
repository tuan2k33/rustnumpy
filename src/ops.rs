//! The first element-wise operation: adding two arrays with broadcasting.
//!
//! This is a mini version of `np.add` — not yet a real ufunc (no dtype
//! dispatch, no `out=`/`where=`), just enough to prove shape → broadcast
//! → element walk works correctly, laying the ground for step 4 (a
//! general ufunc) in the plan.

use crate::error::ShapeError;
use crate::ndarray::NdArray;
use crate::shape::{broadcast_shapes, IndexIter};
use crate::view::ArrayView;

/// `result[idx] = a[idx] + b[idx]` after broadcasting `a` and `b` to a
/// common shape — exactly like NumPy's `a + b` when both are real arrays
/// (not accounting for the Python scalar "weak type" case from NEP 50 yet).
pub fn add_broadcast(a: &ArrayView, b: &ArrayView) -> Result<NdArray, ShapeError> {
    let out_shape = broadcast_shapes(a.shape(), b.shape()).ok_or_else(|| ShapeError::NotBroadcastable {
        lhs: a.shape().to_vec(),
        rhs: b.shape().to_vec(),
    })?;
    let a_b = a.broadcast_to(&out_shape)?;
    let b_b = b.broadcast_to(&out_shape)?;

    let data: Vec<f64> = IndexIter::new(&out_shape)
        .map(|idx| a_b.get(&idx).unwrap() + b_b.get(&idx).unwrap())
        .collect();
    NdArray::from_vec(data, &out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray::NdArray;

    #[test]
    fn add_same_shape() {
        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = NdArray::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        let out = add_broadcast(&a.view(), &b.view()).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.get(&[0, 0]), Some(11.0));
        assert_eq!(out.get(&[1, 1]), Some(44.0));
    }

    #[test]
    fn add_broadcasts_row_vector_over_matrix() {
        // (2,3) + (3,) -> each row adds the same vector, just like NumPy
        let a = NdArray::from_vec(vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0], &[2, 3]).unwrap();
        let b = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let out = add_broadcast(&a.view(), &b.view()).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.get(&[0, 0]), Some(1.0));
        assert_eq!(out.get(&[0, 2]), Some(3.0));
        assert_eq!(out.get(&[1, 0]), Some(11.0));
        assert_eq!(out.get(&[1, 2]), Some(13.0));
    }

    #[test]
    fn add_incompatible_shapes_errs() {
        let a = NdArray::zeros(&[2, 3]);
        let b = NdArray::zeros(&[4]);
        assert!(add_broadcast(&a.view(), &b.view()).is_err());
    }
}
