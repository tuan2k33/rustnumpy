//! Step 4: a small generic ufunc engine (NEP 5), built on the broadcasting
//! from step 1. `add`/`mul`/`sub` below aren't separate implementations —
//! they're the same `zip_with` engine called with a different closure,
//! exactly like real NumPy's ufuncs share one dispatch machinery and only
//! swap out the inner loop function (`resolve_descriptors`/`get_loop`/
//! `strided_loop` from NEP 41/42, see NumPy.md).
//!
//! Still f64-only and still missing real NumPy ufunc features on purpose:
//! no `where=`, no reduce/accumulate/outer, and no generalized core
//! dimensions (NEP 20) — those come later. What *is* here: broadcasting,
//! a closure-driven element loop, and the `out=` pattern (writing into a
//! caller-owned buffer instead of allocating a new one).

use crate::error::ShapeError;
use crate::ndarray::NdArray;
use crate::shape::{broadcast_shapes, IndexIter};
use crate::view::{ArrayView, ArrayViewMut};

/// The core binary ufunc engine: broadcast `a` and `b` to a common shape,
/// then apply `f` element by element, allocating a new `NdArray` for the
/// result.
///
/// `f` is a plain closure (`Fn(f64, f64) -> f64`), not a trait object —
/// each call site (`add`, `mul`, ...) gets its own monomorphized copy of
/// this function with `f` inlined, so the closure call has zero overhead
/// compared to hand-writing the loop body directly. This is the same
/// "generic over the operation" idea NumPy's C core reaches for function
/// pointers to get (at the cost of an indirect call it can't inline).
pub fn zip_with(
    a: &ArrayView,
    b: &ArrayView,
    f: impl Fn(f64, f64) -> f64,
) -> Result<NdArray, ShapeError> {
    let out_shape = broadcast_shapes(a.shape(), b.shape()).ok_or_else(|| ShapeError::NotBroadcastable {
        lhs: a.shape().to_vec(),
        rhs: b.shape().to_vec(),
    })?;
    let a_b = a.broadcast_to(&out_shape)?;
    let b_b = b.broadcast_to(&out_shape)?;

    let data: Vec<f64> = IndexIter::new(&out_shape)
        .map(|idx| f(a_b.get(&idx).unwrap(), b_b.get(&idx).unwrap()))
        .collect();
    NdArray::from_vec(data, &out_shape)
}

/// The `out=` variant: instead of allocating a new `NdArray`, write
/// results directly into a caller-provided `ArrayViewMut`.
///
/// Unlike NumPy's `out=`, which happily lets `out` alias one of the
/// inputs (and documents the cases where that's safe), this signature
/// makes that aliasing impossible to express: `out` is `&mut`, `a`/`b`
/// are `&`, and Rust's borrow checker won't let a `&mut ArrayViewMut` and
/// a `&ArrayView` point at overlapping memory at the same time. Real
/// NumPy has to reason about aliasing manually per ufunc; here the
/// compiler does it, at the cost of ruling out a pattern (`np.add(a, b,
/// out=a)`) that NumPy explicitly supports.
///
/// `out.shape()` must already equal the broadcast result shape — no
/// broadcasting happens on the output side, matching NumPy's own rule
/// that `out=` must have exactly the right shape.
pub fn zip_with_into(
    out: &mut ArrayViewMut,
    a: &ArrayView,
    b: &ArrayView,
    f: impl Fn(f64, f64) -> f64,
) -> Result<(), ShapeError> {
    let out_shape = broadcast_shapes(a.shape(), b.shape()).ok_or_else(|| ShapeError::NotBroadcastable {
        lhs: a.shape().to_vec(),
        rhs: b.shape().to_vec(),
    })?;
    if out.shape() != out_shape.as_slice() {
        return Err(ShapeError::DataShapeMismatch {
            data_len: out.shape().iter().product(),
            shape: out_shape,
        });
    }
    let a_b = a.broadcast_to(&out_shape)?;
    let b_b = b.broadcast_to(&out_shape)?;

    for idx in IndexIter::new(&out_shape) {
        let value = f(a_b.get(&idx).unwrap(), b_b.get(&idx).unwrap());
        out.set(&idx, value)?;
    }
    Ok(())
}

/// A unary ufunc engine: apply `f` to every element of `a`, no
/// broadcasting needed since there's only one shape involved.
pub fn map(a: &ArrayView, f: impl Fn(f64) -> f64) -> NdArray {
    let data: Vec<f64> = IndexIter::new(a.shape()).map(|idx| f(a.get(&idx).unwrap())).collect();
    NdArray::from_vec(data, a.shape()).expect("data.len() always matches a.shape().iter().product()")
}

/// `a + b`, broadcasting like NumPy.
pub fn add(a: &ArrayView, b: &ArrayView) -> Result<NdArray, ShapeError> {
    zip_with(a, b, |x, y| x + y)
}

/// `a - b`, broadcasting like NumPy.
pub fn sub(a: &ArrayView, b: &ArrayView) -> Result<NdArray, ShapeError> {
    zip_with(a, b, |x, y| x - y)
}

/// `a * b`, broadcasting like NumPy.
pub fn mul(a: &ArrayView, b: &ArrayView) -> Result<NdArray, ShapeError> {
    zip_with(a, b, |x, y| x * y)
}

/// Kept as the original step-1 name so earlier examples/tests don't need
/// to change; it's now just `add` under the hood.
pub fn add_broadcast(a: &ArrayView, b: &ArrayView) -> Result<NdArray, ShapeError> {
    add(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray::NdArray;

    #[test]
    fn add_same_shape() {
        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = NdArray::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        let out = add(&a.view(), &b.view()).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.get(&[0, 0]), Some(11.0));
        assert_eq!(out.get(&[1, 1]), Some(44.0));
    }

    #[test]
    fn add_broadcasts_row_vector_over_matrix() {
        // (2,3) + (3,) -> each row adds the same vector, just like NumPy
        let a = NdArray::from_vec(vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0], &[2, 3]).unwrap();
        let b = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let out = add(&a.view(), &b.view()).unwrap();
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
        assert!(add(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn mul_broadcasts_column_vector_over_matrix() {
        // (3,1) * (3,4): np.array([[1],[2],[3]]) * np.ones((3,4))
        let col = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
        let ones = NdArray::from_vec(vec![1.0; 12], &[3, 4]).unwrap();
        let out = mul(&col.view(), &ones.view()).unwrap();
        assert_eq!(out.shape(), &[3, 4]);
        for c in 0..4 {
            assert_eq!(out.get(&[0, c]), Some(1.0));
            assert_eq!(out.get(&[1, c]), Some(2.0));
            assert_eq!(out.get(&[2, c]), Some(3.0));
        }
    }

    #[test]
    fn sub_same_shape() {
        let a = NdArray::from_vec(vec![5.0, 5.0], &[2]).unwrap();
        let b = NdArray::from_vec(vec![2.0, 3.0], &[2]).unwrap();
        let out = sub(&a.view(), &b.view()).unwrap();
        assert_eq!(out.as_slice(), &[3.0, 2.0]);
    }

    #[test]
    fn map_applies_a_unary_closure_elementwise() {
        let a = NdArray::from_vec(vec![1.0, 4.0, 9.0, 16.0], &[4]).unwrap();
        let roots = map(&a.view(), f64::sqrt);
        assert_eq!(roots.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn zip_with_into_writes_through_out_param() {
        // Same result as add(), but written into a pre-allocated buffer —
        // this is the `out=` pattern (`np.add(a, b, out=result)`).
        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = NdArray::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        let mut out = NdArray::zeros(&[2, 2]);
        zip_with_into(&mut out.view_mut(), &a.view(), &b.view(), |x, y| x + y).unwrap();
        assert_eq!(out.as_slice(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn zip_with_into_rejects_wrong_output_shape() {
        let a = NdArray::zeros(&[2, 2]);
        let b = NdArray::zeros(&[2, 2]);
        let mut out = NdArray::zeros(&[3, 3]); // wrong shape on purpose
        let err = zip_with_into(&mut out.view_mut(), &a.view(), &b.view(), |x, y| x + y).unwrap_err();
        assert!(matches!(err, ShapeError::DataShapeMismatch { .. }));
    }
}
