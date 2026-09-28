//! Step 4: a small generic ufunc engine (NEP 5), built on the broadcasting
//! from step 1. `add`/`mul`/`sub` below aren't separate implementations —
//! they're the same `zip_with` engine called with a different closure,
//! exactly like real NumPy's ufuncs share one dispatch machinery and only
//! swap out the inner loop function (`resolve_descriptors`/`get_loop`/
//! `strided_loop` from NEP 41/42, see NumPy.md).
//!
//! Generic over the element type `T` as of step 20 (`add`/`sub`/`mul` work
//! on any numeric `T` the standard arithmetic traits are implemented for —
//! `i8`..`i64`, `u8`..`u64`, `f32`/`f64`, `Complex<f32>`/`Complex<f64>` all
//! monomorphize cleanly), still missing real NumPy ufunc features on
//! purpose: no `where=`, no reduce/accumulate/outer, and no generalized
//! core dimensions (NEP 20) — those come later. What *is* here:
//! broadcasting, a closure-driven element loop, and the `out=` pattern
//! (writing into a caller-owned buffer instead of allocating a new one).
//!
//! Note what genericizing `add`/`sub`/`mul` deliberately does *not* do:
//! it does not implement NEP 50's promotion (`i32 + f64 -> f64`) — every
//! call still needs both operands to already be the *same* concrete `T`,
//! the same way `zip_with`'s closure `Fn(T, T) -> T` always has. Mixed-type
//! promotion (`common_dtype`/`can_cast` in `dtype.rs`) is a runtime-`Kind`
//! decision about *which* concrete type a value should be converted to;
//! actually performing that conversion and dispatching to the right
//! monomorphized instance is a separate, harder problem (real NumPy's
//! `resolve_descriptors`/`get_loop`) not solved here.

use crate::error::ShapeError;
use crate::ndarray::NdArray;
use crate::shape::{broadcast_shapes, unravel_index, IndexIter};
use crate::view::{ArrayView, ArrayViewMut};
use rayon::prelude::*;

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
pub fn zip_with<T: Copy>(
    a: &ArrayView<T>,
    b: &ArrayView<T>,
    f: impl Fn(T, T) -> T,
) -> Result<NdArray<T>, ShapeError> {
    let out_shape = broadcast_shapes(a.shape(), b.shape()).ok_or_else(|| ShapeError::NotBroadcastable {
        lhs: a.shape().to_vec(),
        rhs: b.shape().to_vec(),
    })?;
    let a_b = a.broadcast_to(&out_shape)?;
    let b_b = b.broadcast_to(&out_shape)?;

    let data: Vec<T> = IndexIter::new(&out_shape)
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
pub fn zip_with_into<T: Copy>(
    out: &mut ArrayViewMut<T>,
    a: &ArrayView<T>,
    b: &ArrayView<T>,
    f: impl Fn(T, T) -> T,
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
pub fn map<T: Copy>(a: &ArrayView<T>, f: impl Fn(T) -> T) -> NdArray<T> {
    let data: Vec<T> = IndexIter::new(a.shape()).map(|idx| f(a.get(&idx).unwrap())).collect();
    NdArray::from_vec(data, a.shape()).expect("data.len() always matches a.shape().iter().product()")
}

/// Step 7: the same binary ufunc engine as [`zip_with`], but the element
/// loop runs across a Rayon thread pool instead of one thread.
///
/// This is real NumPy's biggest structural weakness, and this project's
/// clearest chance to actually beat it: outside of BLAS-backed `linalg`
/// calls, NumPy's own ufunc loops are single-threaded C — there's no
/// `rayon` equivalent wired into `np.add`. Rust gets safe, easy
/// parallelism almost for free here specifically *because* `ArrayView` is
/// `Send + Sync` by construction (it only ever hands out `f64` by value
/// from `get`, never a reference into the buffer), so splitting the index
/// range across threads has nothing to race on.
///
/// `IndexIter` (used by the sequential [`zip_with`]) can't be parallelized
/// directly — Rayon's work-stealing needs to jump straight to a given
/// flat index without walking every index before it, which is exactly
/// what [`unravel_index`] provides and a plain sequential `Iterator`
/// doesn't. So the parallel path walks `0..len` through Rayon and calls
/// `unravel_index` per element instead of reusing `IndexIter`.
///
/// `f` needs `Sync` here (on top of `zip_with`'s plain `Fn`) since the
/// *same* closure value is called concurrently from multiple threads —
/// this bound is Rayon (and Rust's data-race prevention) surfacing
/// directly in the function signature, not something you could forget to
/// handle and only find out about at runtime the way a data race in C
/// would show up as an intermittent, unreproducible bug.
///
/// There's no threshold here that falls back to sequential for small
/// arrays — see `examples/step7_rayon.rs` for why that matters and what a
/// real threshold would need to account for.
///
/// Known cost this shares with [`zip_with`], worth being honest about
/// rather than hiding: every element still gets its multi-index as a
/// freshly heap-allocated `Vec<usize>` (from [`unravel_index`], the same
/// way [`zip_with`] gets one from `IndexIter` per step) — one allocation
/// and one deallocation per element, on top of the actual arithmetic.
/// Sequentially that's already wasteful; here it's worse, because the
/// global allocator is one shared, lock-contended resource that every
/// thread has to fight over, which is exactly why `examples/step7_rayon.rs`
/// measures well under the 16x speedup 16 cores would suggest — this is
/// the concrete, measured reason NEP 10's real `NpyIter` design (cache-
/// coherent traversal, no repeated allocation) is worth having, not an
/// abstract one.
pub fn zip_with_parallel<T: Copy + Send + Sync>(
    a: &ArrayView<T>,
    b: &ArrayView<T>,
    f: impl Fn(T, T) -> T + Sync,
) -> Result<NdArray<T>, ShapeError> {
    let out_shape = broadcast_shapes(a.shape(), b.shape()).ok_or_else(|| ShapeError::NotBroadcastable {
        lhs: a.shape().to_vec(),
        rhs: b.shape().to_vec(),
    })?;
    let a_b = a.broadcast_to(&out_shape)?;
    let b_b = b.broadcast_to(&out_shape)?;
    let len: usize = out_shape.iter().product();

    let data: Vec<T> = (0..len)
        .into_par_iter()
        .map(|flat| {
            let idx = unravel_index(flat, &out_shape);
            f(a_b.get(&idx).unwrap(), b_b.get(&idx).unwrap())
        })
        .collect();
    NdArray::from_vec(data, &out_shape)
}

/// The parallel counterpart to [`map`], for the same reason
/// [`zip_with_parallel`] exists alongside [`zip_with`].
pub fn map_parallel<T: Copy + Send + Sync>(a: &ArrayView<T>, f: impl Fn(T) -> T + Sync) -> NdArray<T> {
    let shape = a.shape();
    let len: usize = shape.iter().product();
    let data: Vec<T> = (0..len)
        .into_par_iter()
        .map(|flat| f(a.get(&unravel_index(flat, shape)).unwrap()))
        .collect();
    NdArray::from_vec(data, shape).expect("data.len() always matches shape.iter().product()")
}

/// `a + b`, broadcasting like NumPy. Generic over any `T` with `Add` (every
/// built-in numeric type this crate's `DType` covers: `i8`..`i64`,
/// `u8`..`u64`, `f32`/`f64`, `Complex<f32>`/`Complex<f64>`).
pub fn add<T: Copy + std::ops::Add<Output = T>>(
    a: &ArrayView<T>,
    b: &ArrayView<T>,
) -> Result<NdArray<T>, ShapeError> {
    zip_with(a, b, |x, y| x + y)
}

/// `a - b`, broadcasting like NumPy.
///
/// Named `sub` (not the Python Array API standard's own `subtract`) since
/// it predates step 18's audit against that standard — [`subtract`] is
/// the standard-aligned name added there, a thin wrapper around this.
pub fn sub<T: Copy + std::ops::Sub<Output = T>>(
    a: &ArrayView<T>,
    b: &ArrayView<T>,
) -> Result<NdArray<T>, ShapeError> {
    zip_with(a, b, |x, y| x - y)
}

/// `a * b`, broadcasting like NumPy.
///
/// Named `mul` (not the Python Array API standard's own `multiply`) for
/// the same reason [`sub`] is — see [`multiply`].
pub fn mul<T: Copy + std::ops::Mul<Output = T>>(
    a: &ArrayView<T>,
    b: &ArrayView<T>,
) -> Result<NdArray<T>, ShapeError> {
    zip_with(a, b, |x, y| x * y)
}

/// Step 18 (NEP 56 / Array API standard v2022.12) audit: the standard's
/// own elementwise function is named `subtract`, not `sub` — this crate
/// kept `sub` from step 4 for its own history's sake, so `subtract` is
/// added as the standard-named entry point instead of a disruptive rename
/// (matching the precedent [`add_broadcast`] already set: an old name
/// stays callable, the new name is the one the docs point newcomers at).
pub fn subtract<T: Copy + std::ops::Sub<Output = T>>(
    a: &ArrayView<T>,
    b: &ArrayView<T>,
) -> Result<NdArray<T>, ShapeError> {
    sub(a, b)
}

/// See [`subtract`] — the standard's own name for [`mul`].
pub fn multiply<T: Copy + std::ops::Mul<Output = T>>(
    a: &ArrayView<T>,
    b: &ArrayView<T>,
) -> Result<NdArray<T>, ShapeError> {
    mul(a, b)
}

/// The Rayon-parallel version of [`add`].
pub fn add_parallel<T: Copy + Send + Sync + std::ops::Add<Output = T>>(
    a: &ArrayView<T>,
    b: &ArrayView<T>,
) -> Result<NdArray<T>, ShapeError> {
    zip_with_parallel(a, b, |x, y| x + y)
}

/// The Rayon-parallel version of [`mul`].
pub fn mul_parallel<T: Copy + Send + Sync + std::ops::Mul<Output = T>>(
    a: &ArrayView<T>,
    b: &ArrayView<T>,
) -> Result<NdArray<T>, ShapeError> {
    zip_with_parallel(a, b, |x, y| x * y)
}

/// Kept as the original step-1 name so earlier examples/tests don't need
/// to change; it's now just `add` under the hood.
pub fn add_broadcast<T: Copy + std::ops::Add<Output = T>>(
    a: &ArrayView<T>,
    b: &ArrayView<T>,
) -> Result<NdArray<T>, ShapeError> {
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
    fn step_20_generic_ufunc_engine_works_on_non_f64_numeric_types() {
        // The point of genericizing add/sub/mul: this is not `NdArray<f64>`
        // anywhere below, but the exact same `zip_with` engine as every
        // f64 test above -- monomorphized fresh per concrete `T`.
        let a = NdArray::from_vec(vec![1i32, 2, 3, 4], &[2, 2]).unwrap();
        let b = NdArray::from_vec(vec![10i32, 20, 30, 40], &[2, 2]).unwrap();
        assert_eq!(add(&a.view(), &b.view()).unwrap().as_slice(), &[11, 22, 33, 44]);
        assert_eq!(sub(&b.view(), &a.view()).unwrap().as_slice(), &[9, 18, 27, 36]);
        assert_eq!(mul(&a.view(), &b.view()).unwrap().as_slice(), &[10, 40, 90, 160]);

        let ua = NdArray::from_vec(vec![1u64, 2, 3], &[3]).unwrap();
        let ub = NdArray::from_vec(vec![100u64, 200, 300], &[3]).unwrap();
        assert_eq!(add(&ua.view(), &ub.view()).unwrap().as_slice(), &[101, 202, 303]);

        let ca = NdArray::from_vec(
            vec![crate::fft::Complex64::new(1.0, 1.0), crate::fft::Complex64::new(2.0, 0.0)],
            &[2],
        )
        .unwrap();
        let cb = NdArray::from_vec(
            vec![crate::fft::Complex64::new(0.0, 1.0), crate::fft::Complex64::new(1.0, 1.0)],
            &[2],
        )
        .unwrap();
        assert_eq!(
            add(&ca.view(), &cb.view()).unwrap().as_slice(),
            &[crate::fft::Complex64::new(1.0, 2.0), crate::fft::Complex64::new(3.0, 1.0)]
        );

        // add_parallel/mul_parallel too -- T just needs Send + Sync on top,
        // which every one of these already satisfies.
        assert_eq!(add_parallel(&a.view(), &b.view()).unwrap(), add(&a.view(), &b.view()).unwrap());
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
        let a: NdArray = NdArray::zeros(&[2, 3]);
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
    fn subtract_and_multiply_are_the_array_api_standard_names_for_sub_and_mul() {
        let a = NdArray::from_vec(vec![5.0, 5.0], &[2]).unwrap();
        let b = NdArray::from_vec(vec![2.0, 3.0], &[2]).unwrap();
        assert_eq!(
            subtract(&a.view(), &b.view()).unwrap(),
            sub(&a.view(), &b.view()).unwrap()
        );
        assert_eq!(
            multiply(&a.view(), &b.view()).unwrap(),
            mul(&a.view(), &b.view()).unwrap()
        );
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
        let a: NdArray = NdArray::zeros(&[2, 2]);
        let b = NdArray::zeros(&[2, 2]);
        let mut out = NdArray::zeros(&[3, 3]); // wrong shape on purpose
        let err = zip_with_into(&mut out.view_mut(), &a.view(), &b.view(), |x, y| x + y).unwrap_err();
        assert!(matches!(err, ShapeError::DataShapeMismatch { .. }));
    }

    #[test]
    fn add_parallel_matches_sequential_add() {
        // 10_000 elements: small enough to run fast in a unit test, large
        // enough to actually get split across more than one Rayon thread
        // on this machine (16 cores) rather than trivially running on one.
        let data_a: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
        let data_b: Vec<f64> = (0..10_000).map(|i| (i as f64) * 0.5).collect();
        let a = NdArray::from_vec(data_a, &[100, 100]).unwrap();
        let b = NdArray::from_vec(data_b, &[100, 100]).unwrap();

        let sequential = add(&a.view(), &b.view()).unwrap();
        let parallel = add_parallel(&a.view(), &b.view()).unwrap();
        assert_eq!(sequential, parallel);
    }

    #[test]
    fn mul_parallel_matches_sequential_mul_with_broadcasting() {
        // Same (3,1)*(3,4) broadcast as mul_broadcasts_column_vector_over_matrix,
        // just checked against the parallel path instead of hardcoded values.
        let col = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
        let ones = NdArray::from_vec(vec![1.0; 12], &[3, 4]).unwrap();
        let sequential = mul(&col.view(), &ones.view()).unwrap();
        let parallel = mul_parallel(&col.view(), &ones.view()).unwrap();
        assert_eq!(sequential, parallel);
    }

    #[test]
    fn map_parallel_matches_sequential_map() {
        let data: Vec<f64> = (0..5_000).map(|i| i as f64 * 0.1).collect();
        let a = NdArray::from_vec(data, &[5_000]).unwrap();
        let sequential = map(&a.view(), f64::sqrt);
        let parallel = map_parallel(&a.view(), f64::sqrt);
        assert_eq!(sequential, parallel);
    }

    #[test]
    fn zip_with_parallel_propagates_shape_errors() {
        let a: NdArray = NdArray::zeros(&[2, 3]);
        let b = NdArray::zeros(&[4]);
        assert!(add_parallel(&a.view(), &b.view()).is_err());
    }
}
