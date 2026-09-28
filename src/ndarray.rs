//! `NdArray<T>`: an owned N-dimensional array, generic over its element
//! type `T` — defaulted to `f64` (`NdArray<T = f64>`) so every existing
//! piece of this crate that writes the bare name `NdArray` (a function
//! signature, a local binding, `ufunc.rs`'s whole f64-only engine, ...)
//! keeps meaning exactly what it always meant, `NdArray<f64>`, without
//! being touched.
//!
//! This mirrors how real NumPy's C core is one struct definition
//! (`PyArrayObject`) plus a per-dtype `.c.src` *template*, expanded by a
//! preprocessor into a separate compiled function per concrete dtype
//! (`float64_add`, `int32_add`, ...) — one design, many generated
//! implementations. Rust's generics do the same job at the language level
//! instead of a separate build-time templating tool: `NdArray<i32>`,
//! `NdArray<f64>`, `NdArray<Complex64>` are all the *same* struct
//! definition and the *same* method bodies, and the compiler
//! monomorphizes a separate, fully specialized copy of each method per
//! concrete `T` actually used — no `dyn` dispatch, no runtime branch on
//! "which dtype is this", exactly like a `.c.src`-generated `float64_add`
//! has no idea `int32_add` exists.
//!
//! This is the "owner" struct holding the real buffer — every
//! `ArrayView`/`ArrayViewMut` only borrows (`&`/`&mut`) data from it, never
//! copying when a view is created. This is the first ownership lesson:
//! `NdArray` is the sole owner of the `Vec<T>`; a view is just a
//! reference whose lifetime is tied to it, so the compiler guarantees a
//! view can't outlive the original array (no need for a garbage collector
//! or refcounting the way CPython has to with `PyArrayObject`).
//!
//! **Scope note**: only the *container* is generic here — the numeric
//! algorithms built on top (`ufunc`'s `add`/`mul`, `reductions`,
//! `linalg`, `fft`, `random`) are still `f64`-only by design, each for
//! its own documented reason (see their own doc comments). Making
//! `NdArray<T>` itself generic is what lets a future step give those
//! algorithms their own `T`-generic (or `T: DType`-bounded) versions
//! without first having to touch the container at all.

use crate::error::ShapeError;
use crate::shape::{c_contiguous_strides, index_in_bounds, offset_of};
use crate::view::{ArrayView, ArrayViewMut};

#[derive(Debug, Clone, PartialEq)]
pub struct NdArray<T = f64> {
    /// The real data buffer, always C-contiguous (row-major) since
    /// `NdArray` is the "root" array — only a view can carry unusual
    /// strides (e.g. stride 0 after broadcasting, or non-contiguous
    /// strides after slicing).
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<isize>,
}

impl<T: Default + Clone> NdArray<T> {
    /// Create an array filled with `T::default()` for the given `shape`
    /// (`0.0` for `f64`, `0` for `i32`, `0+0i` for `Complex64`, ...).
    pub fn zeros(shape: &[usize]) -> Self {
        let len = shape.iter().product();
        Self {
            data: vec![T::default(); len],
            strides: c_contiguous_strides(shape),
            shape: shape.to_vec(),
        }
    }
}

impl<T> NdArray<T> {
    /// Create an array from flat (row-major) data plus a declared shape.
    ///
    /// Returns `Err` if `data.len()` doesn't match the product of `shape`
    /// — similar to NumPy raising `ValueError: cannot reshape array of
    /// size X into shape Y`, but here as an explicit error instead of a
    /// runtime exception.
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self, ShapeError> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(ShapeError::DataShapeMismatch {
                data_len: data.len(),
                shape: shape.to_vec(),
            });
        }
        Ok(Self {
            data,
            strides: c_contiguous_strides(shape),
            shape: shape.to_vec(),
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// An `NdArray` allocates its own buffer so it's always C-contiguous;
    /// this method is kept (instead of a `true` constant) so `ArrayView`
    /// can share the same logic when both need to check "is the buffer
    /// contiguous in row-major order".
    pub fn is_c_contiguous(&self) -> bool {
        self.strides == c_contiguous_strides(&self.shape)
    }

    /// The raw buffer, in the row-major order implied by `shape` — always
    /// valid since `NdArray` is guaranteed C-contiguous from the
    /// constructor onwards. Used by `npy::write_npy` to write bytes
    /// straight out to a file without walking each index through
    /// `IndexIter` (only a view with unusual strides needs that kind of walk).
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Borrow the whole array immutably as an `ArrayView`.
    ///
    /// The view's lifetime `'_` is tied by the compiler to `&self` — it
    /// can't be kept alive longer than the original `NdArray`. This is
    /// how Rust solves NumPy's "a whole basket of views aliasing the same
    /// buffer" problem without runtime refcounting: a wrong lifetime is a
    /// compile error, not a runtime bug (use-after-free) like in C.
    pub fn view(&self) -> ArrayView<'_, T> {
        ArrayView::new(&self.data, self.shape.clone(), self.strides.clone(), 0)
    }

    /// Borrow the whole array mutably. Since Rust only allows **one**
    /// `&mut` at a time, calling `view_mut()` locks `self` out of every
    /// other access (including an immutable `view()`) until this view
    /// goes out of scope — this is exactly the "&mut uniqueness"
    /// constraint NumPy.md calls out when comparing against NumPy's C
    /// core, which allows multiple views to write at once (`out=`).
    pub fn view_mut(&mut self) -> ArrayViewMut<'_, T> {
        ArrayViewMut::new(&mut self.data, self.shape.clone(), self.strides.clone(), 0)
    }

    /// Basic indexing via a half-open range per axis, e.g.
    /// `arr.slice(&[0..2, 1..3])`. Always returns a **view** (no copy) —
    /// matching NumPy's "basic indexing" semantics: only `shape`/offset
    /// change, on the same buffer.
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Result<ArrayView<'_, T>, ShapeError> {
        self.view().slice(ranges)
    }
}

impl<T: Copy> NdArray<T> {
    /// Read one element by a full-dimensional index, e.g. `get(&[1, 2])`
    /// for a 2-D array. Returns `None` if the index has the wrong number
    /// of dimensions or is out of bounds — using `Option` instead of a
    /// panic lets the caller decide how to handle the error.
    pub fn get(&self, index: &[usize]) -> Option<T> {
        if !index_in_bounds(index, &self.shape) {
            return None;
        }
        let off = offset_of(index, &self.strides);
        self.data.get(off as usize).copied()
    }
}

impl<T> NdArray<T> {
    /// Write one element. Takes `&mut self` — this is the `&mut` lesson:
    /// the compiler guarantees only one place in the code can write to
    /// `self.data` at a time, so there's no way for two threads to mutate
    /// the buffer without synchronization the way C has to be manually
    /// disciplined about.
    pub fn set(&mut self, index: &[usize], value: T) -> Result<(), ShapeError> {
        if !index_in_bounds(index, &self.shape) {
            return Err(ShapeError::IndexOutOfBounds {
                index: index.to_vec(),
                shape: self.shape.clone(),
            });
        }
        let off = offset_of(index, &self.strides);
        self.data[off as usize] = value;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_has_right_shape_and_strides() {
        let a: NdArray = NdArray::zeros(&[2, 3]);
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.strides(), &[3, 1]);
        assert_eq!(a.len(), 6);
        assert!(a.is_c_contiguous());
    }

    #[test]
    fn from_vec_rejects_mismatched_shape() {
        let err = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[2, 2]).unwrap_err();
        assert_eq!(
            err,
            ShapeError::DataShapeMismatch { data_len: 3, shape: vec![2, 2] }
        );
    }

    #[test]
    fn get_set_roundtrip() {
        let mut a = NdArray::zeros(&[2, 2]);
        a.set(&[1, 0], 7.0).unwrap();
        assert_eq!(a.get(&[1, 0]), Some(7.0));
        assert_eq!(a.get(&[0, 0]), Some(0.0));
        assert_eq!(a.get(&[9, 9]), None); // out of bounds -> None, no panic
    }

    #[test]
    fn row_major_layout_matches_numpy() {
        // np.arange(6).reshape(2, 3) -> [[0,1,2],[3,4,5]]
        let a = NdArray::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]).unwrap();
        assert_eq!(a.get(&[0, 0]), Some(0.0));
        assert_eq!(a.get(&[0, 2]), Some(2.0));
        assert_eq!(a.get(&[1, 0]), Some(3.0));
        assert_eq!(a.get(&[1, 2]), Some(5.0));
    }

    #[test]
    fn generic_over_non_f64_element_types() {
        // The same struct/methods, monomorphized for i32 instead of the
        // default f64 -- no separate "IntNdArray" type needed.
        let mut a: NdArray<i32> = NdArray::zeros(&[2, 2]);
        a.set(&[0, 1], 7).unwrap();
        assert_eq!(a.get(&[0, 1]), Some(7));
        assert_eq!(a.get(&[1, 1]), Some(0));

        let b = NdArray::from_vec(vec!["x".to_string(), "y".to_string()], &[2]).unwrap();
        assert_eq!(b.as_slice(), &["x".to_string(), "y".to_string()]);
    }
}
