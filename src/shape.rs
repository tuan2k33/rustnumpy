//! Pure shape/stride utilities, shared by `NdArray` and `ArrayView`.
//!
//! Simplification compared to real NumPy: here `strides` are counted in
//! **elements**, not bytes — since step 1 only has one fixed dtype
//! (`f64`) there's no need to know `size_of::<T>()`. Once a multi-dtype
//! system is added (NEP 41/42), strides will have to switch to bytes so
//! each array knows how to advance the pointer by its own element size.

/// Compute C-contiguous (row-major) strides for a given `shape`.
///
/// Rule: the trailing axis always has stride 1; each preceding axis has
/// a stride equal to the stride of the next axis times that axis's size.
///
/// Example: shape `[2, 3, 4]` → strides `[12, 4, 1]`.
pub fn c_contiguous_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; shape.len()];
    let mut acc: isize = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i] as isize;
    }
    strides
}

/// Offset (in elements) of a full-dimensional index, according to `strides`.
///
/// This is the core operation turning a logical `(i, j, k, ...)` into a
/// flat position in the buffer — it doesn't care whether the buffer is
/// C-contiguous or not, since all the layout information already lives
/// in `strides`.
pub fn offset_of(index: &[usize], strides: &[isize]) -> isize {
    index
        .iter()
        .zip(strides.iter())
        .map(|(&i, &s)| i as isize * s)
        .sum()
}

/// Check whether `index` is valid for `shape`: same number of dimensions,
/// and each coordinate falls within `[0, shape[axis])`.
pub fn index_in_bounds(index: &[usize], shape: &[usize]) -> bool {
    index.len() == shape.len() && index.iter().zip(shape.iter()).all(|(&i, &s)| i < s)
}

/// Apply NumPy's broadcasting rule to two shapes, returning the resulting
/// shape if they're compatible.
///
/// Rule (compared from the trailing axis backwards): each pair of sizes
/// must be equal, or one of them must be 1 (in which case it "stretches"
/// to match the other), or one side has run out of axes (treated as size
/// 1). The result has `max(a.len(), b.len())` dimensions.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let ndim = a.len().max(b.len());
    let mut result = vec![0usize; ndim];
    for i in 0..ndim {
        // Walk from the trailing axis backwards: axis `i` is counted from the right.
        let da = a.len().checked_sub(1 + i).map(|idx| a[idx]).unwrap_or(1);
        let db = b.len().checked_sub(1 + i).map(|idx| b[idx]).unwrap_or(1);
        let out = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return None,
        };
        result[ndim - 1 - i] = out;
    }
    Some(result)
}

/// Compute strides to "broadcast" an array with the given original
/// `shape`/`strides` to `target_shape` (already checked compatible via
/// [`broadcast_shapes`]), exactly the way NumPy does it: a stretched axis
/// (original size 1, target size > 1) gets **stride 0** — meaning every
/// index on that axis reads the same memory location, with no data copy.
/// A missing leading axis (the array has fewer dimensions) is treated as
/// size 1 and also gets stride 0.
///
/// Returns `None` if `target_shape` isn't a valid broadcast result of
/// `shape` (i.e. there's an axis whose original size is neither 1 nor
/// equal to the target).
pub fn broadcast_strides(
    shape: &[usize],
    strides: &[isize],
    target_shape: &[usize],
) -> Option<Vec<isize>> {
    let ndim = target_shape.len();
    if shape.len() > ndim {
        return None;
    }
    let offset = ndim - shape.len();
    let mut result = vec![0isize; ndim];
    for i in 0..ndim {
        if i < offset {
            // "Virtual" leading axis — always stride 0.
            result[i] = 0;
        } else {
            let orig_dim = shape[i - offset];
            let orig_stride = strides[i - offset];
            let target_dim = target_shape[i];
            if orig_dim == target_dim {
                result[i] = orig_stride;
            } else if orig_dim == 1 {
                result[i] = 0;
            } else {
                return None;
            }
        }
    }
    Some(result)
}

/// Walk every valid multi-index of a `shape` in sequence, in row-major
/// order (trailing axis moves fastest) — the same order NumPy uses when
/// walking a C-contiguous array with `for x in np.nditer(arr)`.
///
/// Used by operations that need to "walk the whole array by shape logic"
/// regardless of whether the underlying buffer is contiguous (for example
/// materializing a view with arbitrary strides into a new `NdArray` in
/// `to_owned()`).
pub struct IndexIter<'a> {
    shape: &'a [usize],
    current: Option<Vec<usize>>,
}

impl<'a> IndexIter<'a> {
    pub fn new(shape: &'a [usize]) -> Self {
        let start = if shape.contains(&0) {
            None // an axis of size 0 -> empty array, no indices at all
        } else {
            Some(vec![0usize; shape.len()])
        };
        Self { shape, current: start }
    }
}

impl<'a> Iterator for IndexIter<'a> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Vec<usize>> {
        let current = self.current.take()?;
        if self.shape.is_empty() {
            // 0-dimensional array: exactly one empty index `[]`, then done.
            self.current = None;
            return Some(current);
        }
        let mut next = current.clone();
        for axis in (0..self.shape.len()).rev() {
            next[axis] += 1;
            if next[axis] < self.shape[axis] {
                self.current = Some(next);
                return Some(current);
            }
            next[axis] = 0;
        }
        // Every axis "overflowed" back to 0 -> fully walked.
        self.current = None;
        Some(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_contiguous_strides_row_major() {
        assert_eq!(c_contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(c_contiguous_strides(&[5]), vec![1]);
        assert_eq!(c_contiguous_strides(&[]), Vec::<isize>::new());
    }

    #[test]
    fn broadcast_shapes_matches_numpy_rules() {
        assert_eq!(broadcast_shapes(&[3, 4], &[4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shapes(&[3, 1], &[1, 4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shapes(&[8, 1, 6], &[7, 1, 5]), None);
        assert_eq!(broadcast_shapes(&[5, 4], &[1]), Some(vec![5, 4]));
        assert_eq!(broadcast_shapes(&[], &[3]), Some(vec![3]));
    }

    #[test]
    fn broadcast_strides_zeroes_stretched_axes() {
        // shape (3,1) strides (1,1) -> broadcast to (3,4): trailing axis (size 1->4) gets stride 0
        let strides = broadcast_strides(&[3, 1], &[1, 1], &[3, 4]).unwrap();
        assert_eq!(strides, vec![1, 0]);

        // shape (4,) strides (1,) -> broadcast to (3,4): new leading axis, stride 0
        let strides = broadcast_strides(&[4], &[1], &[3, 4]).unwrap();
        assert_eq!(strides, vec![0, 1]);
    }

    #[test]
    fn index_iter_is_row_major() {
        let shape = vec![2, 3];
        let all: Vec<_> = IndexIter::new(&shape).collect();
        assert_eq!(
            all,
            vec![
                vec![0, 0], vec![0, 1], vec![0, 2],
                vec![1, 0], vec![1, 1], vec![1, 2],
            ]
        );
    }

    #[test]
    fn index_iter_zero_size_axis_is_empty() {
        let shape = vec![0, 3];
        assert_eq!(IndexIter::new(&shape).count(), 0);
    }

    #[test]
    fn index_iter_zero_dim_yields_single_empty_index() {
        let shape: Vec<usize> = vec![];
        let all: Vec<_> = IndexIter::new(&shape).collect();
        assert_eq!(all, vec![Vec::<usize>::new()]);
    }
}
