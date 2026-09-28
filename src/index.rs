//! Step 8: advanced indexing (fancy integer-array indexing, boolean mask
//! indexing), plus the explicit `.oindex()`/`.vindex()` split that NEP 21
//! proposed for real NumPy but never shipped (see `NumPy.md`'s "Indexing
//! Semantics" section).
//!
//! Real NumPy overloads one `[]` operator for both kinds of indexing, which
//! is exactly what makes mixed fancy indexing so confusing: when the fancy
//! (integer-array) axes are *not* adjacent, NumPy silently moves the
//! resulting axis to the front of the array — a rule almost nobody
//! remembers correctly. Since this is a fresh design with no backward
//! compatibility to protect, we split it into two explicit methods instead:
//!
//! - [`NdArray::vindex`] — *vectorized* indexing: today's NumPy advanced-
//!   indexing behavior (index arrays broadcast together and walk in
//!   lock-step), but restricted to **adjacent** fancy axes so there's no
//!   axis-jump rule to remember at all — a non-adjacent request is a
//!   plain `Err`, not a silent axis reshuffle.
//! - [`NdArray::oindex`] — *outer/orthogonal* indexing: each axis's index
//!   array applies independently, like `numpy.ix_` or MATLAB/Fortran
//!   indexing. No broadcasting, no adjacency requirement, no ambiguity.
//!
//! Both always return an owned copy (never a view) — matching NumPy's rule
//! that advanced indexing copies, only basic indexing ([`NdArray::slice`])
//! views.

use crate::error::ShapeError;
use crate::ndarray::NdArray;
use crate::shape::IndexIter;

/// One axis's index specification, passed as a slice (one per dimension)
/// to [`NdArray::oindex`] / [`NdArray::vindex`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxisIndex {
    /// Keep the whole axis, e.g. Python's `:`.
    Full,
    /// A half-open range, e.g. Python's `1:4`. Basic indexing: never
    /// broadcast, never advanced.
    Slice(std::ops::Range<usize>),
    /// A single position — drops this axis from the result (its dimension
    /// disappears), e.g. `a[2]` instead of `a[2:3]`.
    Single(usize),
    /// An integer array — advanced indexing. Each value selects one
    /// position along this axis; the axis's *count* of positions can
    /// differ from the axis's own size (unlike `Slice`/`Single`).
    Fancy(Vec<usize>),
}

/// How one *source* axis maps into the *output* array, precomputed once per
/// call so the actual walk (`IndexIter` over the output shape) is a plain
/// lookup per element, not a re-derivation. `out_axis` is `None` for a
/// dropped (`Single`) axis; otherwise it names which output axis this
/// source axis reads its coordinate from — normally its own (as in
/// `oindex`), but under `vindex` every fancy axis that got merged together
/// shares the *same* `out_axis`.
enum AxisPlan {
    Direct { axis: usize, start: usize, out_axis: usize },
    Fixed { axis: usize, value: usize },
    Lookup { axis: usize, table: Vec<usize>, out_axis: usize },
}

impl NdArray {
    /// Outer (orthogonal) advanced indexing: each `Fancy` axis's index
    /// array applies independently to its own axis, with no broadcasting
    /// between axes — matching `numpy.ix_(...)`, not `arr[...]` directly.
    ///
    /// Example: `a.oindex(&[Fancy(vec![0,2]), Fancy(vec![1,3,4])])` on a
    /// `(4,6)` array picks rows `{0,2}` × columns `{1,3,4}`, giving a
    /// `(2,3)` result — equivalent to `a[np.ix_([0,2],[1,3,4])]`.
    pub fn oindex(&self, spec: &[AxisIndex]) -> Result<NdArray, ShapeError> {
        if spec.len() != self.ndim() {
            return Err(ShapeError::IndexRankMismatch { expected: self.ndim(), got: spec.len() });
        }
        let mut plans = Vec::with_capacity(spec.len());
        let mut out_shape = Vec::new();
        for (axis, s) in spec.iter().enumerate() {
            match s {
                AxisIndex::Full => {
                    plans.push(AxisPlan::Direct { axis, start: 0, out_axis: out_shape.len() });
                    out_shape.push(self.shape()[axis]);
                }
                AxisIndex::Slice(r) => {
                    validate_range(r, self.shape()[axis])?;
                    plans.push(AxisPlan::Direct { axis, start: r.start, out_axis: out_shape.len() });
                    out_shape.push(r.end - r.start);
                }
                AxisIndex::Single(i) => {
                    validate_index(axis, *i, self.shape()[axis])?;
                    plans.push(AxisPlan::Fixed { axis, value: *i });
                }
                AxisIndex::Fancy(indices) => {
                    for &i in indices {
                        validate_index(axis, i, self.shape()[axis])?;
                    }
                    plans.push(AxisPlan::Lookup {
                        axis,
                        table: indices.clone(),
                        out_axis: out_shape.len(),
                    });
                    out_shape.push(indices.len());
                }
            }
        }
        Ok(self.gather(&out_shape, &plans))
    }

    /// Vectorized advanced indexing: every `Fancy` axis's index array is
    /// broadcast together (NumPy's actual `arr[...]` rule) and walked in
    /// lock-step, collapsing into a **single** output axis positioned where
    /// the fancy axes sit in `spec`.
    ///
    /// Deliberately narrower than real NumPy: the `Fancy` axes in `spec`
    /// must be **adjacent** (e.g. axes 0,1 of a 3-D array, not axes 0 and 2
    /// with a `Slice`/`Single` in between). Real NumPy allows non-adjacent
    /// fancy axes but then silently moves the merged axis to the front — a
    /// rule this port intentionally does not implement; use
    /// [`NdArray::oindex`] instead for that shape of query, or reorder axes
    /// so the fancy ones are next to each other.
    pub fn vindex(&self, spec: &[AxisIndex]) -> Result<NdArray, ShapeError> {
        if spec.len() != self.ndim() {
            return Err(ShapeError::IndexRankMismatch { expected: self.ndim(), got: spec.len() });
        }

        let fancy_axes: Vec<usize> = spec
            .iter()
            .enumerate()
            .filter_map(|(axis, s)| matches!(s, AxisIndex::Fancy(_)).then_some(axis))
            .collect();

        if fancy_axes.len() > 1 && !fancy_axes.windows(2).all(|w| w[1] == w[0] + 1) {
            return Err(ShapeError::NonAdjacentFancyIndices { axes: fancy_axes });
        }

        // Broadcast all fancy index arrays' lengths together (each treated
        // as a 1-D array, so "broadcasting" just means every length is `n`,
        // or `1`, matching NumPy's own rule for fancy index arrays of
        // different lengths).
        let fancy_lens: Vec<usize> = fancy_axes
            .iter()
            .map(|&a| match &spec[a] {
                AxisIndex::Fancy(v) => v.len(),
                _ => unreachable!(),
            })
            .collect();
        let merged_len = fancy_lens
            .iter()
            .copied()
            .filter(|&n| n != 1)
            .max()
            .or(fancy_lens.first().copied())
            .unwrap_or(0);
        if fancy_lens.iter().any(|&n| n != 1 && n != merged_len) {
            return Err(ShapeError::FancyIndexNotBroadcastable { lengths: fancy_lens });
        }

        // All merged fancy axes read from the same output axis; that axis
        // is positioned wherever the first fancy axis appears among the
        // *kept* (non-`Single`) axes.
        let merge_out_axis = fancy_axes.first().map(|&first_fancy| {
            spec[..first_fancy].iter().filter(|s| !matches!(s, AxisIndex::Single(_))).count()
        });

        let mut plans = Vec::with_capacity(spec.len());
        let mut out_shape = Vec::new();
        for (axis, s) in spec.iter().enumerate() {
            match s {
                AxisIndex::Full => {
                    plans.push(AxisPlan::Direct { axis, start: 0, out_axis: out_shape.len() });
                    out_shape.push(self.shape()[axis]);
                }
                AxisIndex::Slice(r) => {
                    validate_range(r, self.shape()[axis])?;
                    plans.push(AxisPlan::Direct { axis, start: r.start, out_axis: out_shape.len() });
                    out_shape.push(r.end - r.start);
                }
                AxisIndex::Single(i) => {
                    validate_index(axis, *i, self.shape()[axis])?;
                    plans.push(AxisPlan::Fixed { axis, value: *i });
                }
                AxisIndex::Fancy(indices) => {
                    for &i in indices {
                        validate_index(axis, i, self.shape()[axis])?;
                    }
                    let table: Vec<usize> = (0..merged_len)
                        .map(|k| if indices.len() == 1 { indices[0] } else { indices[k] })
                        .collect();
                    let out_axis = merge_out_axis.expect("a Fancy axis implies merge_out_axis is Some");
                    plans.push(AxisPlan::Lookup { axis, table, out_axis });
                    if out_axis == out_shape.len() {
                        out_shape.push(merged_len);
                    }
                }
            }
        }
        Ok(self.gather(&out_shape, &plans))
    }

    /// Boolean mask indexing over the **whole** array: `mask` must have
    /// exactly `self.len()` entries, one per element in row-major order
    /// (i.e. built to match `self.shape()` elementwise, like
    /// `mask = arr % 2 == 0` in NumPy). Returns a flat 1-D array of every
    /// element where `mask` is `true`, in row-major order — matching
    /// `arr[mask]` for a full-shape boolean mask.
    ///
    /// Narrower than NumPy on purpose: NumPy also allows a boolean mask
    /// that only covers a *prefix* of the axes (e.g. `arr[row_mask, :]`
    /// selecting whole rows); that's not implemented here — for that,
    /// combine [`NdArray::vindex`]/[`NdArray::oindex`] with your own
    /// nonzero-index computation, or select whole rows via [`NdArray::slice`]
    /// in a loop.
    pub fn boolean_index(&self, mask: &[bool]) -> Result<NdArray, ShapeError> {
        if mask.len() != self.len() {
            return Err(ShapeError::BooleanMaskShapeMismatch {
                mask_len: mask.len(),
                array_shape: self.shape().to_vec(),
            });
        }
        let selected: Vec<f64> = IndexIter::new(self.shape())
            .zip(mask.iter())
            .filter_map(|(idx, &keep)| {
                keep.then(|| self.get(&idx).expect("IndexIter only yields valid indices"))
            })
            .collect();
        let n = selected.len();
        NdArray::from_vec(selected, &[n])
    }

    /// Walk every index of `out_shape` in row-major order, using `plans` to
    /// map each output multi-index back to a source multi-index, and
    /// collect the gathered values into a new owned array. Shared by
    /// `oindex` and `vindex` — both only differ in how `plans` is built.
    fn gather(&self, out_shape: &[usize], plans: &[AxisPlan]) -> NdArray {
        let data: Vec<f64> = IndexIter::new(out_shape)
            .map(|out_idx| {
                let mut src_idx = vec![0usize; self.ndim()];
                for plan in plans {
                    match plan {
                        AxisPlan::Fixed { axis, value } => src_idx[*axis] = *value,
                        AxisPlan::Direct { axis, start, out_axis } => {
                            src_idx[*axis] = start + out_idx[*out_axis];
                        }
                        AxisPlan::Lookup { axis, table, out_axis } => {
                            src_idx[*axis] = table[out_idx[*out_axis]];
                        }
                    }
                }
                self.get(&src_idx).expect("plans only produce validated in-bounds source indices")
            })
            .collect();
        NdArray::from_vec(data, out_shape).expect("data.len() always matches out_shape.iter().product()")
    }
}

fn validate_index(axis: usize, i: usize, dim: usize) -> Result<(), ShapeError> {
    if i >= dim {
        Err(ShapeError::FancyIndexOutOfBounds { axis, index: i, dim })
    } else {
        Ok(())
    }
}

fn validate_range(r: &std::ops::Range<usize>, dim: usize) -> Result<(), ShapeError> {
    if r.start > r.end || r.end > dim {
        Err(ShapeError::InvalidSlice { shape: vec![dim], ranges: vec![(r.start, r.end)] })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ShapeError;

    fn arange(shape: &[usize]) -> NdArray {
        let len: usize = shape.iter().product();
        let data: Vec<f64> = (0..len).map(|i| i as f64).collect();
        NdArray::from_vec(data, shape).unwrap()
    }

    // -- oindex --------------------------------------------------------

    #[test]
    fn oindex_two_fancy_axes_matches_numpy_ix() {
        // a = np.arange(24).reshape(4,6); a[np.ix_([0,2],[1,3,4])]
        // -> [[ 1, 3, 4], [13,15,16]]
        let a = arange(&[4, 6]);
        let out = a
            .oindex(&[AxisIndex::Fancy(vec![0, 2]), AxisIndex::Fancy(vec![1, 3, 4])])
            .unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.as_slice(), &[1.0, 3.0, 4.0, 13.0, 15.0, 16.0]);
    }

    #[test]
    fn oindex_fancy_with_single_drops_dimension() {
        let a = arange(&[4, 6]);
        // a[2, [1,3,4]] -> row 2, columns {1,3,4} -> 1-D result
        let out = a.oindex(&[AxisIndex::Single(2), AxisIndex::Fancy(vec![1, 3, 4])]).unwrap();
        assert_eq!(out.shape(), &[3]);
        assert_eq!(out.as_slice(), &[13.0, 15.0, 16.0]);
    }

    #[test]
    fn oindex_out_of_bounds_index_errs() {
        let a = arange(&[4, 6]);
        let err = a.oindex(&[AxisIndex::Fancy(vec![0, 9]), AxisIndex::Full]).unwrap_err();
        assert_eq!(err, ShapeError::FancyIndexOutOfBounds { axis: 0, index: 9, dim: 4 });
    }

    // -- vindex ----------------------------------------------------------

    #[test]
    fn vindex_single_fancy_axis_matches_numpy() {
        // a[[0,2,3]] -> rows 0,2,3
        let a = arange(&[4, 6]);
        let out = a.vindex(&[AxisIndex::Fancy(vec![0, 2, 3]), AxisIndex::Full]).unwrap();
        assert_eq!(out.shape(), &[3, 6]);
        assert_eq!(
            out.as_slice(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
        );
    }

    #[test]
    fn vindex_fancy_plus_slice_matches_numpy() {
        // a[[0,2], 1:4] -> [[1,2,3],[13,14,15]]
        let a = arange(&[4, 6]);
        let out = a.vindex(&[AxisIndex::Fancy(vec![0, 2]), AxisIndex::Slice(1..4)]).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn vindex_two_adjacent_fancy_axes_broadcast_together() {
        // a[[0,1,2],[0,1,2]] -> diagonal-ish: [a[0,0], a[1,1], a[2,2]] = [0,7,14]
        let a = arange(&[4, 6]);
        let out = a
            .vindex(&[AxisIndex::Fancy(vec![0, 1, 2]), AxisIndex::Fancy(vec![0, 1, 2])])
            .unwrap();
        assert_eq!(out.shape(), &[3]);
        assert_eq!(out.as_slice(), &[0.0, 7.0, 14.0]);
    }

    #[test]
    fn vindex_single_before_fancy_positions_merged_axis_correctly() {
        // a = np.arange(60).reshape(3,4,5); a[1, [0,2], :]
        // -> [[20,21,22,23,24],[30,31,32,33,34]], shape (2,5)
        let a = arange(&[3, 4, 5]);
        let out = a.vindex(&[AxisIndex::Single(1), AxisIndex::Fancy(vec![0, 2]), AxisIndex::Full]).unwrap();
        assert_eq!(out.shape(), &[2, 5]);
        assert_eq!(
            out.as_slice(),
            &[20.0, 21.0, 22.0, 23.0, 24.0, 30.0, 31.0, 32.0, 33.0, 34.0]
        );
    }

    #[test]
    fn vindex_non_adjacent_fancy_axes_is_explicitly_unsupported() {
        // np.arange(24).reshape(2,3,4)[[0,1],:,[0,1]] would jump the merged
        // axis to the front in real NumPy; this port refuses instead.
        let a = arange(&[2, 3, 4]);
        let err = a
            .vindex(&[AxisIndex::Fancy(vec![0, 1]), AxisIndex::Full, AxisIndex::Fancy(vec![0, 1])])
            .unwrap_err();
        assert_eq!(err, ShapeError::NonAdjacentFancyIndices { axes: vec![0, 2] });
    }

    #[test]
    fn vindex_mismatched_fancy_lengths_err() {
        let a = arange(&[4, 6]);
        let err = a
            .vindex(&[AxisIndex::Fancy(vec![0, 1, 2]), AxisIndex::Fancy(vec![0, 1])])
            .unwrap_err();
        assert_eq!(err, ShapeError::FancyIndexNotBroadcastable { lengths: vec![3, 2] });
    }

    // -- boolean_index -----------------------------------------------------

    #[test]
    fn boolean_index_matches_numpy_full_mask() {
        // a = np.arange(24).reshape(4,6); a[a % 2 == 0] -> evens, flattened
        let a = arange(&[4, 6]);
        let mask: Vec<bool> = a.as_slice().iter().map(|&x| x as i64 % 2 == 0).collect();
        let out = a.boolean_index(&mask).unwrap();
        assert_eq!(out.shape(), &[12]);
        let expected: Vec<f64> = (0..24).filter(|&i| i % 2 == 0).map(|i| i as f64).collect();
        assert_eq!(out.as_slice(), expected.as_slice());
    }

    #[test]
    fn boolean_index_wrong_length_errs() {
        let a = arange(&[4, 6]);
        let err = a.boolean_index(&[true, false]).unwrap_err();
        assert_eq!(err, ShapeError::BooleanMaskShapeMismatch { mask_len: 2, array_shape: vec![4, 6] });
    }
}
