use crate::error::ShapeError;
use crate::ndarray::NdArray;
use crate::shape::IndexIter;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxisIndex {

    Full,

    Slice(std::ops::Range<usize>),

    Single(usize),

    Fancy(Vec<usize>),
}

enum AxisPlan {
    Direct { axis: usize, start: usize, out_axis: usize },
    Fixed { axis: usize, value: usize },
    Lookup { axis: usize, table: Vec<usize>, out_axis: usize },
}

impl<T: Copy> NdArray<T> {

    pub fn oindex(&self, spec: &[AxisIndex]) -> Result<NdArray<T>, ShapeError> {
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

    pub fn vindex(&self, spec: &[AxisIndex]) -> Result<NdArray<T>, ShapeError> {
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

    pub fn boolean_index(&self, mask: &[bool]) -> Result<NdArray<T>, ShapeError> {
        if mask.len() != self.len() {
            return Err(ShapeError::BooleanMaskShapeMismatch {
                mask_len: mask.len(),
                array_shape: self.shape().to_vec(),
            });
        }
        let selected: Vec<T> = IndexIter::new(self.shape())
            .zip(mask.iter())
            .filter_map(|(idx, &keep)| {
                keep.then(|| self.get(&idx).expect("IndexIter only yields valid indices"))
            })
            .collect();
        let n = selected.len();
        NdArray::from_vec(selected, &[n])
    }

    fn gather(&self, out_shape: &[usize], plans: &[AxisPlan]) -> NdArray<T> {
        let data: Vec<T> = IndexIter::new(out_shape)
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

    #[test]
    fn oindex_two_fancy_axes_matches_numpy_ix() {

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

    #[test]
    fn vindex_single_fancy_axis_matches_numpy() {

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

        let a = arange(&[4, 6]);
        let out = a.vindex(&[AxisIndex::Fancy(vec![0, 2]), AxisIndex::Slice(1..4)]).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn vindex_two_adjacent_fancy_axes_broadcast_together() {

        let a = arange(&[4, 6]);
        let out = a
            .vindex(&[AxisIndex::Fancy(vec![0, 1, 2]), AxisIndex::Fancy(vec![0, 1, 2])])
            .unwrap();
        assert_eq!(out.shape(), &[3]);
        assert_eq!(out.as_slice(), &[0.0, 7.0, 14.0]);
    }

    #[test]
    fn vindex_single_before_fancy_positions_merged_axis_correctly() {

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

    #[test]
    fn boolean_index_matches_numpy_full_mask() {

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
