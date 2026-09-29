use crate::error::ShapeError;
use crate::ndarray::NdArray;
use crate::shape::IndexIter;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxisIndex {

    Full,

    Slice(std::ops::Range<usize>),

    Single(usize),

    Fancy(Vec<usize>),

    Mask(Vec<bool>),
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
        let spec = resolve_masks(spec, self.shape())?;
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
                AxisIndex::Mask(_) => unreachable!("masks were resolved to Fancy above"),
            }
        }
        Ok(self.gather(&out_shape, &plans))
    }

    pub fn vindex(&self, spec: &[AxisIndex]) -> Result<NdArray<T>, ShapeError> {
        if spec.len() != self.ndim() {
            return Err(ShapeError::IndexRankMismatch { expected: self.ndim(), got: spec.len() });
        }
        let spec = resolve_masks(spec, self.shape())?;
        let has_fancy = spec.iter().any(|s| matches!(s, AxisIndex::Fancy(_)));
        let advanced: Vec<usize> = spec
            .iter()
            .enumerate()
            .filter_map(|(axis, s)| match s {
                AxisIndex::Fancy(_) => Some(axis),
                AxisIndex::Single(_) if has_fancy => Some(axis),
                _ => None,
            })
            .collect();

        let fancy_lens: Vec<usize> = spec
            .iter()
            .filter_map(|s| match s {
                AxisIndex::Fancy(v) => Some(v.len()),
                _ => None,
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

        let adjacent = advanced.windows(2).all(|w| w[1] == w[0] + 1);
        let merge_pos = match advanced.first() {
            Some(&first) if adjacent => {
                spec[..first].iter().filter(|s| matches!(s, AxisIndex::Full | AxisIndex::Slice(_))).count()
            }
            _ => 0,
        };

        let mut plans = Vec::with_capacity(spec.len());
        let mut basic_dims: Vec<usize> = Vec::new();
        let mut basic_axes: Vec<usize> = Vec::new();
        for (axis, s) in spec.iter().enumerate() {
            match s {
                AxisIndex::Full => {
                    basic_axes.push(plans.len());
                    plans.push(AxisPlan::Direct { axis, start: 0, out_axis: 0 });
                    basic_dims.push(self.shape()[axis]);
                }
                AxisIndex::Slice(r) => {
                    validate_range(r, self.shape()[axis])?;
                    basic_axes.push(plans.len());
                    plans.push(AxisPlan::Direct { axis, start: r.start, out_axis: 0 });
                    basic_dims.push(r.end - r.start);
                }
                AxisIndex::Single(i) => {
                    validate_index(axis, *i, self.shape()[axis])?;
                    if has_fancy {
                        plans.push(AxisPlan::Lookup { axis, table: vec![*i; merged_len], out_axis: merge_pos });
                    } else {
                        plans.push(AxisPlan::Fixed { axis, value: *i });
                    }
                }
                AxisIndex::Fancy(indices) => {
                    for &i in indices {
                        validate_index(axis, i, self.shape()[axis])?;
                    }
                    let table: Vec<usize> = (0..merged_len)
                        .map(|k| if indices.len() == 1 { indices[0] } else { indices[k] })
                        .collect();
                    plans.push(AxisPlan::Lookup { axis, table, out_axis: merge_pos });
                }
                AxisIndex::Mask(_) => unreachable!("masks were resolved to Fancy above"),
            }
        }
        let mut out_shape = basic_dims.clone();
        if has_fancy {
            out_shape.insert(merge_pos, merged_len);
        }
        for (k, &plan_idx) in basic_axes.iter().enumerate() {
            if let AxisPlan::Direct { out_axis, .. } = &mut plans[plan_idx] {
                *out_axis = if has_fancy && k >= merge_pos { k + 1 } else { k };
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

    pub fn boolean_index_nd(&self, mask: &NdArray<bool>) -> Result<NdArray<T>, ShapeError> {
        let k = self.check_prefix_mask(mask)?;
        let row_len: usize = self.shape()[k..].iter().product();
        let mut data = Vec::new();
        let mut selected = 0;
        for (flat, &keep) in mask.as_slice().iter().enumerate() {
            if keep {
                data.extend_from_slice(&self.as_slice()[flat * row_len..(flat + 1) * row_len]);
                selected += 1;
            }
        }
        let mut shape = vec![selected];
        shape.extend_from_slice(&self.shape()[k..]);
        NdArray::from_vec(data, &shape)
    }

    pub fn boolean_set(&mut self, mask: &NdArray<bool>, value: T) -> Result<(), ShapeError> {
        let k = self.check_prefix_mask(mask)?;
        let row_len: usize = self.shape()[k..].iter().product();
        let data = self.as_mut_slice();
        for (flat, &keep) in mask.as_slice().iter().enumerate() {
            if keep {
                data[flat * row_len..(flat + 1) * row_len].fill(value);
            }
        }
        Ok(())
    }

    fn check_prefix_mask(&self, mask: &NdArray<bool>) -> Result<usize, ShapeError> {
        if mask.ndim() > self.ndim() {
            return Err(ShapeError::IndexRankMismatch { expected: self.ndim(), got: mask.ndim() });
        }
        for (axis, (&want, &got)) in self.shape().iter().zip(mask.shape()).enumerate() {
            if want != got {
                return Err(ShapeError::BooleanAxisMismatch { axis, expected: want, got });
            }
        }
        Ok(mask.ndim())
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

fn resolve_masks(spec: &[AxisIndex], shape: &[usize]) -> Result<Vec<AxisIndex>, ShapeError> {
    spec.iter()
        .enumerate()
        .map(|(axis, s)| match s {
            AxisIndex::Mask(mask) => {
                if mask.len() != shape[axis] {
                    return Err(ShapeError::BooleanAxisMismatch { axis, expected: shape[axis], got: mask.len() });
                }
                Ok(AxisIndex::Fancy(mask.iter().enumerate().filter_map(|(i, &keep)| keep.then_some(i)).collect()))
            }
            other => Ok(other.clone()),
        })
        .collect()
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

    fn f(v: &[usize]) -> AxisIndex {
        AxisIndex::Fancy(v.to_vec())
    }

    fn check(out: &NdArray, shape: &[usize], head: &[f64]) {
        assert_eq!(out.shape(), shape);
        assert_eq!(&out.as_slice()[..head.len()], head);
    }

    #[test]
    fn vindex_non_adjacent_fancy_axes_move_the_broadcast_dim_to_the_front_like_numpy() {
        let a = arange(&[3, 4, 5]);
        let full = AxisIndex::Full;
        check(&a.vindex(&[f(&[0, 1]), full.clone(), f(&[0, 1])]).unwrap(), &[2, 4], &[0.0, 5.0, 10.0, 15.0, 21.0, 26.0]);
        check(&a.vindex(&[f(&[0, 2]), full.clone(), AxisIndex::Single(0)]).unwrap(), &[2, 4], &[0.0, 5.0, 10.0, 15.0, 40.0, 45.0]);
        check(
            &a.vindex(&[f(&[0, 1]), AxisIndex::Slice(1..3), f(&[2, 3])]).unwrap(),
            &[2, 2],
            &[7.0, 12.0, 28.0, 33.0],
        );
        let b = arange(&[2, 3, 4]);
        check(&b.vindex(&[f(&[0, 1]), full, f(&[0, 1])]).unwrap(), &[2, 3], &[0.0, 4.0, 8.0, 13.0, 17.0, 21.0]);
    }

    #[test]
    fn vindex_integers_count_as_advanced_indices_once_a_fancy_index_is_present() {
        let a = arange(&[3, 4, 5]);
        let full = AxisIndex::Full;
        check(&a.vindex(&[AxisIndex::Single(0), full.clone(), f(&[0, 1])]).unwrap(), &[2, 4], &[0.0, 5.0, 10.0, 15.0, 1.0, 6.0]);
        check(&a.vindex(&[full.clone(), AxisIndex::Single(0), f(&[0, 1])]).unwrap(), &[3, 2], &[0.0, 1.0, 20.0, 21.0, 40.0, 41.0]);
        check(&a.vindex(&[f(&[0, 1]), AxisIndex::Single(0), full.clone()]).unwrap(), &[2, 5], &[0.0, 1.0, 2.0, 3.0, 4.0, 20.0]);
        check(&a.vindex(&[AxisIndex::Single(0), f(&[1, 2]), AxisIndex::Single(0)]).unwrap(), &[2], &[5.0, 10.0]);
        check(&a.vindex(&[AxisIndex::Single(0), full.clone(), AxisIndex::Single(0)]).unwrap(), &[4], &[0.0, 5.0, 10.0, 15.0]);
        check(&a.vindex(&[f(&[0, 1]), f(&[0, 2]), full.clone()]).unwrap(), &[2, 5], &[0.0, 1.0, 2.0, 3.0, 4.0, 30.0]);
        check(&a.vindex(&[AxisIndex::Slice(1..3), f(&[0, 1]), full]).unwrap(), &[2, 2, 5], &[20.0, 21.0, 22.0, 23.0, 24.0, 25.0]);
    }

    #[test]
    fn boolean_masks_over_one_axis_become_fancy_indices() {
        let a = arange(&[3, 4, 5]);
        let rows = AxisIndex::Mask(vec![true, false, true]);
        check(&a.oindex(&[rows.clone(), AxisIndex::Full, AxisIndex::Full]).unwrap(), &[2, 4, 5], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        check(
            &a.oindex(&[AxisIndex::Full, AxisIndex::Mask(vec![true, false, true, true]), AxisIndex::Full]).unwrap(),
            &[3, 3, 5],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 10.0],
        );
        check(&a.vindex(&[rows.clone(), AxisIndex::Full, f(&[0, 1])]).unwrap(), &[2, 4], &[0.0, 5.0, 10.0, 15.0, 41.0, 46.0]);
        check(&a.vindex(&[rows, f(&[1, 2]), AxisIndex::Full]).unwrap(), &[2, 5], &[5.0, 6.0, 7.0, 8.0, 9.0, 50.0]);
        let err = a.oindex(&[AxisIndex::Mask(vec![true, false]), AxisIndex::Full, AxisIndex::Full]).unwrap_err();
        assert_eq!(err, ShapeError::BooleanAxisMismatch { axis: 0, expected: 3, got: 2 });
        let err = a.vindex(&[AxisIndex::Full, AxisIndex::Mask(vec![true, false]), AxisIndex::Full]).unwrap_err();
        assert_eq!(err, ShapeError::BooleanAxisMismatch { axis: 1, expected: 4, got: 2 });
    }

    fn mask(data: Vec<bool>, shape: &[usize]) -> NdArray<bool> {
        NdArray::from_vec(data, shape).unwrap()
    }

    #[test]
    fn boolean_mask_over_an_axis_prefix_selects_whole_trailing_blocks() {
        let a = arange(&[3, 4, 5]);
        let m2 = mask(vec![true, false, true, false, false, true, false, false, true, true, false, true], &[3, 4]);
        check(&a.boolean_index_nd(&m2).unwrap(), &[6, 5], &[0.0, 1.0, 2.0, 3.0, 4.0, 10.0]);
        let m1 = mask(vec![true, false, true], &[3]);
        check(&a.boolean_index_nd(&m1).unwrap(), &[2, 4, 5], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let full = mask(vec![true; 60], &[3, 4, 5]);
        check(&a.boolean_index_nd(&full).unwrap(), &[60], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let none = mask(vec![false; 12], &[3, 4]);
        let empty = a.boolean_index_nd(&none).unwrap();
        assert_eq!(empty.shape(), &[0, 5]);
        assert!(empty.is_empty());
        assert_eq!(
            a.boolean_index_nd(&mask(vec![true, false], &[2])).unwrap_err(),
            ShapeError::BooleanAxisMismatch { axis: 0, expected: 3, got: 2 }
        );
        assert_eq!(
            a.boolean_index_nd(&mask(vec![true; 15], &[3, 5])).unwrap_err(),
            ShapeError::BooleanAxisMismatch { axis: 1, expected: 4, got: 5 }
        );
        assert_eq!(
            a.boolean_index_nd(&mask(vec![true; 120], &[3, 4, 5, 2])).unwrap_err(),
            ShapeError::IndexRankMismatch { expected: 3, got: 4 }
        );
        let flat_mask: Vec<bool> = m2.as_slice().to_vec();
        let square = arange(&[3, 4]);
        assert_eq!(square.boolean_index_nd(&m2).unwrap().as_slice(), square.boolean_index(&flat_mask).unwrap().as_slice());
    }

    #[test]
    fn boolean_set_writes_whole_blocks_like_numpy_assignment() {
        let mut c = arange(&[3, 4]);
        c.boolean_set(&mask(vec![true, false, true], &[3]), -1.0).unwrap();
        assert_eq!(c.as_slice(), &[-1.0, -1.0, -1.0, -1.0, 4.0, 5.0, 6.0, 7.0, -1.0, -1.0, -1.0, -1.0]);
        let mut d = arange(&[2, 2]);
        d.boolean_set(&mask(vec![false, true, true, false], &[2, 2]), 9.0).unwrap();
        assert_eq!(d.as_slice(), &[0.0, 9.0, 9.0, 3.0]);
        assert!(d.boolean_set(&mask(vec![true; 3], &[3]), 0.0).is_err());
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
