use crate::error::ShapeError;
use crate::ndarray::NdArray;
use crate::shape::{broadcast_strides, index_in_bounds, offset_of, IndexIter};
use std::ops::Range;

#[derive(Debug, Clone, PartialEq)]
pub struct ArrayView<'a, T = f64> {
    data: &'a [T],
    shape: Vec<usize>,
    strides: Vec<isize>,

    offset: usize,
}

impl<'a, T> ArrayView<'a, T> {
    pub(crate) fn new(data: &'a [T], shape: Vec<usize>, strides: Vec<isize>, offset: usize) -> Self {
        Self { data, shape, strides, offset }
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
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn slice(&self, ranges: &[Range<usize>]) -> Result<ArrayView<'a, T>, ShapeError> {
        if ranges.len() != self.shape.len() {
            return self.invalid_slice(ranges);
        }
        for (axis, r) in ranges.iter().enumerate() {
            if r.start > r.end || r.end > self.shape[axis] {
                return self.invalid_slice(ranges);
            }
        }
        let new_shape: Vec<usize> = ranges.iter().map(|r| r.end - r.start).collect();
        let extra_offset: isize = ranges
            .iter()
            .zip(self.strides.iter())
            .map(|(r, &s)| r.start as isize * s)
            .sum();
        Ok(ArrayView {
            data: self.data,
            shape: new_shape,
            strides: self.strides.clone(),
            offset: (self.offset as isize + extra_offset) as usize,
        })
    }

    fn invalid_slice(&self, ranges: &[Range<usize>]) -> Result<ArrayView<'a, T>, ShapeError> {
        Err(ShapeError::InvalidSlice {
            shape: self.shape.clone(),
            ranges: ranges.iter().map(|r| (r.start, r.end)).collect(),
        })
    }

    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<ArrayView<'a, T>, ShapeError> {
        match broadcast_strides(&self.shape, &self.strides, target_shape) {
            Some(new_strides) => Ok(ArrayView {
                data: self.data,
                shape: target_shape.to_vec(),
                strides: new_strides,
                offset: self.offset,
            }),
            None => Err(ShapeError::NotBroadcastable {
                lhs: self.shape.clone(),
                rhs: target_shape.to_vec(),
            }),
        }
    }
}

impl<'a, T: Copy> ArrayView<'a, T> {

    pub fn get(&self, index: &[usize]) -> Option<T> {
        if !index_in_bounds(index, &self.shape) {
            return None;
        }
        let rel = offset_of(index, &self.strides);
        let abs = self.offset as isize + rel;
        if abs < 0 {
            return None;
        }
        self.data.get(abs as usize).copied()
    }

    pub fn to_owned(&self) -> NdArray<T> {
        let data: Vec<T> = IndexIter::new(&self.shape)
            .map(|idx| self.get(&idx).expect("IndexIter only produces valid indices"))
            .collect();
        NdArray::from_vec(data, &self.shape).expect("data.len() always matches shape.iter().product()")
    }
}

#[derive(Debug, PartialEq)]
pub struct ArrayViewMut<'a, T = f64> {
    data: &'a mut [T],
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
}

impl<'a, T> ArrayViewMut<'a, T> {
    pub(crate) fn new(
        data: &'a mut [T],
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: usize,
    ) -> Self {
        Self { data, shape, strides, offset }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn set(&mut self, index: &[usize], value: T) -> Result<(), ShapeError> {
        if !index_in_bounds(index, &self.shape) {
            return Err(ShapeError::IndexOutOfBounds {
                index: index.to_vec(),
                shape: self.shape.clone(),
            });
        }
        let rel = offset_of(index, &self.strides);
        let abs = self.offset as isize + rel;
        self.data[abs as usize] = value;
        Ok(())
    }
}

impl<'a, T: Copy> ArrayViewMut<'a, T> {

    pub fn get(&self, index: &[usize]) -> Option<T> {
        if !index_in_bounds(index, &self.shape) {
            return None;
        }
        let rel = offset_of(index, &self.strides);
        let abs = self.offset as isize + rel;
        if abs < 0 {
            return None;
        }
        self.data.get(abs as usize).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray::NdArray;

    fn arange(shape: &[usize]) -> NdArray {
        let len: usize = shape.iter().product();
        let data: Vec<f64> = (0..len).map(|i| i as f64).collect();
        NdArray::from_vec(data, shape).unwrap()
    }

    #[test]
    fn slice_is_a_view_not_a_copy() {

        let a = arange(&[3, 4]);
        let v = a.slice(&[1..3, 1..3]).unwrap();
        assert_eq!(v.shape(), &[2, 2]);
        assert_eq!(v.get(&[0, 0]), Some(5.0));
        assert_eq!(v.get(&[0, 1]), Some(6.0));
        assert_eq!(v.get(&[1, 0]), Some(9.0));
        assert_eq!(v.get(&[1, 1]), Some(10.0));

        assert_eq!(v.strides(), a.strides());
    }

    #[test]
    fn slice_rejects_out_of_bounds_range() {
        let a = arange(&[3, 4]);
        let err = a.slice(&[0..5, 0..1]).unwrap_err();
        matches!(err, ShapeError::InvalidSlice { .. });
    }

    #[test]
    fn broadcast_to_repeats_without_copying() {

        let a = arange(&[3, 1]);
        let v = a.view().broadcast_to(&[3, 4]).unwrap();
        assert_eq!(v.shape(), &[3, 4]);
        for col in 0..4 {
            assert_eq!(v.get(&[0, col]), Some(0.0));
            assert_eq!(v.get(&[1, col]), Some(1.0));
            assert_eq!(v.get(&[2, col]), Some(2.0));
        }

        assert_eq!(v.strides()[1], 0);
    }

    #[test]
    fn broadcast_to_incompatible_shape_errs() {
        let a = arange(&[3, 4]);
        let err = a.view().broadcast_to(&[3, 5]).unwrap_err();
        assert_eq!(
            err,
            ShapeError::NotBroadcastable { lhs: vec![3, 4], rhs: vec![3, 5] }
        );
    }

    #[test]
    fn to_owned_materializes_broadcast_view() {
        let a = arange(&[3, 1]);
        let broadcasted = a.view().broadcast_to(&[3, 2]).unwrap().to_owned();
        assert_eq!(broadcasted.shape(), &[3, 2]);
        assert!(broadcasted.is_c_contiguous());
        assert_eq!(broadcasted.get(&[0, 0]), Some(0.0));
        assert_eq!(broadcasted.get(&[0, 1]), Some(0.0));
        assert_eq!(broadcasted.get(&[2, 1]), Some(2.0));
    }

    #[test]
    fn view_mut_set_then_view_sees_change() {
        let mut a = NdArray::zeros(&[2, 2]);
        {
            let mut vm = a.view_mut();
            vm.set(&[0, 1], 42.0).unwrap();
        }
        assert_eq!(a.get(&[0, 1]), Some(42.0));
    }

    #[test]
    fn view_is_generic_over_non_f64_element_types() {
        let a = NdArray::from_vec(vec![1i32, 2, 3, 4], &[2, 2]).unwrap();
        let v = a.view();
        assert_eq!(v.get(&[1, 1]), Some(4));
        assert_eq!(v.slice(&[0..1, 0..2]).unwrap().to_owned().as_slice(), &[1, 2]);
    }
}
