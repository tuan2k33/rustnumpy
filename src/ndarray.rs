use crate::error::ShapeError;
use crate::shape::{c_contiguous_strides, index_in_bounds, offset_of};
use crate::view::{ArrayView, ArrayViewMut};

#[derive(Debug, Clone, PartialEq)]
pub struct NdArray<T = f64> {

    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<isize>,
}

impl<T: Default + Clone> NdArray<T> {

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

    pub fn is_c_contiguous(&self) -> bool {
        self.strides == c_contiguous_strides(&self.shape)
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn view(&self) -> ArrayView<'_, T> {
        ArrayView::new(&self.data, self.shape.clone(), self.strides.clone(), 0)
    }

    pub fn view_mut(&mut self) -> ArrayViewMut<'_, T> {
        ArrayViewMut::new(&mut self.data, self.shape.clone(), self.strides.clone(), 0)
    }

    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Result<ArrayView<'_, T>, ShapeError> {
        self.view().slice(ranges)
    }
}

impl<T: Copy> NdArray<T> {

    pub fn get(&self, index: &[usize]) -> Option<T> {
        if !index_in_bounds(index, &self.shape) {
            return None;
        }
        let off = offset_of(index, &self.strides);
        self.data.get(off as usize).copied()
    }
}

impl<T> NdArray<T> {

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
        assert_eq!(a.get(&[9, 9]), None);
    }

    #[test]
    fn row_major_layout_matches_numpy() {

        let a = NdArray::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]).unwrap();
        assert_eq!(a.get(&[0, 0]), Some(0.0));
        assert_eq!(a.get(&[0, 2]), Some(2.0));
        assert_eq!(a.get(&[1, 0]), Some(3.0));
        assert_eq!(a.get(&[1, 2]), Some(5.0));
    }

    #[test]
    fn generic_over_non_f64_element_types() {

        let mut a: NdArray<i32> = NdArray::zeros(&[2, 2]);
        a.set(&[0, 1], 7).unwrap();
        assert_eq!(a.get(&[0, 1]), Some(7));
        assert_eq!(a.get(&[1, 1]), Some(0));

        let b = NdArray::from_vec(vec!["x".to_string(), "y".to_string()], &[2]).unwrap();
        assert_eq!(b.as_slice(), &["x".to_string(), "y".to_string()]);
    }
}
