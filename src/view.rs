//! `ArrayView`/`ArrayViewMut`: mượn dữ liệu từ một `NdArray`, không sở hữu
//! buffer, chỉ giữ `shape`/`strides`/`offset` riêng.
//!
//! Đây là câu trả lời của Rust cho vấn đề NumPy.md nêu: "nhiều view alias
//! cùng một buffer". Thay vì để mọi view tự do đọc/ghi một buffer dùng
//! chung (như C con trỏ thô), Rust tách bạch bằng lifetime + borrow
//! checker: `ArrayView<'a>` mượn bất biến (nhiều view cùng lúc, không ai
//! ghi), `ArrayViewMut<'a>` mượn độc quyền (đúng một view, được ghi). Cả
//! hai không thể tồn tại cùng lúc trên cùng dữ liệu — compiler chặn ngay
//! lúc biên dịch, NumPy C phải tự kỷ luật bằng tay (và thỉnh thoảng có bug
//! alias).

use crate::error::ShapeError;
use crate::ndarray::NdArray;
use crate::shape::{broadcast_strides, index_in_bounds, offset_of, IndexIter};
use std::ops::Range;

/// View bất biến: mượn `&'a [f64]`, không thể ghi.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrayView<'a> {
    data: &'a [f64],
    shape: Vec<usize>,
    strides: Vec<isize>,
    /// Offset (phần tử) tính từ đầu `data` tới phần tử `index = [0, 0, ...]`
    /// của view này. Basic indexing (slice) chỉ cần đổi `offset` + `shape`,
    /// không đụng tới `data` — đó là lý do slice không bao giờ copy.
    offset: usize,
}

impl<'a> ArrayView<'a> {
    pub(crate) fn new(data: &'a [f64], shape: Vec<usize>, strides: Vec<isize>, offset: usize) -> Self {
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

    /// Đọc một phần tử theo index đầy đủ chiều (tọa độ tính trên view này,
    /// không phải trên mảng gốc — sau khi slice, `[0, 0]` là phần tử đầu
    /// của *view*, không phải của mảng ban đầu).
    pub fn get(&self, index: &[usize]) -> Option<f64> {
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

    /// Basic indexing: cắt mỗi trục theo `Range<usize>` nửa-mở, trả về
    /// view mới trên **cùng buffer** (chỉ đổi `shape` + `offset`, strides
    /// giữ nguyên) — không copy, đúng ngữ nghĩa "view" của NumPy.
    pub fn slice(&self, ranges: &[Range<usize>]) -> Result<ArrayView<'a>, ShapeError> {
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

    fn invalid_slice(&self, ranges: &[Range<usize>]) -> Result<ArrayView<'a>, ShapeError> {
        Err(ShapeError::InvalidSlice {
            shape: self.shape.clone(),
            ranges: ranges.iter().map(|r| (r.start, r.end)).collect(),
        })
    }

    /// Giãn view sang `target_shape` theo quy tắc broadcasting của NumPy,
    /// không copy dữ liệu: trục bị giãn nhận stride 0 (mọi tọa độ trên
    /// trục đó đọc cùng một ô nhớ). Trả `Err` nếu không broadcast được.
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<ArrayView<'a>, ShapeError> {
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

    /// Vật chất hóa view thành một `NdArray` sở hữu dữ liệu riêng
    /// (C-contiguous), bằng cách duyệt qua từng index logic và copy giá
    /// trị — cần thiết vì sau `slice`/`broadcast_to`, buffer bên dưới có
    /// thể không còn liền mạch (có "lỗ" giữa các phần tử, hoặc stride 0
    /// lặp lại cùng ô nhớ nhiều lần).
    pub fn to_owned(&self) -> NdArray {
        let data: Vec<f64> = IndexIter::new(&self.shape)
            .map(|idx| self.get(&idx).expect("IndexIter chỉ sinh index hợp lệ"))
            .collect();
        NdArray::from_vec(data, &self.shape).expect("data.len() luôn khớp shape.iter().product()")
    }
}

/// View có thể ghi: mượn `&'a mut [f64]`, độc quyền tại một thời điểm.
#[derive(Debug, PartialEq)]
pub struct ArrayViewMut<'a> {
    data: &'a mut [f64],
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
}

impl<'a> ArrayViewMut<'a> {
    pub(crate) fn new(
        data: &'a mut [f64],
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: usize,
    ) -> Self {
        Self { data, shape, strides, offset }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Mượn lại view bất biến từ view có thể ghi — hợp lệ vì `&self` (không
    /// phải `&mut self`) chỉ cho mượn *ngắn hạn*, nhỏ hơn lifetime `'a` gốc;
    /// đây là ví dụ điển hình về "reborrow" trong Rust.
    pub fn get(&self, index: &[usize]) -> Option<f64> {
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

    /// Ghi một phần tử. Nhận `&mut self` để compiler đảm bảo không có view
    /// bất biến nào khác đang sống cùng lúc trỏ vào buffer này.
    pub fn set(&mut self, index: &[usize], value: f64) -> Result<(), ShapeError> {
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
        // np.arange(12).reshape(3,4)[1:3, 1:3] -> [[5,6],[9,10]]
        let a = arange(&[3, 4]);
        let v = a.slice(&[1..3, 1..3]).unwrap();
        assert_eq!(v.shape(), &[2, 2]);
        assert_eq!(v.get(&[0, 0]), Some(5.0));
        assert_eq!(v.get(&[0, 1]), Some(6.0));
        assert_eq!(v.get(&[1, 0]), Some(9.0));
        assert_eq!(v.get(&[1, 1]), Some(10.0));
        // strides không đổi so với mảng gốc -> bằng chứng đây là view, không copy
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
        // shape (3,1) broadcast to (3,4): mỗi hàng lặp lại giá trị cột duy nhất
        let a = arange(&[3, 1]); // [[0],[1],[2]]
        let v = a.view().broadcast_to(&[3, 4]).unwrap();
        assert_eq!(v.shape(), &[3, 4]);
        for col in 0..4 {
            assert_eq!(v.get(&[0, col]), Some(0.0));
            assert_eq!(v.get(&[1, col]), Some(1.0));
            assert_eq!(v.get(&[2, col]), Some(2.0));
        }
        // trục bị giãn phải có stride 0
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
        } // vm hết scope ở đây -> trả quyền mượn lại cho `a`
        assert_eq!(a.get(&[0, 1]), Some(42.0));
    }
}
