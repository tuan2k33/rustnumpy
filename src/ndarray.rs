//! `NdArray`: mảng N-chiều sở hữu dữ liệu (owned), dtype cố định `f64`.
//!
//! Đây là struct "chủ" giữ buffer thật — mọi `ArrayView`/`ArrayViewMut` chỉ
//! mượn (`&`/`&mut`) dữ liệu từ đây, không bao giờ copy khi tạo view. Đây
//! chính là bài học ownership đầu tiên: `NdArray` là chủ sở hữu duy nhất
//! của `Vec<f64>`; view chỉ là tham chiếu có lifetime ràng buộc vào nó, nên
//! compiler đảm bảo view không thể tồn tại lâu hơn mảng gốc (không cần
//! garbage collector hay refcount như CPython phải làm với `PyArrayObject`).

use crate::error::ShapeError;
use crate::shape::{c_contiguous_strides, index_in_bounds, offset_of};
use crate::view::{ArrayView, ArrayViewMut};

#[derive(Debug, Clone, PartialEq)]
pub struct NdArray {
    /// Buffer dữ liệu thật, luôn C-contiguous (row-major) vì `NdArray` là
    /// mảng "gốc" — chỉ view mới có thể mang stride khác thường (ví dụ
    /// stride 0 sau broadcast, hoặc stride không liên tục sau slice).
    data: Vec<f64>,
    shape: Vec<usize>,
    strides: Vec<isize>,
}

impl NdArray {
    /// Tạo mảng toàn số 0 với `shape` cho trước.
    pub fn zeros(shape: &[usize]) -> Self {
        let len = shape.iter().product();
        Self {
            data: vec![0.0; len],
            strides: c_contiguous_strides(shape),
            shape: shape.to_vec(),
        }
    }

    /// Tạo mảng từ dữ liệu phẳng (row-major) + shape đã khai.
    ///
    /// Trả `Err` nếu `data.len()` không khớp tích các chiều trong `shape` —
    /// giống NumPy raise `ValueError: cannot reshape array of size X into
    /// shape Y`, nhưng ở đây là lỗi tường minh thay vì exception runtime.
    pub fn from_vec(data: Vec<f64>, shape: &[usize]) -> Result<Self, ShapeError> {
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

    /// `NdArray` do chính nó cấp phát nên luôn C-contiguous; giữ method này
    /// (thay vì hằng `true`) để `ArrayView` có thể dùng chung logic khi cả
    /// hai đều cần kiểm tra "buffer có liền mạch theo row-major không".
    pub fn is_c_contiguous(&self) -> bool {
        self.strides == c_contiguous_strides(&self.shape)
    }

    /// Đọc một phần tử theo index đầy đủ chiều, ví dụ `get(&[1, 2])` cho
    /// mảng 2 chiều. Trả `None` nếu index sai số chiều hoặc vượt biên —
    /// dùng `Option` thay vì panic để caller tự quyết định xử lý lỗi.
    pub fn get(&self, index: &[usize]) -> Option<f64> {
        if !index_in_bounds(index, &self.shape) {
            return None;
        }
        let off = offset_of(index, &self.strides);
        self.data.get(off as usize).copied()
    }

    /// Ghi một phần tử theo index đầy đủ chiều. Nhận `&mut self` — đây là
    /// điểm học `&mut`: compiler đảm bảo tại một thời điểm chỉ một nơi
    /// trong code có quyền ghi vào `self.data`, không có chuyện hai luồng
    /// cùng sửa buffer mà không qua đồng bộ như C phải tự lo bằng tay.
    pub fn set(&mut self, index: &[usize], value: f64) -> Result<(), ShapeError> {
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

    /// Mượn bất biến toàn bộ mảng dưới dạng `ArrayView`.
    ///
    /// Lifetime `'_` của view bị compiler ràng buộc vào `&self` — không
    /// thể giữ view sống lâu hơn `NdArray` gốc, đây là cách Rust giải
    /// quyết "cả rổ view alias cùng buffer" của NumPy mà không cần
    /// runtime refcounting: sai lifetime là lỗi biên dịch, không phải bug
    /// runtime (use-after-free) như C.
    pub fn view(&self) -> ArrayView<'_> {
        ArrayView::new(&self.data, self.shape.clone(), self.strides.clone(), 0)
    }

    /// Mượn có thể ghi toàn bộ mảng. Vì Rust chỉ cho **một** `&mut`
    /// tại một thời điểm, gọi `view_mut()` sẽ khóa `self` khỏi mọi truy
    /// cập khác (kể cả `view()` bất biến) cho tới khi view này hết scope —
    /// đây chính là giới hạn "&mut uniqueness" mà tài liệu NumPy.md nhắc
    /// tới khi so với việc NumPy C cho phép nhiều view cùng ghi (`out=`).
    pub fn view_mut(&mut self) -> ArrayViewMut<'_> {
        ArrayViewMut::new(&mut self.data, self.shape.clone(), self.strides.clone(), 0)
    }

    /// Basic indexing dạng slice theo range nửa-mở trên mỗi trục, ví dụ
    /// `arr.slice(&[0..2, 1..3])`. Luôn trả về **view** (không copy) —
    /// đúng ngữ nghĩa "basic indexing" của NumPy: chỉ đổi `shape`/offset
    /// trên cùng buffer.
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Result<ArrayView<'_>, ShapeError> {
        self.view().slice(ranges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_has_right_shape_and_strides() {
        let a = NdArray::zeros(&[2, 3]);
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
        assert_eq!(a.get(&[9, 9]), None); // out of bounds -> None, không panic
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
}
