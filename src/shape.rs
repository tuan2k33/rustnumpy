//! Tiện ích thuần về shape/strides, dùng chung cho `NdArray` và `ArrayView`.
//!
//! Ghi chú đơn giản hóa so với NumPy thật: ở đây `strides` tính theo **số
//! phần tử** (element count), không theo byte — vì bước 1 chỉ có một dtype
//! cố định (`f64`) nên không cần biết `size_of::<T>()`. Khi thêm hệ dtype đa
//! kiểu (NEP 41/42), stride sẽ phải đổi sang byte để mỗi mảng tự biết cách
//! nhảy con trỏ theo kích thước phần tử thật của nó.

/// Tính strides C-contiguous (row-major) cho một `shape` cho trước.
///
/// Quy tắc: trục cuối luôn có stride 1; mỗi trục trước đó có stride bằng
/// stride của trục liền sau nhân với kích thước trục đó.
///
/// Ví dụ shape `[2, 3, 4]` → strides `[12, 4, 1]`.
pub fn c_contiguous_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; shape.len()];
    let mut acc: isize = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i] as isize;
    }
    strides
}

/// Offset (tính bằng phần tử) của một index đầy đủ chiều, theo `strides`.
///
/// Đây chính là phép toán lõi biến `(i, j, k, ...)` logic thành một vị trí
/// phẳng trong buffer — không quan tâm buffer có C-contiguous hay không,
/// vì mọi thông tin layout đã nằm trong `strides`.
pub fn offset_of(index: &[usize], strides: &[isize]) -> isize {
    index
        .iter()
        .zip(strides.iter())
        .map(|(&i, &s)| i as isize * s)
        .sum()
}

/// Kiểm tra `index` có hợp lệ với `shape` không: đúng số chiều và mỗi tọa
/// độ nằm trong `[0, shape[axis])`.
pub fn index_in_bounds(index: &[usize], shape: &[usize]) -> bool {
    index.len() == shape.len() && index.iter().zip(shape.iter()).all(|(&i, &s)| i < s)
}

/// Áp quy tắc broadcasting của NumPy cho hai shape, trả về shape kết quả
/// nếu tương thích.
///
/// Quy tắc (so từ trục cuối lên đầu): mỗi cặp kích thước phải bằng nhau,
/// hoặc một trong hai bằng 1 (khi đó "giãn" theo kích thước còn lại), hoặc
/// một bên đã hết trục (coi như kích thước 1). Kết quả có số chiều bằng
/// `max(a.len(), b.len())`.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let ndim = a.len().max(b.len());
    let mut result = vec![0usize; ndim];
    for i in 0..ndim {
        // Đi từ trục cuối lên: trục thứ `i` tính từ phải sang.
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

/// Tính strides để "broadcast" một mảng có `shape`/`strides` gốc sang
/// `target_shape` (đã tương thích qua [`broadcast_shapes`]), theo đúng
/// cách NumPy làm: trục bị giãn (kích thước gốc 1, đích > 1) nhận
/// **stride 0** — nghĩa là mọi chỉ số trên trục đó đọc cùng một vị trí bộ
/// nhớ, không copy dữ liệu. Trục thiếu ở đầu (mảng ít chiều hơn) coi như
/// kích thước 1 và cũng nhận stride 0.
///
/// Trả về `None` nếu `target_shape` không phải kết quả broadcast hợp lệ
/// của `shape` (tức có trục mà kích thước gốc khác 1 và khác đích).
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
            // Trục "ảo" thêm vào phía trước — luôn stride 0.
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

/// Duyệt tuần tự mọi multi-index hợp lệ của một `shape`, theo thứ tự
/// row-major (trục cuối chạy nhanh nhất) — đúng thứ tự NumPy dùng khi
/// duyệt một mảng C-contiguous bằng `for x in np.nditer(arr)`.
///
/// Dùng cho các thao tác cần "duyệt hết mảng theo logic shape" mà không
/// quan tâm buffer bên dưới có liền mạch hay không (ví dụ vật chất hóa
/// một view có stride bất kỳ thành `NdArray` mới trong `to_owned()`).
pub struct IndexIter<'a> {
    shape: &'a [usize],
    current: Option<Vec<usize>>,
}

impl<'a> IndexIter<'a> {
    pub fn new(shape: &'a [usize]) -> Self {
        let start = if shape.contains(&0) {
            None // một trục kích thước 0 -> mảng rỗng, không có index nào
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
            // Mảng 0 chiều: đúng một index rỗng `[]`, rồi dừng.
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
        // Mọi trục đều "tràn" về 0 -> đã duyệt hết.
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
        // shape (3,1) strides (1,1) -> broadcast to (3,4): trục cuối (size 1->4) stride 0
        let strides = broadcast_strides(&[3, 1], &[1, 1], &[3, 4]).unwrap();
        assert_eq!(strides, vec![1, 0]);

        // shape (4,) strides (1,) -> broadcast to (3,4): trục đầu thêm mới, stride 0
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
