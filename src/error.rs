use std::fmt;

/// Mọi lỗi liên quan tới shape: shape/data không khớp số phần tử,
/// index vượt biên, hoặc hai shape không broadcast được với nhau.
///
/// NumPy thật raise `ValueError` với message runtime; ở Rust ta có
/// enum lỗi tường minh để caller match theo từng trường hợp thay vì
/// parse string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    /// Số phần tử trong buffer không khớp tích các chiều trong `shape`.
    DataShapeMismatch { data_len: usize, shape: Vec<usize> },
    /// Index có số chiều khác `ndim`, hoặc một tọa độ vượt quá kích thước trục đó.
    IndexOutOfBounds { index: Vec<usize>, shape: Vec<usize> },
    /// Hai shape không thể broadcast theo quy tắc NumPy (so từ trục cuối,
    /// mỗi cặp phải bằng nhau hoặc một trong hai bằng 1).
    NotBroadcastable { lhs: Vec<usize>, rhs: Vec<usize> },
    /// Slice range vượt quá kích thước trục, hoặc số range khác `ndim`.
    InvalidSlice { shape: Vec<usize>, ranges: Vec<(usize, usize)> },
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeError::DataShapeMismatch { data_len, shape } => write!(
                f,
                "data has {data_len} elements but shape {shape:?} needs {}",
                shape.iter().product::<usize>()
            ),
            ShapeError::IndexOutOfBounds { index, shape } => {
                write!(f, "index {index:?} is out of bounds for shape {shape:?}")
            }
            ShapeError::NotBroadcastable { lhs, rhs } => {
                write!(f, "shapes {lhs:?} and {rhs:?} could not be broadcast together")
            }
            ShapeError::InvalidSlice { shape, ranges } => {
                write!(f, "ranges {ranges:?} are invalid for shape {shape:?}")
            }
        }
    }
}

impl std::error::Error for ShapeError {}
