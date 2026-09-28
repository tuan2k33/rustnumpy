use std::fmt;

/// Any shape-related error: shape/data element count mismatch,
/// out-of-bounds index, or two shapes that can't be broadcast together.
///
/// Real NumPy raises a `ValueError` with a runtime message; in Rust we
/// have an explicit error enum so the caller can match on each case
/// instead of parsing a string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    /// The number of elements in the buffer doesn't match the product of `shape`.
    DataShapeMismatch { data_len: usize, shape: Vec<usize> },
    /// The index has a different number of dimensions than `ndim`, or a coordinate exceeds that axis's size.
    IndexOutOfBounds { index: Vec<usize>, shape: Vec<usize> },
    /// Two shapes can't be broadcast under NumPy's rule (compared from the
    /// trailing axis, each pair must be equal or one of them must be 1).
    NotBroadcastable { lhs: Vec<usize>, rhs: Vec<usize> },
    /// A slice range exceeds the axis size, or the number of ranges differs from `ndim`.
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
