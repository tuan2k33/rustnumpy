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
    /// An `oindex`/`vindex` call passed a different number of `AxisIndex`
    /// entries than the array has dimensions.
    IndexRankMismatch { expected: usize, got: usize },
    /// A `Fancy` index array contained a value out of range for its axis.
    FancyIndexOutOfBounds { axis: usize, index: usize, dim: usize },
    /// `vindex` was given `Fancy` axes that are not adjacent to each other
    /// — real NumPy would silently move the merged axis to the front; this
    /// port refuses instead of implementing that rule (see [`crate::index`]).
    NonAdjacentFancyIndices { axes: Vec<usize> },
    /// `vindex`'s `Fancy` index arrays have lengths that can't be
    /// broadcast together (every length must equal the largest, or be 1).
    FancyIndexNotBroadcastable { lengths: Vec<usize> },
    /// `boolean_index`'s mask doesn't have exactly one entry per element
    /// of the array (`mask.len()` must equal `array.len()`).
    BooleanMaskShapeMismatch { mask_len: usize, array_shape: Vec<usize> },
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
            ShapeError::IndexRankMismatch { expected, got } => {
                write!(f, "expected {expected} axis indices, got {got}")
            }
            ShapeError::FancyIndexOutOfBounds { axis, index, dim } => {
                write!(f, "fancy index {index} is out of bounds for axis {axis} with size {dim}")
            }
            ShapeError::NonAdjacentFancyIndices { axes } => write!(
                f,
                "vindex requires adjacent fancy axes, got non-adjacent axes {axes:?} \
                 (use oindex, or reorder axes so the fancy ones are next to each other)"
            ),
            ShapeError::FancyIndexNotBroadcastable { lengths } => {
                write!(f, "fancy index arrays with lengths {lengths:?} could not be broadcast together")
            }
            ShapeError::BooleanMaskShapeMismatch { mask_len, array_shape } => write!(
                f,
                "boolean mask has {mask_len} entries but array shape {array_shape:?} has {} elements",
                array_shape.iter().product::<usize>()
            ),
        }
    }
}

impl std::error::Error for ShapeError {}
