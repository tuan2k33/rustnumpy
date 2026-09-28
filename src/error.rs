use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {

    DataShapeMismatch { data_len: usize, shape: Vec<usize> },

    IndexOutOfBounds { index: Vec<usize>, shape: Vec<usize> },

    NotBroadcastable { lhs: Vec<usize>, rhs: Vec<usize> },

    InvalidSlice { shape: Vec<usize>, ranges: Vec<(usize, usize)> },

    IndexRankMismatch { expected: usize, got: usize },

    FancyIndexOutOfBounds { axis: usize, index: usize, dim: usize },

    NonAdjacentFancyIndices { axes: Vec<usize> },

    FancyIndexNotBroadcastable { lengths: Vec<usize> },

    BooleanMaskShapeMismatch { mask_len: usize, array_shape: Vec<usize> },

    EmptyArrayList,

    AxisOutOfBounds { axis: usize, ndim: usize },

    ConcatShapeMismatch { axis: usize, shapes: Vec<Vec<usize>> },

    StackShapeMismatch { shapes: Vec<Vec<usize>> },

    NotEvenlyDivisible { axis_len: usize, sections: usize },
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
            ShapeError::EmptyArrayList => write!(f, "need at least one array"),
            ShapeError::AxisOutOfBounds { axis, ndim } => {
                write!(f, "axis {axis} is out of bounds for an array of dimension {ndim}")
            }
            ShapeError::ConcatShapeMismatch { axis, shapes } => write!(
                f,
                "all input arrays must have the same shape except along axis {axis}, got shapes {shapes:?}"
            ),
            ShapeError::StackShapeMismatch { shapes } => {
                write!(f, "all input arrays must have the same shape to stack, got shapes {shapes:?}")
            }
            ShapeError::NotEvenlyDivisible { axis_len, sections } => write!(
                f,
                "array of length {axis_len} along the split axis cannot be split into {sections} equal sections"
            ),
        }
    }
}

impl std::error::Error for ShapeError {}
