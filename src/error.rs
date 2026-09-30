use std::fmt;

#[derive(Debug)]
pub enum Error {
    Shape(ShapeError),
    Op(OpError),
    Linalg(crate::linalg::LinalgError),
    Fft(crate::fft::FftError),
    Random(crate::random::RandomError),
    Reduction(crate::reductions::ReductionError),
    Npy(crate::npy::NpyError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Shape(e) => write!(f, "{e}"),
            Error::Op(e) => write!(f, "{e}"),
            Error::Linalg(e) => write!(f, "{e}"),
            Error::Fft(e) => write!(f, "{e}"),
            Error::Random(e) => write!(f, "{e}"),
            Error::Reduction(e) => write!(f, "{e}"),
            Error::Npy(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Shape(e) => Some(e),
            Error::Op(e) => Some(e),
            Error::Linalg(e) => Some(e),
            Error::Fft(e) => Some(e),
            Error::Random(e) => Some(e),
            Error::Reduction(e) => Some(e),
            Error::Npy(e) => Some(e),
        }
    }
}

impl From<ShapeError> for Error {
    fn from(e: ShapeError) -> Self {
        Error::Shape(e)
    }
}
impl From<OpError> for Error {
    fn from(e: OpError) -> Self {
        Error::Op(e)
    }
}
impl From<crate::linalg::LinalgError> for Error {
    fn from(e: crate::linalg::LinalgError) -> Self {
        Error::Linalg(e)
    }
}
impl From<crate::fft::FftError> for Error {
    fn from(e: crate::fft::FftError) -> Self {
        Error::Fft(e)
    }
}
impl From<crate::random::RandomError> for Error {
    fn from(e: crate::random::RandomError) -> Self {
        Error::Random(e)
    }
}
impl From<crate::reductions::ReductionError> for Error {
    fn from(e: crate::reductions::ReductionError) -> Self {
        Error::Reduction(e)
    }
}
impl From<crate::npy::NpyError> for Error {
    fn from(e: crate::npy::NpyError) -> Self {
        Error::Npy(e)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    DataShapeMismatch { data_len: usize, shape: Vec<usize> },
    IndexOutOfBounds { index: Vec<usize>, shape: Vec<usize> },
    NotBroadcastable { lhs: Vec<usize>, rhs: Vec<usize> },
    InvalidSlice { shape: Vec<usize>, ranges: Vec<(usize, usize)> },
    IndexRankMismatch { expected: usize, got: usize },
    FancyIndexOutOfBounds { axis: usize, index: usize, dim: usize },
    BooleanAxisMismatch { axis: usize, expected: usize, got: usize },
    FancyIndexNotBroadcastable { lengths: Vec<usize> },
    BooleanMaskShapeMismatch { mask_len: usize, array_shape: Vec<usize> },
    EmptyArrayList,
    AxisOutOfBounds { axis: usize, ndim: usize },
    ConcatShapeMismatch { axis: usize, shapes: Vec<Vec<usize>> },
    StackShapeMismatch { shapes: Vec<Vec<usize>> },
    NotEvenlyDivisible { axis_len: usize, sections: usize },
    ReshapeMismatch { size: usize, shape: Vec<isize> },
    MultipleUnknownDims,
    NotContiguous { shape: Vec<usize>, strides: Vec<isize> },
    ZeroSections,
    ChoiceShapeMismatch { shapes: Vec<Vec<usize>> },
    ZeroDimOperand,
    ContractionMismatch { lhs: Vec<usize>, rhs: Vec<usize> },
    InvalidAxis { axis: isize, ndim: usize },
    RepeatedAxis,
    PermutationMismatch { axes: usize, ndim: usize },
    SqueezeNotOne { axis: usize, size: usize },
    AxisCountMismatch { source: usize, destination: usize },
    RepeatLengthMismatch { repeats: usize, len: usize },
    RollShiftAxisMismatch { shifts: usize, axes: usize },
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
            ShapeError::BooleanAxisMismatch { axis, expected, got } => write!(
                f,
                "boolean index did not match indexed array along axis {axis}; size of axis is {expected} but size of corresponding boolean axis is {got}"
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
            ShapeError::ReshapeMismatch { size, shape } => {
                write!(f, "cannot reshape array of size {size} into shape {shape:?}")
            }
            ShapeError::MultipleUnknownDims => write!(f, "can only specify one unknown dimension"),
            ShapeError::NotContiguous { shape, strides } => write!(
                f,
                "cannot reshape a non-contiguous view without copying (shape {shape:?}, strides {strides:?}); call to_owned() first"
            ),
            ShapeError::ZeroSections => write!(f, "number sections must be larger than 0"),
            ShapeError::ZeroDimOperand => write!(f, "operand does not have enough dimensions (has 0, needs at least 1)"),
            ShapeError::ContractionMismatch { lhs, rhs } => {
                write!(f, "shapes {lhs:?} and {rhs:?} are not aligned for contraction")
            }
            ShapeError::InvalidAxis { axis, ndim } => {
                write!(f, "axis {axis} is out of bounds for array of dimension {ndim}")
            }
            ShapeError::RepeatedAxis => write!(f, "repeated axis"),
            ShapeError::PermutationMismatch { axes, ndim } => {
                write!(f, "axes don't match array: {axes} axes given for {ndim} dimensions")
            }
            ShapeError::SqueezeNotOne { axis, size } => write!(
                f,
                "cannot select an axis to squeeze out which has size not equal to one (axis {axis} has size {size})"
            ),
            ShapeError::AxisCountMismatch { source, destination } => write!(
                f,
                "`source` and `destination` arguments must have the same number of elements ({source} vs {destination})"
            ),
            ShapeError::RepeatLengthMismatch { repeats, len } => {
                write!(f, "operands could not be broadcast together: {repeats} repeats for axis of length {len}")
            }
            ShapeError::RollShiftAxisMismatch { shifts, axes } => {
                write!(f, "'shift' and 'axis' should be scalars or 1D sequences of the same length ({shifts} vs {axes})")
            }
            ShapeError::ChoiceShapeMismatch { shapes } => {
                write!(f, "shapes {shapes:?} could not be broadcast together")
            }
        }
    }
}

impl std::error::Error for ShapeError {}

#[derive(Debug, Clone, PartialEq)]
pub enum OpError {
    Shape(ShapeError),
    Reduction(crate::reductions::ReductionError),
    InvalidEinsum { reason: String },
    InvalidGufunc { reason: String },
    NegativeIntegerPower,
    InvalidArange,
    ChoiceIndexOutOfBounds { index: i64, choices: usize },
}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpError::Shape(e) => write!(f, "{e}"),
            OpError::Reduction(e) => write!(f, "{e}"),
            OpError::InvalidEinsum { reason } => write!(f, "invalid einsum: {reason}"),
            OpError::InvalidGufunc { reason } => write!(f, "invalid gufunc call: {reason}"),
            OpError::NegativeIntegerPower => write!(f, "integers to negative integer powers are not allowed"),
            OpError::InvalidArange => write!(f, "arange needs finite start/stop and a non-zero finite step"),
            OpError::ChoiceIndexOutOfBounds { index, choices } => {
                write!(f, "invalid entry {index} in choice array with {choices} choices")
            }
        }
    }
}

impl std::error::Error for OpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OpError::Shape(e) => Some(e),
            OpError::Reduction(e) => Some(e),
            _ => None,
        }
    }
}

impl From<crate::reductions::ReductionError> for OpError {
    fn from(e: crate::reductions::ReductionError) -> Self {
        OpError::Reduction(e)
    }
}

impl From<ShapeError> for OpError {
    fn from(e: ShapeError) -> Self {
        OpError::Shape(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray::NdArray;

    fn compose() -> Result<f64, Error> {
        let a = NdArray::from_vec(vec![1.0, 2.0, 4.0], &[1, 3])?;
        let b = crate::linalg::solve(&a, &a)?;
        Ok(b.get(&[0, 0]).unwrap())
    }

    #[test]
    fn error_composes_across_domains_through_question_mark() {
        let err = compose().unwrap_err();
        assert!(matches!(err, Error::Linalg(_)));
        assert!(err.to_string().contains("square"));
        assert!(std::error::Error::source(&err).is_some());
    }
}
