pub mod allocator;
pub mod contraction;
pub mod dispatch;
pub mod dtype;
pub mod error;
pub mod fft;
pub mod gufunc;
pub mod index;
pub mod linalg;
pub mod mathfunc;
pub mod ndarray;
pub mod npy;
pub mod polynomial;
pub mod promote;
pub mod random;
pub mod reductions;
pub mod selection;
pub mod shape;
pub mod sorting;
pub mod testing;
pub mod ufunc;
pub mod utils;
pub mod view;

pub use allocator::{AllocError, Allocator, BumpArena, PooledVec, System};
pub use contraction::{dot, einsum, matmul, outer, tensordot, tensordot_n};
pub use dtype::{can_cast, common_dtype, common_dtype_of, CastSafety, DType, Kind, WeakScalar};
pub use error::{Error, ShapeError};
pub use fft::{
    fft, fft2, fftfreq, fftn, fftshift, ifft, ifft2, ifftn, ifftshift, irfft, rfft, rfftfreq,
    Complex64, ComplexArray, FftError,
};
pub use gufunc::{gufunc, vecdot, Signature};
pub use index::AxisIndex;
pub use linalg::{
    cholesky, det, eigh, eigvals, eigvalsh, frobenius_norm, inv, matrix_norm, matrix_power, qr,
    solve, svd, vector_norm, LinalgError, VecNormOrd,
};
pub use mathfunc::Arith;
pub use ndarray::NdArray;
pub use npy::{load_npy, save_npy, NpyError};
pub use polynomial::{Polynomial, PolynomialKind};
pub use random::{Generator, RandomError};
pub use reductions::{
    cov, cov_default, corrcoef, histogram, max, mean, median, min, nanmax, nanmean, nanmedian,
    nanmin, nanstd, nanstd_default, nansum, nanvar, nanvar_default, percentile, std, std_default,
    sum, var, var_default, ReductionError,
};
pub use selection::{choose, select, where_cond, ChooseMode};
pub use sorting::{argsort, searchsorted, sort, Side};
pub use testing::{
    assert_allclose, assert_allclose_default, assert_array_almost_equal,
    assert_array_almost_equal_default, assert_array_equal, ArrayAssertionError,
};
pub use ufunc::{
    add, add_broadcast, add_parallel, map, map_parallel, mul, mul_parallel, multiply, sub,
    subtract, zip_with, zip_with_into, zip_with_parallel,
};
pub use utils::{
    array_split, array_split_at, concat, concatenate, gradient, intersect1d, interp, split, stack, tile, union1d, unique,
};
pub use view::{ArrayView, ArrayViewMut};
