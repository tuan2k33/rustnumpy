pub mod allocator;
pub mod contraction;
pub mod creation;
pub mod dispatch;
pub mod dtype;
pub mod error;
pub mod fft;
pub mod gufunc;
pub mod index;
pub mod linalg;
pub mod linalg_complex;
pub mod logic;
pub mod manipulation;
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
pub mod ufunc;
pub mod view;

pub use allocator::{AllocError, Allocator, BumpArena, PooledVec, System};
pub use contraction::{dot, einsum, matmul, outer, tensordot, tensordot_n};
pub use dtype::{can_cast, common_dtype, common_dtype_of, CastSafety, DType, Kind, WeakScalar};
pub use error::{Error, OpError, ShapeError};
pub use fft::{FftFloat, 
    fft, fft2, fftfreq, fftn, fftshift, hfft, ifft, ifft2, ifftn, ifftshift, ihfft, irfft, irfft2,
    irfftn, rfft, rfft2, rfftfreq, rfftn, to_complex, Complex64, FftError,
};
pub use gufunc::{gufunc, vecdot, Signature};
pub use index::AxisIndex;
pub use linalg::{
    cholesky, cond, det, eig, eigh, eigvals, eigvalsh, frobenius_norm, inv, lstsq, matrix_norm,
    matrix_norm_ord, matrix_power, matrix_rank, pinv, qr, slogdet, solve, svd, svdvals,
    vector_norm, LinalgError, Lstsq, MatNormOrd, VecNormOrd,
};
pub use contraction::{cross, kron, trace};
pub use manipulation::{
    array_split, array_split_at, broadcast_arrays, concat, concatenate, expand_dims, gradient, interp, intersect1d, repeat, roll,
    split, stack, tile, union1d, unique, unique_all, unique_counts, unique_inverse, unique_values, unstack, UniqueAll,
};
pub use mathfunc::Arith;
pub use ndarray::NdArray;
pub use npy::{load_npy, save_npy, NpyError};
pub use polynomial::{Polynomial, PolynomialKind};
pub use random::{default_rng, mvn_factor, Generator, MvnMethod, RandomError, SeedSequence};
pub use reductions::{
    cov, cov_default, corrcoef, histogram, max, mean, median, min, nanmax, nanmean, nanmedian,
    nanmin, nanstd, nanstd_default, nansum, nanvar, nanvar_default, percentile, std, std_default,
    sum, var, var_default, ReductionError,
};
pub use selection::{choose, select, where_cond, ChooseMode};
pub use sorting::{argsort, searchsorted, sort, Side};
pub use ufunc::{
    add, add_broadcast, add_parallel, map, map_parallel, mul, mul_parallel, multiply, sub,
    subtract, zip_with, zip_with_into, zip_with_parallel,
};
pub use view::{ArrayView, ArrayViewMut};
