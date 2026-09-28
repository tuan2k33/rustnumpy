//! rustnumpy — steps 1–16 of the plan to port NumPy to Rust (see `NumPy.md`).
//! Targets NumPy >= 2.5 semantics only — deprecated/backward-compat-only
//! NumPy behavior is out of scope by design (see `NumPy.md`'s "NumPy Parts
//! Worth Dropping" section).
//!
//! Covered so far: a basic `NdArray` (shape/strides/buffer, fixed `f64`
//! dtype), manual immutable/mutable views (slicing, broadcasting), `.npy`
//! read/write (NEP 1) cross-checked byte-for-byte against real NumPy, a
//! DType trait + the NEP 50 promotion algorithm, a small generic ufunc
//! engine (broadcasting + closures + the `out=` pattern) with both
//! sequential and Rayon-parallel element loops, a custom `Allocator`
//! trait (NEP 49) with a system and a bump-arena implementation, PyO3
//! bindings (in the separate `python/` crate) so the array is callable
//! from real Python, advanced indexing (fancy integer-array indexing,
//! boolean mask indexing) split into explicit `.oindex()`/`.vindex()`
//! methods per NEP 21's never-shipped proposal (`index.rs`), a packed
//! structured/record dtype (`structured.rs`), fixed-ratio-unit
//! `datetime64`/`timedelta64` with NaT semantics matching real NumPy
//! (`datetime.rs`), `StringDType` (NEP 55) plus a `numpy.strings`-shaped
//! subset of string ufuncs, including its missing-data sentinel
//! (`strings.rs`), a `numpy.testing` equivalent (`assert_array_equal`,
//! `assert_allclose`, `assert_array_almost_equal`) so this project's own
//! tests never depend on a running NumPy (`testing.rs`), whole-array
//! reductions/statistics (`sum`/`mean`/`var`/`std`/`median`/`percentile`,
//! their `nan*` variants, `histogram`, `cov`/`corrcoef`) in `reductions.rs`,
//! `lib/`-layer utilities (`unique`/`intersect1d`/`union1d`,
//! `concatenate`/`stack`/`split`/`tile`, `interp`, `gradient`) in `utils.rs`,
//! and full `linalg` (`solve`/`inv`/`det`/`qr`/`cholesky`/`eigh`/`eigvals`/
//! `svd`/norms/`matrix_power`) in `linalg.rs`, delegating the actual
//! numerics to the pure-Rust `faer` crate (no LAPACK/FFI) per NumPy.md's
//! "depend on a crate, don't hand-convert" decision for numerical tools
//! NumPy merely borrows, `fft` (`fft`/`ifft`/`rfft`/`irfft`/
//! `fftfreq`/`rfftfreq`/`fftshift`/`ifftshift`) in `fft.rs`, delegating to
//! the pure-Rust `rustfft` crate the same way, and a NEP 19 `Generator`
//! (`random`/`uniform`/`integers`/`standard_normal`/`normal`/
//! `exponential`/`gamma`/`beta`/`binomial`/`poisson`/`dirichlet`) in
//! `random.rs`, mapping BitGenerator/Generator onto `rand_pcg::Pcg64` +
//! `rand_distr` (statistically, not bit-stream, equivalent to NumPy's own
//! `Generator` — see `random.rs`'s doc comment).
//!
//! Deliberately **not yet** present: the ufunc engine dispatching over
//! multiple dtypes (still f64-only — step 3's `DType` isn't wired into
//! `NdArray` yet), `NdArray` itself isn't generic over `Allocator` yet
//! (that's `allocator.rs`'s own standalone `PooledVec`), a cache-optimized
//! iterator (NEP 10), generalized core-dimension ufuncs (NEP 20), boolean
//! masks over a prefix of axes, non-adjacent fancy indices in `vindex`,
//! `align=True` structured dtypes, calendar (`Y`/`M`) datetime units,
//! ISO-8601 date-string parsing, most of `numpy.strings` (only a
//! representative subset is implemented), most of `numpy.testing`
//! (no `assert_raises`-equivalent, no generic `assert_array_compare`),
//! `axis=`-parameterized reductions (whole-array only for now),
//! `array_split` (uneven splitting; `utils.rs`'s `split` requires an exact
//! division), and complex eigenvectors for a non-symmetric matrix (no
//! complex dtype yet — `linalg::eigvals` returns `(re, im)` pairs instead
//! of full `eig`, see `linalg.rs`'s doc comment), and `fft2`/`fftn`
//! (`fft.rs` is 1-D only, and works on `Vec<Complex64>`/`Vec<f64>` rather
//! than `NdArray` for the same no-complex-dtype reason), and `RandomState`/
//! the legacy `np.random.seed()` API (deliberately dropped, not a gap —
//! see `random.rs`'s doc comment and `NumPy.md`'s "Parts Worth Dropping")
//! — those are later steps (or, for the deprecated/calendar-dependent
//! pieces, explicit non-goals) in `NumPy.md`.

pub mod allocator;
pub mod datetime;
pub mod dtype;
pub mod error;
pub mod fft;
pub mod index;
pub mod linalg;
pub mod ndarray;
pub mod npy;
pub mod random;
pub mod reductions;
pub mod shape;
pub mod strings;
pub mod structured;
pub mod testing;
pub mod ufunc;
pub mod utils;
pub mod view;

pub use allocator::{AllocError, Allocator, BumpArena, PooledVec, System};
pub use datetime::{Datetime64, TimeError, TimeUnit, Timedelta64};
pub use dtype::{can_cast, common_dtype, common_dtype_of, CastSafety, DType, Kind, WeakScalar};
pub use error::ShapeError;
pub use fft::{fft, fftfreq, fftshift, ifft, ifftshift, irfft, rfft, rfftfreq, Complex64, FftError};
pub use index::AxisIndex;
pub use linalg::{
    cholesky, det, eigh, eigvals, eigvalsh, frobenius_norm, inv, matrix_power, qr, solve, svd,
    vector_norm, LinalgError, VecNormOrd,
};
pub use ndarray::NdArray;
pub use npy::{load_npy, save_npy, NpyError};
pub use random::{Generator, RandomError};
pub use reductions::{
    cov, cov_default, corrcoef, histogram, max, mean, median, min, nanmax, nanmean, nanmedian,
    nanmin, nanstd, nanstd_default, nansum, nanvar, nanvar_default, percentile, std, std_default,
    sum, var, var_default, ReductionError,
};
pub use strings::{StringArray, StringError};
pub use structured::{Field, RecordArray, RecordDType, RecordError};
pub use testing::{
    assert_allclose, assert_allclose_default, assert_array_almost_equal,
    assert_array_almost_equal_default, assert_array_equal, ArrayAssertionError,
};
pub use ufunc::{
    add, add_broadcast, add_parallel, map, map_parallel, mul, mul_parallel, sub, zip_with,
    zip_with_into, zip_with_parallel,
};
pub use utils::{concatenate, gradient, intersect1d, interp, split, stack, tile, union1d, unique};
pub use view::{ArrayView, ArrayViewMut};
