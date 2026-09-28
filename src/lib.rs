//! rustnumpy — steps 1–12 of the plan to port NumPy to Rust (see `NumPy.md`).
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
//! tests never depend on a running NumPy (`testing.rs`), and whole-array
//! reductions/statistics (`sum`/`mean`/`var`/`std`/`median`/`percentile`,
//! their `nan*` variants, `histogram`, `cov`/`corrcoef`) in `reductions.rs`.
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
//! (no `assert_raises`-equivalent, no generic `assert_array_compare`), and
//! `axis=`-parameterized reductions (whole-array only for now) — those are
//! later steps (or, for the deprecated/calendar-dependent pieces, explicit
//! non-goals) in `NumPy.md`.

pub mod allocator;
pub mod datetime;
pub mod dtype;
pub mod error;
pub mod index;
pub mod ndarray;
pub mod npy;
pub mod reductions;
pub mod shape;
pub mod strings;
pub mod structured;
pub mod testing;
pub mod ufunc;
pub mod view;

pub use allocator::{AllocError, Allocator, BumpArena, PooledVec, System};
pub use datetime::{Datetime64, TimeError, TimeUnit, Timedelta64};
pub use dtype::{can_cast, common_dtype, common_dtype_of, CastSafety, DType, Kind, WeakScalar};
pub use error::ShapeError;
pub use index::AxisIndex;
pub use ndarray::NdArray;
pub use npy::{load_npy, save_npy, NpyError};
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
pub use view::{ArrayView, ArrayViewMut};
