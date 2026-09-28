//! rustnumpy — steps 1–5 of the plan to port NumPy to Rust (see `NumPy.md`).
//!
//! Covered so far: a basic `NdArray` (shape/strides/buffer, fixed `f64`
//! dtype), manual immutable/mutable views (slicing, broadcasting), `.npy`
//! read/write (NEP 1) cross-checked byte-for-byte against real NumPy, a
//! DType trait + the NEP 50 promotion algorithm, a small generic ufunc
//! engine (broadcasting + closures + the `out=` pattern), and a custom
//! `Allocator` trait (NEP 49) with a system and a bump-arena implementation.
//!
//! Deliberately **not yet** present: the ufunc engine dispatching over
//! multiple dtypes (still f64-only — step 3's `DType` isn't wired into
//! `NdArray` yet), `NdArray` itself isn't generic over `Allocator` yet
//! (that's `allocator.rs`'s own standalone `PooledVec`), a cache-optimized
//! iterator (NEP 10), generalized core-dimension ufuncs (NEP 20) — those
//! are later steps in `NumPy.md`.

pub mod allocator;
pub mod dtype;
pub mod error;
pub mod ndarray;
pub mod npy;
pub mod shape;
pub mod ufunc;
pub mod view;

pub use allocator::{AllocError, Allocator, BumpArena, PooledVec, System};
pub use dtype::{can_cast, common_dtype, common_dtype_of, CastSafety, DType, Kind, WeakScalar};
pub use error::ShapeError;
pub use ndarray::NdArray;
pub use npy::{load_npy, save_npy, NpyError};
pub use ufunc::{add, add_broadcast, map, mul, sub, zip_with, zip_with_into};
pub use view::{ArrayView, ArrayViewMut};
