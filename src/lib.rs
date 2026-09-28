//! rustnumpy — steps 1–2 of the plan to port NumPy to Rust (see `NumPy.md`).
//!
//! Covered so far: a basic `NdArray` (shape/strides/buffer, fixed `f64`
//! dtype), manual immutable/mutable views (slicing, broadcasting), a mini
//! element-wise add to prove it all fits together, and `.npy` read/write
//! (NEP 1) cross-checked byte-for-byte against real NumPy.
//!
//! Deliberately **not yet** present: multiple dtypes (NEP 41/42), real
//! ufunc dispatch, a cache-optimized iterator (NEP 10), a custom
//! allocator (NEP 49) — those are later steps in `NumPy.md`.

pub mod error;
pub mod ndarray;
pub mod npy;
pub mod ops;
pub mod shape;
pub mod view;

pub use error::ShapeError;
pub use ndarray::NdArray;
pub use npy::{load_npy, save_npy, NpyError};
pub use ops::add_broadcast;
pub use view::{ArrayView, ArrayViewMut};
