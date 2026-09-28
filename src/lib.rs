//! rustnumpy — bước 1 của plan port NumPy sang Rust (xem `NumPy.md`).
//!
//! Nội dung bước này: `NdArray` cơ bản (shape/strides/buffer, dtype cố
//! định `f64`), view bất biến/khả biến thủ công (slicing, broadcasting),
//! và một phép cộng element-wise mini để chứng minh mọi thứ ăn khớp.
//!
//! Cố tình **chưa** có: đa dtype (NEP 41/42), ufunc dispatch thật, iterator
//! tối ưu cache (NEP 10), allocator tùy biến (NEP 49) — những phần đó là
//! các bước sau trong `NumPy.md`.

pub mod error;
pub mod ndarray;
pub mod ops;
pub mod shape;
pub mod view;

pub use error::ShapeError;
pub use ndarray::NdArray;
pub use ops::add_broadcast;
pub use view::{ArrayView, ArrayViewMut};
