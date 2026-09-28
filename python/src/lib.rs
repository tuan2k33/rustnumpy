//! Step 6: PyO3 bindings, so a real Python interpreter can create, read,
//! and operate on the Rust `NdArray` from step 1 onwards.
//!
//! Kept as its own crate (`python/`, package name `rustnumpy-python`)
//! rather than folded into the main `rustnumpy` crate, so the core
//! library's `cargo test`/`cargo clippy` never has to know PyO3 exists —
//! this crate just depends on it via a path dependency.
//!
//! No performance comparison against real NumPy yet, on purpose: the core
//! only has `f64`, no reductions, no generalized ufuncs — a benchmark run
//! now would just be measuring "how fast is one closure-driven loop",
//! not "is this a viable NumPy replacement". That comparison is worth
//! doing once there's enough surface area for it to mean something.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use rustnumpy::{NdArray, ShapeError};

/// Turn our own `ShapeError` into a Python `ValueError` with the same
/// message `Display` already produces — Python code sees a normal
/// exception instead of having to know about a Rust-specific error type.
fn shape_err_to_py(e: ShapeError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn npy_err_to_py(e: rustnumpy::NpyError) -> PyErr {
    PyIOError::new_err(e.to_string())
}

/// The Python-visible wrapper around `rustnumpy::NdArray`. PyO3 needs an
/// owned struct behind every `#[pyclass]` (Python objects are
/// reference-counted and garbage-collected, so it can't just hand out
/// borrows into some Rust-side value with a shorter lifetime) — this is
/// exactly the ownership boundary NumPy.md's memory-model section flags
/// as the hard part of embedding: Python's object model assumes shared,
/// GC'd ownership, Rust's assumes single-owner-plus-borrows, and a
/// binding layer is where those two models have to meet.
#[pyclass(name = "NdArray")]
struct PyNdArray {
    inner: NdArray,
}

#[pymethods]
impl PyNdArray {
    /// `NdArray.zeros([2, 3])`
    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> Self {
        Self { inner: NdArray::zeros(&shape) }
    }

    /// `NdArray.from_list([1.0, 2.0, 3.0, 4.0], [2, 2])` — `data` is flat,
    /// row-major, matching `NdArray::from_vec`.
    #[staticmethod]
    fn from_list(data: Vec<f64>, shape: Vec<usize>) -> PyResult<Self> {
        NdArray::from_vec(data, &shape).map(|inner| Self { inner }).map_err(shape_err_to_py)
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Flatten back to a plain Python list, row-major — the round-trip
    /// partner of `from_list`, and the easiest way for Python-side test
    /// code to check a result without walking `shape` by hand.
    fn to_list(&self) -> Vec<f64> {
        self.inner.as_slice().to_vec()
    }

    fn get(&self, index: Vec<usize>) -> Option<f64> {
        self.inner.get(&index)
    }

    fn __repr__(&self) -> String {
        format!("NdArray(shape={:?}, data={:?})", self.inner.shape(), self.inner.as_slice())
    }

    fn __eq__(&self, other: &PyNdArray) -> bool {
        self.inner == other.inner
    }
}

/// `rustnumpy.add(a, b)` — broadcasting element-wise add, backed by the
/// exact same `ufunc::add` step 4 already tested against real NumPy.
#[pyfunction]
fn add(a: &PyNdArray, b: &PyNdArray) -> PyResult<PyNdArray> {
    rustnumpy::add(&a.inner.view(), &b.inner.view())
        .map(|inner| PyNdArray { inner })
        .map_err(shape_err_to_py)
}

#[pyfunction]
fn sub(a: &PyNdArray, b: &PyNdArray) -> PyResult<PyNdArray> {
    rustnumpy::sub(&a.inner.view(), &b.inner.view())
        .map(|inner| PyNdArray { inner })
        .map_err(shape_err_to_py)
}

#[pyfunction]
fn mul(a: &PyNdArray, b: &PyNdArray) -> PyResult<PyNdArray> {
    rustnumpy::mul(&a.inner.view(), &b.inner.view())
        .map(|inner| PyNdArray { inner })
        .map_err(shape_err_to_py)
}

/// `rustnumpy.save_npy(path, arr)` — the step 2 writer, byte-identical to
/// real NumPy's own `.npy` output (see `npy::tests::writer_output_is_byte_identical_to_real_numpy`).
#[pyfunction]
fn save_npy(path: &str, arr: &PyNdArray) -> PyResult<()> {
    rustnumpy::save_npy(path, &arr.inner).map_err(npy_err_to_py)
}

/// `rustnumpy.load_npy(path)` — reads a `.npy` file, including ones
/// written by real NumPy's own `np.save`.
#[pyfunction]
fn load_npy(path: &str) -> PyResult<PyNdArray> {
    rustnumpy::load_npy(path).map(|inner| PyNdArray { inner }).map_err(npy_err_to_py)
}

#[pymodule]
fn rustnumpy_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNdArray>()?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(save_npy, m)?)?;
    m.add_function(wrap_pyfunction!(load_npy, m)?)?;
    Ok(())
}
