#![allow(clippy::chunks_exact_to_as_chunks, clippy::redundant_guards)]

mod arrayfns;
mod casting;
mod dynarray;
mod linalgfns;
mod ops;
mod rngfns;
mod shapefns;

pub use dynarray::{unsupported, Unsupported};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use rustnumpy::{NdArray, ShapeError};

fn shape_err_to_py(e: ShapeError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn npy_err_to_py(e: rustnumpy::NpyError) -> PyErr {
    PyIOError::new_err(e.to_string())
}

#[pyclass(name = "NdArray")]
struct PyNdArray {
    inner: NdArray,
}

#[pymethods]
impl PyNdArray {

    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> Self {
        Self { inner: NdArray::zeros(&shape) }
    }

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

#[pyfunction]
fn save_npy(path: &str, arr: &PyNdArray) -> PyResult<()> {
    rustnumpy::save_npy(path, &arr.inner).map_err(npy_err_to_py)
}

#[pyfunction]
fn load_npy(path: &str) -> PyResult<PyNdArray> {
    rustnumpy::load_npy(path).map(|inner| PyNdArray { inner }).map_err(npy_err_to_py)
}

#[pymodule]
fn rustnumpy_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNdArray>()?;
    m.add("Unsupported", m.py().get_type::<Unsupported>())?;
    ops::register(m)?;
    arrayfns::register(m)?;
    shapefns::register(m)?;
    linalgfns::register(m)?;
    rngfns::register(m)?;
    m.add_function(wrap_pyfunction!(save_npy, m)?)?;
    m.add_function(wrap_pyfunction!(load_npy, m)?)?;
    Ok(())
}
