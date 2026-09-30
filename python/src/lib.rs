#![allow(clippy::chunks_exact_to_as_chunks, clippy::redundant_guards)]

mod arrayfns;
mod casting;
mod clinalgfns;
mod createfns;
mod cxkey;
mod dlpack;
mod dtypes;
mod dynarray;
mod linalgfns;
mod logicfns;
mod ops;
mod pyarray;
mod pyindex;
mod pymethods;
mod pyops;
mod rngfns;
mod shapefns;
mod typefns;
mod umath;
mod viewfns;

pub use dynarray::{unsupported, Unsupported};
use dynarray::Arr;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

fn npy_err_to_py(e: rustnumpy::NpyError) -> PyErr {
    PyIOError::new_err(e.to_string())
}

#[pyfunction]
fn save(py: Python<'_>, path: &str, arr: &Bound<'_, PyAny>) -> PyResult<()> {
    match crate::casting::astype(&Arr::from_object(py, arr)?, "float64")? {
        Arr::F64(a) => rustnumpy::save_npy::<f64, _>(path, &a).map_err(npy_err_to_py),
        _ => unreachable!("cast to float64"),
    }
}

#[pyfunction]
fn load(py: Python<'_>, path: &str) -> PyResult<Py<PyAny>> {
    let a = rustnumpy::load_npy::<f64, _>(path).map_err(npy_err_to_py)?;
    ops::out_array(py, Arr::from(a))
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn tune_malloc() {
    extern "C" {
        fn mallopt(param: std::os::raw::c_int, value: std::os::raw::c_int) -> std::os::raw::c_int;
    }
    const M_TRIM_THRESHOLD: std::os::raw::c_int = -1;
    const M_MMAP_THRESHOLD: std::os::raw::c_int = -3;
    if std::env::var_os("RUSTNUMPY_NO_MALLOC_TUNING").is_some() {
        return;
    }
    // SAFETY: mallopt only adjusts glibc's allocator parameters and is safe to call at any time.
    unsafe {
        mallopt(M_MMAP_THRESHOLD, 32 << 20);
        mallopt(M_TRIM_THRESHOLD, 128 << 20);
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn tune_malloc() {}

#[pymodule(gil_used = true)]
#[pyo3(name = "_core")]
fn rustnumpy_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    tune_malloc();
    m.add_class::<pyarray::PyArray>()?;
    m.add("Unsupported", m.py().get_type::<Unsupported>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    dtypes::register(m)?;
    ops::register(m)?;
    arrayfns::register(m)?;
    shapefns::register(m)?;
    viewfns::register(m)?;
    createfns::register(m)?;
    logicfns::register(m)?;
    linalgfns::register(m)?;
    clinalgfns::register(m)?;
    rngfns::register(m)?;
    typefns::register(m)?;
    umath::register(m)?;
    m.add_function(wrap_pyfunction!(dlpack::from_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(save, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    Ok(())
}
