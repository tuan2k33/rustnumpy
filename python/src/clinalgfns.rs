use crate::casting::astype;
use crate::dynarray::{Arr, C64};
use crate::linalgfns::LinAlgError;
use crate::ops::{out, out_array, shape_err};
use pyo3::prelude::*;
use rustnumpy::linalg::LinalgError;
use rustnumpy::linalg_complex as lc;
use rustnumpy::NdArray;

fn err(e: LinalgError) -> PyErr {
    if matches!(e, LinalgError::Empty) {
        return crate::dynarray::unsupported("empty matrix input is not bound");
    }
    LinAlgError::new_err(e.to_string())
}

struct In {
    a: NdArray<C64>,
    single: bool,
}

fn input(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<In> {
    let arr = Arr::from_object(py, obj)?;
    let single = matches!(arr, Arr::C64(_));
    let Arr::C128(a) = astype(&arr, "complex128")? else { unreachable!("cast to complex128") };
    Ok(In { a, single })
}

fn back(a: NdArray<C64>, single: bool) -> PyResult<Arr> {
    let arr = Arr::from(a);
    if single { astype(&arr, "complex64") } else { Ok(arr) }
}

fn real_back(values: Vec<f64>, single: bool) -> PyResult<Arr> {
    let n = values.len();
    let arr = Arr::from(NdArray::from_vec(values, &[n]).map_err(shape_err)?);
    if single { astype(&arr, "float32") } else { Ok(arr) }
}

#[pyfunction]
pub fn c_solve(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (ia, ib) = (input(py, a)?, input(py, b)?);
    out_array(py, back(lc::solve(&ia.a, &ib.a).map_err(err)?, ia.single && ib.single)?)
}

#[pyfunction]
pub fn c_inv(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let i = input(py, a)?;
    out_array(py, back(lc::inv(&i.a).map_err(err)?, i.single)?)
}

#[pyfunction]
pub fn c_det(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let i = input(py, a)?;
    let d = lc::det(&i.a).map_err(err)?;
    out(py, back(NdArray::from_vec(vec![d], &[]).map_err(shape_err)?, i.single)?)
}

#[pyfunction]
pub fn c_slogdet(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let i = input(py, a)?;
    let (s, l) = lc::slogdet(&i.a).map_err(err)?;
    let sign = back(NdArray::from_vec(vec![s], &[]).map_err(shape_err)?, i.single)?;
    let log = Arr::from(NdArray::from_vec(vec![l], &[]).map_err(shape_err)?);
    let log = if i.single { astype(&log, "float32")? } else { log };
    Ok((out(py, sign)?, out(py, log)?))
}

#[pyfunction]
pub fn c_qr(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let i = input(py, a)?;
    let (q, r) = lc::qr(&i.a).map_err(err)?;
    Ok((out_array(py, back(q, i.single)?)?, out_array(py, back(r, i.single)?)?))
}

#[pyfunction]
pub fn c_cholesky(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let i = input(py, a)?;
    out_array(py, back(lc::cholesky(&i.a).map_err(err)?, i.single)?)
}

#[pyfunction]
#[pyo3(signature = (a, uplo="L"))]
pub fn c_eigh(py: Python<'_>, a: &Bound<'_, PyAny>, uplo: &str) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let i = input(py, a)?;
    let (w, v) = lc::eigh(&i.a, uplo != "U").map_err(err)?;
    Ok((out_array(py, real_back(w, i.single)?)?, out_array(py, back(v, i.single)?)?))
}

#[pyfunction]
pub fn c_eig(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let i = input(py, a)?;
    let (w, v) = lc::eig(&i.a).map_err(err)?;
    let n = w.len();
    let wa = NdArray::from_vec(w, &[n]).map_err(shape_err)?;
    Ok((out_array(py, back(wa, i.single)?)?, out_array(py, back(v, i.single)?)?))
}

#[pyfunction]
pub fn c_svd(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let i = input(py, a)?;
    let (u, s, vh) = lc::svd(&i.a).map_err(err)?;
    Ok((out_array(py, back(u, i.single)?)?, out_array(py, real_back(s, i.single)?)?, out_array(py, back(vh, i.single)?)?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! reg {
        ($($f:ident),* $(,)?) => {$( m.add_function(wrap_pyfunction!($f, m)?)?; )*};
    }
    reg!(c_solve, c_inv, c_det, c_slogdet, c_qr, c_cholesky, c_eigh, c_eig, c_svd);
    Ok(())
}
