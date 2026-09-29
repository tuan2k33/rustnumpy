use crate::arrayfns::{ints, norm_axis};
use crate::pyarray::{as_array, wrap, PyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[pyfunction]
pub fn reshape(py: Python<'_>, a: &Bound<'_, PyAny>, shape: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    wrap(py, as_array(py, a)?.reshaped(&ints(shape)?)?)
}

#[pyfunction]
pub fn ravel(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    wrap(py, as_array(py, a)?.reshaped(&[-1])?)
}

#[pyfunction]
#[pyo3(signature = (a, axes=None))]
pub fn transpose(py: Python<'_>, a: &Bound<'_, PyAny>, axes: Option<Vec<isize>>) -> PyResult<Py<PyAny>> {
    let arr = as_array(py, a)?;
    wrap(py, arr.derive(|v| match &axes {
        None => Ok(v.transpose()),
        Some(ax) => v.permute_dims(ax),
    })?)
}

#[pyfunction]
pub fn permute_dims(py: Python<'_>, a: &Bound<'_, PyAny>, axes: Vec<isize>) -> PyResult<Py<PyAny>> {
    wrap(py, as_array(py, a)?.derive(|v| v.permute_dims(&axes))?)
}

#[pyfunction]
pub fn swapaxes(py: Python<'_>, a: &Bound<'_, PyAny>, axis1: isize, axis2: isize) -> PyResult<Py<PyAny>> {
    let arr = as_array(py, a)?;
    let (a1, a2) = (norm_axis(axis1, arr.shape.len())?, norm_axis(axis2, arr.shape.len())?);
    wrap(py, arr.derive(|v| v.swap_axes(a1, a2))?)
}

#[pyfunction]
pub fn moveaxis(py: Python<'_>, a: &Bound<'_, PyAny>, source: &Bound<'_, PyAny>, destination: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (s, d) = (ints(source)?, ints(destination)?);
    wrap(py, as_array(py, a)?.derive(|v| v.moveaxis(&s, &d))?)
}

#[pyfunction]
pub fn expand_dims(py: Python<'_>, a: &Bound<'_, PyAny>, axis: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let axes = ints(axis)?;
    wrap(py, as_array(py, a)?.derive(|v| v.expand_dims(&axes))?)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None))]
pub fn squeeze(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let axes = axis.map(ints).transpose()?;
    wrap(py, as_array(py, a)?.derive(|v| v.squeeze(axes.as_deref()))?)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None))]
pub fn flip(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let axes = axis.map(ints).transpose()?;
    wrap(py, as_array(py, a)?.derive(|v| v.flip(axes.as_deref()))?)
}

#[pyfunction]
pub fn broadcast_to(py: Python<'_>, a: &Bound<'_, PyAny>, shape: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let dims: Vec<usize> = ints(shape)?
        .into_iter()
        .map(|d| usize::try_from(d).map_err(|_| PyValueError::new_err("all elements of broadcast shape must be non-negative")))
        .collect::<PyResult<_>>()?;
    wrap(py, as_array(py, a)?.derive(|v| v.broadcast_to(&dims))?)
}

#[pyfunction]
#[pyo3(signature = (*args))]
pub fn broadcast_arrays(py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let arrs: Vec<PyArray> = args.iter().map(|o| as_array(py, &o)).collect::<PyResult<_>>()?;
    let mut shape: Vec<usize> = Vec::new();
    for a in &arrs {
        shape = rustnumpy::shape::broadcast_shapes(&shape, &a.shape)
            .ok_or_else(|| PyValueError::new_err("shape mismatch: objects cannot be broadcast to a single shape"))?;
    }
    let outs: Vec<Py<PyAny>> = arrs.iter().map(|a| wrap(py, a.derive(|v| v.broadcast_to(&shape))?)).collect::<PyResult<_>>()?;
    Ok(PyTuple::new(py, outs)?.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (a, axis=0))]
pub fn unstack(py: Python<'_>, a: &Bound<'_, PyAny>, axis: isize) -> PyResult<Py<PyAny>> {
    let arr = as_array(py, a)?;
    if arr.shape.is_empty() {
        return Err(PyValueError::new_err("Input array must be at least 1-d."));
    }
    let ax = norm_axis(axis, arr.shape.len())?;
    let outs: Vec<Py<PyAny>> = (0..arr.shape[ax])
        .map(|i| wrap(py, arr.derive(|v| v.index_axis(ax as isize, i))?))
        .collect::<PyResult<_>>()?;
    Ok(PyTuple::new(py, outs)?.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (a, offset=0))]
pub fn diagonal(py: Python<'_>, a: &Bound<'_, PyAny>, offset: isize) -> PyResult<Py<PyAny>> {
    wrap(py, as_array(py, a)?.derive(|v| v.diagonal(offset))?)
}

#[pyfunction]
#[pyo3(signature = (a, dtype=None))]
pub fn asarray(py: Python<'_>, a: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let arr = as_array(py, a)?;
    match dtype {
        None => wrap(py, arr),
        Some(d) => {
            let name = crate::dtypes::parse_dtype(d)?;
            if name == arr.dtype_name() {
                wrap(py, arr)
            } else {
                wrap(py, PyArray::from_arr(crate::casting::astype(&arr.to_arr(), name)?))
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature = (obj, dtype=None, copy=true))]
pub fn array(py: Python<'_>, obj: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>, copy: bool) -> PyResult<Py<PyAny>> {
    let arr = as_array(py, obj)?;
    let name = match dtype {
        Some(d) => crate::dtypes::parse_dtype(d)?,
        None => arr.dtype_name(),
    };
    if !copy && name == arr.dtype_name() {
        return wrap(py, arr);
    }
    wrap(py, PyArray::from_arr(crate::casting::astype(&arr.to_arr(), name)?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! reg {
        ($($f:ident),* $(,)?) => {$( m.add_function(wrap_pyfunction!($f, m)?)?; )*};
    }
    reg!(
        reshape, ravel, transpose, permute_dims, swapaxes, moveaxis, expand_dims, squeeze, flip, broadcast_to, broadcast_arrays,
        unstack, diagonal, asarray, array
    );
    Ok(())
}
