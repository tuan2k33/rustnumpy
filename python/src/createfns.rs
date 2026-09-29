use crate::casting::astype;
use crate::dtypes::parse_dtype;
use crate::dynarray::{unsupported, Arr};
use crate::ops::{out_array, shape_err, Operand};
use crate::pyarray::as_array;
use crate::with_arr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustnumpy::NdArray;

fn shape_of(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let dims = crate::arrayfns::ints(obj)?;
    dims.into_iter().map(|d| usize::try_from(d).map_err(|_| PyValueError::new_err("negative dimensions are not allowed"))).collect()
}

fn dtype_or(dtype: Option<&Bound<'_, PyAny>>, default: &'static str) -> PyResult<&'static str> {
    dtype.map_or(Ok(default), parse_dtype)
}

fn available_bytes() -> usize {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|t| {
            t.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<usize>().ok())
        })
        .map_or(usize::MAX / 2, |kb| kb.saturating_mul(1024))
}

pub fn alloc_guard(elems: usize, bytes_per: usize) -> PyResult<()> {
    let bytes = elems.checked_mul(bytes_per).filter(|&b| b <= isize::MAX as usize);
    let ok = bytes.is_some_and(|b| b <= available_bytes() / 2);
    if ok {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyMemoryError::new_err(format!(
            "Unable to allocate an array with {elems} elements of {bytes_per} bytes"
        )))
    }
}

pub fn shape_elems(shape: &[usize]) -> PyResult<usize> {
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d)).ok_or_else(|| PyValueError::new_err("array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size."))
}

pub fn filled(shape: &[usize], value: f64, dtype: &str) -> PyResult<Arr> {
    let n = shape_elems(shape)?;
    alloc_guard(n, 16)?;
    let base = Arr::from(NdArray::from_vec(vec![value; n], shape).map_err(shape_err)?);
    astype(&base, dtype)
}

#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
pub fn zeros(py: Python<'_>, shape: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    out_array(py, filled(&shape_of(shape)?, 0.0, dtype_or(dtype, "float64")?)?)
}

#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
pub fn ones(py: Python<'_>, shape: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    out_array(py, filled(&shape_of(shape)?, 1.0, dtype_or(dtype, "float64")?)?)
}

#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
pub fn empty(py: Python<'_>, shape: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    zeros(py, shape, dtype)
}

fn full_like_arr(py: Python<'_>, shape: &[usize], value: &Bound<'_, PyAny>, dtype: Option<&'static str>) -> PyResult<Arr> {
    let v = Operand::parse(py, value)?.into_arr(py)?;
    let v = match dtype {
        Some(d) => astype(&v, d)?,
        None => v,
    };
    if v.ndim() != 0 {
        return Err(unsupported("full() needs a scalar fill value"));
    }
    alloc_guard(shape_elems(shape)?, 16)?;
    Ok(with_arr!(&v, a => Arr::from(rustnumpy::creation::full(shape, a.as_slice()[0]))))
}

#[pyfunction]
#[pyo3(signature = (shape, fill_value, dtype=None))]
pub fn full(py: Python<'_>, shape: &Bound<'_, PyAny>, fill_value: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let d = dtype.map(parse_dtype).transpose()?;
    out_array(py, full_like_arr(py, &shape_of(shape)?, fill_value, d)?)
}

macro_rules! like_fn {
    ($name:ident, $value:expr) => {
        #[pyfunction]
        #[pyo3(signature = (a, dtype=None))]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
            let arr = as_array(py, a)?;
            let name = dtype_or(dtype, arr.dtype_name())?;
            out_array(py, filled(&arr.shape, $value, name)?)
        }
    };
}
like_fn!(zeros_like, 0.0);
like_fn!(ones_like, 1.0);
like_fn!(empty_like, 0.0);

#[pyfunction]
#[pyo3(signature = (a, fill_value, dtype=None))]
pub fn full_like(py: Python<'_>, a: &Bound<'_, PyAny>, fill_value: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let arr = as_array(py, a)?;
    let name = dtype_or(dtype, arr.dtype_name())?;
    out_array(py, full_like_arr(py, &arr.shape, fill_value, Some(name))?)
}

#[pyfunction]
#[pyo3(signature = (start, stop=None, step=None, dtype=None))]
pub fn arange(
    py: Python<'_>,
    start: &Bound<'_, PyAny>,
    stop: Option<&Bound<'_, PyAny>>,
    step: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let (lo, hi) = match stop {
        Some(s) => (start, s),
        None => (&0i64.into_pyobject(py)?.into_any(), start),
    };
    let one = 1i64.into_pyobject(py)?.into_any();
    let step = step.unwrap_or(&one);
    let ops = [Operand::parse(py, lo)?, Operand::parse(py, hi)?, Operand::parse(py, step)?];
    let all_int = ops.iter().all(|o| match o {
        Operand::WeakInt(_) => true,
        Operand::Arr(a) => a.ndim() == 0 && matches!(a, Arr::I8(_) | Arr::I16(_) | Arr::I32(_) | Arr::I64(_) | Arr::U8(_) | Arr::U16(_) | Arr::U32(_) | Arr::U64(_)),
        Operand::WeakFloat(_) | Operand::WeakComplex(..) => false,
    });
    let as_f64 = |o: &Operand| -> PyResult<f64> {
        match o {
            Operand::WeakInt(v) => Ok(*v as f64),
            Operand::WeakFloat(v) => Ok(*v),
            Operand::WeakComplex(..) => Err(pyo3::exceptions::PyTypeError::new_err("arange does not accept complex bounds")),
            Operand::Arr(a) => match astype(a, "float64")? {
                Arr::F64(x) if x.ndim() == 0 => Ok(x.as_slice()[0]),
                _ => Err(unsupported("arange bounds must be scalars")),
            },
        }
    };
    let result = if all_int {
        let as_i64 = |o: &Operand| -> PyResult<i64> {
            match o {
                Operand::WeakInt(v) => Ok(*v),
                other => Ok(as_f64(other)? as i64),
            }
        };
        Arr::from(rustnumpy::creation::arange_i64(as_i64(&ops[0])?, as_i64(&ops[1])?, as_i64(&ops[2])?).map_err(shape_err)?)
    } else {
        Arr::from(rustnumpy::creation::arange_f64(as_f64(&ops[0])?, as_f64(&ops[1])?, as_f64(&ops[2])?).map_err(shape_err)?)
    };
    let result = match dtype {
        Some(d) => astype(&result, parse_dtype(d)?)?,
        None => result,
    };
    out_array(py, result)
}

#[pyfunction]
#[pyo3(signature = (start, stop, num=50, endpoint=true, dtype=None))]
pub fn linspace(py: Python<'_>, start: f64, stop: f64, num: usize, endpoint: bool, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    alloc_guard(num, 16)?;
    let r = Arr::from(rustnumpy::creation::linspace(start, stop, num, endpoint));
    let r = match dtype {
        Some(d) => astype(&r, parse_dtype(d)?)?,
        None => r,
    };
    out_array(py, r)
}

#[pyfunction]
#[pyo3(signature = (n, m=None, k=0, dtype=None))]
pub fn eye(py: Python<'_>, n: usize, m: Option<usize>, k: isize, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    alloc_guard(n.saturating_mul(m.unwrap_or(n)), 16)?;
    let r = Arr::from(rustnumpy::creation::eye(n, m.unwrap_or(n), k, 1.0f64));
    let r = match dtype {
        Some(d) => astype(&r, parse_dtype(d)?)?,
        None => r,
    };
    out_array(py, r)
}

#[pyfunction]
#[pyo3(signature = (n, dtype=None))]
pub fn identity(py: Python<'_>, n: usize, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    eye(py, n, None, 0, dtype)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! reg {
        ($($f:ident),* $(,)?) => {$( m.add_function(wrap_pyfunction!($f, m)?)?; )*};
    }
    reg!(zeros, ones, empty, full, zeros_like, ones_like, empty_like, full_like, arange, linspace, eye, identity);
    Ok(())
}
