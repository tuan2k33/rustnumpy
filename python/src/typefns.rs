use crate::casting::kind_name;
use crate::dtypes::{parse_dtype, PyDtype};
use crate::pyarray::PyArray;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyComplex, PyFloat, PyInt, PyType};
use rustnumpy::{can_cast as core_can_cast, common_dtype, CastSafety, Kind, Weak};

pub fn kind_of_name(name: &str) -> Kind {
    match name {
        "bool" => Kind::Bool,
        "int8" => Kind::Int(8),
        "int16" => Kind::Int(16),
        "int32" => Kind::Int(32),
        "int64" => Kind::Int(64),
        "uint8" => Kind::Uint(8),
        "uint16" => Kind::Uint(16),
        "uint32" => Kind::Uint(32),
        "uint64" => Kind::Uint(64),
        "float16" => Kind::Float(16),
        "float32" => Kind::Float(32),
        "float64" => Kind::Float(64),
        "complex64" => Kind::Complex(32),
        _ => Kind::Complex(64),
    }
}

fn strong_kind(obj: &Bound<'_, PyAny>) -> PyResult<Kind> {
    if let Ok(a) = obj.extract::<PyRef<'_, PyArray>>() {
        return Ok(kind_of_name(a.dtype_name()));
    }
    if !obj.is_instance_of::<PyType>() {
        if let Ok(dt) = obj.getattr("dtype") {
            return Ok(kind_of_name(parse_dtype(&dt)?));
        }
    }
    Ok(kind_of_name(parse_dtype(obj)?))
}

fn casting_ok(safety: CastSafety, rule: &str) -> PyResult<bool> {
    Ok(match rule {
        "no" | "equiv" => safety == CastSafety::Equivalent,
        "safe" => matches!(safety, CastSafety::Equivalent | CastSafety::Safe),
        "same_kind" => !matches!(safety, CastSafety::Unsafe),
        "unsafe" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe', got '{other}'"
            )))
        }
    })
}

#[pyfunction]
#[pyo3(signature = (from_, to, casting="safe"))]
pub fn can_cast(from_: &Bound<'_, PyAny>, to: &Bound<'_, PyAny>, casting: &str) -> PyResult<bool> {
    if from_.is_instance_of::<PyInt>() || from_.is_instance_of::<PyFloat>() || from_.is_instance_of::<PyComplex>() {
        return Err(PyTypeError::new_err(
            "can_cast() does not support Python ints, floats, and complex because the result used to depend on the value.",
        ));
    }
    casting_ok(core_can_cast(strong_kind(from_)?, strong_kind(to)?), casting)
}

#[pyfunction]
pub fn promote_types(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyDtype> {
    let k = common_dtype(strong_kind(a)?, strong_kind(b)?);
    Ok(PyDtype { name: kind_name(k) })
}

#[pyfunction]
#[pyo3(signature = (*args))]
pub fn result_type(args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<PyDtype> {
    if args.is_empty() {
        return Err(PyValueError::new_err("at least one array or dtype is required"));
    }
    let mut strong: Vec<Kind> = Vec::new();
    let mut weak: Vec<Weak> = Vec::new();
    for a in args.iter() {
        if a.is_exact_instance_of::<PyBool>() {
            strong.push(Kind::Bool);
        } else if a.is_exact_instance_of::<PyInt>() {
            weak.push(Weak::Int);
        } else if a.is_exact_instance_of::<PyFloat>() {
            weak.push(Weak::Float);
        } else if a.is_exact_instance_of::<PyComplex>() {
            weak.push(Weak::Complex);
        } else {
            strong.push(strong_kind(&a)?);
        }
    }
    let kind = rustnumpy::result_type(strong, weak).expect("at least one argument");
    Ok(PyDtype { name: kind_name(kind) })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(can_cast, m)?)?;
    m.add_function(wrap_pyfunction!(promote_types, m)?)?;
    m.add_function(wrap_pyfunction!(result_type, m)?)?;
    Ok(())
}
