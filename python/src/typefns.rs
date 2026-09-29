use crate::casting::kind_name;
use crate::dtypes::{parse_dtype, PyDtype};
use crate::pyarray::PyArray;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyComplex, PyFloat, PyInt, PyType};
use rustnumpy::{can_cast as core_can_cast, common_dtype, CastSafety, Kind};

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

#[derive(Clone, Copy, PartialEq)]
pub enum Weak {
    Int,
    Float,
    Complex,
}

fn category(k: Kind) -> u8 {
    match k {
        Kind::Bool => 0,
        Kind::Int(_) | Kind::Uint(_) => 1,
        Kind::Float(_) => 2,
        Kind::Complex(_) => 3,
    }
}

pub fn weak_rank(w: Weak) -> u8 {
    match w {
        Weak::Int => 1,
        Weak::Float => 2,
        Weak::Complex => 3,
    }
}

pub fn with_weak(k: Kind, w: Weak) -> Kind {
    let default = match w {
        Weak::Int => Kind::Int(64),
        Weak::Float => Kind::Float(64),
        Weak::Complex => Kind::Complex(64),
    };
    if category(k) >= weak_rank(w) {
        k
    } else if weak_rank(w) == 3 && category(k) == 2 {
        Kind::Complex(if k == Kind::Float(64) { 64 } else { 32 })
    } else {
        default
    }
}

#[pyfunction]
#[pyo3(signature = (*args))]
pub fn result_type(args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<PyDtype> {
    if args.is_empty() {
        return Err(PyValueError::new_err("at least one array or dtype is required"));
    }
    let mut strong: Option<Kind> = None;
    let mut weak: Vec<Weak> = Vec::new();
    for a in args.iter() {
        if a.is_exact_instance_of::<PyBool>() {
            strong = Some(strong.map_or(Kind::Bool, |s| common_dtype(s, Kind::Bool)));
        } else if a.is_exact_instance_of::<PyInt>() {
            weak.push(Weak::Int);
        } else if a.is_exact_instance_of::<PyFloat>() {
            weak.push(Weak::Float);
        } else if a.is_exact_instance_of::<PyComplex>() {
            weak.push(Weak::Complex);
        } else {
            let k = strong_kind(&a)?;
            strong = Some(strong.map_or(k, |s| common_dtype(s, k)));
        }
    }
    let widest = weak.iter().copied().max_by_key(|&w| weak_rank(w));
    let kind = match (strong, widest) {
        (Some(k), Some(w)) => with_weak(k, w),
        (Some(k), None) => k,
        (None, Some(w)) => with_weak(Kind::Bool, w),
        (None, None) => unreachable!("at least one argument"),
    };
    Ok(PyDtype { name: kind_name(kind) })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(can_cast, m)?)?;
    m.add_function(wrap_pyfunction!(promote_types, m)?)?;
    m.add_function(wrap_pyfunction!(result_type, m)?)?;
    Ok(())
}
