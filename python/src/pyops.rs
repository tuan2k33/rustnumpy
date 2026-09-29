use crate::dynarray::Arr;
use crate::ops::{self};
use crate::pyarray::PyArray;
use crate::{shapefns, umath};
use pyo3::basic::CompareOp;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rustnumpy::{can_cast, CastSafety};

type Bin = fn(Python<'_>, &Bound<'_, PyAny>, &Bound<'_, PyAny>) -> PyResult<Py<PyAny>>;

fn binop(slf: &Bound<'_, PyArray>, other: &Bound<'_, PyAny>, f: Bin, reflected: bool) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    let result = if reflected { f(py, other, slf.as_any()) } else { f(py, slf.as_any(), other) };
    match result {
        Err(e) if e.is_instance_of::<PyTypeError>(py) || e.is_instance(py, &py.get_type::<crate::dynarray::Unsupported>()) => {
            Ok(py.NotImplemented())
        }
        other => other,
    }
}

fn inplace(slf: &PyArray, py: Python<'_>, other: &Bound<'_, PyAny>, f: Bin, name: &str) -> PyResult<()> {
    let alias = Bound::new(
        py,
        PyArray { storage: std::sync::Arc::clone(&slf.storage), shape: slf.shape.clone(), strides: slf.strides.clone(), offset: slf.offset },
    )?;
    let result = f(py, alias.as_any(), other)?;
    let value = Arr::from_object(py, result.bind(py))?;
    let safety = can_cast(value.kind(), slf.storage.arr().kind());
    if !matches!(safety, CastSafety::Equivalent | CastSafety::Safe | CastSafety::SameKind) {
        return Err(PyTypeError::new_err(format!(
            "Cannot cast ufunc '{name}' output from dtype('{}') to dtype('{}') with casting rule 'same_kind'",
            value.dtype_name(),
            slf.dtype_name()
        )));
    }
    let positions = slf.flat_positions();
    let shaped = crate::pyindex::broadcast_arr(&value, &slf.shape)?;
    slf.write_positions(&positions, &shaped)
}

fn unary(slf: &Bound<'_, PyArray>, f: fn(Python<'_>, &Bound<'_, PyAny>) -> PyResult<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    f(slf.py(), slf.as_any())
}

macro_rules! binary_methods {
    ($(($fwd:ident, $rev:ident, $inp:ident, $f:path, $name:expr)),* $(,)?) => {
        #[pymethods]
        impl PyArray {
            $(
                fn $fwd(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> { binop(slf, other, $f, false) }
                fn $rev(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> { binop(slf, other, $f, true) }
                fn $inp(slf: PyRefMut<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> { let py = other.py(); inplace(&slf, py, other, $f, $name) }
            )*
        }
    };
}

binary_methods!(
    (__add__, __radd__, __iadd__, ops::add, "add"),
    (__sub__, __rsub__, __isub__, ops::subtract, "subtract"),
    (__mul__, __rmul__, __imul__, ops::multiply, "multiply"),
    (__truediv__, __rtruediv__, __itruediv__, umath::divide, "divide"),
    (__floordiv__, __rfloordiv__, __ifloordiv__, umath::floor_divide, "floor_divide"),
    (__mod__, __rmod__, __imod__, umath::remainder, "remainder"),
    (__and__, __rand__, __iand__, umath::bitwise_and, "bitwise_and"),
    (__or__, __ror__, __ior__, umath::bitwise_or, "bitwise_or"),
    (__xor__, __rxor__, __ixor__, umath::bitwise_xor, "bitwise_xor"),
    (__lshift__, __rlshift__, __ilshift__, umath::left_shift, "left_shift"),
    (__rshift__, __rrshift__, __irshift__, umath::right_shift, "right_shift"),
);

#[pymethods]
impl PyArray {
    fn __pow__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        if modulo.is_some_and(|m| !m.is_none()) {
            return Ok(slf.py().NotImplemented());
        }
        binop(slf, other, umath::power, false)
    }

    fn __rpow__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        if modulo.is_some_and(|m| !m.is_none()) {
            return Ok(slf.py().NotImplemented());
        }
        binop(slf, other, umath::power, true)
    }

    fn __ipow__(slf: PyRefMut<'_, Self>, other: &Bound<'_, PyAny>, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let py = other.py();
        inplace(&slf, py, other, umath::power, "power")
    }

    fn __matmul__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        binop(slf, other, shapefns::matmul, false)
    }

    fn __rmatmul__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        binop(slf, other, shapefns::matmul, true)
    }

    fn __neg__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        unary(slf, umath::negative)
    }

    fn __pos__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        Ok(slf.clone().into_any().unbind())
    }

    fn __abs__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        unary(slf, umath::absolute)
    }

    fn __invert__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        unary(slf, umath::invert)
    }

    fn __richcmp__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
        let f: Bin = match op {
            CompareOp::Eq => umath::equal,
            CompareOp::Ne => umath::not_equal,
            CompareOp::Lt => umath::less,
            CompareOp::Le => umath::less_equal,
            CompareOp::Gt => umath::greater,
            CompareOp::Ge => umath::greater_equal,
        };
        binop(slf, other, f, false)
    }
}
