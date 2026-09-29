use crate::dynarray::Arr;
use crate::ops::{self};
use crate::pyarray::PyArray;
use crate::{arrayfns, logicfns, shapefns};
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
    (__truediv__, __rtruediv__, __itruediv__, logicfns::divide, "divide"),
    (__floordiv__, __rfloordiv__, __ifloordiv__, ops::floor_divide, "floor_divide"),
    (__mod__, __rmod__, __imod__, ops::remainder, "remainder"),
    (__and__, __rand__, __iand__, logicfns::bitwise_and, "bitwise_and"),
    (__or__, __ror__, __ior__, logicfns::bitwise_or, "bitwise_or"),
    (__xor__, __rxor__, __ixor__, logicfns::bitwise_xor, "bitwise_xor"),
    (__lshift__, __rlshift__, __ilshift__, logicfns::left_shift, "left_shift"),
    (__rshift__, __rrshift__, __irshift__, logicfns::right_shift, "right_shift"),
);

#[pymethods]
impl PyArray {
    fn __pow__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        if modulo.is_some_and(|m| !m.is_none()) {
            return Ok(slf.py().NotImplemented());
        }
        binop(slf, other, ops::power, false)
    }

    fn __rpow__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        if modulo.is_some_and(|m| !m.is_none()) {
            return Ok(slf.py().NotImplemented());
        }
        binop(slf, other, ops::power, true)
    }

    fn __ipow__(slf: PyRefMut<'_, Self>, other: &Bound<'_, PyAny>, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let py = other.py();
        inplace(&slf, py, other, ops::power, "power")
    }

    fn __matmul__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        binop(slf, other, shapefns::matmul, false)
    }

    fn __rmatmul__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        binop(slf, other, shapefns::matmul, true)
    }

    fn __neg__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        unary(slf, ops::negative)
    }

    fn __pos__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        Ok(slf.clone().into_any().unbind())
    }

    fn __abs__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        unary(slf, ops::absolute)
    }

    fn __invert__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        unary(slf, logicfns::invert)
    }

    fn __richcmp__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
        let f: Bin = match op {
            CompareOp::Eq => logicfns::equal,
            CompareOp::Ne => logicfns::not_equal,
            CompareOp::Lt => logicfns::less,
            CompareOp::Le => logicfns::less_equal,
            CompareOp::Gt => logicfns::greater,
            CompareOp::Ge => logicfns::greater_equal,
        };
        binop(slf, other, f, false)
    }

    #[pyo3(signature = (axis=None, keepdims=false))]
    fn sum(slf: &Bound<'_, Self>, axis: Option<isize>, keepdims: bool) -> PyResult<Py<PyAny>> {
        arrayfns::sum(slf.py(), slf.as_any(), axis, keepdims)
    }

    #[pyo3(signature = (axis=None, keepdims=false))]
    fn prod(slf: &Bound<'_, Self>, axis: Option<isize>, keepdims: bool) -> PyResult<Py<PyAny>> {
        arrayfns::prod(slf.py(), slf.as_any(), axis, keepdims)
    }

    #[pyo3(signature = (axis=None, keepdims=false))]
    fn max(slf: &Bound<'_, Self>, axis: Option<isize>, keepdims: bool) -> PyResult<Py<PyAny>> {
        arrayfns::max(slf.py(), slf.as_any(), axis, keepdims)
    }

    #[pyo3(signature = (axis=None, keepdims=false))]
    fn min(slf: &Bound<'_, Self>, axis: Option<isize>, keepdims: bool) -> PyResult<Py<PyAny>> {
        arrayfns::min(slf.py(), slf.as_any(), axis, keepdims)
    }

    #[pyo3(signature = (axis=None))]
    fn mean(slf: &Bound<'_, Self>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
        arrayfns::mean(slf.py(), slf.as_any(), axis)
    }

    #[pyo3(signature = (axis=None, ddof=0))]
    fn var(slf: &Bound<'_, Self>, axis: Option<isize>, ddof: usize) -> PyResult<Py<PyAny>> {
        arrayfns::var(slf.py(), slf.as_any(), axis, ddof)
    }

    #[pyo3(signature = (axis=None, ddof=0))]
    fn std(slf: &Bound<'_, Self>, axis: Option<isize>, ddof: usize) -> PyResult<Py<PyAny>> {
        arrayfns::std_(slf.py(), slf.as_any(), axis, ddof)
    }

    #[pyo3(signature = (axis=None))]
    fn argmax(slf: &Bound<'_, Self>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
        logicfns::argmax(slf.py(), slf.as_any(), axis)
    }

    #[pyo3(signature = (axis=None))]
    fn argmin(slf: &Bound<'_, Self>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
        logicfns::argmin(slf.py(), slf.as_any(), axis)
    }

    #[pyo3(signature = (axis=None, keepdims=false))]
    fn any(slf: &Bound<'_, Self>, axis: Option<isize>, keepdims: bool) -> PyResult<Py<PyAny>> {
        logicfns::any(slf.py(), slf.as_any(), axis, keepdims)
    }

    #[pyo3(signature = (axis=None, keepdims=false))]
    fn all(slf: &Bound<'_, Self>, axis: Option<isize>, keepdims: bool) -> PyResult<Py<PyAny>> {
        logicfns::all(slf.py(), slf.as_any(), axis, keepdims)
    }

    #[pyo3(signature = (axis=None))]
    fn cumsum(slf: &Bound<'_, Self>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
        logicfns::cumsum(slf.py(), slf.as_any(), axis)
    }

    #[pyo3(signature = (axis=None))]
    fn cumprod(slf: &Bound<'_, Self>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
        logicfns::cumprod(slf.py(), slf.as_any(), axis)
    }

    #[pyo3(signature = (a_min=None, a_max=None))]
    fn clip(slf: &Bound<'_, Self>, a_min: Option<&Bound<'_, PyAny>>, a_max: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        logicfns::clip(slf.py(), slf.as_any(), a_min, a_max)
    }

    #[pyo3(signature = (axis=-1))]
    fn argsort(slf: &Bound<'_, Self>, axis: isize) -> PyResult<Py<PyAny>> {
        arrayfns::argsort(slf.py(), slf.as_any(), axis)
    }

    #[pyo3(signature = (axis=-1))]
    fn sort(slf: &Bound<'_, Self>, axis: isize) -> PyResult<()> {
        let py = slf.py();
        let sorted = arrayfns::sort(py, slf.as_any(), axis)?;
        let value = Arr::from_object(py, sorted.bind(py))?;
        let this = slf.borrow();
        this.write_positions(&this.flat_positions(), &value)
    }

    fn dot(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        shapefns::dot(slf.py(), slf.as_any(), other)
    }

    #[pyo3(signature = (offset=0))]
    fn trace(slf: &Bound<'_, Self>, offset: isize) -> PyResult<Py<PyAny>> {
        shapefns::trace(slf.py(), slf.as_any(), offset)
    }

    #[pyo3(signature = (offset=0))]
    fn diagonal(slf: &Bound<'_, Self>, offset: isize) -> PyResult<Py<PyAny>> {
        crate::viewfns::diagonal(slf.py(), slf.as_any(), offset)
    }

    #[pyo3(signature = (repeats, axis=None))]
    fn repeat(slf: &Bound<'_, Self>, repeats: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
        shapefns::repeat(slf.py(), slf.as_any(), repeats, axis)
    }
}
