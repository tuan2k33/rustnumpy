use crate::dtypes::PyDtype;
use crate::dynarray::Arr;
use num_complex::Complex;
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyAttributeError, PyTypeError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyComplex, PyDict, PyFloat, PyInt, PyString, PyTuple};

#[pyclass(extends = PyFloat, frozen, module = "rustnumpy", name = "float64")]
pub struct F64;

#[pyclass(extends = PyComplex, frozen, module = "rustnumpy", name = "complex128")]
pub struct C128;

#[pyclass(frozen, module = "rustnumpy", name = "int64")]
pub struct I64(pub i64);

#[pyclass(frozen, module = "rustnumpy", name = "bool")]
pub struct Bool(pub bool);

#[derive(Clone, Copy)]
enum Num {
    F(f64),
    I(i64, bool),
    B(bool),
    C(f64, f64),
}

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Add,
    Sub,
    Mul,
    TrueDiv,
    FloorDiv,
    Mod,
    Pow,
    LShift,
    RShift,
    And,
    Or,
    Xor,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Op {
    fn core_name(self) -> &'static str {
        match self {
            Op::Add => "add",
            Op::Sub => "subtract",
            Op::Mul => "multiply",
            Op::TrueDiv => "divide",
            Op::FloorDiv => "floor_divide",
            Op::Mod => "remainder",
            Op::Pow => "power",
            Op::LShift => "left_shift",
            Op::RShift => "right_shift",
            Op::And => "bitwise_and",
            Op::Or => "bitwise_or",
            Op::Xor => "bitwise_xor",
            Op::Eq => "equal",
            Op::Ne => "not_equal",
            Op::Lt => "less",
            Op::Le => "less_equal",
            Op::Gt => "greater",
            Op::Ge => "greater_equal",
        }
    }

    fn dunder(self, reflected: bool) -> String {
        let short = match self {
            Op::Add => "add",
            Op::Sub => "sub",
            Op::Mul => "mul",
            Op::TrueDiv => "truediv",
            Op::FloorDiv => "floordiv",
            Op::Mod => "mod",
            Op::Pow => "pow",
            Op::LShift => "lshift",
            Op::RShift => "rshift",
            Op::And => "and",
            Op::Or => "or",
            Op::Xor => "xor",
            Op::Eq => return "__eq__".into(),
            Op::Ne => return "__ne__".into(),
            Op::Lt => return "__lt__".into(),
            Op::Le => return "__le__".into(),
            Op::Gt => return "__gt__".into(),
            Op::Ge => return "__ge__".into(),
        };
        format!("__{}{short}__", if reflected { "r" } else { "" })
    }

    fn is_comparison(self) -> bool {
        matches!(self, Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge)
    }
}

pub fn new_f64(py: Python<'_>, v: f64) -> PyResult<Py<PyAny>> {
    let tp = py.get_type::<F64>();
    // SAFETY: tp_alloc returns a zeroed instance of the float subclass (the zeroed PyO3 contents of this field-less
    // class mean "never borrowed, no dict"); ob_fval is the float payload and nothing else references the object yet.
    unsafe {
        let raw = ffi::PyType_GenericAlloc(tp.as_type_ptr(), 0);
        if raw.is_null() {
            return Err(PyErr::fetch(py));
        }
        (*(raw as *mut ffi::PyFloatObject)).ob_fval = v;
        Ok(Py::from_owned_ptr(py, raw))
    }
}

pub fn new_c128(py: Python<'_>, re: f64, im: f64) -> PyResult<Py<PyAny>> {
    let tp = py.get_type::<C128>();
    // SAFETY: as in `new_f64`, for the complex payload.
    unsafe {
        let raw = ffi::PyType_GenericAlloc(tp.as_type_ptr(), 0);
        if raw.is_null() {
            return Err(PyErr::fetch(py));
        }
        (*(raw as *mut ffi::PyComplexObject)).cval = ffi::Py_complex { real: re, imag: im };
        Ok(Py::from_owned_ptr(py, raw))
    }
}

pub fn new_i64(py: Python<'_>, v: i64) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, I64(v))?.into_any())
}

pub fn new_bool(py: Python<'_>, v: bool) -> PyResult<Py<PyAny>> {
    static BOOLS: PyOnceLock<[Py<Bool>; 2]> = PyOnceLock::new();
    let both = BOOLS.get_or_try_init(py, || -> PyResult<[Py<Bool>; 2]> { Ok([Py::new(py, Bool(false))?, Py::new(py, Bool(true))?]) })?;
    Ok(both[usize::from(v)].clone_ref(py).into_any())
}

fn num_py(py: Python<'_>, n: Num) -> PyResult<Py<PyAny>> {
    match n {
        Num::F(v) => new_f64(py, v),
        Num::I(v, _) => new_i64(py, v),
        Num::B(v) => new_bool(py, v),
        Num::C(re, im) => new_c128(py, re, im),
    }
}

fn num_of(o: &Bound<'_, PyAny>) -> Option<Num> {
    if let Ok(f) = o.downcast_exact::<F64>() {
        return Some(Num::F(f.as_super().value()));
    }
    if let Ok(f) = o.downcast_exact::<PyFloat>() {
        return Some(Num::F(f.value()));
    }
    if let Ok(i) = o.downcast_exact::<I64>() {
        return Some(Num::I(i.get().0, false));
    }
    if let Ok(i) = o.downcast_exact::<PyInt>() {
        return i.extract::<i64>().ok().map(|v| Num::I(v, true));
    }
    if let Ok(b) = o.downcast_exact::<Bool>() {
        return Some(Num::B(b.get().0));
    }
    if let Ok(b) = o.downcast_exact::<PyBool>() {
        return Some(Num::B(b.is_true()));
    }
    if let Ok(c) = o.downcast_exact::<C128>() {
        let c = c.as_super();
        return Some(Num::C(c.real(), c.imag()));
    }
    if let Ok(c) = o.downcast_exact::<PyComplex>() {
        return Some(Num::C(c.real(), c.imag()));
    }
    None
}

fn is_number_like(o: &Bound<'_, PyAny>) -> bool {
    o.is_exact_instance_of::<PyInt>()
        || o.is_exact_instance_of::<PyBool>()
        || o.is_exact_instance_of::<PyFloat>()
        || o.is_exact_instance_of::<PyComplex>()
        || o.is_exact_instance_of::<F64>()
        || o.is_exact_instance_of::<I64>()
        || o.is_exact_instance_of::<Bool>()
        || o.is_exact_instance_of::<C128>()
}

fn as_complex(n: Num) -> (f64, f64) {
    match n {
        Num::F(v) => (v, 0.0),
        Num::I(v, _) => (v as f64, 0.0),
        Num::B(v) => (f64::from(u8::from(v)), 0.0),
        Num::C(re, im) => (re, im),
    }
}

fn as_float(n: Num) -> f64 {
    as_complex(n).0
}

fn as_int(n: Num) -> i64 {
    match n {
        Num::I(v, _) => v,
        Num::B(v) => i64::from(v),
        _ => unreachable!("integer operands only"),
    }
}

fn cmp_ord<T: PartialOrd>(op: Op, a: T, b: T) -> Option<bool> {
    Some(match op {
        Op::Eq => a == b,
        Op::Ne => a != b,
        Op::Lt => a < b,
        Op::Le => a <= b,
        Op::Gt => a > b,
        Op::Ge => a >= b,
        _ => return None,
    })
}

fn eval(op: Op, l: Num, r: Num) -> Option<Num> {
    let any_c = matches!(l, Num::C(..)) || matches!(r, Num::C(..));
    let any_f = matches!(l, Num::F(_)) || matches!(r, Num::F(_));
    if any_c {
        let ((a, b), (c, d)) = (as_complex(l), as_complex(r));
        return Some(match op {
            Op::Add => Num::C(a + c, b + d),
            Op::Sub => Num::C(a - c, b - d),
            Op::Mul => Num::C(a * c - b * d, a * d + b * c),
            Op::Eq => Num::B(a == c && b == d),
            Op::Ne => Num::B(!(a == c && b == d)),
            _ => return None,
        });
    }
    if any_f {
        let python_int = |n: Num| matches!(n, Num::I(_, true));
        if op.is_comparison() && (python_int(l) || python_int(r)) {
            return None;
        }
        let (a, b) = (as_float(l), as_float(r));
        return Some(match op {
            Op::Add => Num::F(a + b),
            Op::Sub => Num::F(a - b),
            Op::Mul => Num::F(a * b),
            Op::TrueDiv => Num::F(a / b),
            Op::Pow => Num::F(a.powf(b)),
            _ => Num::B(cmp_ord(op, a, b)?),
        });
    }
    if let (Num::B(a), Num::B(b)) = (l, r) {
        return Some(Num::B(match op {
            Op::And => a & b,
            Op::Or => a | b,
            Op::Xor => a ^ b,
            _ => cmp_ord(op, a, b)?,
        }));
    }
    let (a, b) = (as_int(l), as_int(r));
    Some(match op {
        Op::Add => Num::I(a.checked_add(b)?, false),
        Op::Sub => Num::I(a.checked_sub(b)?, false),
        Op::Mul => Num::I(a.checked_mul(b)?, false),
        Op::TrueDiv => Num::F(a as f64 / b as f64),
        Op::FloorDiv => {
            let q = a.checked_div(b)?;
            let r = a.checked_rem(b)?;
            Num::I(if r != 0 && ((r < 0) != (b < 0)) { q - 1 } else { q }, false)
        }
        Op::Mod => {
            let r = a.checked_rem(b)?;
            Num::I(if r != 0 && ((r < 0) != (b < 0)) { r + b } else { r }, false)
        }
        Op::And => Num::I(a & b, false),
        Op::Or => Num::I(a | b, false),
        Op::Xor => Num::I(a ^ b, false),
        Op::Pow | Op::LShift | Op::RShift => return None,
        _ => Num::B(cmp_ord(op, a, b)?),
    })
}

fn core(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    static CORE: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    Ok(CORE.get_or_try_init(py, || py.import("rustnumpy._core").map(Bound::unbind))?.bind(py))
}

fn plain(py: Python<'_>, n: Num) -> PyResult<Py<PyAny>> {
    Ok(match n {
        Num::F(v) => PyFloat::new(py, v).into_any().unbind(),
        Num::I(v, _) => v.into_pyobject(py)?.into_any().unbind(),
        Num::B(v) => PyBool::new(py, v).to_owned().into_any().unbind(),
        Num::C(re, im) => PyComplex::from_doubles(py, re, im).into_any().unbind(),
    })
}

fn array_of(py: Python<'_>, n: Num) -> PyResult<Py<PyAny>> {
    let arr = match n {
        Num::F(v) => Arr::scalar(v),
        Num::I(v, _) => Arr::scalar(v),
        Num::B(v) => Arr::scalar(v),
        Num::C(re, im) => Arr::scalar(Complex::new(re, im)),
    };
    crate::ops::out_array(py, arr)
}

fn me(slf: &Bound<'_, PyAny>) -> Num {
    num_of(slf).expect("scalar methods are only defined on the scalar classes")
}

fn through_array(slf: &Bound<'_, PyAny>, name: &str, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    Ok(array_of(py, me(slf))?.bind(py).call_method1(name, (other,))?.unbind())
}

fn binary(slf: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>, op: Op, reflected: bool) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    let this = me(slf);
    if let Some(o) = num_of(other) {
        let (l, r) = if reflected { (o, this) } else { (this, o) };
        if let Some(v) = eval(op, l, r) {
            return num_py(py, v);
        }
    }
    if is_number_like(other) {
        let f = core(py)?.getattr(op.core_name())?;
        return Ok(if reflected { f.call1((other, slf))? } else { f.call1((slf, other))? }.unbind());
    }
    through_array(slf, &op.dunder(reflected), other)
}

fn compare(slf: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
    let op = match op {
        CompareOp::Eq => Op::Eq,
        CompareOp::Ne => Op::Ne,
        CompareOp::Lt => Op::Lt,
        CompareOp::Le => Op::Le,
        CompareOp::Gt => Op::Gt,
        CompareOp::Ge => Op::Ge,
    };
    binary(slf, other, op, false)
}

fn unary(slf: &Bound<'_, PyAny>, name: &str) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    let fast = match (name, me(slf)) {
        ("__neg__", Num::F(v)) => Some(Num::F(-v)),
        ("__neg__", Num::I(v, _)) => v.checked_neg().map(|r| Num::I(r, false)),
        ("__neg__", Num::C(re, im)) => Some(Num::C(-re, -im)),
        ("__abs__", Num::F(v)) => Some(Num::F(v.abs())),
        ("__abs__", Num::I(v, _)) => v.checked_abs().map(|r| Num::I(r, false)),
        ("__abs__", Num::C(re, im)) => Some(Num::F(re.hypot(im))),
        ("__invert__", Num::I(v, _)) => Some(Num::I(!v, false)),
        ("__invert__", Num::B(v)) => Some(Num::B(!v)),
        _ => None,
    };
    if let Some(v) = fast {
        return num_py(py, v);
    }
    Ok(array_of(py, me(slf))?.bind(py).call_method0(name)?.unbind())
}

fn dtype_name(n: Num) -> &'static str {
    match n {
        Num::F(_) => "float64",
        Num::I(..) => "int64",
        Num::B(_) => "bool",
        Num::C(..) => "complex128",
    }
}

macro_rules! scalar_methods {
    ($ty:ident) => {
        #[pymethods]
        impl $ty {
            #[classattr]
            fn _rnp_scalar() -> bool {
                true
            }

            #[classattr]
            fn _rnp_dtype(py: Python<'_>) -> PyResult<Py<PyString>> {
                let name = match stringify!($ty) {
                    "F64" => "float64",
                    "C128" => "complex128",
                    "I64" => "int64",
                    _ => "bool",
                };
                Ok(PyString::new(py, name).unbind())
            }

            #[classattr]
            fn ndim() -> usize {
                0
            }

            #[classattr]
            fn size() -> usize {
                1
            }

            #[classattr]
            fn shape(py: Python<'_>) -> PyResult<Py<PyTuple>> {
                Ok(PyTuple::empty(py).unbind())
            }

            #[getter]
            fn dtype(slf: &Bound<'_, Self>) -> PyResult<Py<PyDtype>> {
                Py::new(slf.py(), PyDtype { name: dtype_name(me(slf.as_any())) })
            }

            #[getter]
            fn itemsize(slf: &Bound<'_, Self>) -> usize {
                crate::dtypes::itemsize(dtype_name(me(slf.as_any())))
            }

            #[getter]
            fn nbytes(slf: &Bound<'_, Self>) -> usize {
                crate::dtypes::itemsize(dtype_name(me(slf.as_any())))
            }

            #[pyo3(signature = (*_args))]
            fn item(slf: &Bound<'_, Self>, _args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
                plain(slf.py(), me(slf.as_any()))
            }

            fn tolist(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                plain(slf.py(), me(slf.as_any()))
            }

            #[pyo3(signature = (dtype, *args, **kwargs))]
            fn astype(slf: &Bound<'_, Self>, dtype: &Bound<'_, PyAny>, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
                let py = slf.py();
                let mut all = vec![dtype.clone()];
                all.extend(args.iter());
                Ok(array_of(py, me(slf.as_any()))?.bind(py).call_method("astype", PyTuple::new(py, all)?, kwargs)?.unbind())
            }

            fn __getitem__(slf: &Bound<'_, Self>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                let py = slf.py();
                if index.downcast::<PyTuple>().is_ok_and(|t| t.is_empty()) {
                    return Ok(slf.clone().into_any().unbind());
                }
                Ok(array_of(py, me(slf.as_any()))?.bind(py).get_item(index)?.unbind())
            }

            fn __getattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<Py<PyAny>> {
                let py = slf.py();
                if name.starts_with("__") {
                    return Err(PyAttributeError::new_err(name.to_string()));
                }
                Ok(array_of(py, me(slf.as_any()))?.bind(py).getattr(name)?.unbind())
            }

            #[getter]
            fn __array_interface__(slf: &Bound<'_, Self>) -> PyResult<Py<PyDict>> {
                let py = slf.py();
                let (typestr, data): (&str, Vec<u8>) = match me(slf.as_any()) {
                    Num::F(v) => ("<f8", v.to_le_bytes().to_vec()),
                    Num::I(v, _) => ("<i8", v.to_le_bytes().to_vec()),
                    Num::B(v) => ("|b1", vec![u8::from(v)]),
                    Num::C(re, im) => ("<c16", [re.to_le_bytes(), im.to_le_bytes()].concat()),
                };
                let d = PyDict::new(py);
                d.set_item("version", 3)?;
                d.set_item("shape", PyTuple::empty(py))?;
                d.set_item("typestr", typestr)?;
                d.set_item("data", pyo3::types::PyBytes::new(py, &data))?;
                Ok(d.unbind())
            }

            #[pyo3(signature = (*, api_version=None))]
            fn __array_namespace__(slf: &Bound<'_, Self>, api_version: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
                let py = slf.py();
                let kw = PyDict::new(py);
                kw.set_item("api_version", api_version)?;
                Ok(array_of(py, me(slf.as_any()))?.bind(py).call_method("__array_namespace__", (), Some(&kw))?.unbind())
            }

            fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                let py = slf.py();
                let n = me(slf.as_any());
                let factory = core(py)?.getattr("scalar")?;
                Ok((factory, (dtype_name(n), plain(py, n)?)).into_pyobject(py)?.into_any().unbind())
            }

            fn __richcmp__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
                compare(slf.as_any(), other, op)
            }

            fn __hash__(slf: &Bound<'_, Self>) -> PyResult<isize> {
                let py = slf.py();
                plain(py, me(slf.as_any()))?.bind(py).hash()
            }

            fn __neg__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                unary(slf.as_any(), "__neg__")
            }

            fn __pos__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                match me(slf.as_any()) {
                    Num::B(_) => unary(slf.as_any(), "__pos__"),
                    _ => Ok(slf.clone().into_any().unbind()),
                }
            }

            fn __abs__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                unary(slf.as_any(), "__abs__")
            }

            fn __invert__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                unary(slf.as_any(), "__invert__")
            }

            fn __add__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::Add, false)
            }
            fn __radd__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::Add, true)
            }
            fn __sub__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::Sub, false)
            }
            fn __rsub__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::Sub, true)
            }
            fn __mul__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::Mul, false)
            }
            fn __rmul__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::Mul, true)
            }
            fn __truediv__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::TrueDiv, false)
            }
            fn __rtruediv__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::TrueDiv, true)
            }
            fn __floordiv__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::FloorDiv, false)
            }
            fn __rfloordiv__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::FloorDiv, true)
            }
            fn __mod__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::Mod, false)
            }
            fn __rmod__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::Mod, true)
            }
            fn __pow__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::Pow, false)
            }
            fn __rpow__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::Pow, true)
            }
            fn __lshift__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::LShift, false)
            }
            fn __rlshift__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::LShift, true)
            }
            fn __rshift__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::RShift, false)
            }
            fn __rrshift__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::RShift, true)
            }
            fn __and__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::And, false)
            }
            fn __rand__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::And, true)
            }
            fn __or__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::Or, false)
            }
            fn __ror__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::Or, true)
            }
            fn __xor__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::Xor, false)
            }
            fn __rxor__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                binary(slf.as_any(), o, Op::Xor, true)
            }
            fn __divmod__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                through_array(slf.as_any(), "__divmod__", o)
            }
            fn __rdivmod__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                through_array(slf.as_any(), "__rdivmod__", o)
            }
            fn __matmul__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                through_array(slf.as_any(), "__matmul__", o)
            }
            fn __rmatmul__(slf: &Bound<'_, Self>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                through_array(slf.as_any(), "__rmatmul__", o)
            }
        }
    };
}

scalar_methods!(F64);
scalar_methods!(C128);
scalar_methods!(I64);
scalar_methods!(Bool);

macro_rules! python_number_methods {
    ($ty:ident) => {
        #[pymethods]
        impl $ty {
            fn __bool__(slf: &Bound<'_, Self>) -> bool {
                match me(slf.as_any()) {
                    Num::I(v, _) => v != 0,
                    Num::B(v) => v,
                    _ => true,
                }
            }

            fn __int__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                let py = slf.py();
                match me(slf.as_any()) {
                    Num::I(v, _) => Ok(v.into_pyobject(py)?.into_any().unbind()),
                    Num::B(v) => Ok(i64::from(v).into_pyobject(py)?.into_any().unbind()),
                    _ => Err(PyTypeError::new_err("not an integer scalar")),
                }
            }

            fn __index__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                Self::__int__(slf)
            }

            fn __float__(slf: &Bound<'_, Self>) -> f64 {
                as_float(me(slf.as_any()))
            }

            fn __complex__(slf: &Bound<'_, Self>) -> Py<PyComplex> {
                let (re, im) = as_complex(me(slf.as_any()));
                PyComplex::from_doubles(slf.py(), re, im).unbind()
            }

            fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
                let py = slf.py();
                plain(py, me(slf.as_any()))?.bind(py).repr()?.extract()
            }

            fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
                let py = slf.py();
                plain(py, me(slf.as_any()))?.bind(py).str()?.extract()
            }

            fn __format__(slf: &Bound<'_, Self>, spec: &str) -> PyResult<String> {
                let py = slf.py();
                plain(py, me(slf.as_any()))?.bind(py).call_method1("__format__", (spec,))?.extract()
            }

            #[pyo3(signature = (ndigits=None))]
            fn __round__(slf: &Bound<'_, Self>, ndigits: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
                let py = slf.py();
                let builtins = py.import("builtins")?;
                Ok(match ndigits {
                    Some(n) => builtins.getattr("round")?.call1((plain(py, me(slf.as_any()))?, n))?,
                    None => builtins.getattr("round")?.call1((plain(py, me(slf.as_any()))?,))?,
                }
                .unbind())
            }

            fn __trunc__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                Self::__int__(slf)
            }

            fn __floor__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                Self::__int__(slf)
            }

            fn __ceil__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                Self::__int__(slf)
            }
        }
    };
}

python_number_methods!(I64);
python_number_methods!(Bool);

pub fn as_arr(obj: &Bound<'_, PyAny>) -> Option<Arr> {
    Some(match num_of_ours(obj)? {
        Num::F(v) => Arr::scalar(v),
        Num::I(v, _) => Arr::scalar(v),
        Num::B(v) => Arr::scalar(v),
        Num::C(re, im) => Arr::scalar(Complex::new(re, im)),
    })
}

fn num_of_ours(o: &Bound<'_, PyAny>) -> Option<Num> {
    if let Ok(f) = o.downcast_exact::<F64>() {
        return Some(Num::F(f.as_super().value()));
    }
    if let Ok(i) = o.downcast_exact::<I64>() {
        return Some(Num::I(i.get().0, false));
    }
    if let Ok(b) = o.downcast_exact::<Bool>() {
        return Some(Num::B(b.get().0));
    }
    if let Ok(c) = o.downcast_exact::<C128>() {
        let c = c.as_super();
        return Some(Num::C(c.real(), c.imag()));
    }
    None
}

pub fn bool_value(o: &Bound<'_, PyAny>) -> Option<bool> {
    o.downcast_exact::<Bool>().ok().map(|b| b.get().0)
}

#[pyfunction]
fn scalar(py: Python<'_>, dtype: &str, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    match dtype {
        "float64" => new_f64(py, value.extract()?),
        "int64" => new_i64(py, value.extract()?),
        "bool" => new_bool(py, value.is_truthy()?),
        "complex128" => {
            let c = py.get_type::<PyComplex>().call1((value,))?;
            let c = c.downcast::<PyComplex>()?;
            new_c128(py, c.real(), c.imag())
        }
        other => Err(PyTypeError::new_err(format!("no scalar class for dtype {other}"))),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scalar, m)?)?;
    Ok(())
}
