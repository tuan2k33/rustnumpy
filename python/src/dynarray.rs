use num_complex::Complex;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use rustnumpy::NdArray;

pyo3::create_exception!(rustnumpy, Unsupported, PyNotImplementedError);

pub fn unsupported(what: impl Into<String>) -> PyErr {
    Unsupported::new_err(what.into())
}

pub fn value_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

pub type C32 = Complex<f32>;
pub type C64 = Complex<f64>;

pub enum Arr {
    Bool(NdArray<bool>),
    I8(NdArray<i8>),
    I16(NdArray<i16>),
    I32(NdArray<i32>),
    I64(NdArray<i64>),
    U8(NdArray<u8>),
    U16(NdArray<u16>),
    U32(NdArray<u32>),
    U64(NdArray<u64>),
    F16(NdArray<half::f16>),
    F32(NdArray<f32>),
    F64(NdArray<f64>),
    C64(NdArray<C32>),
    C128(NdArray<C64>),
}

#[macro_export]
macro_rules! with_arr {
    ($arr:expr, $a:ident => $body:expr) => {
        match $arr {
            Arr::Bool($a) => $body,
            Arr::I8($a) => $body,
            Arr::I16($a) => $body,
            Arr::I32($a) => $body,
            Arr::I64($a) => $body,
            Arr::U8($a) => $body,
            Arr::U16($a) => $body,
            Arr::U32($a) => $body,
            Arr::U64($a) => $body,
            Arr::F16($a) => $body,
            Arr::F32($a) => $body,
            Arr::F64($a) => $body,
            Arr::C64($a) => $body,
            Arr::C128($a) => $body,
        }
    };
}

#[macro_export]
macro_rules! dispatch_b {
    ($b:expr, $x:ident, $y:ident => $body:expr) => {
        match $b {
            Arr::Bool($y) => $body,
            Arr::I8($y) => $body,
            Arr::I16($y) => $body,
            Arr::I32($y) => $body,
            Arr::I64($y) => $body,
            Arr::U8($y) => $body,
            Arr::U16($y) => $body,
            Arr::U32($y) => $body,
            Arr::U64($y) => $body,
            Arr::F16($y) => $body,
            Arr::F32($y) => $body,
            Arr::F64($y) => $body,
            Arr::C64($y) => $body,
            Arr::C128($y) => $body,
        }
    };
}

#[macro_export]
macro_rules! dispatch2 {
    ($a:expr, $b:expr, $x:ident, $y:ident => $body:expr) => {
        match $a {
            Arr::Bool($x) => $crate::dispatch_b!($b, $x, $y => $body),
            Arr::I8($x) => $crate::dispatch_b!($b, $x, $y => $body),
            Arr::I16($x) => $crate::dispatch_b!($b, $x, $y => $body),
            Arr::I32($x) => $crate::dispatch_b!($b, $x, $y => $body),
            Arr::I64($x) => $crate::dispatch_b!($b, $x, $y => $body),
            Arr::U8($x) => $crate::dispatch_b!($b, $x, $y => $body),
            Arr::U16($x) => $crate::dispatch_b!($b, $x, $y => $body),
            Arr::U32($x) => $crate::dispatch_b!($b, $x, $y => $body),
            Arr::U64($x) => $crate::dispatch_b!($b, $x, $y => $body),
            Arr::F16($x) => $crate::dispatch_b!($b, $x, $y => $body),
            Arr::F32($x) => $crate::dispatch_b!($b, $x, $y => $body),
            Arr::F64($x) => $crate::dispatch_b!($b, $x, $y => $body),
            Arr::C64($x) => $crate::dispatch_b!($b, $x, $y => $body),
            Arr::C128($x) => $crate::dispatch_b!($b, $x, $y => $body),
        }
    };
}

macro_rules! impl_from {
    ($($v:ident, $t:ty);*) => {$(
        impl From<NdArray<$t>> for Arr {
            fn from(a: NdArray<$t>) -> Self { Arr::$v(a) }
        }
    )*};
}
impl_from!(Bool, bool; I8, i8; I16, i16; I32, i32; I64, i64; U8, u8; U16, u16; U32, u32; U64, u64; F16, half::f16; F32, f32; F64, f64; C64, C32; C128, C64);

impl Arr {
    pub fn dtype_name(&self) -> &'static str {
        match self {
            Arr::Bool(_) => "bool",
            Arr::I8(_) => "int8",
            Arr::I16(_) => "int16",
            Arr::I32(_) => "int32",
            Arr::I64(_) => "int64",
            Arr::U8(_) => "uint8",
            Arr::U16(_) => "uint16",
            Arr::U32(_) => "uint32",
            Arr::U64(_) => "uint64",
            Arr::F16(_) => "float16",
            Arr::F32(_) => "float32",
            Arr::F64(_) => "float64",
            Arr::C64(_) => "complex64",
            Arr::C128(_) => "complex128",
        }
    }

    pub fn clone_arr(&self) -> Arr {
        with_arr!(self, a => Arr::from(a.clone()))
    }

    pub fn shape(&self) -> Vec<usize> {
        with_arr!(self, a => a.shape().to_vec())
    }

    pub fn ndim(&self) -> usize {
        with_arr!(self, a => a.ndim())
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, Arr::Bool(_))
    }

    pub fn is_int(&self) -> bool {
        matches!(
            self,
            Arr::I8(_) | Arr::I16(_) | Arr::I32(_) | Arr::I64(_) | Arr::U8(_) | Arr::U16(_) | Arr::U32(_) | Arr::U64(_)
        )
    }

    pub fn is_complex(&self) -> bool {
        matches!(self, Arr::C64(_) | Arr::C128(_))
    }

    pub fn kind(&self) -> rustnumpy::Kind {
        use rustnumpy::Kind;
        match self {
            Arr::Bool(_) => Kind::Bool,
            Arr::I8(_) => Kind::Int(8),
            Arr::I16(_) => Kind::Int(16),
            Arr::I32(_) => Kind::Int(32),
            Arr::I64(_) => Kind::Int(64),
            Arr::U8(_) => Kind::Uint(8),
            Arr::U16(_) => Kind::Uint(16),
            Arr::U32(_) => Kind::Uint(32),
            Arr::U64(_) => Kind::Uint(64),
            Arr::F16(_) => Kind::Float(16),
            Arr::F32(_) => Kind::Float(32),
            Arr::F64(_) => Kind::Float(64),
            Arr::C64(_) => Kind::Complex(32),
            Arr::C128(_) => Kind::Complex(64),
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        macro_rules! ints {
            ($a:expr) => {
                $a.as_slice().iter().flat_map(|x| x.to_le_bytes()).collect()
            };
        }
        match self {
            Arr::Bool(a) => a.as_slice().iter().map(|&b| u8::from(b)).collect(),
            Arr::I8(a) => ints!(a),
            Arr::I16(a) => ints!(a),
            Arr::I32(a) => ints!(a),
            Arr::I64(a) => ints!(a),
            Arr::U8(a) => ints!(a),
            Arr::U16(a) => ints!(a),
            Arr::U32(a) => ints!(a),
            Arr::U64(a) => ints!(a),
            Arr::F16(a) => ints!(a),
            Arr::F32(a) => ints!(a),
            Arr::F64(a) => ints!(a),
            Arr::C64(a) => a.as_slice().iter().flat_map(|c| c.re.to_le_bytes().into_iter().chain(c.im.to_le_bytes())).collect(),
            Arr::C128(a) => a.as_slice().iter().flat_map(|c| c.re.to_le_bytes().into_iter().chain(c.im.to_le_bytes())).collect(),
        }
    }

    pub fn from_bytes(dtype: &str, shape: &[usize], bytes: &[u8]) -> PyResult<Arr> {
        macro_rules! parse {
            ($t:ty, $v:ident) => {{
                const W: usize = std::mem::size_of::<$t>();
                let data: Vec<$t> = bytes.chunks_exact(W).map(|c| <$t>::from_le_bytes(c.try_into().unwrap())).collect();
                Arr::$v(NdArray::from_vec(data, shape).map_err(value_err)?)
            }};
        }
        Ok(match dtype {
            "bool" => Arr::Bool(NdArray::from_vec(bytes.iter().map(|&b| b != 0).collect(), shape).map_err(value_err)?),
            "int8" => parse!(i8, I8),
            "int16" => parse!(i16, I16),
            "int32" => parse!(i32, I32),
            "int64" => parse!(i64, I64),
            "uint8" => parse!(u8, U8),
            "uint16" => parse!(u16, U16),
            "uint32" => parse!(u32, U32),
            "uint64" => parse!(u64, U64),
            "float16" => parse!(half::f16, F16),
            "float32" => parse!(f32, F32),
            "float64" => parse!(f64, F64),
            "complex64" => {
                let data: Vec<C32> = bytes
                    .chunks_exact(8)
                    .map(|c| Complex::new(f32::from_le_bytes(c[..4].try_into().unwrap()), f32::from_le_bytes(c[4..].try_into().unwrap())))
                    .collect();
                Arr::C64(NdArray::from_vec(data, shape).map_err(value_err)?)
            }
            "complex128" => {
                let data: Vec<C64> = bytes
                    .chunks_exact(16)
                    .map(|c| Complex::new(f64::from_le_bytes(c[..8].try_into().unwrap()), f64::from_le_bytes(c[8..].try_into().unwrap())))
                    .collect();
                Arr::C128(NdArray::from_vec(data, shape).map_err(value_err)?)
            }
            other => return Err(unsupported(format!("dtype {other} is not supported"))),
        })
    }

    pub fn scalar<T>(v: T) -> Arr
    where
        Arr: From<NdArray<T>>,
    {
        Arr::from(NdArray::from_vec(vec![v], &[]).expect("a 0-d array holds one element"))
    }

    pub fn to_native_scalar(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match self {
            Arr::Bool(a) => pyo3::types::PyBool::new(py, a.as_slice()[0]).to_owned().into_any().unbind(),
            Arr::I8(a) => i64::from(a.as_slice()[0]).into_pyobject(py)?.into_any().unbind(),
            Arr::I16(a) => i64::from(a.as_slice()[0]).into_pyobject(py)?.into_any().unbind(),
            Arr::I32(a) => i64::from(a.as_slice()[0]).into_pyobject(py)?.into_any().unbind(),
            Arr::I64(a) => a.as_slice()[0].into_pyobject(py)?.into_any().unbind(),
            Arr::U8(a) => u64::from(a.as_slice()[0]).into_pyobject(py)?.into_any().unbind(),
            Arr::U16(a) => u64::from(a.as_slice()[0]).into_pyobject(py)?.into_any().unbind(),
            Arr::U32(a) => u64::from(a.as_slice()[0]).into_pyobject(py)?.into_any().unbind(),
            Arr::U64(a) => a.as_slice()[0].into_pyobject(py)?.into_any().unbind(),
            Arr::F16(a) => f64::from(a.as_slice()[0]).into_pyobject(py)?.into_any().unbind(),
            Arr::F32(a) => f64::from(a.as_slice()[0]).into_pyobject(py)?.into_any().unbind(),
            Arr::F64(a) => a.as_slice()[0].into_pyobject(py)?.into_any().unbind(),
            Arr::C64(a) => {
                let c = a.as_slice()[0];
                pyo3::types::PyComplex::from_doubles(py, f64::from(c.re), f64::from(c.im)).into_any().unbind()
            }
            Arr::C128(a) => {
                let c = a.as_slice()[0];
                pyo3::types::PyComplex::from_doubles(py, c.re, c.im).into_any().unbind()
            }
        })
    }

    pub fn to_py_scalar(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if self.ndim() != 0 {
            return Ok(None);
        }
        let (dtype, value) = match self {
            Arr::Bool(a) => ("bool", pyo3::types::PyBool::new(py, a.as_slice()[0]).to_owned().into_any()),
            Arr::I64(a) => ("int64", a.as_slice()[0].into_pyobject(py)?.into_any()),
            Arr::F64(a) => ("float64", a.as_slice()[0].into_pyobject(py)?.into_any()),
            Arr::C128(a) => ("complex128", pyo3::types::PyComplex::from_doubles(py, a.as_slice()[0].re, a.as_slice()[0].im).into_any()),
            _ => return Ok(None),
        };
        Ok(Some(py.import("rustnumpy._scalars")?.getattr("scalar")?.call1((dtype, value))?.unbind()))
    }

    pub fn from_scalar_class(obj: &Bound<'_, PyAny>) -> PyResult<Option<Arr>> {
        use pyo3::types::{PyComplex, PyFloat, PyInt};
        if obj.is_exact_instance_of::<PyFloat>() || obj.is_exact_instance_of::<PyInt>() || obj.is_exact_instance_of::<PyComplex>() {
            return Ok(None);
        }
        let Ok(name) = obj.get_type().getattr("_rnp_dtype") else { return Ok(None) };
        Ok(Some(match name.extract::<String>()?.as_str() {
            "bool" => Arr::scalar(obj.is_truthy()?),
            "int64" => Arr::scalar(obj.extract::<i64>()?),
            "float64" => Arr::scalar(obj.extract::<f64>()?),
            _ => {
                let c = obj.downcast::<PyComplex>()?;
                Arr::scalar(Complex::new(c.real(), c.imag()))
            }
        }))
    }

    pub fn from_object(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Arr> {
        if let Ok(a) = obj.downcast::<crate::pyarray::PyArray>() {
            let this = a.borrow();
            crate::createfns::alloc_guard(this.size(), crate::dtypes::itemsize(this.dtype_name()) + 8)?;
            return Ok(this.to_arr());
        }
        if let Some(a) = Arr::from_scalar_class(obj)? {
            return Ok(a);
        }
        if obj.is_exact_instance_of::<pyo3::types::PyBool>() {
            return Ok(Arr::scalar(obj.extract::<bool>()?));
        }
        if obj.is_exact_instance_of::<pyo3::types::PyInt>() {
            return match obj.extract::<i64>() {
                Ok(v) => Ok(Arr::scalar(v)),
                Err(_) => match obj.extract::<u64>() {
                    Ok(v) => Ok(Arr::scalar(v)),
                    Err(_) => Err(unsupported("Python int outside the int64/uint64 range")),
                },
            };
        }
        if obj.is_exact_instance_of::<pyo3::types::PyFloat>() {
            return Ok(Arr::scalar(obj.extract::<f64>()?));
        }
        if obj.is_exact_instance_of::<pyo3::types::PyComplex>() {
            let c = obj.downcast::<pyo3::types::PyComplex>()?;
            return Ok(Arr::scalar(Complex::new(c.real(), c.imag())));
        }
        if obj.is_instance_of::<pyo3::types::PyList>() || obj.is_instance_of::<pyo3::types::PyTuple>() {
            return Arr::from_sequence(py, obj);
        }
        if obj.is_instance_of::<pyo3::types::PyString>() || obj.is_instance_of::<pyo3::types::PyBytes>() {
            return Err(unsupported("string and bytes data are not supported"));
        }
        if let Some(a) = Arr::from_buffer(py, obj)? {
            return Ok(a);
        }
        if let Some(a) = Arr::from_array_interface(obj)? {
            return Ok(a);
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "cannot convert {} to an array (expected a number, nested sequence, or an object with the buffer protocol or __array_interface__)",
            obj.get_type().name()?
        )))
    }

    fn from_sequence(py: Python<'_>, seq: &Bound<'_, PyAny>) -> PyResult<Arr> {
        let items: Vec<Bound<'_, PyAny>> = seq.try_iter()?.collect::<PyResult<_>>()?;
        if items.is_empty() {
            return Ok(Arr::F64(NdArray::from_vec(vec![], &[0]).map_err(value_err)?));
        }
        let plain = items.iter().all(|i| {
            i.is_exact_instance_of::<pyo3::types::PyBool>()
                || i.is_exact_instance_of::<pyo3::types::PyInt>()
                || i.is_exact_instance_of::<pyo3::types::PyFloat>()
                || i.is_exact_instance_of::<pyo3::types::PyComplex>()
        });
        if plain {
            return Arr::from_plain_scalars(&items);
        }
        let parts: Vec<Arr> = items.iter().map(|i| Arr::from_object(py, i)).collect::<PyResult<_>>()?;
        let first_shape = parts[0].shape();
        if parts.iter().any(|p| p.shape() != first_shape) {
            return Err(PyValueError::new_err("setting an array element with a sequence: the requested array has an inhomogeneous shape"));
        }
        let parts = crate::shapefns::common_all(&parts)?;
        let dtype = parts[0].dtype_name();
        let bytes: Vec<u8> = parts.iter().flat_map(|p| p.to_bytes()).collect();
        let mut shape = vec![parts.len()];
        shape.extend(first_shape);
        Arr::from_bytes(dtype, &shape, &bytes)
    }

    fn from_plain_scalars(items: &[Bound<'_, PyAny>]) -> PyResult<Arr> {
        let n = items.len();
        let any_complex = items.iter().any(|i| i.is_exact_instance_of::<pyo3::types::PyComplex>());
        let any_float = items.iter().any(|i| i.is_exact_instance_of::<pyo3::types::PyFloat>());
        let any_int = items.iter().any(|i| i.is_exact_instance_of::<pyo3::types::PyInt>());
        if any_complex {
            let data: Vec<C64> = items
                .iter()
                .map(|i| match i.downcast::<pyo3::types::PyComplex>() {
                    Ok(c) => Ok(Complex::new(c.real(), c.imag())),
                    Err(_) => i.extract::<f64>().map(|v| Complex::new(v, 0.0)),
                })
                .collect::<PyResult<_>>()?;
            return Ok(Arr::C128(NdArray::from_vec(data, &[n]).map_err(value_err)?));
        }
        if any_float {
            let data: Vec<f64> = items.iter().map(|i| i.extract::<f64>()).collect::<PyResult<_>>()?;
            return Ok(Arr::F64(NdArray::from_vec(data, &[n]).map_err(value_err)?));
        }
        if any_int {
            let as_i64: Result<Vec<i64>, _> = items.iter().map(|i| i.extract::<i64>()).collect();
            if let Ok(data) = as_i64 {
                return Ok(Arr::I64(NdArray::from_vec(data, &[n]).map_err(value_err)?));
            }
            let as_u64: Result<Vec<u64>, _> = items.iter().map(|i| i.extract::<u64>()).collect();
            return match as_u64 {
                Ok(data) => Ok(Arr::U64(NdArray::from_vec(data, &[n]).map_err(value_err)?)),
                Err(_) => Err(unsupported("Python integers outside the int64/uint64 range")),
            };
        }
        let data: Vec<bool> = items.iter().map(|i| i.extract::<bool>()).collect::<PyResult<_>>()?;
        Ok(Arr::Bool(NdArray::from_vec(data, &[n]).map_err(value_err)?))
    }

    fn from_buffer(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Option<Arr>> {
        let mv = match py.import("builtins")?.call_method1("memoryview", (obj,)) {
            Ok(m) => m,
            Err(_) => return Ok(None),
        };
        let format: String = mv.getattr("format")?.extract()?;
        let itemsize: usize = mv.getattr("itemsize")?.extract()?;
        let shape: Vec<usize> = mv.getattr("shape")?.extract()?;
        if format.starts_with(['>', '!']) {
            return Err(unsupported("big-endian buffers are not supported"));
        }
        let code = format.trim_start_matches(['<', '=', '@']);
        let dtype = match (code, itemsize) {
            ("?", _) => "bool",
            ("b", _) => "int8",
            ("B", _) => "uint8",
            ("h", _) => "int16",
            ("H", _) => "uint16",
            ("i", _) => "int32",
            ("I", _) => "uint32",
            ("q", _) | ("l", 8) => "int64",
            ("Q", _) | ("L", 8) => "uint64",
            ("l", 4) => "int32",
            ("L", 4) => "uint32",
            ("e", _) => "float16",
            ("f", _) => "float32",
            ("d", _) => "float64",
            ("Zf", _) => "complex64",
            ("Zd", _) => "complex128",
            _ => return Err(unsupported(format!("buffer format {format:?} is not supported"))),
        };
        let bytes: Vec<u8> = mv.call_method0("tobytes")?.extract()?;
        Ok(Some(Arr::from_bytes(dtype, &shape, &bytes)?))
    }

    fn from_array_interface(obj: &Bound<'_, PyAny>) -> PyResult<Option<Arr>> {
        let Ok(iface) = obj.getattr("__array_interface__") else { return Ok(None) };
        let typestr: String = iface.get_item("typestr")?.extract()?;
        let shape: Vec<usize> = iface.get_item("shape")?.extract()?;
        let dtype = match typestr.as_str() {
            "|b1" => "bool",
            "|i1" => "int8",
            "<i2" => "int16",
            "<i4" => "int32",
            "<i8" => "int64",
            "|u1" => "uint8",
            "<u2" => "uint16",
            "<u4" => "uint32",
            "<u8" => "uint64",
            "<f2" => "float16",
            "<f4" => "float32",
            "<f8" => "float64",
            "<c8" => "complex64",
            "<c16" => "complex128",
            other => return Err(unsupported(format!("__array_interface__ typestr {other:?} is not supported"))),
        };
        let itemsize = crate::dtypes::itemsize(dtype);
        let data = iface.get_item("data")?;
        let count: usize = shape.iter().product();
        let strides: Option<Vec<isize>> = match iface.get_item("strides") {
            Ok(s) if !s.is_none() => Some(s.extract()?),
            _ => None,
        };
        let bytes: Vec<u8> = if let Ok((ptr, _ro)) = data.extract::<(usize, bool)>() {
            let strides = strides.unwrap_or_else(|| {
                let mut st = vec![0isize; shape.len()];
                let mut acc = itemsize as isize;
                for i in (0..shape.len()).rev() {
                    st[i] = acc;
                    acc *= shape[i] as isize;
                }
                st
            });
            let mut out = Vec::with_capacity(count * itemsize);
            for idx in rustnumpy::shape::IndexIter::new(&shape) {
                let off: isize = idx.iter().zip(&strides).map(|(&i, &s)| i as isize * s).sum();
                // SAFETY: the producer of __array_interface__ promises `ptr` addresses a live buffer covering
                // every element reachable through `shape` and `strides`; `obj` is kept alive by the caller.
                let src = unsafe { std::slice::from_raw_parts((ptr as isize + off) as *const u8, itemsize) };
                out.extend_from_slice(src);
            }
            out
        } else {
            data.extract::<Vec<u8>>()?
        };
        Ok(Some(Arr::from_bytes(dtype, &shape, &bytes)?))
    }
}
