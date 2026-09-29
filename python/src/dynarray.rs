use num_complex::Complex;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use rustnumpy::NdArray;

pyo3::create_exception!(rustnumpy_python, Unsupported, PyNotImplementedError);

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
impl_from!(Bool, bool; I8, i8; I16, i16; I32, i32; I64, i64; U8, u8; U16, u16; U32, u32; U64, u64; F32, f32; F64, f64; C64, C32; C128, C64);

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
            Arr::F32(_) => "float32",
            Arr::F64(_) => "float64",
            Arr::C64(_) => "complex64",
            Arr::C128(_) => "complex128",
        }
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

    pub fn from_numpy(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Arr> {
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (obj,))?;
        let dtype = arr.getattr("dtype")?;
        let name: String = dtype.getattr("name")?.extract()?;
        let byteorder: String = dtype.getattr("byteorder")?.extract()?;
        let native = if byteorder == ">" { arr.call_method1("astype", (dtype.call_method1("newbyteorder", ("=",))?,))? } else { arr };
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("order", "C")?;
        let contiguous = np.call_method("asarray", (native,), Some(&kwargs))?;
        let shape: Vec<usize> = contiguous.getattr("shape")?.extract()?;
        let bytes: Vec<u8> = contiguous.call_method0("tobytes")?.extract()?;
        Arr::from_bytes(&name, &shape, &bytes)
    }

    pub fn to_numpy<'py>(&self, py: Python<'py>, scalar_if_0d: bool) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        let shape = PyTuple::new(py, self.shape())?;
        let flat = np.call_method1("frombuffer", (PyBytes::new(py, &self.to_bytes()), self.dtype_name()))?;
        let arr = flat.call_method1("reshape", (shape,))?.call_method0("copy")?;
        if scalar_if_0d && self.ndim() == 0 {
            return arr.get_item(PyTuple::empty(py));
        }
        Ok(arr)
    }
}
