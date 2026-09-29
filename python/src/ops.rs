use crate::casting::astype;
use crate::dynarray::{unsupported, value_err, Arr, C32, C64};
use crate::dispatch2;
use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyInt};
use rustnumpy::dispatch::{zip_with_promoted, Common, Out};
use rustnumpy::{NdArray, ShapeError};

pub fn shape_err(e: ShapeError) -> PyErr {
    match e {
        ShapeError::WeakScalarOverflow { .. } => PyOverflowError::new_err(e.to_string()),
        other => value_err(other),
    }
}

pub enum Operand {
    Arr(Arr),
    WeakInt(i64),
    WeakFloat(f64),
    WeakComplex(f64, f64),
}

impl Operand {
    pub fn parse(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Operand> {
        if obj.is_exact_instance_of::<PyInt>() {
            return match obj.extract::<i64>() {
                Ok(v) => Ok(Operand::WeakInt(v)),
                Err(_) => match obj.extract::<u64>() {
                    Ok(v) => Ok(Operand::Arr(Arr::scalar(v))),
                    Err(_) => Err(unsupported("Python int outside the int64/uint64 range")),
                },
            };
        }
        if obj.is_exact_instance_of::<PyFloat>() {
            return Ok(Operand::WeakFloat(obj.extract::<f64>()?));
        }
        if obj.is_exact_instance_of::<pyo3::types::PyComplex>() {
            let c = obj.downcast::<pyo3::types::PyComplex>()?;
            return Ok(Operand::WeakComplex(c.real(), c.imag()));
        }
        Ok(Operand::Arr(Arr::from_object(py, obj)?))
    }

    pub fn into_arr(self, _py: Python<'_>) -> PyResult<Arr> {
        match self {
            Operand::Arr(a) => Ok(a),
            Operand::WeakInt(v) => Ok(Arr::scalar(v)),
            Operand::WeakFloat(v) => Ok(Arr::scalar(v)),
            Operand::WeakComplex(re, im) => Ok(Arr::scalar(C64::new(re, im))),
        }
    }

    fn weak(&self) -> Option<crate::typefns::Weak> {
        use crate::typefns::Weak;
        match self {
            Operand::Arr(_) => None,
            Operand::WeakInt(_) => Some(Weak::Int),
            Operand::WeakFloat(_) => Some(Weak::Float),
            Operand::WeakComplex(..) => Some(Weak::Complex),
        }
    }

    pub fn strong_dtype(&self) -> Option<&'static str> {
        match self {
            Operand::Arr(a) => Some(a.dtype_name()),
            _ => None,
        }
    }
}

pub fn common_name(a: &Operand, b: &Operand) -> &'static str {
    use crate::typefns::{kind_of_name, weak_rank, with_weak};
    let strong = [a.strong_dtype(), b.strong_dtype()].into_iter().flatten().map(kind_of_name).reduce(rustnumpy::common_dtype);
    let widest = [a.weak(), b.weak()].into_iter().flatten().max_by_key(|&w| weak_rank(w));
    let kind = match (strong, widest) {
        (Some(k), Some(w)) => with_weak(k, w),
        (Some(k), None) => k,
        (None, Some(w)) => with_weak(rustnumpy::Kind::Bool, w),
        (None, None) => unreachable!("an operand is either strong or weak"),
    };
    crate::casting::kind_name(kind)
}

pub fn materialize(op: Operand, target: &'static str) -> PyResult<Arr> {
    if let Operand::WeakInt(v) = &op {
        if let Some((lo, hi)) = int_range(target) {
            if (*v as i128) < lo || (*v as i128) > hi {
                return Err(PyOverflowError::new_err(format!("Python integer {v} out of bounds for {target}")));
            }
        }
    }
    let arr = match op {
        Operand::Arr(a) => a,
        Operand::WeakInt(v) => Arr::scalar(v),
        Operand::WeakFloat(v) => Arr::scalar(v),
        Operand::WeakComplex(re, im) => Arr::scalar(C64::new(re, im)),
    };
    if arr.dtype_name() == target { Ok(arr) } else { astype(&arr, target) }
}

pub fn out(py: Python<'_>, a: Arr) -> PyResult<Py<PyAny>> {
    if let Some(v) = a.to_py_scalar(py)? {
        return Ok(v);
    }
    out_array(py, a)
}

pub fn out_array(py: Python<'_>, a: Arr) -> PyResult<Py<PyAny>> {
    crate::pyarray::wrap(py, crate::pyarray::PyArray::from_arr(a))
}

pub trait BSub: Copy {
    fn bsub(self, rhs: Self) -> Self;
}
macro_rules! bsub_ints {
    ($($t:ty),*) => {$(impl BSub for $t { fn bsub(self, r: Self) -> Self { self.wrapping_sub(r) } })*};
}
bsub_ints!(i8, i16, i32, i64, u8, u16, u32, u64);
macro_rules! bsub_plain {
    ($($t:ty),*) => {$(impl BSub for $t { fn bsub(self, r: Self) -> Self { self - r } })*};
}
bsub_plain!(half::f16, f32, f64, C32, C64);
impl BSub for bool {
    fn bsub(self, _: Self) -> Self {
        unreachable!("boolean subtract is rejected before dispatch")
    }
}

pub fn sub_promoted<A, B>(a: &rustnumpy::ArrayView<A>, b: &rustnumpy::ArrayView<B>) -> Result<NdArray<Out<A, B>>, ShapeError>
where
    A: Common<B>,
    B: Copy,
    Out<A, B>: BSub,
{
    zip_with_promoted(a, b, |x, y| x.bsub(y))
}

macro_rules! arith_fn {
    ($name:ident, $prom:expr, $bool_check:expr) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let (x, y) = resolve_binary(py, a, b)?;
            if $bool_check && x.is_bool() && y.is_bool() {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.",
                ));
            }
            let result: Arr = dispatch2!(&x, &y, p, q => Arr::from($prom(&p.view(), &q.view()).map_err(shape_err)?));
            out(py, result)
        }
    };
}

arith_fn!(add, rustnumpy::add, false);
arith_fn!(multiply, rustnumpy::mul, false);
arith_fn!(subtract, sub_promoted, true);

#[pyfunction]
pub fn astype_(py: Python<'_>, a: &Bound<'_, PyAny>, dtype: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = Arr::from_object(py, a)?;
    out_array(py, astype(&arr, crate::dtypes::parse_dtype(dtype)?)?)
}

fn int_range(dtype: &str) -> Option<(i128, i128)> {
    Some(match dtype {
        "int8" => (i8::MIN.into(), i8::MAX.into()),
        "int16" => (i16::MIN.into(), i16::MAX.into()),
        "int32" => (i32::MIN.into(), i32::MAX.into()),
        "int64" => (i64::MIN.into(), i64::MAX.into()),
        "uint8" => (0, u8::MAX.into()),
        "uint16" => (0, u16::MAX.into()),
        "uint32" => (0, u32::MAX.into()),
        "uint64" => (0, u64::MAX.into()),
        _ => return None,
    })
}

pub fn resolve_binary(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<(Arr, Arr)> {
    let (oa, ob) = (Operand::parse(py, a)?, Operand::parse(py, b)?);
    let name = common_name(&oa, &ob);
    Ok((materialize(oa, name)?, materialize(ob, name)?))
}

pub fn resolve_loop(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    pick: impl Fn(&'static str, [Option<&'static str>; 2]) -> PyResult<&'static str>,
) -> PyResult<(Arr, Arr)> {
    let (oa, ob) = (Operand::parse(py, a)?, Operand::parse(py, b)?);
    let name = common_name(&oa, &ob);
    let target = pick(name, [oa.strong_dtype(), ob.strong_dtype()])?;
    Ok((materialize(oa, target)?, materialize(ob, target)?))
}

#[pyfunction]
pub fn real(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = Arr::from_object(py, a)?;
    let result = match &arr {
        Arr::C64(x) => Arr::from(NdArray::from_vec(x.as_slice().iter().map(|c| c.re).collect(), x.shape()).map_err(shape_err)?),
        Arr::C128(x) => Arr::from(NdArray::from_vec(x.as_slice().iter().map(|c| c.re).collect(), x.shape()).map_err(shape_err)?),
        _ => return crate::pyarray::wrap(py, crate::pyarray::as_array(py, a)?),
    };
    out_array(py, result)
}

#[pyfunction]
pub fn imag(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = Arr::from_object(py, a)?;
    let result = match &arr {
        Arr::C64(x) => Arr::from(NdArray::from_vec(x.as_slice().iter().map(|c| c.im).collect(), x.shape()).map_err(shape_err)?),
        Arr::C128(x) => Arr::from(NdArray::from_vec(x.as_slice().iter().map(|c| c.im).collect(), x.shape()).map_err(shape_err)?),
        other => astype(&Arr::from(NdArray::<f64>::zeros(&other.shape())), other.dtype_name())?,
    };
    out_array(py, result)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(real, m)?)?;
    m.add_function(wrap_pyfunction!(imag, m)?)?;
    macro_rules! reg {
        ($($f:ident),* $(,)?) => {$( m.add_function(wrap_pyfunction!($f, m)?)?; )*};
    }
    reg!(add, subtract, multiply, astype_);
    Ok(())
}
