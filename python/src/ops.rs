use crate::casting::{astype, common_arrs};
use crate::dynarray::{unsupported, value_err, Arr, C32, C64};
use crate::{dispatch2, with_arr};
use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyInt};
use rustnumpy::dispatch::{map_weak_float, map_weak_int, zip_with_promoted, Common, Out, WrapAdd, WrapMul};
use rustnumpy::mathfunc::{self, Arith};
use rustnumpy::reductions::FloatIsh;
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
}

impl Operand {
    pub fn parse(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Operand> {
        if obj.is_exact_instance_of::<PyInt>() {
            return match obj.extract::<i64>() {
                Ok(v) => Ok(Operand::WeakInt(v)),
                Err(_) => Err(unsupported("Python int outside the int64 range")),
            };
        }
        if obj.is_exact_instance_of::<PyFloat>() {
            return Ok(Operand::WeakFloat(obj.extract::<f64>()?));
        }
        Ok(Operand::Arr(Arr::from_numpy(py, obj)?))
    }

    pub fn into_arr(self, py: Python<'_>) -> PyResult<Arr> {
        match self {
            Operand::Arr(a) => Ok(a),
            Operand::WeakInt(v) => {
                let np = py.import("numpy")?;
                Arr::from_numpy(py, &np.call_method1("asarray", (v,))?)
            }
            Operand::WeakFloat(v) => {
                let np = py.import("numpy")?;
                Arr::from_numpy(py, &np.call_method1("asarray", (v,))?)
            }
        }
    }
}

pub fn out(py: Python<'_>, a: Arr) -> PyResult<Py<PyAny>> {
    Ok(a.to_numpy(py, true)?.unbind())
}

pub fn out_array(py: Python<'_>, a: Arr) -> PyResult<Py<PyAny>> {
    Ok(a.to_numpy(py, false)?.unbind())
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
bsub_plain!(f32, f64, C32, C64);
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
    ($name:ident, $prom:expr, $wint:expr, $wfloat:expr, $rev_int:expr, $rev_float:expr, $bool_check:expr) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let (oa, ob) = (Operand::parse(py, a)?, Operand::parse(py, b)?);
            let result: Arr = match (oa, ob) {
                (Operand::Arr(x), Operand::Arr(y)) => {
                    if $bool_check && x.is_bool() && y.is_bool() {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "numpy boolean subtract, the `-` operator, is not supported",
                        ));
                    }
                    dispatch2!(&x, &y, p, q => Arr::from($prom(&p.view(), &q.view()).map_err(shape_err)?))
                }
                (Operand::Arr(x), Operand::WeakInt(s)) => {
                    with_arr!(&x, p => Arr::from(map_weak_int(&p.view(), s, $wint).map_err(shape_err)?))
                }
                (Operand::Arr(x), Operand::WeakFloat(s)) => {
                    with_arr!(&x, p => Arr::from(map_weak_float(&p.view(), s, $wfloat).map_err(shape_err)?))
                }
                (Operand::WeakInt(s), Operand::Arr(x)) => {
                    with_arr!(&x, p => Arr::from(map_weak_int(&p.view(), s, $rev_int).map_err(shape_err)?))
                }
                (Operand::WeakFloat(s), Operand::Arr(x)) => {
                    with_arr!(&x, p => Arr::from(map_weak_float(&p.view(), s, $rev_float).map_err(shape_err)?))
                }
                (x, y) => {
                    let (x, y) = (x.into_arr(py)?, y.into_arr(py)?);
                    dispatch2!(&x, &y, p, q => Arr::from($prom(&p.view(), &q.view()).map_err(shape_err)?))
                }
            };
            out(py, result)
        }
    };
}

arith_fn!(
    add,
    rustnumpy::add,
    |x, y| x.wrap_add(y),
    |x, y| x.wrap_add(y),
    |x, y| y.wrap_add(x),
    |x, y| y.wrap_add(x),
    false
);
arith_fn!(
    multiply,
    rustnumpy::mul,
    |x, y| x.wrap_mul(y),
    |x, y| x.wrap_mul(y),
    |x, y| y.wrap_mul(x),
    |x, y| y.wrap_mul(x),
    false
);
arith_fn!(
    subtract,
    sub_promoted,
    |x, y| x.bsub(y),
    |x, y| x.bsub(y),
    |x, y| y.bsub(x),
    |x, y| y.bsub(x),
    true
);

#[pyfunction]
pub fn astype_(py: Python<'_>, a: &Bound<'_, PyAny>, dtype: &str) -> PyResult<Py<PyAny>> {
    let arr = Arr::from_numpy(py, a)?;
    out_array(py, astype(&arr, dtype)?)
}

macro_rules! unary_float {
    ($($name:ident => $f:path),* $(,)?) => {$(
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let arr = Arr::from_numpy(py, a)?;
            let result = match &arr {
                Arr::F32(x) => Arr::from($f(&x.view())),
                Arr::F64(x) => Arr::from($f(&x.view())),
                _ => return Err(unsupported("float math is bound for float32/float64 only")),
            };
            out(py, result)
        }
    )*};
}

unary_float! {
    sqrt => mathfunc::sqrt, cbrt => mathfunc::cbrt, exp => mathfunc::exp, exp2 => mathfunc::exp2,
    expm1 => mathfunc::expm1, log => mathfunc::log, log2 => mathfunc::log2, log10 => mathfunc::log10,
    log1p => mathfunc::log1p, sin => mathfunc::sin, cos => mathfunc::cos, tan => mathfunc::tan,
    arcsin => mathfunc::arcsin, arccos => mathfunc::arccos, arctan => mathfunc::arctan,
    sinh => mathfunc::sinh, cosh => mathfunc::cosh, tanh => mathfunc::tanh,
    arcsinh => mathfunc::arcsinh, arccosh => mathfunc::arccosh, arctanh => mathfunc::arctanh,
    floor => mathfunc::floor, ceil => mathfunc::ceil, trunc => mathfunc::trunc, rint => mathfunc::rint,
    reciprocal => mathfunc::reciprocal, degrees => mathfunc::degrees, radians => mathfunc::radians,
}

macro_rules! unary_arith {
    ($($name:ident => $f:path),* $(,)?) => {$(
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let arr = Arr::from_numpy(py, a)?;
            let result = match &arr {
                Arr::I8(x) => Arr::from($f(&x.view())),
                Arr::I16(x) => Arr::from($f(&x.view())),
                Arr::I32(x) => Arr::from($f(&x.view())),
                Arr::I64(x) => Arr::from($f(&x.view())),
                Arr::U8(x) => Arr::from($f(&x.view())),
                Arr::U16(x) => Arr::from($f(&x.view())),
                Arr::U32(x) => Arr::from($f(&x.view())),
                Arr::U64(x) => Arr::from($f(&x.view())),
                Arr::F32(x) => Arr::from($f(&x.view())),
                Arr::F64(x) => Arr::from($f(&x.view())),
                _ => return Err(unsupported("bound for real integer and float dtypes only")),
            };
            out(py, result)
        }
    )*};
}

unary_arith! { absolute => mathfunc::abs, negative => mathfunc::negative, square => mathfunc::square, sign => mathfunc::sign }

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

fn weak_scalar_like(py: Python<'_>, target: &str, value: &Bound<'_, PyAny>) -> PyResult<Arr> {
    let np = py.import("numpy")?;
    let scalar = Arr::from_numpy(py, &np.call_method1("asarray", (value,))?)?;
    astype(&scalar, target)
}

pub fn resolve_binary(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<(Arr, Arr)> {
    let (oa, ob) = (Operand::parse(py, a)?, Operand::parse(py, b)?);
    match (oa, ob) {
        (Operand::Arr(x), Operand::Arr(y)) => common_arrs(&x, &y),
        (Operand::Arr(x), w @ (Operand::WeakInt(_) | Operand::WeakFloat(_))) => {
            let (target, value) = weak_target(&x, &w, b)?;
            let y = weak_scalar_like(py, target, &value)?;
            Ok((astype(&x, target)?, y))
        }
        (w @ (Operand::WeakInt(_) | Operand::WeakFloat(_)), Operand::Arr(y)) => {
            let (target, value) = weak_target(&y, &w, a)?;
            let x = weak_scalar_like(py, target, &value)?;
            Ok((x, astype(&y, target)?))
        }
        (x, y) => common_arrs(&x.into_arr(py)?, &y.into_arr(py)?),
    }
}

fn weak_target<'py>(arr: &Arr, weak: &Operand, original: &Bound<'py, PyAny>) -> PyResult<(&'static str, Bound<'py, PyAny>)> {
    let name = arr.dtype_name();
    let target = match weak {
        Operand::WeakInt(_) => if arr.is_bool() { "int64" } else { name },
        _ => if arr.is_bool() || arr.is_int() { "float64" } else { name },
    };
    if let (Operand::WeakInt(v), Some((lo, hi))) = (weak, int_range(target)) {
        if (*v as i128) < lo || (*v as i128) > hi {
            return Err(PyOverflowError::new_err(format!("Python integer {v} out of bounds for {target}")));
        }
    }
    Ok((target, original.clone()))
}

macro_rules! binary_same_type {
    ($($name:ident => $f:path);* $(;)?) => {$(
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let (x, y) = resolve_binary(py, a, b)?;
            let result = match (&x, &y) {
                (Arr::F32(p), Arr::F32(q)) => Arr::from($f(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::F64(p), Arr::F64(q)) => Arr::from($f(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::I8(p), Arr::I8(q)) => Arr::from($f(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::I16(p), Arr::I16(q)) => Arr::from($f(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::I32(p), Arr::I32(q)) => Arr::from($f(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::I64(p), Arr::I64(q)) => Arr::from($f(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::U8(p), Arr::U8(q)) => Arr::from($f(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::U16(p), Arr::U16(q)) => Arr::from($f(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::U32(p), Arr::U32(q)) => Arr::from($f(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::U64(p), Arr::U64(q)) => Arr::from($f(&p.view(), &q.view()).map_err(shape_err)?),
                _ => return Err(unsupported("bound for real integer and float dtypes only")),
            };
            out(py, result)
        }
    )*};
}

fn maximum_f<T: FloatIsh>(a: &rustnumpy::ArrayView<T>, b: &rustnumpy::ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    mathfunc::maximum(a, b)
}
fn minimum_f<T: FloatIsh>(a: &rustnumpy::ArrayView<T>, b: &rustnumpy::ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    mathfunc::minimum(a, b)
}
fn fmax_f<T: FloatIsh>(a: &rustnumpy::ArrayView<T>, b: &rustnumpy::ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    mathfunc::fmax(a, b)
}
fn fmin_f<T: FloatIsh>(a: &rustnumpy::ArrayView<T>, b: &rustnumpy::ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    mathfunc::fmin(a, b)
}
fn floor_divide_f<T: Arith>(a: &rustnumpy::ArrayView<T>, b: &rustnumpy::ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    mathfunc::floor_divide(a, b)
}
fn remainder_f<T: Arith>(a: &rustnumpy::ArrayView<T>, b: &rustnumpy::ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    mathfunc::remainder(a, b)
}
fn power_f<T: Arith>(a: &rustnumpy::ArrayView<T>, b: &rustnumpy::ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    mathfunc::power(a, b)
}

binary_same_type! {
    maximum => maximum_f;
    minimum => minimum_f;
    fmax => fmax_f;
    fmin => fmin_f;
    floor_divide => floor_divide_f;
    remainder => remainder_f;
    power => power_f;
}

macro_rules! binary_float_only {
    ($($name:ident => $f:path);* $(;)?) => {$(
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let (x, y) = resolve_binary(py, a, b)?;
            let result = match (&x, &y) {
                (Arr::F32(p), Arr::F32(q)) => Arr::from($f(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::F64(p), Arr::F64(q)) => Arr::from($f(&p.view(), &q.view()).map_err(shape_err)?),
                _ => return Err(unsupported("bound for float32/float64 operands only")),
            };
            out(py, result)
        }
    )*};
}

binary_float_only! {
    arctan2 => mathfunc::arctan2;
    hypot => mathfunc::hypot;
    copysign => mathfunc::copysign;
    fmod => mathfunc::fmod;
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! reg {
        ($($f:ident),* $(,)?) => {$( m.add_function(wrap_pyfunction!($f, m)?)?; )*};
    }
    reg!(
        add, subtract, multiply, astype_, sqrt, cbrt, exp, exp2, expm1, log, log2, log10, log1p, sin, cos, tan, arcsin,
        arccos, arctan, sinh, cosh, tanh, arcsinh, arccosh, arctanh, floor, ceil, trunc, rint, reciprocal, degrees,
        radians, absolute, negative, square, sign, maximum, minimum, fmax, fmin, floor_divide, remainder, power, arctan2,
        hypot, copysign, fmod
    );
    Ok(())
}
