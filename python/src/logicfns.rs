use crate::arrayfns::{accumulator_name, arr_of, fold_axis, norm_axis};
use crate::casting::astype;
use crate::dynarray::{unsupported, Arr};
use crate::ops::{out, out_array, resolve_binary, shape_err};
use crate::shapefns::common_all;
use crate::with_arr;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use rustnumpy::logic;
use rustnumpy::mathfunc;
use rustnumpy::NdArray;

macro_rules! same_arms {
    ($x:expr, $y:expr, $p:ident, $q:ident => $body:expr, complex: $complex:expr) => {
        match ($x, $y) {
            (Arr::Bool($p), Arr::Bool($q)) => $body,
            (Arr::I8($p), Arr::I8($q)) => $body,
            (Arr::I16($p), Arr::I16($q)) => $body,
            (Arr::I32($p), Arr::I32($q)) => $body,
            (Arr::I64($p), Arr::I64($q)) => $body,
            (Arr::U8($p), Arr::U8($q)) => $body,
            (Arr::U16($p), Arr::U16($q)) => $body,
            (Arr::U32($p), Arr::U32($q)) => $body,
            (Arr::U64($p), Arr::U64($q)) => $body,
            (Arr::F32($p), Arr::F32($q)) => $body,
            (Arr::F64($p), Arr::F64($q)) => $body,
            _ => return Err(unsupported("complex operands are not bound for this comparison")),
        }
    };
}

macro_rules! compare_fn {
    ($name:ident, $core:path, eq) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let (x, y) = resolve_binary(py, a, b)?;
            let r = match (&x, &y) {
                (Arr::C64(p), Arr::C64(q)) => $core(&p.view(), &q.view()).map_err(shape_err)?,
                (Arr::C128(p), Arr::C128(q)) => $core(&p.view(), &q.view()).map_err(shape_err)?,
                _ => same_arms!(&x, &y, p, q => $core(&p.view(), &q.view()).map_err(shape_err)?, complex: false),
            };
            out(py, Arr::from(r))
        }
    };
    ($name:ident, $core:path, ord) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let (x, y) = resolve_binary(py, a, b)?;
            let r = same_arms!(&x, &y, p, q => $core(&p.view(), &q.view()).map_err(shape_err)?, complex: false);
            out(py, Arr::from(r))
        }
    };
}
compare_fn!(equal, logic::equal, eq);
compare_fn!(not_equal, logic::not_equal, eq);
compare_fn!(less, logic::less, ord);
compare_fn!(less_equal, logic::less_equal, ord);
compare_fn!(greater, logic::greater, ord);
compare_fn!(greater_equal, logic::greater_equal, ord);

macro_rules! truthy_binary {
    ($name:ident, $core:path) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let (x, y) = resolve_binary(py, a, b)?;
            let r = match (&x, &y) {
                (Arr::C64(p), Arr::C64(q)) => $core(&p.view(), &q.view()).map_err(shape_err)?,
                (Arr::C128(p), Arr::C128(q)) => $core(&p.view(), &q.view()).map_err(shape_err)?,
                _ => same_arms!(&x, &y, p, q => $core(&p.view(), &q.view()).map_err(shape_err)?, complex: false),
            };
            out(py, Arr::from(r))
        }
    };
}
truthy_binary!(logical_and, logic::logical_and);
truthy_binary!(logical_or, logic::logical_or);
truthy_binary!(logical_xor, logic::logical_xor);

#[pyfunction]
pub fn logical_not(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = arr_of(py, a)?;
    out(py, Arr::from(with_arr!(&arr, x => logic::logical_not(&x.view()))))
}

macro_rules! int_binary {
    ($name:ident, $core:path) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let (x, y) = resolve_binary(py, a, b)?;
            let r = match (&x, &y) {
                (Arr::Bool(p), Arr::Bool(q)) => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::I8(p), Arr::I8(q)) => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::I16(p), Arr::I16(q)) => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::I32(p), Arr::I32(q)) => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::I64(p), Arr::I64(q)) => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::U8(p), Arr::U8(q)) => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::U16(p), Arr::U16(q)) => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::U32(p), Arr::U32(q)) => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::U64(p), Arr::U64(q)) => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?),
                _ => return Err(PyTypeError::new_err("ufunc only supports integer and boolean inputs")),
            };
            out(py, r)
        }
    };
}
int_binary!(bitwise_and, logic::bitwise_and);
int_binary!(bitwise_or, logic::bitwise_or);
int_binary!(bitwise_xor, logic::bitwise_xor);

macro_rules! shift_binary {
    ($name:ident, $core:path) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let (x, y) = resolve_binary(py, a, b)?;
            let r = match (&x, &y) {
                (Arr::I8(p), Arr::I8(q)) => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::I16(p), Arr::I16(q)) => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::I32(p), Arr::I32(q)) => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::I64(p), Arr::I64(q)) => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::U8(p), Arr::U8(q)) => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::U16(p), Arr::U16(q)) => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::U32(p), Arr::U32(q)) => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?),
                (Arr::U64(p), Arr::U64(q)) => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?),
                _ => return Err(PyTypeError::new_err("ufunc only supports integer inputs")),
            };
            out(py, r)
        }
    };
}
shift_binary!(left_shift, logic::left_shift);
shift_binary!(right_shift, logic::right_shift);

#[pyfunction]
pub fn invert(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = arr_of(py, a)?;
    let r = match &arr {
        Arr::Bool(x) => Arr::from(logic::invert(&x.view())),
        Arr::I8(x) => Arr::from(logic::invert(&x.view())),
        Arr::I16(x) => Arr::from(logic::invert(&x.view())),
        Arr::I32(x) => Arr::from(logic::invert(&x.view())),
        Arr::I64(x) => Arr::from(logic::invert(&x.view())),
        Arr::U8(x) => Arr::from(logic::invert(&x.view())),
        Arr::U16(x) => Arr::from(logic::invert(&x.view())),
        Arr::U32(x) => Arr::from(logic::invert(&x.view())),
        Arr::U64(x) => Arr::from(logic::invert(&x.view())),
        _ => return Err(PyTypeError::new_err("ufunc 'invert' not supported for the input types")),
    };
    out(py, r)
}

macro_rules! class_fn {
    ($name:ident, $core:path) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let arr = arr_of(py, a)?;
            out(py, Arr::from(with_arr!(&arr, x => $core(&x.view()))))
        }
    };
}
class_fn!(isnan, logic::isnan);
class_fn!(isinf, logic::isinf);
class_fn!(isfinite, logic::isfinite);

#[pyfunction]
pub fn signbit(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = arr_of(py, a)?;
    let r = match &arr {
        Arr::F32(x) => logic::signbit(&x.view()),
        Arr::F64(x) => logic::signbit(&x.view()),
        _ => return Err(unsupported("signbit is bound for float32/float64")),
    };
    out(py, Arr::from(r))
}

#[pyfunction]
pub fn divide(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (mut x, mut y) = resolve_binary(py, a, b)?;
    if !matches!(x, Arr::F32(_) | Arr::F64(_) | Arr::C64(_) | Arr::C128(_)) {
        x = astype(&x, "float64")?;
        y = astype(&y, "float64")?;
    }
    let r = match (&x, &y) {
        (Arr::F32(p), Arr::F32(q)) => Arr::from(mathfunc::divide(&p.view(), &q.view()).map_err(shape_err)?),
        (Arr::F64(p), Arr::F64(q)) => Arr::from(mathfunc::divide(&p.view(), &q.view()).map_err(shape_err)?),
        (Arr::C64(p), Arr::C64(q)) => Arr::from(mathfunc::divide(&p.view(), &q.view()).map_err(shape_err)?),
        (Arr::C128(p), Arr::C128(q)) => Arr::from(mathfunc::divide(&p.view(), &q.view()).map_err(shape_err)?),
        _ => return Err(unsupported("unreachable divide dtype")),
    };
    out(py, r)
}

fn truth_arr(arr: &Arr) -> PyResult<NdArray<bool>> {
    match astype(arr, "bool")? {
        Arr::Bool(m) => Ok(m),
        _ => unreachable!("cast to bool"),
    }
}

macro_rules! any_all {
    ($name:ident, $identity:expr, $op:expr) => {
        #[pyfunction]
        #[pyo3(signature = (a, axis=None, keepdims=false))]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>, keepdims: bool) -> PyResult<Py<PyAny>> {
            let arr = arr_of(py, a)?;
            let m = truth_arr(&arr)?;
            let axis = if arr.ndim() == 0 && matches!(axis, Some(0) | Some(-1)) { None } else { axis };
            let r = fold_axis(&m, axis, keepdims, Some($identity), $op)?;
            out(py, Arr::from(r))
        }
    };
}
any_all!(any, false, |x: bool, y: bool| x || y);
any_all!(all, true, |x: bool, y: bool| x && y);

#[pyfunction]
pub fn count_nonzero(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = arr_of(py, a)?;
    let n = with_arr!(&arr, x => logic::count_nonzero(&x.view()));
    out(py, Arr::scalar(n as i64))
}

fn arg_like(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>, want_max: bool) -> PyResult<Py<PyAny>> {
    let mut arr = arr_of(py, a)?;
    if arr.is_bool() {
        arr = astype(&arr, "uint8")?;
    }
    if arr.is_complex() {
        return Err(unsupported("complex argmax/argmin is not bound"));
    }
    let ax = axis.map(|x| norm_axis(x, arr.ndim())).transpose()?;
    let r = with_real_all!(&arr, x => logic::arg_extreme(&x.view(), ax, want_max).map_err(|e| match e {
        rustnumpy::ShapeError::ReduceNoIdentity => PyValueError::new_err("attempt to get argmax of an empty sequence"),
        other => shape_err(other),
    })?);
    let data: Vec<i64> = r.as_slice().iter().map(|&i| i as i64).collect();
    out(py, Arr::from(NdArray::from_vec(data, r.shape()).map_err(shape_err)?))
}

macro_rules! with_real_all {
    ($arr:expr, $a:ident => $body:expr) => {
        match $arr {
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
            _ => return Err(unsupported("bound for real integer and float dtypes only")),
        }
    };
}
use with_real_all;

#[pyfunction]
#[pyo3(signature = (a, axis=None))]
pub fn argmax(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
    arg_like(py, a, axis, true)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None))]
pub fn argmin(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
    arg_like(py, a, axis, false)
}

macro_rules! cumulative {
    ($name:ident, $core:path) => {
        #[pyfunction]
        #[pyo3(signature = (a, axis=None))]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
            let arr = arr_of(py, a)?;
            let acc = astype(&arr, accumulator_name(&arr))?;
            let ax = axis.map(|x| norm_axis(x, acc.ndim().max(1))).transpose()?;
            let ax = if acc.ndim() == 0 { None } else { ax };
            let r = match &acc {
                Arr::I64(x) => Arr::from($core(&x.view(), ax).map_err(shape_err)?),
                Arr::U64(x) => Arr::from($core(&x.view(), ax).map_err(shape_err)?),
                Arr::F32(x) => Arr::from($core(&x.view(), ax).map_err(shape_err)?),
                Arr::F64(x) => Arr::from($core(&x.view(), ax).map_err(shape_err)?),
                Arr::C64(x) => Arr::from($core(&x.view(), ax).map_err(shape_err)?),
                Arr::C128(x) => Arr::from($core(&x.view(), ax).map_err(shape_err)?),
                _ => return Err(unsupported("unreachable accumulator dtype")),
            };
            out_array_or_scalar(py, r)
        }
    };
}

fn out_array_or_scalar(py: Python<'_>, a: Arr) -> PyResult<Py<PyAny>> {
    out_array(py, a)
}
cumulative!(cumsum, logic::cumsum);
cumulative!(cumprod, logic::cumprod);

#[pyfunction]
#[pyo3(signature = (a, a_min=None, a_max=None))]
pub fn clip(py: Python<'_>, a: &Bound<'_, PyAny>, a_min: Option<&Bound<'_, PyAny>>, a_max: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let base = arr_of(py, a)?;
    let mut parts = vec![base];
    let mut order = Vec::new();
    for (slot, bound) in [(0usize, a_min), (1usize, a_max)] {
        if let Some(b) = bound {
            let (x, y) = resolve_binary(py, a, b)?;
            parts[0] = x;
            parts.push(y);
            order.push(slot);
        }
    }
    let parts = common_all(&parts)?;
    let low_idx = order.iter().position(|&s| s == 0).map(|i| i + 1);
    let high_idx = order.iter().position(|&s| s == 1).map(|i| i + 1);
    macro_rules! go {
        ($x:ident) => {{
            let scalar = |i: Option<usize>| -> PyResult<Option<_>> {
                Ok(match i {
                    None => None,
                    Some(k) => match &parts[k] {
                        Arr::$x(v) if v.ndim() == 0 => Some(v.as_slice()[0]),
                        _ => return Err(unsupported("clip is bound for scalar bounds")),
                    },
                })
            };
            match &parts[0] {
                Arr::$x(v) => Arr::from(logic::clip(&v.view(), scalar(low_idx)?, scalar(high_idx)?)),
                _ => unreachable!("common dtype"),
            }
        }};
    }
    let r = match &parts[0] {
        Arr::I8(_) => go!(I8),
        Arr::I16(_) => go!(I16),
        Arr::I32(_) => go!(I32),
        Arr::I64(_) => go!(I64),
        Arr::U8(_) => go!(U8),
        Arr::U16(_) => go!(U16),
        Arr::U32(_) => go!(U32),
        Arr::U64(_) => go!(U64),
        Arr::F32(_) => go!(F32),
        Arr::F64(_) => go!(F64),
        _ => return Err(unsupported("clip is bound for real integer and float dtypes only")),
    };
    out(py, r)
}

fn arrays_of(py: Python<'_>, seq: &Bound<'_, PyAny>) -> PyResult<Vec<Arr>> {
    let items: Vec<Arr> = seq.try_iter()?.map(|i| Arr::from_object(py, &i?)).collect::<PyResult<_>>()?;
    if items.is_empty() {
        return Err(PyValueError::new_err("need at least one array to concatenate"));
    }
    common_all(&items)
}

macro_rules! join_fn {
    ($name:ident, $core:path) => {
        fn $name(arrs: &[Arr], axis: usize) -> PyResult<Arr> {
            macro_rules! per {
                ($v:ident) => {{
                    let list: Vec<&NdArray<_>> = arrs.iter().map(|a| match a { Arr::$v(x) => x, _ => unreachable!("common dtype") }).collect();
                    Arr::from($core(&list, axis).map_err(shape_err)?)
                }};
            }
            Ok(match &arrs[0] {
                Arr::Bool(_) => per!(Bool),
                Arr::I8(_) => per!(I8),
                Arr::I16(_) => per!(I16),
                Arr::I32(_) => per!(I32),
                Arr::I64(_) => per!(I64),
                Arr::U8(_) => per!(U8),
                Arr::U16(_) => per!(U16),
                Arr::U32(_) => per!(U32),
                Arr::U64(_) => per!(U64),
                Arr::F32(_) => per!(F32),
                Arr::F64(_) => per!(F64),
                Arr::C64(_) => per!(C64),
                Arr::C128(_) => per!(C128),
            })
        }
    };
}
join_fn!(join_concat, rustnumpy::concatenate);
join_fn!(join_stack, rustnumpy::stack);

#[pyfunction]
#[pyo3(signature = (arrays, axis=Some(0)))]
pub fn concatenate(py: Python<'_>, arrays: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
    let mut arrs = arrays_of(py, arrays)?;
    let ax = match axis {
        None => {
            for a in arrs.iter_mut() {
                let n = a.shape().iter().product::<usize>() as isize;
                *a = with_arr!(&*a, x => Arr::from(x.clone().into_shape(&[n]).map_err(shape_err)?));
            }
            0
        }
        Some(x) => {
            if arrs[0].ndim() == 0 {
                return Err(PyValueError::new_err("zero-dimensional arrays cannot be concatenated"));
            }
            norm_axis(x, arrs[0].ndim())?
        }
    };
    out_array(py, join_concat(&arrs, ax)?)
}

#[pyfunction]
#[pyo3(signature = (arrays, axis=0))]
pub fn stack(py: Python<'_>, arrays: &Bound<'_, PyAny>, axis: isize) -> PyResult<Py<PyAny>> {
    let arrs = arrays_of(py, arrays)?;
    let ax = norm_axis(axis, arrs[0].ndim() + 1)?;
    out_array(py, join_stack(&arrs, ax)?)
}

fn at_least(arr: Arr, nd: usize) -> PyResult<Arr> {
    let mut shape = arr.shape();
    if shape.len() >= nd {
        return Ok(arr);
    }
    while shape.len() < nd {
        if nd == 2 && shape.len() == 1 && shape.len() < nd {
            shape.insert(0, 1);
        } else if shape.is_empty() {
            shape.push(1);
        } else {
            shape.insert(0, 1);
        }
    }
    let dims: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
    Ok(with_arr!(&arr, x => Arr::from(x.clone().into_shape(&dims).map_err(shape_err)?)))
}

#[pyfunction]
pub fn vstack(py: Python<'_>, arrays: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arrs: Vec<Arr> = arrays_of(py, arrays)?.into_iter().map(|a| at_least(a, 2)).collect::<PyResult<_>>()?;
    out_array(py, join_concat(&arrs, 0)?)
}

#[pyfunction]
pub fn hstack(py: Python<'_>, arrays: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arrs: Vec<Arr> = arrays_of(py, arrays)?.into_iter().map(|a| at_least(a, 1)).collect::<PyResult<_>>()?;
    let axis = if arrs[0].ndim() == 1 { 0 } else { 1 };
    out_array(py, join_concat(&arrs, axis)?)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! reg {
        ($($f:ident),* $(,)?) => {$( m.add_function(wrap_pyfunction!($f, m)?)?; )*};
    }
    reg!(
        equal, not_equal, less, less_equal, greater, greater_equal, logical_and, logical_or, logical_xor, logical_not,
        bitwise_and, bitwise_or, bitwise_xor, left_shift, right_shift, invert, isnan, isinf, isfinite, signbit, divide, any,
        all, count_nonzero, argmax, argmin, cumsum, cumprod, clip, concatenate, stack, vstack, hstack
    );
    m.add("true_divide", wrap_pyfunction!(divide, m)?)?;
    Ok(())
}
