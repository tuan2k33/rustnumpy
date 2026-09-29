use crate::casting::{astype, common_arrs};
use crate::dynarray::{unsupported, value_err, Arr};
use crate::ops::{out, out_array, shape_err, Operand};
use num_traits::One;
use pyo3::prelude::*;
use rustnumpy::dispatch::{reduce, ReduceOptions, WrapAdd, WrapMul};
use rustnumpy::reductions::{AsF64, FloatIsh};
use rustnumpy::{ArrayView, NdArray};

macro_rules! with_real {
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

pub fn norm_axis(axis: isize, ndim: usize) -> PyResult<usize> {
    let n = ndim as isize;
    if axis < -n || axis >= n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "axis {axis} is out of bounds for array of dimension {ndim}"
        )));
    }
    Ok(if axis < 0 { axis + n } else { axis } as usize)
}

pub fn ints(obj: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    if let Ok(v) = obj.extract::<isize>() {
        return Ok(vec![v]);
    }
    obj.extract::<Vec<isize>>()
}

pub fn arr_of(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Arr> {
    Operand::parse(py, obj)?.into_arr(py)
}

fn zero_d_axis(arr: &Arr, axis: Option<isize>) -> Option<isize> {
    if arr.ndim() == 0 && matches!(axis, Some(0) | Some(-1)) { None } else { axis }
}

pub fn accumulator_name(a: &Arr) -> &'static str {
    match a {
        Arr::Bool(_) | Arr::I8(_) | Arr::I16(_) | Arr::I32(_) => "int64",
        Arr::U8(_) | Arr::U16(_) | Arr::U32(_) => "uint64",
        other => other.dtype_name(),
    }
}

pub fn reduced_shape(shape: &[usize], axis: Option<usize>, keepdims: bool) -> Vec<usize> {
    match axis {
        None if keepdims => vec![1; shape.len()],
        None => vec![],
        Some(a) if keepdims => {
            let mut s = shape.to_vec();
            s[a] = 1;
            s
        }
        Some(a) => {
            let mut s = shape.to_vec();
            s.remove(a);
            s
        }
    }
}

pub fn scalar_nd<T>(v: T, shape: &[usize]) -> NdArray<T> {
    NdArray::from_vec(vec![v], shape).expect("a single element fits any all-ones shape")
}

fn flat<T: Copy>(a: &NdArray<T>) -> NdArray<T> {
    a.view().to_owned().into_shape(&[a.len() as isize]).expect("flattening always works")
}

pub fn fold_axis<T: Copy>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
    initial: Option<T>,
    f: impl Fn(T, T) -> T,
) -> PyResult<NdArray<T>> {
    let (data, ax) = match axis {
        None => (flat(a), 0usize),
        Some(x) => (a.clone(), norm_axis(x, a.ndim())?),
    };
    let empty_axis = data.shape()[ax] == 0;
    let opts = ReduceOptions { initial: if empty_axis { initial } else { None }, where_: None, keepdims: false };
    let r = reduce(&data.view(), ax, opts, f).map_err(shape_err)?;
    let target = reduced_shape(a.shape(), axis.map(|_| ax), keepdims);
    r.into_shape(&target.iter().map(|&d| d as isize).collect::<Vec<_>>()).map_err(shape_err)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, keepdims=false))]
pub fn sum(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>, keepdims: bool) -> PyResult<Py<PyAny>> {
    let arr = arr_of(py, a)?;
    let axis = zero_d_axis(&arr, axis);
    let acc = astype(&arr, accumulator_name(&arr))?;
    let result: Arr = match &acc {
        Arr::I64(x) => Arr::from(sum_of(x, axis, keepdims)?),
        Arr::U64(x) => Arr::from(sum_of(x, axis, keepdims)?),
        Arr::F32(x) => Arr::from(sum_of(x, axis, keepdims)?),
        Arr::F64(x) => Arr::from(sum_of(x, axis, keepdims)?),
        Arr::C64(x) => Arr::from(sum_of(x, axis, keepdims)?),
        Arr::C128(x) => Arr::from(sum_of(x, axis, keepdims)?),
        _ => return Err(unsupported("unreachable accumulator dtype")),
    };
    out(py, result)
}

fn sum_of<T: Copy + Default + WrapAdd>(a: &NdArray<T>, axis: Option<isize>, keepdims: bool) -> PyResult<NdArray<T>> {
    if axis.is_none() {
        let s = rustnumpy::sum(&a.view());
        return Ok(scalar_nd(s, &reduced_shape(a.shape(), None, keepdims)));
    }
    fold_axis(a, axis, keepdims, Some(T::default()), |x, y| x.wrap_add(y))
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, keepdims=false))]
pub fn prod(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>, keepdims: bool) -> PyResult<Py<PyAny>> {
    let arr = arr_of(py, a)?;
    let axis = zero_d_axis(&arr, axis);
    let acc = astype(&arr, accumulator_name(&arr))?;
    let result: Arr = match &acc {
        Arr::I64(x) => Arr::from(prod_of(x, axis, keepdims)?),
        Arr::U64(x) => Arr::from(prod_of(x, axis, keepdims)?),
        Arr::F32(x) => Arr::from(prod_of(x, axis, keepdims)?),
        Arr::F64(x) => Arr::from(prod_of(x, axis, keepdims)?),
        Arr::C64(x) => Arr::from(prod_of(x, axis, keepdims)?),
        Arr::C128(x) => Arr::from(prod_of(x, axis, keepdims)?),
        _ => return Err(unsupported("unreachable accumulator dtype")),
    };
    out(py, result)
}

fn prod_of<T: Copy + One + WrapMul>(a: &NdArray<T>, axis: Option<isize>, keepdims: bool) -> PyResult<NdArray<T>> {
    fold_axis(a, axis, keepdims, Some(T::one()), |x, y| x.wrap_mul(y))
}

fn nan_min<T: FloatIsh>(a: T, b: T) -> T {
    if a.is_nan_ish() || (!b.is_nan_ish() && a <= b) { a } else { b }
}

fn nan_max<T: FloatIsh>(a: T, b: T) -> T {
    if a.is_nan_ish() || (!b.is_nan_ish() && a >= b) { a } else { b }
}

fn extreme_of<T: FloatIsh>(a: &NdArray<T>, axis: Option<isize>, keepdims: bool, want_max: bool) -> PyResult<NdArray<T>> {
    if a.is_empty() && axis.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err("zero-size array to reduction operation which has no identity"));
    }
    if axis.is_none() {
        let v = if want_max { rustnumpy::max(&a.view()) } else { rustnumpy::min(&a.view()) };
        return Ok(scalar_nd(v.map_err(value_err)?, &reduced_shape(a.shape(), None, keepdims)));
    }
    if want_max { fold_axis(a, axis, keepdims, None, nan_max) } else { fold_axis(a, axis, keepdims, None, nan_min) }
}

macro_rules! extreme_fn {
    ($name:ident, $alias:ident, $max:expr) => {
        #[pyfunction]
        #[pyo3(signature = (a, axis=None, keepdims=false))]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>, keepdims: bool) -> PyResult<Py<PyAny>> {
            let arr = arr_of(py, a)?;
            let axis = zero_d_axis(&arr, axis);
            let result = with_real!(&arr, x => Arr::from(extreme_of(x, axis, keepdims, $max)?));
            out(py, result)
        }
        #[pyfunction]
        #[pyo3(signature = (a, axis=None, keepdims=false))]
        pub fn $alias(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>, keepdims: bool) -> PyResult<Py<PyAny>> {
            $name(py, a, axis, keepdims)
        }
    };
}
extreme_fn!(max, amax, true);
extreme_fn!(min, amin, false);

fn float_result(py: Python<'_>, arr: &Arr, value: f64) -> PyResult<Py<PyAny>> {
    let r = if matches!(arr, Arr::F32(_)) { Arr::from(scalar_nd(value as f32, &[])) } else { Arr::from(scalar_nd(value, &[])) };
    out(py, r)
}

fn stat_input(py: Python<'_>, a: &Bound<'_, PyAny>, axis: &Option<isize>) -> PyResult<Arr> {
    if axis.is_some() {
        return Err(unsupported("axis is not bound for this statistic yet"));
    }
    let arr = arr_of(py, a)?;
    if arr.is_bool() { astype(&arr, "uint8") } else { Ok(arr) }
}

macro_rules! stat_fn {
    ($name:ident, $core:path $(, $extra:ident : $ty:ty = $default:expr)*) => {
        #[pyfunction]
        #[pyo3(signature = (a, axis=None $(, $extra=$default)*))]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>, $($extra: $ty),*) -> PyResult<Py<PyAny>> {
            let arr = stat_input(py, a, &axis)?;
            let v = with_real!(&arr, x => $core(&x.view() $(, $extra)*));
            float_result(py, &arr, v)
        }
    };
}

fn mean_c<T: AsF64>(v: &ArrayView<T>) -> f64 {
    rustnumpy::mean(v)
}
fn nanmean_c<T: FloatIsh + AsF64>(v: &ArrayView<T>) -> f64 {
    rustnumpy::nanmean(v)
}
fn var_c<T: AsF64>(v: &ArrayView<T>, ddof: usize) -> f64 {
    rustnumpy::var(v, ddof)
}
fn std_c<T: AsF64>(v: &ArrayView<T>, ddof: usize) -> f64 {
    rustnumpy::std(v, ddof)
}
fn nanvar_c<T: FloatIsh + AsF64>(v: &ArrayView<T>, ddof: usize) -> f64 {
    rustnumpy::nanvar(v, ddof)
}
fn nanstd_c<T: FloatIsh + AsF64>(v: &ArrayView<T>, ddof: usize) -> f64 {
    rustnumpy::nanstd(v, ddof)
}

stat_fn!(mean, mean_c);
stat_fn!(var, var_c, ddof: usize = 0);
stat_fn!(std_, std_c, ddof: usize = 0);
stat_fn!(nanmean, nanmean_c);
stat_fn!(nanvar, nanvar_c, ddof: usize = 0);
stat_fn!(nanstd, nanstd_c, ddof: usize = 0);

#[pyfunction]
#[pyo3(signature = (a, axis=None))]
pub fn median(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
    let arr = stat_input(py, a, &axis)?;
    let v = with_real!(&arr, x => match rustnumpy::median(&x.view()) {
        Ok(v) => v,
        Err(rustnumpy::ReductionError::EmptyInput) => f64::NAN,
        Err(e) => return Err(value_err(e)),
    });
    float_result(py, &arr, v)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None))]
pub fn nanmedian(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
    let arr = stat_input(py, a, &axis)?;
    let v = with_real!(&arr, x => match rustnumpy::nanmedian(&x.view()) {
        Ok(v) => v,
        Err(rustnumpy::ReductionError::EmptyInput) => f64::NAN,
        Err(e) => return Err(value_err(e)),
    });
    float_result(py, &arr, v)
}

#[pyfunction]
#[pyo3(signature = (a, q, axis=None))]
pub fn percentile(py: Python<'_>, a: &Bound<'_, PyAny>, q: f64, axis: Option<isize>) -> PyResult<Py<PyAny>> {
    let arr = stat_input(py, a, &axis)?;
    let v = with_real!(&arr, x => match rustnumpy::percentile(&x.view(), q) {
        Ok(v) => v,
        Err(rustnumpy::ReductionError::EmptyInput) => return Err(pyo3::exceptions::PyIndexError::new_err("index -1 is out of bounds for axis 0 with size 0")),
        Err(e) => return Err(value_err(e)),
    });
    float_result(py, &arr, v)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, keepdims=false))]
pub fn nansum(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>, keepdims: bool) -> PyResult<Py<PyAny>> {
    if axis.is_some() {
        return Err(unsupported("axis is not bound for nansum yet"));
    }
    let arr = arr_of(py, a)?;
    let acc = astype(&arr, accumulator_name(&arr))?;
    let shape = reduced_shape(&acc.shape(), None, keepdims);
    let result: Arr = match &acc {
        Arr::I64(x) => Arr::from(scalar_nd(rustnumpy::nansum(&x.view()), &shape)),
        Arr::U64(x) => Arr::from(scalar_nd(rustnumpy::nansum(&x.view()), &shape)),
        Arr::F32(x) => Arr::from(scalar_nd(rustnumpy::nansum(&x.view()), &shape)),
        Arr::F64(x) => Arr::from(scalar_nd(rustnumpy::nansum(&x.view()), &shape)),
        _ => return Err(unsupported("nansum is bound for real dtypes only")),
    };
    out(py, result)
}

macro_rules! nan_extreme {
    ($name:ident, $core:path) => {
        #[pyfunction]
        #[pyo3(signature = (a, axis=None))]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
            if axis.is_some() {
                return Err(unsupported("axis is not bound for this function yet"));
            }
            let arr = arr_of(py, a)?;
            let result = with_real!(&arr, x => Arr::from(scalar_nd($core(&x.view()).map_err(value_err)?, &[])));
            out(py, result)
        }
    };
}
nan_extreme!(nanmin, rustnumpy::nanmin);
nan_extreme!(nanmax, rustnumpy::nanmax);

fn cov_like(py: Python<'_>, m: &Bound<'_, PyAny>, ddof: usize, correlation: bool, as_scalar: bool) -> PyResult<Py<PyAny>> {
    let arr = arr_of(py, m)?;
    if arr.is_complex() {
        return Err(unsupported("complex covariance is not bound"));
    }
    let empty = arr.shape().iter().product::<usize>() == 0;
    if empty && arr.ndim() != 1 {
        return Err(unsupported("empty multi-dimensional covariance input is not bound"));
    }
    let observations = arr.shape().last().copied().unwrap_or(1);
    if !correlation && !empty && ddof >= observations {
        return Err(unsupported("ddof >= number of observations (NumPy returns inf/nan with a warning; the core returns an error)"));
    }
    if empty {
        let nan = Arr::from(scalar_nd(f64::NAN, &[]));
        return if as_scalar { out(py, nan) } else { out_array(py, nan) };
    }
    let mat = as_f64_matrix(py, m)?;
    let result = if correlation {
        rustnumpy::corrcoef(&mat.view()).map_err(value_err)?
    } else {
        rustnumpy::cov(&mat.view(), ddof).map_err(value_err)?
    };
    let result = if result.len() == 1 { result.into_shape(&[]).map_err(shape_err)? } else { result };
    if as_scalar && result.ndim() == 0 { out(py, Arr::from(result)) } else { out_array(py, Arr::from(result)) }
}

#[pyfunction]
#[pyo3(signature = (m, ddof=1))]
pub fn cov(py: Python<'_>, m: &Bound<'_, PyAny>, ddof: usize) -> PyResult<Py<PyAny>> {
    cov_like(py, m, ddof, false, false)
}

#[pyfunction]
pub fn corrcoef(py: Python<'_>, m: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    cov_like(py, m, 1, true, true)
}

pub fn as_f64_matrix(py: Python<'_>, m: &Bound<'_, PyAny>) -> PyResult<NdArray<f64>> {
    let arr = astype(&arr_of(py, m)?, "float64")?;
    let Arr::F64(x) = arr else { unreachable!("cast to float64") };
    match x.ndim() {
        2 => Ok(x),
        1 => {
            let n = x.len() as isize;
            x.into_shape(&[1, n]).map_err(shape_err)
        }
        _ => Err(unsupported("only 1-D and 2-D inputs are bound")),
    }
}

#[pyfunction]
#[pyo3(signature = (a, bins=10, range=None))]
pub fn histogram(py: Python<'_>, a: &Bound<'_, PyAny>, bins: usize, range: Option<(f64, f64)>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let arr = astype(&arr_of(py, a)?, "float64")?;
    let Arr::F64(x) = arr else { unreachable!("cast to float64") };
    let (counts, edges) = rustnumpy::histogram(x.as_slice(), bins, range).map_err(value_err)?;
    let counts: Vec<i64> = counts.into_iter().map(|c| c as i64).collect();
    let n = counts.len();
    let ne = edges.len();
    Ok((
        out_array(py, Arr::from(NdArray::from_vec(counts, &[n]).map_err(shape_err)?))?,
        out_array(py, Arr::from(NdArray::from_vec(edges, &[ne]).map_err(shape_err)?))?,
    ))
}

fn sort_like(py: Python<'_>, a: &Bound<'_, PyAny>, axis: isize, indices: bool) -> PyResult<Py<PyAny>> {
    let arr = arr_of(py, a)?;
    if arr.ndim() == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Cannot sort a 0-d array"));
    }
    let ax = norm_axis(axis, arr.ndim())?;
    let result = if indices {
        with_real!(&arr, x => {
            let r = rustnumpy::argsort(&x.view(), ax).map_err(shape_err)?;
            let data: Vec<i64> = r.as_slice().iter().map(|&i| i as i64).collect();
            Arr::from(NdArray::from_vec(data, r.shape()).map_err(shape_err)?)
        })
    } else {
        with_real!(&arr, x => Arr::from(rustnumpy::sort(&x.view(), ax).map_err(shape_err)?))
    };
    out_array(py, result)
}

#[pyfunction]
#[pyo3(signature = (a, axis=-1))]
pub fn sort(py: Python<'_>, a: &Bound<'_, PyAny>, axis: isize) -> PyResult<Py<PyAny>> {
    sort_like(py, a, axis, false)
}

#[pyfunction]
#[pyo3(signature = (a, axis=-1))]
pub fn argsort(py: Python<'_>, a: &Bound<'_, PyAny>, axis: isize) -> PyResult<Py<PyAny>> {
    sort_like(py, a, axis, true)
}

#[pyfunction]
#[pyo3(signature = (a, v, side="left"))]
pub fn searchsorted(py: Python<'_>, a: &Bound<'_, PyAny>, v: &Bound<'_, PyAny>, side: &str) -> PyResult<Py<PyAny>> {
    let side = match side {
        "left" => rustnumpy::Side::Left,
        "right" => rustnumpy::Side::Right,
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!("side must be 'left' or 'right', got {other}"))),
    };
    let (sorted, values) = common_arrs(&arr_of(py, a)?, &arr_of(py, v)?)?;
    if sorted.ndim() != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err("object too deep for desired array"));
    }
    let shape = values.shape();
    macro_rules! go {
        ($s:expr, $v:expr) => {{
            let idx: Vec<i64> = rustnumpy::searchsorted($s.as_slice(), $v.as_slice(), side).into_iter().map(|i| i as i64).collect();
            NdArray::from_vec(idx, &shape).map_err(shape_err)?
        }};
    }
    let result = match (&sorted, &values) {
        (Arr::I8(s), Arr::I8(x)) => go!(s, x),
        (Arr::I16(s), Arr::I16(x)) => go!(s, x),
        (Arr::I32(s), Arr::I32(x)) => go!(s, x),
        (Arr::I64(s), Arr::I64(x)) => go!(s, x),
        (Arr::U8(s), Arr::U8(x)) => go!(s, x),
        (Arr::U16(s), Arr::U16(x)) => go!(s, x),
        (Arr::U32(s), Arr::U32(x)) => go!(s, x),
        (Arr::U64(s), Arr::U64(x)) => go!(s, x),
        (Arr::F32(s), Arr::F32(x)) => go!(s, x),
        (Arr::F64(s), Arr::F64(x)) => go!(s, x),
        _ => return Err(unsupported("searchsorted is bound for real dtypes only")),
    };
    out(py, Arr::from(result))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! reg {
        ($($f:ident),* $(,)?) => {$( m.add_function(wrap_pyfunction!($f, m)?)?; )*};
    }
    reg!(
        sum, prod, max, amax, min, amin, mean, var, median, nanmedian, percentile, nansum, nanmean, nanvar, nanstd,
        nanmin, nanmax, cov, corrcoef, histogram, sort, argsort, searchsorted
    );
    m.add("std", wrap_pyfunction!(std_, m)?)?;
    Ok(())
}
