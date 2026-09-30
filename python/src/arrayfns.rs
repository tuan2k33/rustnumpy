use crate::casting::{astype, common_arrs};
use crate::dynarray::{unsupported, value_err, Arr, ArrIn};
use crate::ops::{out, out_array, shape_err, Operand};
use num_traits::One;
use pyo3::exceptions::PyTypeError;
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
            Arr::F16($a) => $body,
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

fn strict_int(obj: &Bound<'_, PyAny>) -> PyResult<isize> {
    if obj.is_instance_of::<pyo3::types::PyBool>() {
        return Err(PyTypeError::new_err("'bool' object cannot be interpreted as an integer"));
    }
    obj.extract::<isize>()
}

pub fn ints(obj: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    if let Ok(v) = strict_int(obj) {
        return Ok(vec![v]);
    }
    obj.try_iter()?.map(|x| strict_int(&x?)).collect()
}

pub fn arr_of(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<ArrIn> {
    Operand::parse(py, obj)?.into_input(py)
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
    let arr_in = arr_of(py, a)?;
    let arr: &Arr = &arr_in;
    let axis = zero_d_axis(arr, axis);
    if matches!(arr, Arr::F16(_)) {
        let wide = astype(arr, "float32")?;
        let r = sum_prod_f16(&wide, axis, keepdims, true)?;
        return out(py, astype(&r, "float16")?);
    }
    let mut acc_slot = None;
    let acc = crate::casting::cast_ref(arr, accumulator_name(arr), &mut acc_slot)?;
    let result: Arr = match acc {
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

fn sum_prod_f16(wide: &Arr, axis: Option<isize>, keepdims: bool, add: bool) -> PyResult<Arr> {
    let Arr::F32(x) = wide else { unreachable!("cast to float32") };
    Ok(Arr::from(if add { sum_of(x, axis, keepdims)? } else { prod_of(x, axis, keepdims)? }))
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
    let arr_in = arr_of(py, a)?;
    let arr: &Arr = &arr_in;
    let axis = zero_d_axis(arr, axis);
    if matches!(arr, Arr::F16(_)) {
        let wide = astype(arr, "float32")?;
        let r = sum_prod_f16(&wide, axis, keepdims, false)?;
        return out(py, astype(&r, "float16")?);
    }
    let mut acc_slot = None;
    let acc = crate::casting::cast_ref(arr, accumulator_name(arr), &mut acc_slot)?;
    let result: Arr = match acc {
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

fn bool_extreme(a: &NdArray<bool>, axis: Option<isize>, keepdims: bool, want_max: bool) -> PyResult<NdArray<bool>> {
    if a.is_empty() && axis.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err("zero-size array to reduction operation which has no identity"));
    }
    if want_max { fold_axis(a, axis, keepdims, None, |x, y| x || y) } else { fold_axis(a, axis, keepdims, None, |x, y| x && y) }
}

fn complex_extreme<T: num_traits::Float>(
    a: &NdArray<num_complex::Complex<T>>,
    axis: Option<isize>,
    keepdims: bool,
    want_max: bool,
) -> PyResult<NdArray<num_complex::Complex<T>>> {
    if a.is_empty() && axis.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err("zero-size array to reduction operation which has no identity"));
    }
    if want_max {
        fold_axis(a, axis, keepdims, None, crate::umath::c_maximum)
    } else {
        fold_axis(a, axis, keepdims, None, crate::umath::c_minimum)
    }
}

macro_rules! extreme_fn {
    ($name:ident, $alias:ident, $max:expr) => {
        #[pyfunction]
        #[pyo3(signature = (a, axis=None, keepdims=false))]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>, keepdims: bool) -> PyResult<Py<PyAny>> {
            let arr_in = arr_of(py, a)?;
            let arr: &Arr = &arr_in;
            let axis = zero_d_axis(&arr, axis);
            let result = match &arr {
                Arr::Bool(x) => Arr::from(bool_extreme(x, axis, keepdims, $max)?),
                Arr::C64(x) => Arr::from(complex_extreme(x, axis, keepdims, $max)?),
                Arr::C128(x) => Arr::from(complex_extreme(x, axis, keepdims, $max)?),
                _ => with_real!(&arr, x => Arr::from(extreme_of(x, axis, keepdims, $max)?)),
            };
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

fn float_out(py: Python<'_>, arr: &Arr, values: NdArray<f64>) -> PyResult<Py<PyAny>> {
    let wide = Arr::from(values);
    let target = match arr {
        Arr::F16(_) => "float16",
        Arr::F32(_) => "float32",
        _ => "float64",
    };
    out(py, astype(&wide, target)?)
}

fn stat_input(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<ArrIn> {
    let arr_in = arr_of(py, a)?;
    let arr: &Arr = &arr_in;
    if arr.is_complex() {
        return Err(unsupported("complex statistics are composed at the Python layer"));
    }
    if arr.is_bool() { astype(arr, "uint8").map(ArrIn::Owned) } else { Ok(arr_in) }
}

fn lane_view<T>(lane: &[T]) -> PyResult<ArrayView<'_, T>> {
    ArrayView::from_raw_parts(lane, vec![lane.len()], vec![1], 0).map_err(shape_err)
}

fn lanes_f64<T: Copy>(x: &NdArray<T>, axis: Option<isize>, f: impl Fn(&ArrayView<T>) -> PyResult<f64>) -> PyResult<NdArray<f64>> {
    let Some(axis) = axis else {
        return Ok(scalar_nd(f(&lane_view(x.as_slice())?)?, &[]));
    };
    let ax = norm_axis(axis, x.ndim().max(1))?;
    if x.ndim() == 0 {
        return Ok(scalar_nd(f(&x.view())?, &[]));
    }
    let view = x.view();
    let owned;
    let src = match view.as_slice_c() {
        Some(s) => s,
        None => {
            owned = view.to_owned();
            owned.as_slice()
        }
    };
    let shape = x.shape();
    let n = shape[ax];
    let outer: usize = shape[..ax].iter().product();
    let inner: usize = shape[ax + 1..].iter().product();
    let mut result = Vec::with_capacity(outer * inner);
    if inner == 1 {
        for o in 0..outer {
            result.push(f(&lane_view(&src[o * n..(o + 1) * n])?)?);
        }
    } else if n == 0 {
        let empty: [T; 0] = [];
        for _ in 0..outer * inner {
            result.push(f(&lane_view(&empty)?)?);
        }
    } else {
        const BLOCK: usize = 64;
        let mut buf: Vec<T> = Vec::with_capacity(BLOCK * n);
        for o in 0..outer {
            for i0 in (0..inner).step_by(BLOCK) {
                let width = BLOCK.min(inner - i0);
                buf.clear();
                buf.resize(width * n, src[0]);
                for k in 0..n {
                    let row = &src[(o * n + k) * inner + i0..][..width];
                    for (b, &v) in row.iter().enumerate() {
                        buf[b * n + k] = v;
                    }
                }
                for lane in buf.chunks_exact(n).take(width) {
                    result.push(f(&lane_view(lane)?)?);
                }
            }
        }
    }
    let mut shape = shape.to_vec();
    shape.remove(ax);
    NdArray::from_vec(result, &shape).map_err(shape_err)
}

macro_rules! stat_fn {
    ($name:ident, $core:path $(, $extra:ident : $ty:ty = $default:expr)*) => {
        #[pyfunction]
        #[pyo3(signature = (a, axis=None $(, $extra=$default)*))]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>, $($extra: $ty),*) -> PyResult<Py<PyAny>> {
            let arr_in = stat_input(py, a)?;
            let arr: &Arr = &arr_in;
            let r = with_real!(&arr, x => lanes_f64(x, axis, |v| Ok($core(v $(, $extra)*)))?);
            float_out(py, &arr, r)
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
    let arr_in = stat_input(py, a)?;
    let arr: &Arr = &arr_in;
    let r = with_real!(&arr, x => lanes_f64(x, axis, |v| match rustnumpy::median(v) {
        Ok(v) => Ok(v),
        Err(rustnumpy::ReductionError::EmptyInput) => Ok(f64::NAN),
        Err(e) => Err(value_err(e)),
    })?);
    float_out(py, arr, r)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None))]
pub fn nanmedian(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
    let arr_in = stat_input(py, a)?;
    let arr: &Arr = &arr_in;
    let r = with_real!(&arr, x => lanes_f64(x, axis, |v| match rustnumpy::nanmedian(v) {
        Ok(v) => Ok(v),
        Err(rustnumpy::ReductionError::EmptyInput) => Ok(f64::NAN),
        Err(e) => Err(value_err(e)),
    })?);
    float_out(py, arr, r)
}

#[pyfunction]
#[pyo3(signature = (a, q, axis=None))]
pub fn percentile(py: Python<'_>, a: &Bound<'_, PyAny>, q: f64, axis: Option<isize>) -> PyResult<Py<PyAny>> {
    let arr_in = stat_input(py, a)?;
    let arr: &Arr = &arr_in;
    let r = with_real!(&arr, x => lanes_f64(x, axis, |v| match rustnumpy::percentile(v, q) {
        Ok(v) => Ok(v),
        Err(rustnumpy::ReductionError::EmptyInput) => Err(pyo3::exceptions::PyIndexError::new_err("index -1 is out of bounds for axis 0 with size 0")),
        Err(e) => Err(value_err(e)),
    })?);
    float_out(py, arr, r)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, keepdims=false))]
pub fn nansum(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>, keepdims: bool) -> PyResult<Py<PyAny>> {
    let arr_in = arr_of(py, a)?;
    let arr: &Arr = &arr_in;
    if arr.is_complex() {
        return Err(unsupported("complex nansum is composed at the Python layer"));
    }
    let wide_f16 = matches!(arr, Arr::F16(_));
    let widened;
    let arr: &Arr = if wide_f16 {
        widened = astype(arr, "float32")?;
        &widened
    } else {
        arr
    };
    let mut acc_slot = None;
    let acc = crate::casting::cast_ref(arr, accumulator_name(arr), &mut acc_slot)?;
    let result: Arr = match acc {
        Arr::I64(x) => Arr::from(nan_sum_of(x, axis, keepdims)?),
        Arr::U64(x) => Arr::from(nan_sum_of(x, axis, keepdims)?),
        Arr::F32(x) => Arr::from(nan_sum_of(x, axis, keepdims)?),
        Arr::F64(x) => Arr::from(nan_sum_of(x, axis, keepdims)?),
        _ => return Err(unsupported("unreachable nansum dtype")),
    };
    out(py, if wide_f16 { astype(&result, "float16")? } else { result })
}

fn nan_sum_of<T: FloatIsh + Default + WrapAdd>(a: &NdArray<T>, axis: Option<isize>, keepdims: bool) -> PyResult<NdArray<T>> {
    let zeroed = rustnumpy::logic::map_to(&a.view(), |v: T| if v.is_nan_ish() { T::default() } else { v });
    sum_of(&zeroed, axis, keepdims)
}

macro_rules! nan_extreme {
    ($name:ident, $core:path, $max:expr) => {
        #[pyfunction]
        #[pyo3(signature = (a, axis=None))]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
            let arr_in = arr_of(py, a)?;
            let arr: &Arr = &arr_in;
            if arr.ndim() == 0 || axis.is_none() {
                let result = with_real!(&arr, x => Arr::from(scalar_nd($core(&x.view()).map_err(value_err)?, &[])));
                return out(py, result);
            }
            let axis = axis.expect("axis is Some");
            let result = with_real!(&arr, x => Arr::from(nan_lane_extreme(x, axis, $max)?));
            out(py, result)
        }
    };
}

fn nan_lane_extreme<T: FloatIsh>(x: &NdArray<T>, axis: isize, want_max: bool) -> PyResult<NdArray<T>> {
    let ax = norm_axis(axis, x.ndim())?;
    let owned = x.view().to_owned();
    let n = owned.shape()[ax];
    let outer: usize = owned.shape()[..ax].iter().product();
    let inner: usize = owned.shape()[ax + 1..].iter().product();
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("zero-size array to reduction operation which has no identity"));
    }
    let src = owned.as_slice();
    let mut result = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let lane: Vec<T> = (0..n).map(|k| src[(o * n + k) * inner + i]).collect();
            let lane = NdArray::from_vec(lane, &[n]).map_err(shape_err)?;
            let v = if want_max { rustnumpy::nanmax(&lane.view()) } else { rustnumpy::nanmin(&lane.view()) };
            result.push(v.map_err(value_err)?);
        }
    }
    let mut shape = owned.shape().to_vec();
    shape.remove(ax);
    NdArray::from_vec(result, &shape).map_err(shape_err)
}

nan_extreme!(nanmin, rustnumpy::nanmin, false);
nan_extreme!(nanmax, rustnumpy::nanmax, true);

fn cov_like(py: Python<'_>, m: &Bound<'_, PyAny>, ddof: usize, correlation: bool, as_scalar: bool) -> PyResult<Py<PyAny>> {
    let arr_in = arr_of(py, m)?;
    let arr: &Arr = &arr_in;
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
    let arr = astype(&*arr_of(py, m)?, "float64")?;
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
    let arr = astype(&*arr_of(py, a)?, "float64")?;
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
    let arr_in = arr_of(py, a)?;
    let arr: &Arr = &arr_in;
    if arr.ndim() == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Cannot sort a 0-d array"));
    }
    let ax = norm_axis(axis, arr.ndim())?;
    fn order<K: FloatIsh>(x: &NdArray<K>, ax: usize) -> PyResult<Arr> {
        let r = rustnumpy::argsort(&x.view(), ax).map_err(shape_err)?;
        let data: Vec<i64> = r.as_slice().iter().map(|&i| i as i64).collect();
        Ok(Arr::from(NdArray::from_vec(data, r.shape()).map_err(shape_err)?))
    }
    let result = match (&arr, indices) {
        (Arr::Bool(x), false) => {
            let as_u8 = astype(&Arr::Bool(x.clone()), "uint8")?;
            let sorted = with_real!(&as_u8, y => Arr::from(rustnumpy::sort(&y.view(), ax).map_err(shape_err)?));
            astype(&sorted, "bool")?
        }
        (Arr::Bool(x), true) => order(&rustnumpy::logic::map_to(&x.view(), u8::from), ax)?,
        (Arr::C64(x), false) => {
            let k = crate::cxkey::to_keys(x);
            crate::cxkey::from_key_result(rustnumpy::sort(&k.view(), ax).map_err(shape_err)?)
        }
        (Arr::C128(x), false) => {
            let k = crate::cxkey::to_keys(x);
            crate::cxkey::from_key_result(rustnumpy::sort(&k.view(), ax).map_err(shape_err)?)
        }
        (Arr::C64(x), true) => order(&crate::cxkey::to_keys(x), ax)?,
        (Arr::C128(x), true) => order(&crate::cxkey::to_keys(x), ax)?,
        (_, true) => with_real!(&arr, x => order(x, ax)?),
        (_, false) => with_real!(&arr, x => Arr::from(rustnumpy::sort(&x.view(), ax).map_err(shape_err)?)),
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
    let (sorted, values) = common_arrs(&*arr_of(py, a)?, &*arr_of(py, v)?)?;
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
        (Arr::Bool(s), Arr::Bool(x)) => {
            let (s8, x8) = (rustnumpy::logic::map_to(&s.view(), u8::from), rustnumpy::logic::map_to(&x.view(), u8::from));
            go!(s8, x8)
        }
        (Arr::C64(s), Arr::C64(x)) => {
            let (sk, xk) = (crate::cxkey::to_keys(s), crate::cxkey::to_keys(x));
            go!(sk, xk)
        }
        (Arr::C128(s), Arr::C128(x)) => {
            let (sk, xk) = (crate::cxkey::to_keys(s), crate::cxkey::to_keys(x));
            go!(sk, xk)
        }
        (Arr::I8(s), Arr::I8(x)) => go!(s, x),
        (Arr::I16(s), Arr::I16(x)) => go!(s, x),
        (Arr::I32(s), Arr::I32(x)) => go!(s, x),
        (Arr::I64(s), Arr::I64(x)) => go!(s, x),
        (Arr::U8(s), Arr::U8(x)) => go!(s, x),
        (Arr::U16(s), Arr::U16(x)) => go!(s, x),
        (Arr::U32(s), Arr::U32(x)) => go!(s, x),
        (Arr::U64(s), Arr::U64(x)) => go!(s, x),
        (Arr::F16(s), Arr::F16(x)) => go!(s, x),
        (Arr::F32(s), Arr::F32(x)) => go!(s, x),
        (Arr::F64(s), Arr::F64(x)) => go!(s, x),
        _ => return Err(unsupported("unreachable searchsorted dtype pair")),
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
