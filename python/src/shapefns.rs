use crate::arrayfns::{ints, norm_axis};
use crate::casting::{astype, kind_name};
use crate::dynarray::{unsupported, Arr};
use crate::ops::{out, out_array, shape_err, Operand};
use crate::with_arr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rustnumpy::{ArrayView, ChooseMode, NdArray};

macro_rules! with_list {
    ($arrs:expr, $l:ident => $body:expr) => {
        match &$arrs[0] {
            Arr::Bool(_) => { let $l: Vec<&NdArray<bool>> = $arrs.iter().map(|a| match a { Arr::Bool(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::I8(_) => { let $l: Vec<&NdArray<i8>> = $arrs.iter().map(|a| match a { Arr::I8(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::I16(_) => { let $l: Vec<&NdArray<i16>> = $arrs.iter().map(|a| match a { Arr::I16(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::I32(_) => { let $l: Vec<&NdArray<i32>> = $arrs.iter().map(|a| match a { Arr::I32(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::I64(_) => { let $l: Vec<&NdArray<i64>> = $arrs.iter().map(|a| match a { Arr::I64(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::U8(_) => { let $l: Vec<&NdArray<u8>> = $arrs.iter().map(|a| match a { Arr::U8(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::U16(_) => { let $l: Vec<&NdArray<u16>> = $arrs.iter().map(|a| match a { Arr::U16(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::U32(_) => { let $l: Vec<&NdArray<u32>> = $arrs.iter().map(|a| match a { Arr::U32(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::U64(_) => { let $l: Vec<&NdArray<u64>> = $arrs.iter().map(|a| match a { Arr::U64(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::F16(_) => { let $l: Vec<&NdArray<half::f16>> = $arrs.iter().map(|a| match a { Arr::F16(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::F32(_) => { let $l: Vec<&NdArray<f32>> = $arrs.iter().map(|a| match a { Arr::F32(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::F64(_) => { let $l: Vec<&NdArray<f64>> = $arrs.iter().map(|a| match a { Arr::F64(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::C64(_) => { let $l: Vec<&NdArray<crate::dynarray::C32>> = $arrs.iter().map(|a| match a { Arr::C64(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::C128(_) => { let $l: Vec<&NdArray<crate::dynarray::C64>> = $arrs.iter().map(|a| match a { Arr::C128(v) => v, _ => unreachable!() }).collect(); $body }
        }
    };
}

macro_rules! with_list_nb {
    ($arrs:expr, $l:ident => $body:expr) => {
        match &$arrs[0] {
            Arr::Bool(_) => return Err(unsupported("boolean input is not bound for this function")),
            Arr::I8(_) => { let $l: Vec<&NdArray<i8>> = $arrs.iter().map(|a| match a { Arr::I8(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::I16(_) => { let $l: Vec<&NdArray<i16>> = $arrs.iter().map(|a| match a { Arr::I16(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::I32(_) => { let $l: Vec<&NdArray<i32>> = $arrs.iter().map(|a| match a { Arr::I32(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::I64(_) => { let $l: Vec<&NdArray<i64>> = $arrs.iter().map(|a| match a { Arr::I64(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::U8(_) => { let $l: Vec<&NdArray<u8>> = $arrs.iter().map(|a| match a { Arr::U8(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::U16(_) => { let $l: Vec<&NdArray<u16>> = $arrs.iter().map(|a| match a { Arr::U16(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::U32(_) => { let $l: Vec<&NdArray<u32>> = $arrs.iter().map(|a| match a { Arr::U32(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::U64(_) => { let $l: Vec<&NdArray<u64>> = $arrs.iter().map(|a| match a { Arr::U64(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::F16(_) => { let $l: Vec<&NdArray<half::f16>> = $arrs.iter().map(|a| match a { Arr::F16(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::F32(_) => { let $l: Vec<&NdArray<f32>> = $arrs.iter().map(|a| match a { Arr::F32(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::F64(_) => { let $l: Vec<&NdArray<f64>> = $arrs.iter().map(|a| match a { Arr::F64(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::C64(_) => { let $l: Vec<&NdArray<crate::dynarray::C32>> = $arrs.iter().map(|a| match a { Arr::C64(v) => v, _ => unreachable!() }).collect(); $body }
            Arr::C128(_) => { let $l: Vec<&NdArray<crate::dynarray::C64>> = $arrs.iter().map(|a| match a { Arr::C128(v) => v, _ => unreachable!() }).collect(); $body }
        }
    };
}

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

fn arr_of(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Arr> {
    Operand::parse(py, obj)?.into_arr(py)
}

pub fn common_all(arrs: &[Arr]) -> PyResult<Vec<Arr>> {
    let mut kind = arrs[0].kind();
    for a in &arrs[1..] {
        kind = rustnumpy::common_dtype(kind, a.kind());
    }
    let name = kind_name(kind);
    arrs.iter().map(|a| astype(a, name)).collect()
}

fn list_of(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<Arr>> {
    obj.try_iter()?.map(|item| arr_of(py, &item?)).collect()
}

fn to_list(py: Python<'_>, items: Vec<Arr>, scalar: bool) -> PyResult<Py<PyAny>> {
    let objs: Vec<Py<PyAny>> = items
        .into_iter()
        .map(|a| if scalar { out(py, a) } else { out_array(py, a) })
        .collect::<PyResult<_>>()?;
    Ok(PyList::new(py, objs)?.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (a, shift, axis=None))]
pub fn roll(py: Python<'_>, a: &Bound<'_, PyAny>, shift: &Bound<'_, PyAny>, axis: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let arr = arr_of(py, a)?;
    let shifts = ints(shift)?;
    let axes = axis.map(ints).transpose()?;
    out_array(py, with_arr!(&arr, x => Arr::from(rustnumpy::roll(&x.view(), &shifts, axes.as_deref()).map_err(shape_err)?)))
}

#[pyfunction]
#[pyo3(signature = (a, repeats, axis=None))]
pub fn repeat(py: Python<'_>, a: &Bound<'_, PyAny>, repeats: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
    let arr = arr_of(py, a)?;
    let reps: Vec<usize> = ints(repeats)?
        .into_iter()
        .map(|r| usize::try_from(r).map_err(|_| PyValueError::new_err("negative dimensions are not allowed")))
        .collect::<PyResult<_>>()?;
    let total: usize = if reps.len() == 1 { reps[0].saturating_mul(arr.shape().iter().product::<usize>().max(1)) } else { reps.iter().sum() };
    crate::createfns::alloc_guard(total, 16)?;
    out_array(py, with_arr!(&arr, x => Arr::from(rustnumpy::repeat(&x.view(), &reps, axis).map_err(shape_err)?)))
}

fn acc_name(a: &Arr) -> &'static str {
    match a {
        Arr::Bool(_) | Arr::I8(_) | Arr::I16(_) | Arr::I32(_) => "int64",
        Arr::U8(_) | Arr::U16(_) | Arr::U32(_) => "uint64",
        other => other.dtype_name(),
    }
}

#[pyfunction]
#[pyo3(signature = (a, offset=0))]
pub fn trace(py: Python<'_>, a: &Bound<'_, PyAny>, offset: isize) -> PyResult<Py<PyAny>> {
    let arr = arr_of(py, a)?;
    let acc = astype(&arr, acc_name(&arr))?;
    let result = with_arr!(&acc, x => Arr::from(NdArray::from_vec(vec![rustnumpy::trace(&x.view(), offset).map_err(shape_err)?], &[]).map_err(shape_err)?));
    out(py, result)
}

#[pyfunction]
pub fn kron(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let pair = common_all(&[arr_of(py, a)?, arr_of(py, b)?])?;
    let result = with_list!(pair, l => Arr::from(rustnumpy::kron(&l[0].view(), &l[1].view()).map_err(shape_err)?));
    out_array(py, result)
}

#[pyfunction]
pub fn cross(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let pair = common_all(&[arr_of(py, a)?, arr_of(py, b)?])?;
    let result = with_list_nb!(pair, l => Arr::from(rustnumpy::cross(&l[0].view(), &l[1].view()).map_err(shape_err)?));
    out_array(py, result)
}

macro_rules! contract2 {
    ($name:ident, $core:path) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let pair = common_all(&[arr_of(py, a)?, arr_of(py, b)?])?;
            let result = with_list!(pair, l => Arr::from($core(&l[0].view(), &l[1].view()).map_err(shape_err)?));
            out(py, result)
        }
    };
}
contract2!(matmul, rustnumpy::matmul);
contract2!(dot, rustnumpy::dot);
contract2!(outer, rustnumpy::outer);

#[pyfunction]
pub fn vecdot(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let pair = common_all(&[arr_of(py, a)?, arr_of(py, b)?])?;
    if pair[0].is_complex() {
        return Err(unsupported("vecdot conjugates its first argument for complex input; not bound"));
    }
    let result = with_list!(pair, l => Arr::from(rustnumpy::vecdot(&l[0].view(), &l[1].view()).map_err(shape_err)?));
    out(py, result)
}

#[pyfunction]
#[pyo3(signature = (a, b, axes=None))]
pub fn tensordot(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, axes: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let pair = common_all(&[arr_of(py, a)?, arr_of(py, b)?])?;
    let (nd_a, nd_b) = (pair[0].ndim(), pair[1].ndim());
    let (axes_a, axes_b): (Vec<isize>, Vec<isize>) = match axes {
        None => (((nd_a as isize - 2)..nd_a as isize).collect(), (0..2).collect()),
        Some(obj) => match obj.extract::<isize>() {
            Ok(n) => (((nd_a as isize - n)..nd_a as isize).collect(), (0..n).collect()),
            Err(_) => {
                let (x, y): (Bound<'_, PyAny>, Bound<'_, PyAny>) = obj.extract()?;
                (ints(&x)?, ints(&y)?)
            }
        },
    };
    let norm = |v: &[isize], nd: usize| -> PyResult<Vec<usize>> { v.iter().map(|&x| norm_axis(x, nd)).collect() };
    let (ua, ub) = (norm(&axes_a, nd_a)?, norm(&axes_b, nd_b)?);
    let has_dup = |v: &[usize]| v.iter().enumerate().any(|(i, x)| v[..i].contains(x));
    if has_dup(&ua) || has_dup(&ub) {
        return Err(PyValueError::new_err("duplicate axes are not allowed in tensordot"));
    }
    let result = with_list!(pair, l => Arr::from(rustnumpy::tensordot(&l[0].view(), &l[1].view(), &ua, &ub).map_err(shape_err)?));
    out(py, result)
}

#[pyfunction]
#[pyo3(signature = (subscripts, *operands))]
pub fn einsum(py: Python<'_>, subscripts: &str, operands: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Py<PyAny>> {
    let arrs: Vec<Arr> = operands.iter().map(|o| arr_of(py, &o)).collect::<PyResult<_>>()?;
    if arrs.is_empty() {
        return Err(PyValueError::new_err("No input operands"));
    }
    let arrs = common_all(&arrs)?;
    let result = with_list!(arrs, l => {
        let views: Vec<ArrayView<_>> = l.iter().map(|x| x.view()).collect();
        let refs: Vec<&ArrayView<_>> = views.iter().collect();
        Arr::from(rustnumpy::einsum(subscripts, &refs).map_err(shape_err)?)
    });
    out(py, result)
}

#[pyfunction]
pub fn where_(py: Python<'_>, cond: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let c = astype(&arr_of(py, cond)?, "bool")?;
    let Arr::Bool(mask) = c else { unreachable!("cast to bool") };
    let (px, py_) = crate::ops::resolve_binary(py, x, y)?;
    let pair = [px, py_];
    let result = with_list!(pair, l => Arr::from(rustnumpy::where_cond(&mask.view(), &l[0].view(), &l[1].view()).map_err(shape_err)?));
    out_array(py, result)
}

#[pyfunction]
#[pyo3(signature = (condlist, choicelist, default=None))]
pub fn select(py: Python<'_>, condlist: &Bound<'_, PyAny>, choicelist: &Bound<'_, PyAny>, default: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let cond_arrs = list_of(py, condlist)?;
    let mut conds: Vec<NdArray<bool>> = Vec::new();
    for (i, c) in cond_arrs.into_iter().enumerate() {
        match c {
            Arr::Bool(m) => conds.push(m),
            _ => {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "invalid entry {i} in condlist: should be boolean ndarray"
                )))
            }
        }
    }
    let choices = common_all(&list_of(py, choicelist)?)?;
    let default_operand = match default {
        Some(d) => Operand::parse(py, d)?,
        None => Operand::WeakInt(0),
    };
    let (target, dflt) = match default_operand {
        Operand::WeakInt(v) => {
            let name = if choices[0].is_bool() { "int64" } else { choices[0].dtype_name() };
            (name, astype(&Arr::scalar(v), name)?)
        }
        Operand::WeakFloat(v) => {
            let name = if choices[0].is_bool() || choices[0].is_int() { "float64" } else { choices[0].dtype_name() };
            (name, astype(&Arr::scalar(v), name)?)
        }
        Operand::WeakComplex(re, im) => {
            let name = if choices[0].is_complex() { choices[0].dtype_name() } else { "complex128" };
            (name, astype(&Arr::scalar(crate::dynarray::C64::new(re, im)), name)?)
        }
        Operand::Arr(a) => {
            let mut both = choices.iter().map(|c| astype(c, c.dtype_name())).collect::<PyResult<Vec<_>>>()?;
            both.push(a);
            let both = common_all(&both)?;
            let name = both[0].dtype_name();
            (name, astype(both.last().expect("pushed above"), name)?)
        }
    };
    let choices: Vec<Arr> = choices.iter().map(|c| astype(c, target)).collect::<PyResult<_>>()?;
    let mut all = choices;
    if dflt.ndim() != 0 {
        conds.push(NdArray::from_vec(vec![true], &[]).map_err(shape_err)?);
        all.push(astype(&dflt, target)?);
    }
    all.push(dflt);
    let n = all.len() - 1;
    let cviews: Vec<ArrayView<bool>> = conds.iter().map(|c| c.view()).collect();
    let crefs: Vec<&ArrayView<bool>> = cviews.iter().collect();
    let result = with_list!(all, l => {
        let default_value = l[l.len() - 1].as_slice()[0];
        let views: Vec<ArrayView<_>> = l[..n].iter().map(|x| x.view()).collect();
        let refs: Vec<&ArrayView<_>> = views.iter().collect();
        Arr::from(rustnumpy::select(&crefs, &refs, default_value).map_err(shape_err)?)
    });
    out_array(py, result)
}

#[pyfunction]
#[pyo3(signature = (a, choices, mode="raise"))]
pub fn choose(py: Python<'_>, a: &Bound<'_, PyAny>, choices: &Bound<'_, PyAny>, mode: &str) -> PyResult<Py<PyAny>> {
    let mode = match mode {
        "raise" => ChooseMode::Raise,
        "wrap" => ChooseMode::Wrap,
        "clip" => ChooseMode::Clip,
        other => return Err(PyValueError::new_err(format!("clipmode must be one of 'clip', 'raise', or 'wrap' (got '{other}')"))),
    };
    let idx = astype(&arr_of(py, a)?, "int64")?;
    let Arr::I64(idx) = idx else { unreachable!("cast to int64") };
    let chs = common_all(&list_of(py, choices)?)?;
    let result = with_list!(chs, l => {
        let views: Vec<ArrayView<_>> = l.iter().map(|x| x.view()).collect();
        let refs: Vec<&ArrayView<_>> = views.iter().collect();
        Arr::from(rustnumpy::choose(&idx.view(), &refs, mode).map_err(shape_err)?)
    });
    out_array(py, result)
}

#[pyfunction]
#[pyo3(signature = (ary, indices_or_sections, axis=0))]
pub fn array_split(py: Python<'_>, ary: &Bound<'_, PyAny>, indices_or_sections: &Bound<'_, PyAny>, axis: isize) -> PyResult<Py<PyAny>> {
    let arr = arr_of(py, ary)?;
    let ax = norm_axis(axis, arr.ndim())?;
    let parts: Vec<Arr> = match indices_or_sections.extract::<usize>() {
        Ok(n) => with_arr!(&arr, x => rustnumpy::array_split(x, n, ax).map_err(shape_err)?.into_iter().map(Arr::from).collect()),
        Err(_) => {
            let idx: Vec<usize> = indices_or_sections.extract()?;
            with_arr!(&arr, x => rustnumpy::array_split_at(x, &idx, ax).map_err(shape_err)?.into_iter().map(Arr::from).collect())
        }
    };
    to_list(py, parts, false)
}

fn i64_arr(v: Vec<usize>) -> PyResult<Arr> {
    let n = v.len();
    Ok(Arr::from(NdArray::from_vec(v.into_iter().map(|i| i as i64).collect::<Vec<_>>(), &[n]).map_err(shape_err)?))
}

fn unique_of<K: rustnumpy::reductions::FloatIsh>(
    x: &NdArray<K>,
    back: impl Fn(Vec<K>) -> PyResult<Arr>,
) -> PyResult<(Arr, Vec<usize>, Arr, Vec<usize>)> {
    let u = rustnumpy::unique_all(&x.view());
    let inv = &u.inverse_indices;
    let inverse = Arr::from(NdArray::from_vec(inv.as_slice().iter().map(|&i| i as i64).collect::<Vec<_>>(), inv.shape()).map_err(shape_err)?);
    Ok((back(u.values)?, u.indices, inverse, u.counts))
}

fn unique_parts(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<(Arr, Vec<usize>, Arr, Vec<usize>)> {
    let arr = arr_of(py, a)?;
    fn flat<T>(values: Vec<T>) -> PyResult<Arr>
    where
        Arr: From<NdArray<T>>,
    {
        let n = values.len();
        Ok(Arr::from(NdArray::from_vec(values, &[n]).map_err(shape_err)?))
    }
    match &arr {
        Arr::Bool(x) => {
            let (values, i, inv, c) = unique_of(&rustnumpy::logic::map_to(&x.view(), u8::from), flat)?;
            Ok((astype(&values, "bool")?, i, inv, c))
        }
        Arr::C64(x) => unique_of(&crate::cxkey::to_keys(x), |v| flat(v.into_iter().map(|k| k.0).collect())),
        Arr::C128(x) => unique_of(&crate::cxkey::to_keys(x), |v| flat(v.into_iter().map(|k| k.0).collect())),
        _ => with_real!(&arr, x => unique_of(x, flat)),
    }
}

fn named<'py>(py: Python<'py>, name: &str, fields: &[&str], values: Vec<Py<PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let nt = py.import("collections")?.call_method1("namedtuple", (name, fields.to_vec()))?;
    nt.call1(pyo3::types::PyTuple::new(py, values)?)
}

#[pyfunction]
pub fn unique_all(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (values, indices, inverse, counts) = unique_parts(py, a)?;
    let r = named(
        py,
        "UniqueAllResult",
        &["values", "indices", "inverse_indices", "counts"],
        vec![out_array(py, values)?, out_array(py, i64_arr(indices)?)?, out_array(py, inverse)?, out_array(py, i64_arr(counts)?)?],
    )?;
    Ok(r.unbind())
}

#[pyfunction]
pub fn unique_counts(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (values, _, _, counts) = unique_parts(py, a)?;
    Ok(named(py, "UniqueCountsResult", &["values", "counts"], vec![out_array(py, values)?, out_array(py, i64_arr(counts)?)?])?.unbind())
}

#[pyfunction]
pub fn unique_inverse(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (values, _, inverse, _) = unique_parts(py, a)?;
    Ok(named(py, "UniqueInverseResult", &["values", "inverse_indices"], vec![out_array(py, values)?, out_array(py, inverse)?])?.unbind())
}

#[pyfunction]
pub fn unique_values(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (values, _, _, _) = unique_parts(py, a)?;
    out_array(py, values)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! reg {
        ($($f:ident),* $(,)?) => {$( m.add_function(wrap_pyfunction!($f, m)?)?; )*};
    }
    reg!(
        roll, repeat, trace, kron, cross, matmul, dot, outer, vecdot, tensordot, einsum, select, choose,
        array_split, unique_all, unique_counts, unique_inverse, unique_values
    );
    m.add("where", wrap_pyfunction!(where_, m)?)?;
    Ok(())
}

