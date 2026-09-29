use crate::dynarray::{unsupported, Arr};
use crate::ops::{out_array, shape_err};
use crate::pyarray::{wrap, PyArray};
use crate::with_arr;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};
use rustnumpy::{AxisIndex, NdArray};
use std::sync::Arc;

pub fn broadcast_arr(v: &Arr, shape: &[usize]) -> PyResult<Arr> {
    with_arr!(v, a => {
        let view = a.view().broadcast_to(shape).map_err(|_| {
            PyValueError::new_err(format!("could not broadcast input array from shape {:?} into shape {:?}", a.shape(), shape))
        })?;
        Ok(Arr::from(view.to_owned()))
    })
}

enum Item {
    Int(isize),
    Slice { start: isize, step: isize, len: usize },
    RawSlice(Py<PySlice>),
    Full,
    NewAxis,
    Ellipsis,
    Index(Arr),
    Mask(Arr),
}

fn parse(py: Python<'_>, index: &Bound<'_, PyAny>, shape: &[usize]) -> PyResult<Vec<Item>> {
    let raw: Vec<Bound<'_, PyAny>> = match index.downcast::<PyTuple>() {
        Ok(t) => t.iter().collect(),
        Err(_) => vec![index.clone()],
    };
    let mut items = Vec::with_capacity(raw.len());
    for it in &raw {
        if it.is_none() {
            items.push(Item::NewAxis);
            continue;
        }
        if it.is(py.Ellipsis()) {
            items.push(Item::Ellipsis);
            continue;
        }
        if let Ok(s) = it.downcast::<PySlice>() {
            items.push(Item::RawSlice(s.clone().unbind()));
            continue;
        }
        if it.is_exact_instance_of::<pyo3::types::PyBool>() {
            return Err(unsupported("boolean scalar indices are not supported"));
        }
        if it.is_exact_instance_of::<pyo3::types::PyInt>() {
            items.push(Item::Int(it.extract::<isize>()?));
            continue;
        }
        let arr = Arr::from_object(py, it)?;
        if arr.is_bool() {
            items.push(Item::Mask(arr));
        } else if arr.is_int() {
            if arr.ndim() == 0 {
                let v = crate::casting::astype(&arr, "int64")?;
                let Arr::I64(x) = v else { unreachable!("cast to int64") };
                items.push(Item::Int(x.as_slice()[0] as isize));
            } else {
                items.push(Item::Index(arr));
            }
        } else {
            return Err(PyIndexError::new_err("arrays used as indices must be of integer (or boolean) type"));
        }
    }
    let consumed: usize = items
        .iter()
        .map(|i| match i {
            Item::Int(_) | Item::RawSlice(_) | Item::Slice { .. } | Item::Index(_) => 1,
            Item::Mask(m) => m.ndim(),
            _ => 0,
        })
        .sum();
    let nd = shape.len();
    if consumed > nd {
        return Err(PyIndexError::new_err(format!("too many indices for array: array is {nd}-dimensional, but {consumed} were indexed")));
    }
    let ellipses = items.iter().filter(|i| matches!(i, Item::Ellipsis)).count();
    if ellipses > 1 {
        return Err(PyIndexError::new_err("an index can only have a single ellipsis ('...')"));
    }
    let fill = nd - consumed;
    let mut expanded = Vec::new();
    let mut placed = false;
    for it in items {
        if matches!(it, Item::Ellipsis) {
            expanded.extend((0..fill).map(|_| Item::Full));
            placed = true;
        } else {
            expanded.push(it);
        }
    }
    if !placed {
        expanded.extend((0..fill).map(|_| Item::Full));
    }
    let mut axis = 0usize;
    for it in expanded.iter_mut() {
        match it {
            Item::RawSlice(raw) => {
                let ind = raw.bind(py).indices(shape[axis] as isize)?;
                *it = Item::Slice { start: ind.start, step: ind.step, len: ind.slicelength };
                axis += 1;
            }
            Item::Int(_) | Item::Slice { .. } | Item::Index(_) | Item::Full => axis += 1,
            Item::Mask(m) => axis += m.ndim(),
            Item::NewAxis | Item::Ellipsis => {}
        }
    }
    Ok(expanded)
}

fn wrap_index(i: isize, dim: usize, axis: usize) -> PyResult<usize> {
    let n = dim as isize;
    let j = if i < 0 { i + n } else { i };
    if j < 0 || j >= n {
        return Err(PyIndexError::new_err(format!("index {i} is out of bounds for axis {axis} with size {dim}")));
    }
    Ok(j as usize)
}

pub enum Applied {
    View(PyArray),
    Owned(Arr),
    Element(usize),
}

impl PyArray {
    pub fn apply_index(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<Applied> {
        let items = parse(py, index, &self.shape)?;
        let advanced = items.iter().any(|i| matches!(i, Item::Index(_) | Item::Mask(_)));
        if !advanced {
            return self.apply_basic(&items);
        }
        if items.iter().any(|i| matches!(i, Item::NewAxis)) {
            return Err(unsupported("newaxis combined with array indices is not supported"));
        }
        self.apply_advanced(py, items)
    }

    fn apply_basic(&self, items: &[Item]) -> PyResult<Applied> {
        let mut shape = Vec::new();
        let mut strides = Vec::new();
        let mut offset = self.offset as isize;
        let mut axis = 0usize;
        let mut all_int = true;
        for it in items {
            match it {
                Item::Int(i) => {
                    let j = wrap_index(*i, self.shape[axis], axis)?;
                    offset += j as isize * self.strides[axis];
                    axis += 1;
                }
                Item::Slice { start, step, len } => {
                    all_int = false;
                    if *len > 0 {
                        offset += start * self.strides[axis];
                    }
                    shape.push(*len);
                    strides.push(self.strides[axis] * step);
                    axis += 1;
                }
                Item::Full => {
                    all_int = false;
                    shape.push(self.shape[axis]);
                    strides.push(self.strides[axis]);
                    axis += 1;
                }
                Item::NewAxis => {
                    all_int = false;
                    shape.push(1);
                    strides.push(0);
                }
                Item::RawSlice(_) | Item::Ellipsis | Item::Index(_) | Item::Mask(_) => unreachable!("resolved or advanced items are handled elsewhere"),
            }
        }
        if all_int {
            return Ok(Applied::Element(offset as usize));
        }
        Ok(Applied::View(PyArray { storage: Arc::clone(&self.storage), shape, strides, offset: offset as usize }))
    }

    fn apply_advanced(&self, py: Python<'_>, items: Vec<Item>) -> PyResult<Applied> {
        if let [Item::Mask(m), rest @ ..] = items.as_slice() {
            if m.ndim() > 1 && rest.iter().all(|i| matches!(i, Item::Full)) {
                let Arr::Bool(mask) = m else { unreachable!("mask arrays are bool") };
                let base = self.to_arr();
                let picked = with_arr!(&base, a => Arr::from(a.boolean_index_nd(mask).map_err(mask_err)?));
                return Ok(Applied::Owned(picked));
            }
            if m.ndim() > 1 {
                return Err(unsupported("a multi-dimensional boolean mask combined with other indices is not supported"));
            }
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        let mut offset = self.offset as isize;
        let mut spec_kinds: Vec<Option<Vec<usize>>> = Vec::new();
        let mut singles: Vec<Option<usize>> = Vec::new();
        let mut index_shapes: Vec<Vec<usize>> = Vec::new();
        let mut axis = 0usize;
        for it in &items {
            match it {
                Item::Slice { start, step, len } => {
                    if *len > 0 {
                        offset += start * strides[axis];
                    }
                    shape[axis] = *len;
                    strides[axis] *= step;
                    spec_kinds.push(None);
                    singles.push(None);
                    axis += 1;
                }
                Item::Full => {
                    spec_kinds.push(None);
                    singles.push(None);
                    axis += 1;
                }
                Item::Int(i) => {
                    singles.push(Some(wrap_index(*i, shape[axis], axis)?));
                    spec_kinds.push(None);
                    axis += 1;
                }
                Item::Index(a) => {
                    let Arr::I64(v) = crate::casting::astype(a, "int64")? else { unreachable!("cast to int64") };
                    let wrapped: Vec<usize> = v.as_slice().iter().map(|&i| wrap_index(i as isize, shape[axis], axis)).collect::<PyResult<_>>()?;
                    index_shapes.push(v.shape().to_vec());
                    spec_kinds.push(Some(wrapped));
                    singles.push(None);
                    axis += 1;
                }
                Item::Mask(m) => {
                    let Arr::Bool(b) = m else { unreachable!("mask arrays are bool") };
                    if b.len() != shape[axis] {
                        return Err(PyIndexError::new_err(format!(
                            "boolean index did not match indexed array along axis {axis}; size of axis is {} but size of corresponding boolean axis is {}",
                            shape[axis],
                            b.len()
                        )));
                    }
                    let idx: Vec<usize> = b.as_slice().iter().enumerate().filter_map(|(i, &k)| k.then_some(i)).collect();
                    index_shapes.push(vec![idx.len()]);
                    spec_kinds.push(Some(idx));
                    singles.push(None);
                    axis += 1;
                }
                Item::NewAxis | Item::Ellipsis | Item::RawSlice(_) => unreachable!("expanded earlier"),
            }
        }
        let mut broadcast: Vec<usize> = Vec::new();
        for s in &index_shapes {
            broadcast = rustnumpy::shape::broadcast_shapes(&broadcast, s).ok_or_else(|| {
                PyIndexError::new_err(format!("shape mismatch: indexing arrays could not be broadcast together with shapes {:?}", index_shapes))
            })?;
        }
        let total: usize = broadcast.iter().product();
        let mut spec: Vec<AxisIndex> = Vec::with_capacity(items.len());
        let mut kind_iter = index_shapes.iter();
        for (k, it) in items.iter().enumerate() {
            spec.push(match it {
                Item::Slice { .. } | Item::Full => AxisIndex::Full,
                Item::Int(_) => AxisIndex::Single(singles[k].expect("int index resolved above")),
                Item::Index(_) | Item::Mask(_) => {
                    let values = spec_kinds[k].as_ref().expect("array index resolved above");
                    let ishape = kind_iter.next().expect("one shape per array index");
                    let arr = NdArray::from_vec(values.clone(), ishape).map_err(shape_err)?;
                    let flat = arr.view().broadcast_to(&broadcast).map_err(shape_err)?.to_owned();
                    debug_assert_eq!(flat.len(), total);
                    AxisIndex::Fancy(flat.into_vec())
                }
                Item::NewAxis | Item::Ellipsis | Item::RawSlice(_) => unreachable!("expanded earlier"),
            });
        }
        let base_view = PyArray { storage: Arc::clone(&self.storage), shape, strides, offset: offset as usize };
        let base = base_view.to_arr();
        let picked = with_arr!(&base, a => Arr::from(a.vindex(&spec).map_err(shape_err)?));
        let advanced_axes: Vec<usize> = spec
            .iter()
            .enumerate()
            .filter_map(|(i, s)| matches!(s, AxisIndex::Fancy(_) | AxisIndex::Single(_)).then_some(i))
            .collect();
        let adjacent = advanced_axes.windows(2).all(|w| w[1] == w[0] + 1);
        let merge_pos = if adjacent { spec[..advanced_axes[0]].iter().filter(|s| matches!(s, AxisIndex::Full)).count() } else { 0 };
        let _ = py;
        if broadcast.len() == 1 {
            return Ok(Applied::Owned(picked));
        }
        let mut target = picked.shape()[..merge_pos].to_vec();
        target.extend(&broadcast);
        target.extend(&picked.shape()[merge_pos + 1..]);
        let dims: Vec<isize> = target.iter().map(|&d| d as isize).collect();
        Ok(Applied::Owned(with_arr!(&picked, a => Arr::from(a.clone().into_shape(&dims).map_err(shape_err)?))))
    }
}

fn int_bounds(dtype: &str) -> Option<(i128, i128)> {
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

fn mask_err(e: rustnumpy::ShapeError) -> PyErr {
    match e {
        rustnumpy::ShapeError::BooleanAxisMismatch { axis, expected, got } => PyIndexError::new_err(format!(
            "boolean index did not match indexed array along axis {axis}; size of axis is {expected} but size of corresponding boolean axis is {got}"
        )),
        other => shape_err(other),
    }
}

#[pymethods]
impl PyArray {
    fn __getitem__(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match self.apply_index(py, index)? {
            Applied::View(v) => wrap(py, v),
            Applied::Owned(a) => out_array(py, a),
            Applied::Element(pos) => self.read_position(py, pos),
        }
    }

    fn __setitem__(&self, py: Python<'_>, index: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let positions = PyArray::from_arr(Arr::I64(
            NdArray::from_vec(self.flat_positions().into_iter().map(|p| p as i64).collect(), &self.shape).map_err(shape_err)?,
        ));
        let (targets, shape): (Vec<usize>, Vec<usize>) = match positions.apply_index(py, index)? {
            Applied::Element(pos) => {
                let p = match positions.storage.arr() {
                    Arr::I64(a) => a.as_slice()[pos] as usize,
                    _ => unreachable!("positions are int64"),
                };
                (vec![p], vec![])
            }
            Applied::View(v) => match v.to_arr() {
                Arr::I64(a) => (a.as_slice().iter().map(|&p| p as usize).collect(), a.shape().to_vec()),
                _ => unreachable!("positions are int64"),
            },
            Applied::Owned(Arr::I64(a)) => (a.as_slice().iter().map(|&p| p as usize).collect(), a.shape().to_vec()),
            Applied::Owned(_) => unreachable!("positions are int64"),
        };
        if value.is_exact_instance_of::<pyo3::types::PyInt>() && !matches!(self.storage.arr(), Arr::F32(_) | Arr::F64(_) | Arr::C64(_) | Arr::C128(_) | Arr::Bool(_)) {
            if let Ok(v) = value.extract::<i128>() {
                if let Some((lo, hi)) = int_bounds(self.dtype_name()) {
                    if v < lo || v > hi {
                        return Err(pyo3::exceptions::PyOverflowError::new_err(format!("Python integer {v} out of bounds for {}", self.dtype_name())));
                    }
                }
            }
        }
        let val = Arr::from_object(py, value)?;
        let shaped = broadcast_arr(&val, &shape)?;
        self.write_positions(&targets, &shaped)
    }
}
