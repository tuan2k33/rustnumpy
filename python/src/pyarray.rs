use crate::dtypes::{buffer_format, itemsize, parse_dtype, typestr, PyDtype};
use crate::dynarray::Arr;
use crate::ops::shape_err;
use crate::with_arr;
use pyo3::exceptions::{PyBufferError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rustnumpy::shape::{c_contiguous_strides, is_c_contiguous_layout};
use rustnumpy::{ArrayView, NdArray, ShapeError};
use std::cell::UnsafeCell;
use std::os::raw::{c_int, c_void};
use std::sync::Arc;

pub struct Storage {
    cell: UnsafeCell<Arr>,
}

// SAFETY: every access to the storage happens either under the GIL (all pymethods and module functions) or
// through raw pointers handed to consumers of the buffer protocol / __array_interface__, which carry NumPy's
// own contract that the exporter must not be mutated while an export is being read. A free-threaded build
// would need a lock here; this crate targets the GIL build.
unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

impl Storage {
    pub fn new(a: Arr) -> Self {
        Self { cell: UnsafeCell::new(a) }
    }

    pub fn arr(&self) -> &Arr {
        // SAFETY: see the comment on `unsafe impl Sync`.
        unsafe { &*self.cell.get() }
    }

    #[allow(clippy::mut_from_ref)]
    pub fn arr_mut(&self) -> &mut Arr {
        // SAFETY: see the comment on `unsafe impl Sync`; callers hold no other reference across this call.
        unsafe { &mut *self.cell.get() }
    }

    pub fn len(&self) -> usize {
        with_arr!(self.arr(), a => a.len())
    }

    pub fn base_ptr(&self) -> *mut u8 {
        with_arr!(self.arr(), a => a.as_slice().as_ptr() as *mut u8)
    }
}

#[pyclass(name = "ndarray", module = "rustnumpy")]
pub struct PyArray {
    pub storage: Arc<Storage>,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub offset: usize,
}

impl PyArray {
    pub fn from_arr(a: Arr) -> PyArray {
        let shape = a.shape();
        let strides = c_contiguous_strides(&shape);
        PyArray { storage: Arc::new(Storage::new(a)), shape, strides, offset: 0 }
    }

    pub fn dtype_name(&self) -> &'static str {
        self.storage.arr().dtype_name()
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_c_contiguous(&self) -> bool {
        is_c_contiguous_layout(&self.shape, &self.strides)
    }

    pub fn to_arr(&self) -> Arr {
        with_arr!(self.storage.arr(), a => {
            let view = ArrayView::from_raw_parts(a.as_slice(), self.shape.clone(), self.strides.clone(), self.offset)
                .expect("PyArray keeps its shape/strides/offset inside the storage");
            Arr::from(view.to_owned())
        })
    }

    pub fn derive(&self, f: impl for<'a> FnOnce(&ArrayView<'a, ()>) -> Result<ArrayView<'a, ()>, ShapeError>) -> PyResult<PyArray> {
        let units = vec![(); self.storage.len()];
        let view = ArrayView::from_raw_parts(&units, self.shape.clone(), self.strides.clone(), self.offset).map_err(shape_err)?;
        let next = f(&view).map_err(shape_err)?;
        Ok(PyArray {
            storage: Arc::clone(&self.storage),
            shape: next.shape().to_vec(),
            strides: next.strides().to_vec(),
            offset: next.offset(),
        })
    }

    pub fn flat_positions(&self) -> Vec<usize> {
        rustnumpy::shape::IndexIter::new(&self.shape)
            .map(|idx| (self.offset as isize + idx.iter().zip(&self.strides).map(|(&i, &s)| i as isize * s).sum::<isize>()) as usize)
            .collect()
    }

    pub fn read_position(&self, py: Python<'_>, pos: usize) -> PyResult<Py<PyAny>> {
        let a = with_arr!(self.storage.arr(), s => Arr::scalar(s.as_slice()[pos]));
        crate::ops::out(py, a)
    }

    pub fn write_positions(&self, positions: &[usize], values: &Arr) -> PyResult<()> {
        let mut src = crate::casting::astype(values, self.dtype_name())?;
        if src.shape() != [positions.len()] {
            let n = positions.len() as isize;
            src = with_arr!(&src, a => Arr::from(a.clone().into_shape(&[n]).map_err(shape_err)?));
        }
        macro_rules! copy {
            ($d:expr, $s:expr) => {{
                let data = $d.as_mut_slice();
                for (&p, &v) in positions.iter().zip($s.as_slice()) {
                    data[p] = v;
                }
            }};
        }
        match (self.storage.arr_mut(), &src) {
            (Arr::Bool(d), Arr::Bool(s)) => copy!(d, s),
            (Arr::I8(d), Arr::I8(s)) => copy!(d, s),
            (Arr::I16(d), Arr::I16(s)) => copy!(d, s),
            (Arr::I32(d), Arr::I32(s)) => copy!(d, s),
            (Arr::I64(d), Arr::I64(s)) => copy!(d, s),
            (Arr::U8(d), Arr::U8(s)) => copy!(d, s),
            (Arr::U16(d), Arr::U16(s)) => copy!(d, s),
            (Arr::U32(d), Arr::U32(s)) => copy!(d, s),
            (Arr::U64(d), Arr::U64(s)) => copy!(d, s),
            (Arr::F32(d), Arr::F32(s)) => copy!(d, s),
            (Arr::F64(d), Arr::F64(s)) => copy!(d, s),
            (Arr::C64(d), Arr::C64(s)) => copy!(d, s),
            (Arr::C128(d), Arr::C128(s)) => copy!(d, s),
            _ => return Err(PyTypeError::new_err("internal error: dtype mismatch while assigning")),
        }
        Ok(())
    }
}

pub fn wrap(py: Python<'_>, a: PyArray) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, a)?.into_any())
}

pub fn as_array(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    if let Ok(a) = obj.downcast::<PyArray>() {
        let a = a.borrow();
        return Ok(PyArray { storage: Arc::clone(&a.storage), shape: a.shape.clone(), strides: a.strides.clone(), offset: a.offset });
    }
    Ok(PyArray::from_arr(Arr::from_object(py, obj)?))
}

fn shape_tuple<'py>(py: Python<'py>, shape: &[usize]) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, shape.iter().copied())
}

struct BufferAux {
    _shape: Box<[isize]>,
    _strides: Box<[isize]>,
}

#[pymethods]
impl PyArray {
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        shape_tuple(py, &self.shape)
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.shape.len()
    }

    #[getter(size)]
    fn size_py(&self) -> usize {
        self.size()
    }

    #[getter]
    fn dtype(&self) -> PyDtype {
        PyDtype { name: self.dtype_name() }
    }

    #[getter]
    fn itemsize(&self) -> usize {
        itemsize(self.dtype_name())
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.size() * itemsize(self.dtype_name())
    }

    #[getter(strides)]
    fn strides_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let sz = itemsize(self.dtype_name()) as isize;
        PyTuple::new(py, self.strides.iter().map(|s| s * sz))
    }

    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let sz = itemsize(self.dtype_name());
        let ptr = self.storage.base_ptr() as usize + self.offset * sz;
        d.set_item("version", 3)?;
        d.set_item("shape", shape_tuple(py, &self.shape)?)?;
        d.set_item("typestr", typestr(self.dtype_name()))?;
        d.set_item("data", (ptr, false))?;
        d.set_item("strides", PyTuple::new(py, self.strides.iter().map(|s| s * sz as isize))?)?;
        Ok(d)
    }

    unsafe fn __getbuffer__(slf: Bound<'_, Self>, view: *mut ffi::Py_buffer, flags: c_int) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("view is null"));
        }
        let this = slf.borrow();
        let name = this.dtype_name();
        let sz = itemsize(name);
        let wants_strides = flags & ffi::PyBUF_STRIDES == ffi::PyBUF_STRIDES;
        if !wants_strides && !this.is_c_contiguous() {
            return Err(PyBufferError::new_err("ndarray is not C-contiguous"));
        }
        let shape: Box<[isize]> = this.shape.iter().map(|&d| d as isize).collect();
        let strides: Box<[isize]> = this.strides.iter().map(|&s| s * sz as isize).collect();
        let aux = Box::new(BufferAux { _shape: shape, _strides: strides });
        let ptr = this.storage.base_ptr().add(this.offset * sz);
        // SAFETY: `view` is a valid, writable Py_buffer supplied by CPython; every pointer stored in it stays
        // valid until `__releasebuffer__` because the storage is kept alive by `obj` and the boxed shape and
        // strides arrays are owned by `internal`.
        unsafe {
            (*view).buf = ptr as *mut c_void;
            (*view).len = (this.size() * sz) as isize;
            (*view).readonly = 0;
            (*view).itemsize = sz as isize;
            (*view).format = if flags & ffi::PyBUF_FORMAT == ffi::PyBUF_FORMAT { buffer_format(name).as_ptr() as *mut _ } else { std::ptr::null_mut() };
            (*view).ndim = this.shape.len() as c_int;
            (*view).shape = if flags & ffi::PyBUF_ND == ffi::PyBUF_ND { aux._shape.as_ptr() as *mut _ } else { std::ptr::null_mut() };
            (*view).strides = if wants_strides { aux._strides.as_ptr() as *mut _ } else { std::ptr::null_mut() };
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = Box::into_raw(aux) as *mut c_void;
            (*view).obj = slf.clone().into_ptr();
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        // SAFETY: `internal` was produced by `Box::into_raw` in `__getbuffer__` and is released exactly once.
        unsafe { drop(Box::from_raw((*view).internal as *mut BufferAux)) };
    }

    fn __len__(&self) -> PyResult<usize> {
        self.shape.first().copied().ok_or_else(|| PyTypeError::new_err("len() of unsized object"))
    }

    fn __bool__(&self) -> PyResult<bool> {
        if self.size() != 1 {
            return Err(PyValueError::new_err("The truth value of an array with more than one element is ambiguous. Use any() or all()"));
        }
        let pos = self.flat_positions()[0];
        Ok(with_arr!(self.storage.arr(), a => rustnumpy::logic::Truthy::truthy(a.as_slice()[pos])))
    }

    fn item(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.size() != 1 {
            return Err(PyValueError::new_err("can only convert an array of size 1 to a Python scalar"));
        }
        let pos = self.flat_positions()[0];
        let a = with_arr!(self.storage.arr(), s => Arr::scalar(s.as_slice()[pos]));
        a.to_native_scalar(py)
    }

    fn __float__(&self, py: Python<'_>) -> PyResult<f64> {
        self.item(py)?.bind(py).extract::<f64>().or_else(|_| self.item(py)?.bind(py).call_method0("__float__")?.extract())
    }

    fn __int__(&self, py: Python<'_>) -> PyResult<i64> {
        let v = self.item(py)?;
        v.bind(py).call_method0("__int__")?.extract()
    }

    fn __index__(&self, py: Python<'_>) -> PyResult<i64> {
        if !matches!(self.storage.arr(), Arr::I8(_) | Arr::I16(_) | Arr::I32(_) | Arr::I64(_) | Arr::U8(_) | Arr::U16(_) | Arr::U32(_) | Arr::U64(_)) || self.size() != 1 {
            return Err(PyTypeError::new_err("only integer scalar arrays can be converted to a scalar index"));
        }
        self.__int__(py)
    }

    fn __complex__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let v = self.item(py)?;
        Ok(py.get_type::<pyo3::types::PyComplex>().call1((v,))?.unbind())
    }

    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        fn build(this: &PyArray, py: Python<'_>, axis: usize, pos: isize) -> PyResult<Py<PyAny>> {
            if axis == this.shape.len() {
                let a = with_arr!(this.storage.arr(), s => Arr::scalar(s.as_slice()[pos as usize]));
                return a.to_native_scalar(py);
            }
            let mut items = Vec::with_capacity(this.shape[axis]);
            for i in 0..this.shape[axis] {
                items.push(build(this, py, axis + 1, pos + i as isize * this.strides[axis])?);
            }
            Ok(PyList::new(py, items)?.into_any().unbind())
        }
        build(self, py, 0, self.offset as isize)
    }

    fn copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        wrap(py, PyArray::from_arr(self.to_arr()))
    }

    #[pyo3(signature = (dtype, copy=true))]
    fn astype(&self, py: Python<'_>, dtype: &Bound<'_, PyAny>, copy: bool) -> PyResult<Py<PyAny>> {
        let name = parse_dtype(dtype)?;
        if !copy && name == self.dtype_name() {
            return wrap(py, PyArray { storage: Arc::clone(&self.storage), shape: self.shape.clone(), strides: self.strides.clone(), offset: self.offset });
        }
        wrap(py, PyArray::from_arr(crate::casting::astype(&self.to_arr(), name)?))
    }

    #[pyo3(signature = (*shape))]
    fn reshape(&self, py: Python<'_>, shape: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
        let dims: Vec<isize> = if shape.len() == 1 { crate::arrayfns::ints(&shape.get_item(0)?)? } else { shape.extract()? };
        wrap(py, self.reshaped(&dims)?)
    }

    fn ravel(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        wrap(py, self.reshaped(&[-1])?)
    }

    fn flatten(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        wrap(py, PyArray::from_arr(self.reshaped(&[-1])?.to_arr()))
    }

    #[getter(T)]
    fn t(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        wrap(py, self.derive(|v| Ok(v.transpose()))?)
    }

    #[pyo3(signature = (*axes))]
    fn transpose(&self, py: Python<'_>, axes: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
        let list: Vec<isize> = if axes.is_empty() {
            return self.t(py);
        } else if axes.len() == 1 {
            crate::arrayfns::ints(&axes.get_item(0)?)?
        } else {
            axes.extract()?
        };
        wrap(py, self.derive(|v| v.permute_dims(&list))?)
    }

    #[pyo3(signature = (axis=None))]
    fn squeeze(&self, py: Python<'_>, axis: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        let axes = axis.map(crate::arrayfns::ints).transpose()?;
        wrap(py, self.derive(|v| v.squeeze(axes.as_deref()))?)
    }

    fn swapaxes(&self, py: Python<'_>, axis1: isize, axis2: isize) -> PyResult<Py<PyAny>> {
        let n = self.shape.len();
        let a1 = crate::arrayfns::norm_axis(axis1, n)?;
        let a2 = crate::arrayfns::norm_axis(axis2, n)?;
        wrap(py, self.derive(|v| v.swap_axes(a1, a2))?)
    }

    fn fill(&self, fill_value: &Bound<'_, PyAny>) -> PyResult<()> {
        let v = Arr::from_object(fill_value.py(), fill_value)?;
        self.write_positions(&self.flat_positions(), &broadcast_to_len(v, self.size())?)
    }

    #[getter]
    fn real(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::ops::real(py, &Bound::new(py, PyArray { storage: Arc::clone(&self.storage), shape: self.shape.clone(), strides: self.strides.clone(), offset: self.offset })?.into_any())
    }

    #[getter]
    fn imag(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::ops::imag(py, &Bound::new(py, PyArray { storage: Arc::clone(&self.storage), shape: self.shape.clone(), strides: self.strides.clone(), offset: self.offset })?.into_any())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(crate::repr::array_repr(self))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(crate::repr::array_str(self))
    }

    fn __iter__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let this = slf.borrow();
        if this.shape.is_empty() {
            return Err(PyTypeError::new_err("iteration over a 0-d array"));
        }
        let mut items: Vec<Py<PyAny>> = Vec::with_capacity(this.shape[0]);
        for i in 0..this.shape[0] {
            items.push(this.get_axis0(py, i)?);
        }
        Ok(PyList::new(py, items)?.call_method0("__iter__")?.unbind())
    }
}

fn broadcast_to_len(v: Arr, n: usize) -> PyResult<Arr> {
    if v.shape().iter().product::<usize>() == n {
        return Ok(v);
    }
    if v.shape().iter().product::<usize>() == 1 {
        let data = with_arr!(&v, a => Arr::from(NdArray::from_vec(vec![a.as_slice()[0]; n], &[n]).map_err(shape_err)?));
        return Ok(data);
    }
    Err(PyValueError::new_err("could not broadcast input array to the target shape"))
}

impl PyArray {
    pub fn reshaped(&self, dims: &[isize]) -> PyResult<PyArray> {
        let resolved = rustnumpy::shape::resolve_reshape(self.size(), dims).map_err(shape_err)?;
        if self.is_c_contiguous() {
            return Ok(PyArray {
                storage: Arc::clone(&self.storage),
                strides: c_contiguous_strides(&resolved),
                shape: resolved,
                offset: self.offset,
            });
        }
        let copy = with_arr!(&self.to_arr(), a => Arr::from(a.clone().into_shape(dims).map_err(shape_err)?));
        Ok(PyArray::from_arr(copy))
    }

    pub fn get_axis0(&self, py: Python<'_>, i: usize) -> PyResult<Py<PyAny>> {
        if self.shape.len() == 1 {
            let pos = (self.offset as isize + i as isize * self.strides[0]) as usize;
            return self.read_position(py, pos);
        }
        wrap(py, self.derive(|v| v.index_axis(0, i))?)
    }
}
