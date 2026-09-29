use crate::dynarray::Arr;
use crate::dtypes::itemsize;
use crate::pyarray::PyArray;
use pyo3::exceptions::{PyBufferError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use std::ffi::{c_char, c_void};

#[repr(C)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

struct Holder {
    _owner: Py<PyArray>,
    _shape: Vec<i64>,
    _strides: Vec<i64>,
}

const CODE_INT: u8 = 0;
const CODE_UINT: u8 = 1;
const CODE_FLOAT: u8 = 2;
const CODE_COMPLEX: u8 = 5;
const CODE_BOOL: u8 = 6;

fn dl_dtype(name: &str) -> (u8, u8) {
    let bytes = itemsize(name) as u8;
    match name {
        "bool" => (CODE_BOOL, 8),
        n if n.starts_with("uint") => (CODE_UINT, bytes * 8),
        n if n.starts_with("int") => (CODE_INT, bytes * 8),
        n if n.starts_with("float") => (CODE_FLOAT, bytes * 8),
        _ => (CODE_COMPLEX, bytes * 8),
    }
}

unsafe extern "C" fn delete_managed(t: *mut DLManagedTensor) {
    // SAFETY: `t` was produced by `Box::into_raw` in `export` and the deleter is invoked exactly once by the
    // consumer (or by the capsule destructor when nobody consumed the capsule); `manager_ctx` owns a `Holder`.
    unsafe {
        let boxed = Box::from_raw(t);
        drop(Box::from_raw(boxed.manager_ctx as *mut Holder));
    }
}

unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    // SAFETY: called by CPython with the capsule being destroyed; an unconsumed capsule keeps the name "dltensor".
    unsafe {
        let name = c"dltensor".as_ptr();
        if ffi::PyCapsule_IsValid(capsule, name) == 1 {
            let t = ffi::PyCapsule_GetPointer(capsule, name) as *mut DLManagedTensor;
            if !t.is_null() {
                if let Some(d) = (*t).deleter {
                    d(t);
                }
            }
        }
    }
}

pub fn export(slf: &Bound<'_, PyArray>) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    let this = slf.borrow();
    let (code, bits) = dl_dtype(this.dtype_name());
    let sz = itemsize(this.dtype_name());
    let shape: Vec<i64> = this.shape.iter().map(|&d| d as i64).collect();
    let strides: Vec<i64> = this.strides.iter().map(|&s| s as i64).collect();
    let holder = Box::new(Holder { _owner: slf.clone().unbind(), _shape: shape, _strides: strides });
    let shape_ptr = holder._shape.as_ptr() as *mut i64;
    let strides_ptr = holder._strides.as_ptr() as *mut i64;
    let data = this.storage.base_ptr();
    let managed = Box::new(DLManagedTensor {
        dl_tensor: DLTensor {
            data: data as *mut c_void,
            device: DLDevice { device_type: 1, device_id: 0 },
            ndim: this.shape.len() as i32,
            dtype: DLDataType { code, bits, lanes: 1 },
            shape: shape_ptr,
            strides: strides_ptr,
            byte_offset: (this.offset * sz) as u64,
        },
        manager_ctx: Box::into_raw(holder) as *mut c_void,
        deleter: Some(delete_managed),
    });
    let raw = Box::into_raw(managed);
    // SAFETY: `raw` is a valid, heap-allocated DLManagedTensor; ownership passes to the capsule whose destructor
    // (or the consumer's call to `deleter`) releases it exactly once.
    unsafe {
        let capsule = ffi::PyCapsule_New(raw as *mut c_void, c"dltensor".as_ptr(), Some(capsule_destructor));
        if capsule.is_null() {
            delete_managed(raw);
            return Err(PyErr::fetch(py));
        }
        Ok(Bound::from_owned_ptr(py, capsule).unbind())
    }
}

#[pyfunction]
#[pyo3(signature = (x, /, *, device=None, copy=None))]
pub fn from_dlpack(py: Python<'_>, x: &Bound<'_, PyAny>, device: Option<&Bound<'_, PyAny>>, copy: Option<bool>) -> PyResult<Py<PyAny>> {
    let _ = (device, copy);
    let capsule_obj = if x.hasattr("__dlpack__")? {
        x.call_method0("__dlpack__")?
    } else if x.is_instance_of::<PyCapsule>() {
        x.clone()
    } else {
        return Err(PyTypeError::new_err("from_dlpack: the object must implement __dlpack__"));
    };
    let capsule_ptr = capsule_obj.as_ptr();
    let name: *const c_char = c"dltensor".as_ptr();
    // SAFETY: the capsule was produced by a DLPack exporter; while the GIL is held we read the tensor, copy the
    // elements, call the exporter's deleter once and mark the capsule as consumed by renaming it.
    let arr = unsafe {
        if ffi::PyCapsule_IsValid(capsule_ptr, name) != 1 {
            return Err(PyValueError::new_err("from_dlpack: invalid or already consumed DLPack capsule"));
        }
        let t = ffi::PyCapsule_GetPointer(capsule_ptr, name) as *mut DLManagedTensor;
        let tensor = &(*t).dl_tensor;
        if tensor.device.device_type != 1 && tensor.device.device_type != 3 {
            return Err(PyBufferError::new_err("from_dlpack: only CPU tensors are supported"));
        }
        if tensor.dtype.lanes != 1 {
            return Err(PyBufferError::new_err("from_dlpack: vector lanes are not supported"));
        }
        let dtype = match (tensor.dtype.code, tensor.dtype.bits) {
            (CODE_BOOL, _) => "bool",
            (CODE_INT, 8) => "int8",
            (CODE_INT, 16) => "int16",
            (CODE_INT, 32) => "int32",
            (CODE_INT, 64) => "int64",
            (CODE_UINT, 8) => "uint8",
            (CODE_UINT, 16) => "uint16",
            (CODE_UINT, 32) => "uint32",
            (CODE_UINT, 64) => "uint64",
            (CODE_FLOAT, 16) => "float16",
            (CODE_FLOAT, 32) => "float32",
            (CODE_FLOAT, 64) => "float64",
            (CODE_COMPLEX, 64) => "complex64",
            (CODE_COMPLEX, 128) => "complex128",
            (c, b) => return Err(PyBufferError::new_err(format!("from_dlpack: unsupported dtype (code {c}, bits {b})"))),
        };
        let nd = tensor.ndim as usize;
        let shape: Vec<usize> = (0..nd).map(|i| *tensor.shape.add(i) as usize).collect();
        let sz = itemsize(dtype);
        let strides: Vec<isize> = if tensor.strides.is_null() {
            let mut st = vec![0isize; nd];
            let mut acc = 1isize;
            for i in (0..nd).rev() {
                st[i] = acc;
                acc *= shape[i] as isize;
            }
            st
        } else {
            (0..nd).map(|i| *tensor.strides.add(i) as isize).collect()
        };
        let base = (tensor.data as *const u8).add(tensor.byte_offset as usize);
        let count: usize = shape.iter().product();
        let mut bytes = Vec::with_capacity(count * sz);
        for idx in rustnumpy::shape::IndexIter::new(&shape) {
            let off: isize = idx.iter().zip(&strides).map(|(&i, &s)| i as isize * s).sum::<isize>() * sz as isize;
            bytes.extend_from_slice(std::slice::from_raw_parts(base.offset(off), sz));
        }
        let arr = Arr::from_bytes(dtype, &shape, &bytes)?;
        if let Some(d) = (*t).deleter {
            d(t);
        }
        ffi::PyCapsule_SetName(capsule_ptr, c"used_dltensor".as_ptr());
        ffi::PyCapsule_SetDestructor(capsule_ptr, None);
        arr
    };
    crate::ops::out_array(py, arr)
}

