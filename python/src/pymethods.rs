use crate::dynarray::Arr;
use crate::ops::out_array;
use crate::pyarray::PyArray;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

fn lib_call<'py>(
    slf: &Bound<'py, PyArray>,
    module: &str,
    name: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    let f = py.import(module)?.getattr(name)?;
    let mut all: Vec<Bound<'py, PyAny>> = vec![slf.clone().into_any()];
    all.extend(args.iter());
    Ok(f.call(PyTuple::new(py, all)?, kwargs)?.unbind())
}

macro_rules! delegate {
    ($module:literal; $($method:ident => $func:literal),* $(,)?) => {
        #[pymethods]
        impl PyArray {
            $(
                #[pyo3(signature = (*args, **kwargs))]
                fn $method<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
                    lib_call(slf, $module, $func, args, kwargs)
                }
            )*
        }
    };
}

delegate!("rustnumpy";
    sum => "sum", prod => "prod", max => "max", min => "min", mean => "mean", var => "var", std => "std",
    argmax => "argmax", argmin => "argmin", any => "any", all => "all", cumsum => "cumsum", cumprod => "cumprod",
    clip => "clip", argsort => "argsort", dot => "dot", trace => "trace", diagonal => "diagonal", repeat => "repeat",
    take => "take", put => "put", nonzero => "nonzero", compress => "_m_compress", choose => "choose", round => "round",
    conj => "conjugate", conjugate => "conjugate", searchsorted => "searchsorted", partition => "_m_partition",
    argpartition => "argpartition", sort => "_m_sort", resize => "_m_resize", view => "_m_view", ptp => "ptp",
    byteswap => "_m_byteswap", tofile => "_m_tofile", dump => "_m_dump", dumps => "_m_dumps", to_device => "_m_to_device",
    setflags => "_m_setflags", getfield => "_m_unsupported", setfield => "_m_unsupported", tostring => "_m_tobytes_alias",
    ravel_order => "ravel"
);

#[pymethods]
impl PyArray {
    #[getter]
    fn flags(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let this = slf.borrow();
        let c = this.is_c_contiguous();
        let f = rustnumpy::shape::is_c_contiguous_layout(
            &this.shape.iter().rev().copied().collect::<Vec<_>>(),
            &this.strides.iter().rev().copied().collect::<Vec<_>>(),
        );
        let owns = this.owns_storage();
        let flags = py.import("rustnumpy._misc")?.getattr("Flags")?;
        Ok(flags.call1((c, f, owns))?.unbind())
    }

    #[getter]
    fn base(slf: &Bound<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let this = slf.borrow();
        if this.owns_storage() {
            return Ok(None);
        }
        let whole = this.whole_storage_view();
        Ok(Some(Bound::new(py, whole)?.into_any().unbind()))
    }

    #[getter(mT)]
    fn matrix_t(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        py.import("rustnumpy")?.getattr("matrix_transpose")?.call1((slf,)).map(Bound::unbind)
    }

    #[getter]
    fn flat(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        py.import("rustnumpy._misc")?.getattr("flatiter")?.call1((slf,)).map(Bound::unbind)
    }

    #[getter]
    fn data(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        py.import("builtins")?.getattr("memoryview")?.call1((slf,)).map(Bound::unbind)
    }

    #[getter]
    fn device(&self) -> &'static str {
        "cpu"
    }

    #[pyo3(signature = (order=None))]
    fn tobytes(&self, py: Python<'_>, order: Option<&str>) -> PyResult<Py<PyAny>> {
        let _ = order;
        let arr = self.to_arr();
        Ok(pyo3::types::PyBytes::new(py, &arr.to_bytes()).into_any().unbind())
    }

    fn __contains__(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        let eq = slf.as_any().eq(value);
        drop(eq);
        let lib = py.import("rustnumpy")?;
        let mask = lib.getattr("equal")?.call1((slf, value))?;
        lib.getattr("any")?.call1((mask,))?.is_truthy()
    }

    fn __divmod__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match crate::umath::divmod(slf.py(), slf.as_any(), other) {
            Ok((q, r)) => Ok(PyTuple::new(slf.py(), [q, r])?.into_any().unbind()),
            Err(e) if e.is_instance_of::<PyTypeError>(slf.py()) => Ok(slf.py().NotImplemented()),
            Err(e) => Err(e),
        }
    }

    fn __rdivmod__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match crate::umath::divmod(slf.py(), other, slf.as_any()) {
            Ok((q, r)) => Ok(PyTuple::new(slf.py(), [q, r])?.into_any().unbind()),
            Err(e) if e.is_instance_of::<PyTypeError>(slf.py()) => Ok(slf.py().NotImplemented()),
            Err(e) => Err(e),
        }
    }

    fn __imatmul__(slf: PyRefMut<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = other.py();
        let alias = Bound::new(py, slf.alias())?;
        let result = crate::shapefns::matmul(py, alias.as_any(), other)?;
        let value = Arr::from_object(py, result.bind(py))?;
        let shaped = crate::pyindex::broadcast_arr(&value, &slf.shape)?;
        slf.write_positions(&slf.flat_positions(), &shaped)
    }

    #[pyo3(signature = (*, stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__(slf: &Bound<'_, Self>, stream: Option<&Bound<'_, PyAny>>, max_version: Option<&Bound<'_, PyAny>>, dl_device: Option<&Bound<'_, PyAny>>, copy: Option<bool>) -> PyResult<Py<PyAny>> {
        let _ = (stream, max_version, dl_device);
        if copy == Some(true) {
            let copied = crate::ops::out_array(slf.py(), slf.borrow().to_arr())?;
            return crate::dlpack::export(copied.bind(slf.py()).downcast::<PyArray>()?);
        }
        crate::dlpack::export(slf)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (1, 0)
    }

    fn __copy__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        out_array(py, self.to_arr())
    }

    #[pyo3(signature = (memo=None))]
    fn __deepcopy__(&self, py: Python<'_>, memo: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        let _ = memo;
        out_array(py, self.to_arr())
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let arr = self.to_arr();
        let reconstruct = py.import("rustnumpy._io")?.getattr("_reconstruct")?;
        let payload = pyo3::types::PyBytes::new(py, &arr.to_bytes());
        let args = (crate::pyarray::shape_tuple(py, &self.shape)?, self.dtype_name(), payload);
        Ok((reconstruct, args).into_pyobject(py)?.into_any().unbind())
    }

    #[pyo3(signature = (*, api_version=None))]
    fn __array_namespace__(&self, py: Python<'_>, api_version: Option<&str>) -> PyResult<Py<PyAny>> {
        let _ = api_version;
        Ok(py.import("rustnumpy")?.into_any().unbind())
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, pyo3::types::PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().into_any().unbind()
    }
}
