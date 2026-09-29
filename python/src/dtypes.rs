use crate::dynarray::unsupported;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyComplex, PyFloat, PyInt, PyString, PyType};

pub const NAMES: [&str; 13] = [
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64", "complex64",
    "complex128",
];

#[pyclass(name = "dtype", module = "rustnumpy", frozen)]
#[derive(Clone)]
pub struct PyDtype {
    pub name: &'static str,
}

#[pymethods]
impl PyDtype {
    #[new]
    fn new(spec: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self { name: parse_dtype(spec)? })
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.name
    }

    #[getter]
    fn itemsize(&self) -> usize {
        itemsize(self.name)
    }

    #[getter]
    fn kind(&self) -> &'static str {
        kind_char(self.name)
    }

    #[getter]
    fn char(&self) -> &'static str {
        type_char(self.name)
    }

    #[getter]
    fn str(&self) -> &'static str {
        typestr(self.name)
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: pyo3::basic::CompareOp) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let same = match parse_dtype(other) {
            Ok(n) => n == self.name,
            Err(_) => return Ok(py.NotImplemented()),
        };
        match op {
            pyo3::basic::CompareOp::Eq => Ok(pyo3::types::PyBool::new(py, same).to_owned().into_any().unbind()),
            pyo3::basic::CompareOp::Ne => Ok(pyo3::types::PyBool::new(py, !same).to_owned().into_any().unbind()),
            _ => Ok(py.NotImplemented()),
        }
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.name.hash(&mut h);
        h.finish()
    }

    fn __repr__(&self) -> String {
        format!("dtype('{}')", self.name)
    }

    fn __str__(&self) -> &'static str {
        self.name
    }
}

pub fn itemsize(name: &str) -> usize {
    match name {
        "bool" | "int8" | "uint8" => 1,
        "int16" | "uint16" => 2,
        "int32" | "uint32" | "float32" => 4,
        "int64" | "uint64" | "float64" | "complex64" => 8,
        _ => 16,
    }
}

pub fn kind_char(name: &str) -> &'static str {
    match name {
        "bool" => "b",
        n if n.starts_with("uint") => "u",
        n if n.starts_with("int") => "i",
        n if n.starts_with("float") => "f",
        _ => "c",
    }
}

pub fn type_char(name: &str) -> &'static str {
    match name {
        "bool" => "?",
        "int8" => "b",
        "int16" => "h",
        "int32" => "i",
        "int64" => "q",
        "uint8" => "B",
        "uint16" => "H",
        "uint32" => "I",
        "uint64" => "Q",
        "float32" => "f",
        "float64" => "d",
        "complex64" => "F",
        _ => "D",
    }
}

pub fn typestr(name: &str) -> &'static str {
    match name {
        "bool" => "|b1",
        "int8" => "|i1",
        "int16" => "<i2",
        "int32" => "<i4",
        "int64" => "<i8",
        "uint8" => "|u1",
        "uint16" => "<u2",
        "uint32" => "<u4",
        "uint64" => "<u8",
        "float32" => "<f4",
        "float64" => "<f8",
        "complex64" => "<c8",
        _ => "<c16",
    }
}

pub fn buffer_format(name: &str) -> &'static [u8] {
    match name {
        "bool" => b"?\0",
        "int8" => b"b\0",
        "int16" => b"h\0",
        "int32" => b"i\0",
        "int64" => b"q\0",
        "uint8" => b"B\0",
        "uint16" => b"H\0",
        "uint32" => b"I\0",
        "uint64" => b"Q\0",
        "float32" => b"f\0",
        "float64" => b"d\0",
        "complex64" => b"Zf\0",
        _ => b"Zd\0",
    }
}

fn from_text(text: &str) -> Option<&'static str> {
    let t = text.trim_start_matches(['<', '=', '|', '@']);
    Some(match t {
        "bool" | "bool_" | "?" | "b1" => "bool",
        "int8" | "i1" | "b" | "byte" => "int8",
        "int16" | "i2" | "h" | "short" => "int16",
        "int32" | "i4" | "i" | "intc" => "int32",
        "int64" | "i8" | "q" | "l" | "int" | "int_" | "intp" | "long" | "longlong" => "int64",
        "uint8" | "u1" | "B" | "ubyte" => "uint8",
        "uint16" | "u2" | "H" | "ushort" => "uint16",
        "uint32" | "u4" | "I" | "uintc" => "uint32",
        "uint64" | "u8" | "Q" | "L" | "uint" | "uintp" | "ulonglong" | "ulong" => "uint64",
        "float32" | "f4" | "f" | "single" => "float32",
        "float64" | "f8" | "d" | "float" | "double" | "float_" => "float64",
        "complex64" | "c8" | "F" | "csingle" => "complex64",
        "complex128" | "c16" | "D" | "complex" | "cdouble" | "complex_" => "complex128",
        _ => return None,
    })
}

pub fn parse_dtype(obj: &Bound<'_, PyAny>) -> PyResult<&'static str> {
    if let Ok(d) = obj.extract::<PyRef<'_, PyDtype>>() {
        return Ok(d.name);
    }
    if let Ok(s) = obj.downcast::<PyString>() {
        let text = s.to_str()?;
        return from_text(text).ok_or_else(|| PyTypeError::new_err(format!("data type '{text}' not understood")));
    }
    if let Ok(t) = obj.downcast::<PyType>() {
        let py = obj.py();
        if t.is(py.get_type::<PyBool>()) {
            return Ok("bool");
        }
        if t.is(py.get_type::<PyInt>()) {
            return Ok("int64");
        }
        if t.is(py.get_type::<PyFloat>()) {
            return Ok("float64");
        }
        if t.is(py.get_type::<PyComplex>()) {
            return Ok("complex128");
        }
        if let Ok(n) = t.getattr("__name__").and_then(|n| n.extract::<String>()) {
            if let Some(found) = from_text(&n) {
                return Ok(found);
            }
        }
    }
    if let Ok(name) = obj.getattr("name").and_then(|n| n.extract::<String>()) {
        if let Some(n) = from_text(&name) {
            return Ok(n);
        }
    }
    Err(unsupported(format!("cannot interpret {} as a supported dtype", obj.repr()?)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDtype>()?;
    for name in NAMES {
        m.add(name, PyDtype { name }.into_pyobject(m.py())?)?;
    }
    m.add("bool_", PyDtype { name: "bool" }.into_pyobject(m.py())?)?;
    m.add("intp", PyDtype { name: "int64" }.into_pyobject(m.py())?)?;
    m.add("double", PyDtype { name: "float64" }.into_pyobject(m.py())?)?;
    m.add("single", PyDtype { name: "float32" }.into_pyobject(m.py())?)?;
    Ok(())
}
