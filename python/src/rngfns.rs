use crate::casting::astype;
use crate::dynarray::{unsupported, Arr};
use crate::ops::{out, out_array, shape_err};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustnumpy::{MvnMethod, NdArray};

fn rand_err(e: rustnumpy::RandomError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn size_of(size: Option<&Bound<'_, PyAny>>) -> PyResult<(Vec<usize>, bool)> {
    match size {
        None => Ok((vec![], true)),
        Some(o) => {
            if let Ok(n) = o.extract::<usize>() {
                return Ok((vec![n], false));
            }
            Ok((o.extract::<Vec<usize>>()?, false))
        }
    }
}

fn emit(py: Python<'_>, a: NdArray<f64>, scalar: bool) -> PyResult<Py<PyAny>> {
    if scalar { out(py, Arr::from(a)) } else { out_array(py, Arr::from(a)) }
}

#[pyclass(name = "SeedSequence")]
pub struct PySeedSequence {
    inner: rustnumpy::SeedSequence,
}

#[pymethods]
impl PySeedSequence {
    #[new]
    fn new(entropy: u64) -> Self {
        Self { inner: rustnumpy::SeedSequence::new(entropy) }
    }

    #[getter]
    fn spawn_key(&self) -> Vec<u32> {
        self.inner.spawn_key().to_vec()
    }

    #[pyo3(signature = (n_words, dtype="uint32"))]
    fn generate_state(&self, py: Python<'_>, n_words: usize, dtype: &str) -> PyResult<Py<PyAny>> {
        let arr = match dtype {
            "uint64" => Arr::from(NdArray::from_vec(self.inner.generate_state_u64(n_words), &[n_words]).map_err(shape_err)?),
            "uint32" => Arr::from(NdArray::from_vec(self.inner.generate_state_u32(n_words), &[n_words]).map_err(shape_err)?),
            other => return Err(unsupported(format!("dtype {other} is not supported"))),
        };
        out_array(py, arr)
    }

    fn spawn(&mut self, n: usize) -> Vec<PySeedSequence> {
        self.inner.spawn(n).into_iter().map(|inner| PySeedSequence { inner }).collect()
    }
}

#[pyclass(name = "Generator")]
pub struct PyGenerator {
    inner: rustnumpy::Generator,
}

#[pymethods]
impl PyGenerator {
    #[new]
    fn new(seed: u64) -> Self {
        Self { inner: rustnumpy::default_rng(seed) }
    }

    #[staticmethod]
    fn from_seed_sequence(seq: &PySeedSequence) -> Self {
        Self { inner: rustnumpy::Generator::from_seed_sequence(seq.inner.clone()) }
    }

    fn random_raw(&mut self) -> u64 {
        self.inner.random_raw()
    }

    fn spawn(&mut self, n: usize) -> Vec<PyGenerator> {
        self.inner.spawn(n).into_iter().map(|inner| PyGenerator { inner }).collect()
    }

    #[pyo3(signature = (size=None))]
    fn random(&mut self, py: Python<'_>, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        let (shape, scalar) = size_of(size)?;
        emit(py, self.inner.random(&shape), scalar)
    }

    #[pyo3(signature = (low=0.0, high=1.0, size=None))]
    fn uniform(&mut self, py: Python<'_>, low: f64, high: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        let (shape, scalar) = size_of(size)?;
        emit(py, self.inner.uniform(low, high, &shape).map_err(rand_err)?, scalar)
    }

    #[pyo3(signature = (low, high=None, size=None, dtype="int64", endpoint=false))]
    fn integers(&mut self, py: Python<'_>, low: i64, high: Option<i64>, size: Option<&Bound<'_, PyAny>>, dtype: &str, endpoint: bool) -> PyResult<Py<PyAny>> {
        if dtype != "int64" {
            return Err(unsupported("only dtype='int64' is bit-compatible with NumPy's bounded-integer algorithm"));
        }
        let (lo, hi) = match high {
            Some(h) => (low, h),
            None => (0, low),
        };
        let hi = if endpoint { hi + 1 } else { hi };
        let (shape, scalar) = size_of(size)?;
        let a = self.inner.integers(lo, hi, &shape).map_err(rand_err)?;
        let Arr::I64(r) = astype(&Arr::from(a), "int64")? else { unreachable!("cast to int64") };
        if scalar { out(py, Arr::from(r)) } else { out_array(py, Arr::from(r)) }
    }

    #[pyo3(signature = (size=None))]
    fn standard_normal(&mut self, py: Python<'_>, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        let (shape, scalar) = size_of(size)?;
        emit(py, self.inner.standard_normal(&shape), scalar)
    }

    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn normal(&mut self, py: Python<'_>, loc: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        let (shape, scalar) = size_of(size)?;
        emit(py, self.inner.normal(loc, scale, &shape).map_err(rand_err)?, scalar)
    }

    #[pyo3(signature = (scale=1.0, size=None))]
    fn exponential(&mut self, py: Python<'_>, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        let (shape, scalar) = size_of(size)?;
        emit(py, self.inner.exponential(scale, &shape).map_err(rand_err)?, scalar)
    }

    #[pyo3(signature = (shape, scale=1.0, size=None))]
    fn gamma(&mut self, py: Python<'_>, shape: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        let (dims, scalar) = size_of(size)?;
        emit(py, self.inner.gamma(shape, scale, &dims).map_err(rand_err)?, scalar)
    }

    #[pyo3(signature = (a, b, size=None))]
    fn beta(&mut self, py: Python<'_>, a: f64, b: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        let (shape, scalar) = size_of(size)?;
        emit(py, self.inner.beta(a, b, &shape).map_err(rand_err)?, scalar)
    }

    #[pyo3(signature = (n, p, size=None))]
    fn binomial(&mut self, py: Python<'_>, n: u64, p: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        let (shape, scalar) = size_of(size)?;
        let a = self.inner.binomial(n, p, &shape).map_err(rand_err)?;
        let r = astype(&Arr::from(a), "int64")?;
        if scalar { out(py, r) } else { out_array(py, r) }
    }

    #[pyo3(signature = (lam=1.0, size=None))]
    fn poisson(&mut self, py: Python<'_>, lam: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        let (shape, scalar) = size_of(size)?;
        let a = self.inner.poisson(lam, &shape).map_err(rand_err)?;
        let r = astype(&Arr::from(a), "int64")?;
        if scalar { out(py, r) } else { out_array(py, r) }
    }

    #[pyo3(signature = (alpha, size=None))]
    fn dirichlet(&mut self, py: Python<'_>, alpha: Vec<f64>, size: Option<usize>) -> PyResult<Py<PyAny>> {
        let n = size.unwrap_or(1);
        let a = self.inner.dirichlet(&alpha, n).map_err(rand_err)?;
        if size.is_none() {
            let k = alpha.len();
            return out_array(py, Arr::from(a.into_shape(&[k as isize]).map_err(shape_err)?));
        }
        out_array(py, Arr::from(a))
    }

    fn shuffle(&mut self, x: &Bound<'_, PyAny>) -> PyResult<()> {
        let Ok(a) = x.downcast::<crate::pyarray::PyArray>() else {
            return Err(pyo3::exceptions::PyTypeError::new_err("shuffle needs a rustnumpy.ndarray (it shuffles in place)"));
        };
        let this = a.borrow();
        if this.shape.is_empty() {
            return Err(PyValueError::new_err("x must be an array"));
        }
        let mut arr = this.to_arr();
        crate::with_arr!(&mut arr, m => self.inner.shuffle_rows(m));
        this.write_positions(&this.flat_positions(), &arr)
    }

    fn permutation(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(n) = x.extract::<usize>() {
            let p: Vec<i64> = self.inner.permutation(n).into_iter().map(|i| i as i64).collect();
            return out_array(py, Arr::from(NdArray::from_vec(p, &[n]).map_err(shape_err)?));
        }
        let mut arr = Arr::from_object(py, x)?;
        if arr.ndim() == 0 {
            return Err(PyValueError::new_err("x must be an integer or at least 1-dimensional"));
        }
        crate::with_arr!(&mut arr, a => self.inner.shuffle_rows(a));
        out_array(py, arr)
    }

    #[pyo3(signature = (a, size=None, replace=true, p=None))]
    fn choice(&mut self, py: Python<'_>, a: &Bound<'_, PyAny>, size: Option<&Bound<'_, PyAny>>, replace: bool, p: Option<Vec<f64>>) -> PyResult<Py<PyAny>> {
        let (shape, scalar) = size_of(size)?;
        let count: usize = shape.iter().product();
        let (source, pop): (Option<Arr>, usize) = match a.extract::<usize>() {
            Ok(n) => (None, n),
            Err(_) => {
                let arr = Arr::from_object(py, a)?;
                if arr.ndim() != 1 {
                    return Err(unsupported("choice over an N-D array is not bound"));
                }
                let n = arr.shape()[0];
                (Some(arr), n)
            }
        };
        let idx = self.inner.choice_indices(pop, count, replace, p.as_deref()).map_err(rand_err)?;
        let result = match source {
            None => {
                let data: Vec<i64> = idx.into_iter().map(|i| i as i64).collect();
                Arr::from(NdArray::from_vec(data, &shape).map_err(shape_err)?)
            }
            Some(arr) => crate::with_arr!(&arr, x => {
                let data: Vec<_> = idx.iter().map(|&i| x.as_slice()[i]).collect();
                Arr::from(NdArray::from_vec(data, &shape).map_err(shape_err)?)
            }),
        };
        if scalar { out(py, result) } else { out_array(py, result) }
    }

    #[pyo3(signature = (mean, cov, size=None, check_valid="warn", method="svd"))]
    fn multivariate_normal(&mut self, py: Python<'_>, mean: Vec<f64>, cov: &Bound<'_, PyAny>, size: Option<usize>, check_valid: &str, method: &str) -> PyResult<Py<PyAny>> {
        let m = match method {
            "svd" => MvnMethod::Svd,
            "eigh" => MvnMethod::Eigh,
            "cholesky" => MvnMethod::Cholesky,
            other => return Err(PyValueError::new_err(format!("mode must be one of svd, eigh, cholesky, not {other}"))),
        };
        let Arr::F64(c) = astype(&Arr::from_object(py, cov)?, "float64")? else { unreachable!("cast to float64") };
        let n = size.unwrap_or(1);
        let x = self.inner.multivariate_normal(&mean, &c, n, m, check_valid != "ignore").map_err(rand_err)?;
        if size.is_none() {
            let k = mean.len();
            return out_array(py, Arr::from(x.into_shape(&[k as isize]).map_err(shape_err)?));
        }
        out_array(py, Arr::from(x))
    }
}

#[pyfunction]
pub fn default_rng(seed: u64) -> PyGenerator {
    PyGenerator { inner: rustnumpy::default_rng(seed) }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGenerator>()?;
    m.add_class::<PySeedSequence>()?;
    m.add_function(wrap_pyfunction!(default_rng, m)?)?;
    Ok(())
}
