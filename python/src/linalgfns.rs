use crate::arrayfns::norm_axis;
use crate::casting::astype;
use crate::dynarray::{unsupported, Arr};
use crate::ops::{out, out_array, shape_err, Operand};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustnumpy::linalg::{self, MatNormOrd, VecNormOrd};
use rustnumpy::{Complex64, NdArray};

pyo3::create_exception!(rustnumpy_python, LinAlgError, PyValueError);

fn la_err(e: linalg::LinalgError) -> PyErr {
    if matches!(e, linalg::LinalgError::Empty) {
        return unsupported("empty matrix input is not bound");
    }
    LinAlgError::new_err(e.to_string())
}

struct F64Input {
    a: NdArray<f64>,
    single: bool,
}

fn f64_input_any(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<F64Input> {
    let arr = Operand::parse(py, obj)?.into_arr(py)?;
    if arr.is_complex() {
        return Err(unsupported("complex linalg is not bound"));
    }
    let single = matches!(arr, Arr::F32(_));
    let Arr::F64(a) = astype(&arr, "float64")? else { unreachable!("cast to float64") };
    Ok(F64Input { a, single })
}

fn f64_input(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<F64Input> {
    let i = f64_input_any(py, obj)?;
    if i.a.ndim() != 2 {
        return Err(unsupported("stacked or non-2-D linalg input is not bound"));
    }
    Ok(i)
}

fn f64_rhs(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<F64Input> {
    let i = f64_input_any(py, obj)?;
    if !(1..=2).contains(&i.a.ndim()) {
        return Err(unsupported("right-hand side must be 1-D or 2-D"));
    }
    Ok(i)
}

fn ret(py: Python<'_>, a: NdArray<f64>, single: bool) -> PyResult<Py<PyAny>> {
    if single { out_array(py, astype(&Arr::from(a), "float32")?) } else { out_array(py, Arr::from(a)) }
}

fn ret_scalar(py: Python<'_>, v: f64, single: bool) -> PyResult<Py<PyAny>> {
    let a = NdArray::from_vec(vec![v], &[]).map_err(shape_err)?;
    if single { out(py, astype(&Arr::from(a), "float32")?) } else { out(py, Arr::from(a)) }
}

fn ret_vec(py: Python<'_>, v: Vec<f64>, single: bool) -> PyResult<Py<PyAny>> {
    let n = v.len();
    ret(py, NdArray::from_vec(v, &[n]).map_err(shape_err)?, single)
}

fn named(py: Python<'_>, name: &str, fields: &[&str], values: Vec<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    let nt = py.import("collections")?.call_method1("namedtuple", (name, fields.to_vec()))?;
    Ok(nt.call1(pyo3::types::PyTuple::new(py, values)?)?.unbind())
}

macro_rules! matrix_fn {
    ($name:ident, $core:path) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let i = f64_input(py, a)?;
            ret(py, $core(&i.a).map_err(la_err)?, i.single)
        }
    };
}
matrix_fn!(inv, linalg::inv);
matrix_fn!(cholesky, linalg::cholesky);

#[pyfunction]
pub fn det(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let i = f64_input(py, a)?;
    ret_scalar(py, linalg::det(&i.a).map_err(la_err)?, i.single)
}

#[pyfunction]
pub fn slogdet(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let i = f64_input(py, a)?;
    let (sign, logabs) = linalg::slogdet(&i.a).map_err(la_err)?;
    named(py, "SlogdetResult", &["sign", "logabsdet"], vec![ret_scalar(py, sign, i.single)?, ret_scalar(py, logabs, i.single)?])
}

#[pyfunction]
pub fn solve(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (ia, ib) = (f64_input(py, a)?, f64_rhs(py, b)?);
    ret(py, linalg::solve(&ia.a, &ib.a).map_err(la_err)?, ia.single && ib.single)
}

#[pyfunction]
pub fn qr(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let i = f64_input(py, a)?;
    let (q, r) = linalg::qr(&i.a).map_err(la_err)?;
    named(py, "QRResult", &["Q", "R"], vec![ret(py, q, i.single)?, ret(py, r, i.single)?])
}

#[pyfunction]
#[pyo3(signature = (a, UPLO="L"))]
#[allow(non_snake_case)]
pub fn eigh(py: Python<'_>, a: &Bound<'_, PyAny>, UPLO: &str) -> PyResult<Py<PyAny>> {
    if UPLO != "L" {
        return Err(unsupported("only UPLO='L' is bound"));
    }
    let i = f64_input(py, a)?;
    let (w, v) = linalg::eigh(&i.a).map_err(la_err)?;
    named(py, "EighResult", &["eigenvalues", "eigenvectors"], vec![ret_vec(py, w, i.single)?, ret(py, v, i.single)?])
}

#[pyfunction]
#[pyo3(signature = (a, UPLO="L"))]
#[allow(non_snake_case)]
pub fn eigvalsh(py: Python<'_>, a: &Bound<'_, PyAny>, UPLO: &str) -> PyResult<Py<PyAny>> {
    if UPLO != "L" {
        return Err(unsupported("only UPLO='L' is bound"));
    }
    let i = f64_input(py, a)?;
    ret_vec(py, linalg::eigvalsh(&i.a).map_err(la_err)?, i.single)
}

fn eig_values(py: Python<'_>, w: Vec<Complex64>, single: bool) -> PyResult<Py<PyAny>> {
    let n = w.len();
    let arr = Arr::from(NdArray::from_vec(w, &[n]).map_err(shape_err)?);
    out_array(py, if single { astype(&arr, "complex64")? } else { arr })
}

#[pyfunction]
pub fn eigvals(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let i = f64_input(py, a)?;
    let (w, _) = linalg::eig(&i.a).map_err(la_err)?;
    eig_values(py, w, i.single)
}

#[pyfunction]
pub fn eig(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let i = f64_input(py, a)?;
    let (w, v) = linalg::eig(&i.a).map_err(la_err)?;
    let arr = Arr::from(v);
    let vectors = out_array(py, if i.single { astype(&arr, "complex64")? } else { arr })?;
    named(py, "EigResult", &["eigenvalues", "eigenvectors"], vec![eig_values(py, w, i.single)?, vectors])
}

#[pyfunction]
#[pyo3(signature = (a, full_matrices=true, compute_uv=true))]
pub fn svd(py: Python<'_>, a: &Bound<'_, PyAny>, full_matrices: bool, compute_uv: bool) -> PyResult<Py<PyAny>> {
    let i = f64_input(py, a)?;
    if !compute_uv {
        return ret_vec(py, linalg::svdvals(&i.a).map_err(la_err)?, i.single);
    }
    if full_matrices && i.a.ndim() == 2 && i.a.shape()[0] != i.a.shape()[1] {
        return Err(unsupported("full_matrices=True on a non-square matrix is not bound; pass full_matrices=False"));
    }
    let (u, s, vt) = linalg::svd(&i.a).map_err(la_err)?;
    named(py, "SVDResult", &["U", "S", "Vh"], vec![ret(py, u, i.single)?, ret_vec(py, s, i.single)?, ret(py, vt, i.single)?])
}

#[pyfunction]
pub fn svdvals(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let i = f64_input(py, a)?;
    ret_vec(py, linalg::svdvals(&i.a).map_err(la_err)?, i.single)
}

#[pyfunction]
#[pyo3(signature = (a, rcond=None))]
pub fn pinv(py: Python<'_>, a: &Bound<'_, PyAny>, rcond: Option<f64>) -> PyResult<Py<PyAny>> {
    let i = f64_input(py, a)?;
    ret(py, linalg::pinv(&i.a, rcond).map_err(la_err)?, i.single)
}

#[pyfunction]
#[pyo3(signature = (a, tol=None))]
pub fn matrix_rank(py: Python<'_>, a: &Bound<'_, PyAny>, tol: Option<f64>) -> PyResult<Py<PyAny>> {
    let i = f64_input_any(py, a)?;
    if i.a.ndim() > 2 {
        return Err(unsupported("stacked input is not bound"));
    }
    if i.a.is_empty() {
        return out(py, Arr::from(NdArray::from_vec(vec![0i64], &[]).map_err(shape_err)?));
    }
    let r = linalg::matrix_rank(&i.a, tol).map_err(la_err)?;
    out(py, Arr::from(NdArray::from_vec(vec![r as i64], &[]).map_err(shape_err)?))
}

#[pyfunction]
#[pyo3(signature = (a, b, rcond=None))]
pub fn lstsq(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, rcond: Option<f64>) -> PyResult<Py<PyAny>> {
    let (ia, ib) = (f64_input(py, a)?, f64_rhs(py, b)?);
    let single = ia.single && ib.single;
    let rcond = rcond.map(|r| if r < 0.0 { f64::EPSILON } else { r });
    let r = linalg::lstsq(&ia.a, &ib.a, rcond).map_err(la_err)?;
    let res = ret_vec(py, r.residuals, single)?;
    let rank = out(py, Arr::from(NdArray::from_vec(vec![r.rank as i32], &[]).map_err(shape_err)?))?;
    Ok(pyo3::types::PyTuple::new(py, vec![ret(py, r.x, single)?, res, rank, ret_vec(py, r.singular_values, single)?])?.into_any().unbind())
}

fn mat_ord(ord: &Bound<'_, PyAny>) -> PyResult<MatNormOrd> {
    if let Ok(s) = ord.extract::<String>() {
        return match s.as_str() {
            "fro" => Ok(MatNormOrd::Fro),
            "nuc" => Ok(MatNormOrd::Nuc),
            other => Err(PyValueError::new_err(format!("Invalid norm order '{other}' for matrices"))),
        };
    }
    let v: f64 = ord.extract()?;
    Ok(match v {
        x if x == f64::INFINITY => MatNormOrd::Inf,
        x if x == f64::NEG_INFINITY => MatNormOrd::NegInf,
        1.0 => MatNormOrd::One,
        -1.0 => MatNormOrd::NegOne,
        2.0 => MatNormOrd::Two,
        -2.0 => MatNormOrd::NegTwo,
        _ => return Err(PyValueError::new_err("Invalid norm order for matrices.")),
    })
}

#[pyfunction]
#[pyo3(signature = (x, p=None))]
pub fn cond(py: Python<'_>, x: &Bound<'_, PyAny>, p: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let i = f64_input(py, x)?;
    let ord = match p {
        None => MatNormOrd::Two,
        Some(o) => mat_ord(o)?,
    };
    ret_scalar(py, linalg::cond(&i.a, ord).map_err(la_err)?, i.single)
}

#[pyfunction]
#[pyo3(signature = (x, ord=None))]
pub fn norm(py: Python<'_>, x: &Bound<'_, PyAny>, ord: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let i = f64_input_any(py, x)?;
    let value = match i.a.ndim() {
        1 => {
            let o = match ord {
                None => VecNormOrd::Two,
                Some(o) => match o.extract::<f64>() {
                    Ok(v) if v == 2.0 => VecNormOrd::Two,
                    Ok(v) if v == 1.0 => VecNormOrd::One,
                    Ok(v) if v == f64::INFINITY => VecNormOrd::Inf,
                    _ => return Err(unsupported("this vector norm order is not bound")),
                },
            };
            linalg::vector_norm(i.a.as_slice(), o)
        }
        2 => match ord {
            None => linalg::frobenius_norm(&i.a),
            Some(o) => linalg::matrix_norm_ord(&i.a, mat_ord(o)?).map_err(la_err)?,
        },
        _ => return Err(unsupported("norm is bound for 1-D and 2-D input only")),
    };
    ret_scalar(py, value, i.single)
}

#[pyfunction]
pub fn matrix_power(py: Python<'_>, a: &Bound<'_, PyAny>, n: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let raw = Arr::from_numpy(py, a)?;
    if !matches!(raw, Arr::F32(_) | Arr::F64(_)) {
        return Err(unsupported("matrix_power is bound for float32/float64 matrices only"));
    }
    let n: i32 = n.extract().map_err(|_| unsupported("exponent outside the i32 range"))?;
    let i = f64_input(py, a)?;
    ret(py, linalg::matrix_power(&i.a, n).map_err(la_err)?, i.single)
}

fn complex_input(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<(NdArray<Complex64>, bool)> {
    let arr = Operand::parse(py, obj)?.into_arr(py)?;
    let single = matches!(arr, Arr::F32(_) | Arr::C64(_));
    let Arr::C128(a) = astype(&arr, "complex128")? else { unreachable!("cast to complex128") };
    Ok((a, single))
}

fn real_input(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<(NdArray<f64>, bool)> {
    let arr = Operand::parse(py, obj)?.into_arr(py)?;
    if arr.is_complex() {
        return Err(unsupported("real-input transform given complex data"));
    }
    let single = matches!(arr, Arr::F32(_));
    let Arr::F64(a) = astype(&arr, "float64")? else { unreachable!("cast to float64") };
    Ok((a, single))
}

fn ret_complex(py: Python<'_>, a: NdArray<Complex64>, single: bool) -> PyResult<Py<PyAny>> {
    let arr = Arr::from(a);
    out_array(py, if single { astype(&arr, "complex64")? } else { arr })
}

fn ret_real(py: Python<'_>, a: NdArray<f64>, single: bool) -> PyResult<Py<PyAny>> {
    ret(py, a, single)
}

fn check_norm(norm: &Option<String>) -> PyResult<()> {
    match norm.as_deref() {
        None | Some("backward") => Ok(()),
        Some(_) => Err(unsupported("only the default norm='backward' is bound")),
    }
}

fn fft_err(e: rustnumpy::FftError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn lines_apply<T: Copy, U: Copy>(
    a: &NdArray<T>,
    axis: isize,
    f: impl Fn(&[T]) -> PyResult<Vec<U>>,
) -> PyResult<NdArray<U>> {
    let ax = norm_axis(axis, a.ndim())?;
    let moved = a.view().moveaxis(&[ax as isize], &[-1]).map_err(shape_err)?.to_owned();
    let width = *moved.shape().last().expect("ndim >= 1");
    let mut data = Vec::new();
    let mut new_width = 0;
    for line in moved.as_slice().chunks(width.max(1)) {
        let r = f(line)?;
        new_width = r.len();
        data.extend(r);
    }
    if moved.is_empty() {
        return Err(PyValueError::new_err("Invalid number of FFT data points (0) specified."));
    }
    let mut shape = moved.shape().to_vec();
    *shape.last_mut().expect("ndim >= 1") = new_width;
    let result = NdArray::from_vec(data, &shape).map_err(shape_err)?;
    Ok(result.view().moveaxis(&[-1], &[ax as isize]).map_err(shape_err)?.to_owned())
}

fn fit<T: Copy + Default>(line: &[T], n: usize) -> Vec<T> {
    let mut v: Vec<T> = line.iter().copied().take(n).collect();
    v.resize(n, T::default());
    v
}

macro_rules! fft_c2c {
    ($name:ident, $core:path) => {
        #[pyfunction]
        #[pyo3(signature = (a, n=None, axis=-1, norm=None))]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, n: Option<usize>, axis: isize, norm: Option<String>) -> PyResult<Py<PyAny>> {
            check_norm(&norm)?;
            let (x, single) = complex_input(py, a)?;
            let r = lines_apply(&x, axis, |line| {
                let src = match n { Some(k) => fit(line, k), None => line.to_vec() };
                $core(&src).map_err(fft_err)
            })?;
            ret_complex(py, r, single)
        }
    };
}
fft_c2c!(fft, rustnumpy::fft);
fft_c2c!(ifft, rustnumpy::ifft);

#[pyfunction]
#[pyo3(signature = (a, n=None, axis=-1, norm=None))]
pub fn rfft(py: Python<'_>, a: &Bound<'_, PyAny>, n: Option<usize>, axis: isize, norm: Option<String>) -> PyResult<Py<PyAny>> {
    check_norm(&norm)?;
    let (x, single) = real_input(py, a)?;
    let r = lines_apply(&x, axis, |line| {
        let src = match n { Some(k) => fit(line, k), None => line.to_vec() };
        rustnumpy::rfft(&src).map_err(fft_err)
    })?;
    ret_complex(py, r, single)
}

#[pyfunction]
#[pyo3(signature = (a, n=None, axis=-1, norm=None))]
pub fn irfft(py: Python<'_>, a: &Bound<'_, PyAny>, n: Option<usize>, axis: isize, norm: Option<String>) -> PyResult<Py<PyAny>> {
    check_norm(&norm)?;
    let (x, single) = complex_input(py, a)?;
    let r = lines_apply(&x, axis, |line| {
        let len = n.unwrap_or(2 * line.len().saturating_sub(1));
        rustnumpy::irfft(line, len).map_err(fft_err)
    })?;
    ret_real(py, r, single)
}

#[pyfunction]
#[pyo3(signature = (a, n=None, axis=-1, norm=None))]
pub fn hfft(py: Python<'_>, a: &Bound<'_, PyAny>, n: Option<usize>, axis: isize, norm: Option<String>) -> PyResult<Py<PyAny>> {
    check_norm(&norm)?;
    let (x, single) = complex_input(py, a)?;
    let r = lines_apply(&x, axis, |line| rustnumpy::hfft(line, n).map_err(fft_err))?;
    ret_real(py, r, single)
}

#[pyfunction]
#[pyo3(signature = (a, n=None, axis=-1, norm=None))]
pub fn ihfft(py: Python<'_>, a: &Bound<'_, PyAny>, n: Option<usize>, axis: isize, norm: Option<String>) -> PyResult<Py<PyAny>> {
    check_norm(&norm)?;
    let (x, single) = real_input(py, a)?;
    let r = lines_apply(&x, axis, |line| rustnumpy::ihfft(line, n).map_err(fft_err))?;
    ret_complex(py, r, single)
}

macro_rules! fft_nd {
    ($name:ident, $core:path) => {
        #[pyfunction]
        #[pyo3(signature = (a, s=None, axes=None, norm=None))]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, s: Option<Bound<'_, PyAny>>, axes: Option<Bound<'_, PyAny>>, norm: Option<String>) -> PyResult<Py<PyAny>> {
            check_norm(&norm)?;
            if s.is_some() || axes.is_some() {
                return Err(unsupported("s= and axes= are not bound; the transform runs over all axes"));
            }
            let (x, single) = complex_input(py, a)?;
            ret_complex(py, $core(&x).map_err(fft_err)?, single)
        }
    };
}
fft_nd!(fftn, rustnumpy::fftn);
fft_nd!(ifftn, rustnumpy::ifftn);

#[pyfunction]
#[pyo3(signature = (a, s=None, axes=None, norm=None))]
pub fn fft2(py: Python<'_>, a: &Bound<'_, PyAny>, s: Option<Bound<'_, PyAny>>, axes: Option<Bound<'_, PyAny>>, norm: Option<String>) -> PyResult<Py<PyAny>> {
    check_norm(&norm)?;
    if s.is_some() || axes.is_some() {
        return Err(unsupported("s= and axes= are not bound"));
    }
    let (x, single) = complex_input(py, a)?;
    if x.ndim() != 2 {
        return Err(unsupported("fft2 is bound for 2-D input only"));
    }
    ret_complex(py, rustnumpy::fft2(&x).map_err(fft_err)?, single)
}

#[pyfunction]
#[pyo3(signature = (a, s=None, axes=None, norm=None))]
pub fn ifft2(py: Python<'_>, a: &Bound<'_, PyAny>, s: Option<Bound<'_, PyAny>>, axes: Option<Bound<'_, PyAny>>, norm: Option<String>) -> PyResult<Py<PyAny>> {
    check_norm(&norm)?;
    if s.is_some() || axes.is_some() {
        return Err(unsupported("s= and axes= are not bound"));
    }
    let (x, single) = complex_input(py, a)?;
    if x.ndim() != 2 {
        return Err(unsupported("ifft2 is bound for 2-D input only"));
    }
    ret_complex(py, rustnumpy::ifft2(&x).map_err(fft_err)?, single)
}

#[pyfunction]
#[pyo3(signature = (a, s=None, axes=None, norm=None))]
pub fn rfftn(py: Python<'_>, a: &Bound<'_, PyAny>, s: Option<Bound<'_, PyAny>>, axes: Option<Bound<'_, PyAny>>, norm: Option<String>) -> PyResult<Py<PyAny>> {
    check_norm(&norm)?;
    if s.is_some() || axes.is_some() {
        return Err(unsupported("s= and axes= are not bound"));
    }
    let (x, single) = real_input(py, a)?;
    ret_complex(py, rustnumpy::rfftn(&x).map_err(fft_err)?, single)
}

#[pyfunction]
#[pyo3(signature = (a, s=None, axes=None, norm=None))]
pub fn irfftn(py: Python<'_>, a: &Bound<'_, PyAny>, s: Option<Bound<'_, PyAny>>, axes: Option<Bound<'_, PyAny>>, norm: Option<String>) -> PyResult<Py<PyAny>> {
    check_norm(&norm)?;
    if axes.is_some() {
        return Err(unsupported("axes= is not bound"));
    }
    let (x, single) = complex_input(py, a)?;
    let shape: Option<Vec<usize>> = match &s {
        None => None,
        Some(o) => Some(o.extract().map_err(|_| unsupported("s= with negative or non-integer entries is not bound"))?),
    };
    ret_real(py, rustnumpy::irfftn(&x, shape.as_deref()).map_err(fft_err)?, single)
}

#[pyfunction]
#[pyo3(signature = (n, d=1.0))]
pub fn fftfreq(py: Python<'_>, n: usize, d: f64) -> PyResult<Py<PyAny>> {
    ret_vec(py, rustnumpy::fftfreq(n, d).map_err(fft_err)?, false)
}

#[pyfunction]
#[pyo3(signature = (n, d=1.0))]
pub fn rfftfreq(py: Python<'_>, n: usize, d: f64) -> PyResult<Py<PyAny>> {
    ret_vec(py, rustnumpy::rfftfreq(n, d).map_err(fft_err)?, false)
}

macro_rules! shift_fn {
    ($name:ident, $core:path) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let arr = Operand::parse(py, x)?.into_arr(py)?;
            if arr.ndim() != 1 {
                return Err(unsupported("shift is bound for 1-D input only"));
            }
            let result = crate::with_arr!(&arr, a => Arr::from(NdArray::from_vec($core(a.as_slice()), a.shape()).map_err(shape_err)?));
            out_array(py, result)
        }
    };
}
shift_fn!(fftshift, rustnumpy::fftshift);
shift_fn!(ifftshift, rustnumpy::ifftshift);

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! reg {
        ($($f:ident),* $(,)?) => {$( m.add_function(wrap_pyfunction!($f, m)?)?; )*};
    }
    reg!(
        inv, cholesky, det, slogdet, solve, qr, eigh, eigvalsh, eigvals, eig, svd, svdvals, pinv, matrix_rank, lstsq, cond,
        norm, matrix_power, fft, ifft, rfft, irfft, hfft, ihfft, fftn, ifftn, fft2, ifft2, rfftn, irfftn, fftfreq, rfftfreq,
        fftshift, ifftshift
    );
    m.add("LinAlgError", m.py().get_type::<LinAlgError>())?;
    Ok(())
}
