use crate::casting::astype;
use crate::dynarray::{unsupported, Arr, C32, C64};
use crate::ops::{out, resolve_binary, resolve_loop, shape_err, Operand};
use half::f16;
use num_complex::Complex;
use num_traits::Float;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rustnumpy::logic::{self, map_to, zip_map};
use rustnumpy::mathfunc::{self, Arith};
use rustnumpy::ufunc::{map, zip_with};
use rustnumpy::{NdArray, ShapeError};

pub fn float_loop_name(name: &str) -> &'static str {
    match name {
        "bool" | "int8" | "uint8" | "float16" => "float16",
        "int16" | "uint16" | "float32" => "float32",
        "complex64" => "complex64",
        "complex128" => "complex128",
        _ => "float64",
    }
}

fn float_rank(name: &str) -> u8 {
    match float_loop_name(name) {
        "float16" => 0,
        "float32" => 1,
        "float64" => 2,
        "complex64" => 3,
        _ => 4,
    }
}

fn common_static(name: &str) -> &'static str {
    match name {
        "float16" => "float16",
        "float32" => "float32",
        "float64" => "float64",
        "complex64" => "complex64",
        _ => "complex128",
    }
}

fn float_input(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Arr> {
    let arr = Arr::from_object(py, a)?;
    let target = float_loop_name(arr.dtype_name());
    if target == arr.dtype_name() {
        Ok(arr)
    } else {
        astype(&arr, target)
    }
}

fn not_supported(name: &str, dtype: &str) -> PyErr {
    PyTypeError::new_err(format!("ufunc '{name}' not supported for the input types (dtype('{dtype}')), and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''"))
}

macro_rules! float_arms {
    ($arr:expr, $x:ident, $t:ident => $body:expr, else $other:expr) => {
        match $arr {
            Arr::F16($x) => {
                #[allow(dead_code)]
                type $t = f16;
                $body
            }
            Arr::F32($x) => {
                #[allow(dead_code)]
                type $t = f32;
                $body
            }
            Arr::F64($x) => {
                #[allow(dead_code)]
                type $t = f64;
                $body
            }
            _ => $other,
        }
    };
}

macro_rules! real_arms {
    ($arr:expr, $x:ident, $t:ident => $body:expr, else $other:expr) => {
        match $arr {
            Arr::I8($x) => {
                #[allow(dead_code)]
                type $t = i8;
                $body
            }
            Arr::I16($x) => {
                #[allow(dead_code)]
                type $t = i16;
                $body
            }
            Arr::I32($x) => {
                #[allow(dead_code)]
                type $t = i32;
                $body
            }
            Arr::I64($x) => {
                #[allow(dead_code)]
                type $t = i64;
                $body
            }
            Arr::U8($x) => {
                #[allow(dead_code)]
                type $t = u8;
                $body
            }
            Arr::U16($x) => {
                #[allow(dead_code)]
                type $t = u16;
                $body
            }
            Arr::U32($x) => {
                #[allow(dead_code)]
                type $t = u32;
                $body
            }
            Arr::U64($x) => {
                #[allow(dead_code)]
                type $t = u64;
                $body
            }
            Arr::F16($x) => {
                #[allow(dead_code)]
                type $t = f16;
                $body
            }
            Arr::F32($x) => {
                #[allow(dead_code)]
                type $t = f32;
                $body
            }
            Arr::F64($x) => {
                #[allow(dead_code)]
                type $t = f64;
                $body
            }
            _ => $other,
        }
    };
}

macro_rules! int_arms2 {
    ($x:expr, $y:expr, $p:ident, $q:ident, $t:ident => $body:expr, else $other:expr) => {
        match ($x, $y) {
            (Arr::I8($p), Arr::I8($q)) => {
                #[allow(dead_code)]
                type $t = i8;
                $body
            }
            (Arr::I16($p), Arr::I16($q)) => {
                #[allow(dead_code)]
                type $t = i16;
                $body
            }
            (Arr::I32($p), Arr::I32($q)) => {
                #[allow(dead_code)]
                type $t = i32;
                $body
            }
            (Arr::I64($p), Arr::I64($q)) => {
                #[allow(dead_code)]
                type $t = i64;
                $body
            }
            (Arr::U8($p), Arr::U8($q)) => {
                #[allow(dead_code)]
                type $t = u8;
                $body
            }
            (Arr::U16($p), Arr::U16($q)) => {
                #[allow(dead_code)]
                type $t = u16;
                $body
            }
            (Arr::U32($p), Arr::U32($q)) => {
                #[allow(dead_code)]
                type $t = u32;
                $body
            }
            (Arr::U64($p), Arr::U64($q)) => {
                #[allow(dead_code)]
                type $t = u64;
                $body
            }
            _ => $other,
        }
    };
}

macro_rules! real_arms2 {
    ($x:expr, $y:expr, $p:ident, $q:ident, $t:ident => $body:expr, else $other:expr) => {
        match ($x, $y) {
            (Arr::F16($p), Arr::F16($q)) => {
                #[allow(dead_code)]
                type $t = f16;
                $body
            }
            (Arr::F32($p), Arr::F32($q)) => {
                #[allow(dead_code)]
                type $t = f32;
                $body
            }
            (Arr::F64($p), Arr::F64($q)) => {
                #[allow(dead_code)]
                type $t = f64;
                $body
            }
            _ => int_arms2!($x, $y, $p, $q, $t => $body, else $other),
        }
    };
}

macro_rules! all_arms2 {
    ($x:expr, $y:expr, $p:ident, $q:ident, $t:ident => $body:expr, else $other:expr) => {
        match ($x, $y) {
            (Arr::Bool($p), Arr::Bool($q)) => {
                #[allow(dead_code)]
                type $t = bool;
                $body
            }
            (Arr::C64($p), Arr::C64($q)) => {
                #[allow(dead_code)]
                type $t = C32;
                $body
            }
            (Arr::C128($p), Arr::C128($q)) => {
                #[allow(dead_code)]
                type $t = C64;
                $body
            }
            _ => real_arms2!($x, $y, $p, $q, $t => $body, else $other),
        }
    };
}

fn u_sqrt<T: Float>(x: T) -> T {
    x.sqrt()
}
fn u_cbrt<T: Float>(x: T) -> T {
    x.cbrt()
}
fn u_exp<T: Float>(x: T) -> T {
    x.exp()
}
fn u_exp2<T: Float>(x: T) -> T {
    x.exp2()
}
fn u_expm1<T: Float>(x: T) -> T {
    x.exp_m1()
}
fn u_log<T: Float>(x: T) -> T {
    x.ln()
}
fn u_log2<T: Float>(x: T) -> T {
    x.log2()
}
fn u_log10<T: Float>(x: T) -> T {
    x.log10()
}
fn u_log1p<T: Float>(x: T) -> T {
    x.ln_1p()
}
fn u_sin<T: Float>(x: T) -> T {
    x.sin()
}
fn u_cos<T: Float>(x: T) -> T {
    x.cos()
}
fn u_tan<T: Float>(x: T) -> T {
    x.tan()
}
fn u_asin<T: Float>(x: T) -> T {
    x.asin()
}
fn u_acos<T: Float>(x: T) -> T {
    x.acos()
}
fn u_atan<T: Float>(x: T) -> T {
    x.atan()
}
fn u_sinh<T: Float>(x: T) -> T {
    x.sinh()
}
fn u_cosh<T: Float>(x: T) -> T {
    x.cosh()
}
fn u_tanh<T: Float>(x: T) -> T {
    x.tanh()
}
fn u_asinh<T: Float>(x: T) -> T {
    x.asinh()
}
fn u_acosh<T: Float>(x: T) -> T {
    x.acosh()
}
fn u_atanh<T: Float>(x: T) -> T {
    x.atanh()
}
fn u_degrees<T: Float>(x: T) -> T {
    x.to_degrees()
}
fn u_radians<T: Float>(x: T) -> T {
    x.to_radians()
}
fn u_floor<T: Float>(x: T) -> T {
    x.floor()
}
fn u_ceil<T: Float>(x: T) -> T {
    x.ceil()
}
fn u_trunc<T: Float>(x: T) -> T {
    x.trunc()
}
fn u_fabs<T: Float>(x: T) -> T {
    x.abs()
}
fn u_rint<T: Float>(x: T) -> T {
    let r = x.round();
    let two = T::one() + T::one();
    if (x - x.trunc()).abs() == T::one() / two && (r / two).fract() != T::zero() {
        r - x.signum()
    } else {
        r
    }
}
fn ln2<T: Float>() -> T {
    T::from(std::f64::consts::LN_2).expect("ln 2 fits every float")
}

fn c_sqrt_fb<T: Float>(z: Complex<T>) -> Complex<T> {
    z.sqrt()
}
fn c_exp_fb<T: Float>(z: Complex<T>) -> Complex<T> {
    z.exp()
}
fn c_exp2<T: Float>(z: Complex<T>) -> Complex<T> {
    (z * Complex::new(ln2::<T>(), T::zero())).exp()
}
fn c_expm1<T: Float>(z: Complex<T>) -> Complex<T> {
    let two = T::one() + T::one();
    let a = (z.im / two).sin();
    Complex::new(z.re.exp_m1() * z.im.cos() - two * a * a, z.re.exp() * z.im.sin())
}
fn c_log_fb<T: Float>(z: Complex<T>) -> Complex<T> {
    let half = T::one() / (T::one() + T::one());
    let two = T::one() + T::one();
    let (ax, ay) = (z.re.abs(), z.im.abs());
    let m = ax.hypot(ay);
    let re = if m > half && m < two && m.is_finite() {
        half * ((ax - T::one()) * (ax + T::one()) + ay * ay).ln_1p()
    } else {
        m.ln()
    };
    Complex::new(re, z.im.atan2(z.re))
}
fn c_log2<T: Float>(z: Complex<T>) -> Complex<T> {
    let l = c_log(z);
    let k = T::from(std::f64::consts::LOG2_E).expect("log2(e) fits every float");
    Complex::new(l.re * k, l.im * k)
}
fn c_log10<T: Float>(z: Complex<T>) -> Complex<T> {
    let l = c_log(z);
    let k = T::from(std::f64::consts::LOG10_E).expect("log10(e) fits every float");
    Complex::new(l.re * k, l.im * k)
}
fn c_log1p<T: Float>(z: Complex<T>) -> Complex<T> {
    let x1 = z.re + T::one();
    Complex::new(x1.hypot(z.im).ln(), z.im.atan2(x1))
}
fn c_sin_fb<T: Float>(z: Complex<T>) -> Complex<T> {
    z.sin()
}
fn c_cos_fb<T: Float>(z: Complex<T>) -> Complex<T> {
    z.cos()
}
fn c_tan_fb<T: Float>(z: Complex<T>) -> Complex<T> {
    let r = c_tanh(Complex::new(-z.im, z.re));
    Complex::new(r.im, -r.re)
}
fn c_asin_fb<T: Float>(z: Complex<T>) -> Complex<T> {
    let a = Complex::new(T::one() - z.re, -z.im).sqrt();
    let b = Complex::new(T::one() + z.re, z.im).sqrt();
    Complex::new(z.re.atan2((a * b).re), (a.conj() * b).im.asinh())
}
fn c_acos_fb<T: Float>(z: Complex<T>) -> Complex<T> {
    let a = Complex::new(T::one() - z.re, -z.im).sqrt();
    let b = Complex::new(T::one() + z.re, z.im).sqrt();
    let two = T::one() + T::one();
    Complex::new(two * a.re.atan2(b.re), (b.conj() * a).im.asinh())
}
fn c_atan_fb<T: Float>(z: Complex<T>) -> Complex<T> {
    let r = c_atanh(Complex::new(-z.im, z.re));
    Complex::new(r.im, -r.re)
}
fn c_sinh_fb<T: Float>(z: Complex<T>) -> Complex<T> {
    z.sinh()
}
fn c_cosh_fb<T: Float>(z: Complex<T>) -> Complex<T> {
    z.cosh()
}
fn c_tanh_fb<T: Float>(z: Complex<T>) -> Complex<T> {
    let (x, y) = (z.re, z.im);
    let one = T::one();
    let limit = T::from(if std::mem::size_of::<T>() == 4 { 9.0 } else { 22.0 }).expect("limit fits every float");
    if x.abs() >= limit {
        let e = (-x.abs()).exp();
        let four = T::from(4.0).expect("4 fits every float");
        return Complex::new(one.copysign(x), four * y.sin() * y.cos() * e * e);
    }
    let t = y.tan();
    let beta = one + t * t;
    let s = x.sinh();
    let rho = (one + s * s).sqrt();
    let denom = one + beta * s * s;
    Complex::new(beta * rho * s / denom, t / denom)
}
fn c_asinh_fb<T: Float>(z: Complex<T>) -> Complex<T> {
    let r = c_asin(Complex::new(-z.im, z.re));
    Complex::new(r.im, -r.re)
}
fn c_acosh_fb<T: Float>(z: Complex<T>) -> Complex<T> {
    let w = c_acos(z);
    if z.im.is_sign_negative() { Complex::new(w.im, -w.re) } else { Complex::new(-w.im, w.re) }
}
fn c_atanh_fb<T: Float>(z: Complex<T>) -> Complex<T> {
    let x = z.re.abs();
    let y = z.im;
    let one = T::one();
    let two = one + one;
    let four = two + two;
    let quarter = one / four;
    let re = quarter * (four * x / ((one - x) * (one - x) + y * y)).ln_1p();
    let im = (one / two) * (two * y).atan2((one - x) * (one + x) - y * y);
    Complex::new(re.copysign(z.re), im)
}
fn mulinf(a: f64, sign: f64) -> f64 {
    if a == 0.0 { a * sign.signum() } else { (a.signum() * sign.signum()) * f64::INFINITY }
}

fn special64(name: &str, z: Complex<f64>, edom: bool) -> Option<Complex<f64>> {
    let (x, y) = (z.re, z.im);
    if !edom {
        let (sy, cy) = y.sin_cos();
        let (sx, cx) = x.sin_cos();
        return Some(match name {
            "exp" => Complex::new(mulinf(cy, 1.0), mulinf(sy, 1.0)),
            "cosh" => Complex::new(mulinf(cy, 1.0), mulinf(sy, x)),
            "sinh" => Complex::new(mulinf(cy, x), mulinf(sy, 1.0)),
            "sin" => Complex::new(mulinf(sx, 1.0), mulinf(cx, y)),
            "cos" => Complex::new(mulinf(cx, 1.0), -mulinf(sx, y)),
            _ => return None,
        });
    }
    if y.is_infinite() && x.is_finite() {
        return match (name, x == 0.0) {
            ("sinh", true) => Some(Complex::new(x, f64::NAN)),
            ("cosh", true) => Some(Complex::new(f64::NAN, 0.0 * x.signum() * y.signum())),
            ("tanh", true) => Some(Complex::new(x, f64::NAN)),
            ("sinh" | "cosh" | "tanh", false) => Some(Complex::new(f64::NAN, f64::NAN)),
            _ => None,
        };
    }
    if y.is_infinite() && x.is_infinite() {
        return match name {
            "sinh" => Some(Complex::new(x.signum() * f64::INFINITY, f64::NAN)),
            "cosh" => Some(Complex::new(f64::INFINITY, f64::NAN)),
            "tanh" => Some(Complex::new(x.signum(), 0.0 * y.signum())),
            "sin" => Some(Complex::new(f64::NAN, f64::INFINITY)),
            "cos" => Some(Complex::new(f64::INFINITY, f64::NAN)),
            _ => None,
        };
    }
    if (name == "sin" || name == "cos") && x.is_infinite() && y.is_finite() {
        return Some(Complex::new(f64::NAN, if y == 0.0 { y } else { f64::NAN }));
    }
    if name == "tan" && x.is_infinite() && y.is_finite() {
        return Some(Complex::new(f64::NAN, if y == 0.0 { y } else { f64::NAN }));
    }
    None
}

fn via64<T: Float>(name: &str, z: Complex<T>, f: impl Fn(Complex<f64>) -> pymath::Result<Complex<f64>>, fallback: impl Fn(Complex<T>) -> Complex<T>) -> Complex<T> {
    let w = Complex::new(z.re.to_f64().expect("float converts"), z.im.to_f64().expect("float converts"));
    let back = |r: Complex<f64>| Complex::new(T::from(r.re).expect("float converts"), T::from(r.im).expect("float converts"));
    match f(w) {
        Ok(r) => back(r),
        Err(e) => match special64(name, w, e == pymath::Error::EDOM) {
            Some(r) => back(r),
            None => fallback(z),
        },
    }
}

fn c_sqrt<T: Float>(z: Complex<T>) -> Complex<T> {
    via64("sqrt", z, pymath::cmath::sqrt, c_sqrt_fb)
}

fn c_exp<T: Float>(z: Complex<T>) -> Complex<T> {
    via64("exp", z, pymath::cmath::exp, c_exp_fb)
}

fn c_sin<T: Float>(z: Complex<T>) -> Complex<T> {
    via64("sin", z, pymath::cmath::sin, c_sin_fb)
}

fn c_cos<T: Float>(z: Complex<T>) -> Complex<T> {
    via64("cos", z, pymath::cmath::cos, c_cos_fb)
}

fn c_tan<T: Float>(z: Complex<T>) -> Complex<T> {
    via64("tan", z, pymath::cmath::tan, c_tan_fb)
}

fn c_asin<T: Float>(z: Complex<T>) -> Complex<T> {
    via64("asin", z, pymath::cmath::asin, c_asin_fb)
}

fn c_acos<T: Float>(z: Complex<T>) -> Complex<T> {
    via64("acos", z, pymath::cmath::acos, c_acos_fb)
}

fn c_atan<T: Float>(z: Complex<T>) -> Complex<T> {
    via64("atan", z, pymath::cmath::atan, c_atan_fb)
}

fn c_sinh<T: Float>(z: Complex<T>) -> Complex<T> {
    via64("sinh", z, pymath::cmath::sinh, c_sinh_fb)
}

fn c_cosh<T: Float>(z: Complex<T>) -> Complex<T> {
    via64("cosh", z, pymath::cmath::cosh, c_cosh_fb)
}

fn c_tanh<T: Float>(z: Complex<T>) -> Complex<T> {
    via64("tanh", z, pymath::cmath::tanh, c_tanh_fb)
}

fn c_asinh<T: Float>(z: Complex<T>) -> Complex<T> {
    via64("asinh", z, pymath::cmath::asinh, c_asinh_fb)
}

fn c_acosh<T: Float>(z: Complex<T>) -> Complex<T> {
    via64("acosh", z, pymath::cmath::acosh, c_acosh_fb)
}

fn c_atanh<T: Float>(z: Complex<T>) -> Complex<T> {
    via64("atanh", z, pymath::cmath::atanh, c_atanh_fb)
}

fn c_log<T: Float>(z: Complex<T>) -> Complex<T> {
    via64("log", z, |w| pymath::cmath::log(w, None), c_log_fb)
}

fn c_rint<T: Float>(z: Complex<T>) -> Complex<T> {
    Complex::new(u_rint(z.re), u_rint(z.im))
}

macro_rules! unary_math {
    ($name:ident, $real:path, $complex:path) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let arr = float_input(py, a)?;
            let r = match &arr {
                Arr::C64(x) => Arr::from(map(&x.view(), $complex)),
                Arr::C128(x) => Arr::from(map(&x.view(), $complex)),
                other => float_arms!(other, x, T => Arr::from(map(&x.view(), $real)), else unreachable!("float loop input")),
            };
            out(py, r)
        }
    };
    ($name:ident, $real:path) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let arr = float_input(py, a)?;
            let r = match &arr {
                Arr::C64(_) | Arr::C128(_) => return Err(not_supported(stringify!($name), arr.dtype_name())),
                other => float_arms!(other, x, T => Arr::from(map(&x.view(), $real)), else unreachable!("float loop input")),
            };
            out(py, r)
        }
    };
}

unary_math!(sqrt, u_sqrt, c_sqrt);
unary_math!(cbrt, u_cbrt);
unary_math!(exp, u_exp, c_exp);
unary_math!(exp2, u_exp2, c_exp2);
unary_math!(expm1, u_expm1, c_expm1);
unary_math!(log, u_log, c_log);
unary_math!(log2, u_log2, c_log2);
unary_math!(log10, u_log10, c_log10);
unary_math!(log1p, u_log1p, c_log1p);
unary_math!(sin, u_sin, c_sin);
unary_math!(cos, u_cos, c_cos);
unary_math!(tan, u_tan, c_tan);
unary_math!(arcsin, u_asin, c_asin);
unary_math!(arccos, u_acos, c_acos);
unary_math!(arctan, u_atan, c_atan);
unary_math!(sinh, u_sinh, c_sinh);
unary_math!(cosh, u_cosh, c_cosh);
unary_math!(tanh, u_tanh, c_tanh);
unary_math!(arcsinh, u_asinh, c_asinh);
unary_math!(arccosh, u_acosh, c_acosh);
unary_math!(arctanh, u_atanh, c_atanh);
unary_math!(degrees, u_degrees);
unary_math!(radians, u_radians);
unary_math!(fabs, u_fabs);
unary_math!(rint, u_rint, c_rint);

macro_rules! rounding {
    ($name:ident, $real:path) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let arr = Arr::from_object(py, a)?;
            let r = match &arr {
                Arr::C64(_) | Arr::C128(_) => return Err(not_supported(stringify!($name), arr.dtype_name())),
                other => float_arms!(other, x, T => Arr::from(map(&x.view(), $real)), else arr.clone_arr()),
            };
            out(py, r)
        }
    };
}
rounding!(floor, u_floor);
rounding!(ceil, u_ceil);
rounding!(trunc, u_trunc);

fn c_recip<T: Float>(z: Complex<T>) -> Complex<T> {
    let one = T::one();
    if z.im.abs() <= z.re.abs() {
        let rat = z.im / z.re;
        let scl = one / (z.re + z.im * rat);
        Complex::new(scl, -rat * scl)
    } else {
        let rat = z.re / z.im;
        let scl = one / (z.im + z.re * rat);
        Complex::new(rat * scl, -scl)
    }
}

fn bool_as_i8(arr: Arr) -> PyResult<Arr> {
    if arr.is_bool() {
        astype(&arr, "int8")
    } else {
        Ok(arr)
    }
}

trait Recip: Copy {
    fn recip_(self) -> Self;
}
macro_rules! recip_signed {
    ($zero:expr; $($t:ty),*) => {$(impl Recip for $t {
        fn recip_(self) -> Self { if self == 0 { $zero(<$t>::MIN) } else { (1 as $t).wrapping_div(self) } }
    })*};
}
recip_signed!(|_min| 0; i8, i16);
recip_signed!(|min| min; i32, i64);
macro_rules! recip_unsigned {
    ($($t:ty),*) => {$(impl Recip for $t {
        fn recip_(self) -> Self { if self == 0 { 0 } else { 1 / self } }
    })*};
}
recip_unsigned!(u8, u16, u32, u64);
macro_rules! recip_float {
    ($($t:ty),*) => {$(impl Recip for $t { fn recip_(self) -> Self { Float::recip(self) } })*};
}
recip_float!(f16, f32, f64);

#[pyfunction]
pub fn reciprocal(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = bool_as_i8(Arr::from_object(py, a)?)?;
    let r = match &arr {
        Arr::C64(x) => Arr::from(map(&x.view(), c_recip)),
        Arr::C128(x) => Arr::from(map(&x.view(), c_recip)),
        other => real_arms!(other, x, T => Arr::from(map(&x.view(), <T as Recip>::recip_)), else unreachable!("bool cast to int8")),
    };
    out(py, r)
}

#[pyfunction]
pub fn square(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = bool_as_i8(Arr::from_object(py, a)?)?;
    let r = match &arr {
        Arr::C64(x) => Arr::from(map(&x.view(), |z| z * z)),
        Arr::C128(x) => Arr::from(map(&x.view(), |z| z * z)),
        other => real_arms!(other, x, T => Arr::from(mathfunc::square(&x.view())), else unreachable!("bool cast to int8")),
    };
    out(py, r)
}

#[pyfunction]
pub fn negative(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = Arr::from_object(py, a)?;
    let r = match &arr {
        Arr::Bool(_) => {
            return Err(PyTypeError::new_err(
                "The numpy boolean negative, the `-` operator, is not supported, use the `~` operator or the logical_not function instead.",
            ))
        }
        Arr::C64(x) => Arr::from(map(&x.view(), |z| -z)),
        Arr::C128(x) => Arr::from(map(&x.view(), |z| -z)),
        other => real_arms!(other, x, T => Arr::from(mathfunc::negative(&x.view())), else unreachable!("bool rejected")),
    };
    out(py, r)
}

#[pyfunction]
pub fn positive(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = Arr::from_object(py, a)?;
    if arr.is_bool() {
        return Err(not_supported("positive", "bool"));
    }
    out(py, arr)
}

fn c_sign<T: Float>(z: Complex<T>) -> Complex<T> {
    if z.re.is_nan() || z.im.is_nan() {
        return Complex::new(T::nan(), T::nan());
    }
    let n = z.norm();
    if n == T::zero() { Complex::new(T::zero(), T::zero()) } else { Complex::new(z.re / n, z.im / n) }
}

#[pyfunction]
pub fn sign(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = Arr::from_object(py, a)?;
    let r = match &arr {
        Arr::Bool(_) => return Err(not_supported("sign", "bool")),
        Arr::C64(x) => Arr::from(map(&x.view(), c_sign)),
        Arr::C128(x) => Arr::from(map(&x.view(), c_sign)),
        other => real_arms!(other, x, T => Arr::from(mathfunc::sign(&x.view())), else unreachable!("bool rejected")),
    };
    out(py, r)
}

#[pyfunction]
pub fn absolute(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = Arr::from_object(py, a)?;
    let r = match &arr {
        Arr::Bool(_) => arr.clone_arr(),
        Arr::C64(x) => Arr::from(map_to(&x.view(), |z| z.norm())),
        Arr::C128(x) => Arr::from(map_to(&x.view(), |z| z.norm())),
        other => real_arms!(other, x, T => Arr::from(mathfunc::abs(&x.view())), else unreachable!("bool handled")),
    };
    out(py, r)
}

#[pyfunction]
pub fn conjugate(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = bool_as_i8(Arr::from_object(py, a)?)?;
    let r = match &arr {
        Arr::C64(x) => Arr::from(map(&x.view(), |z| z.conj())),
        Arr::C128(x) => Arr::from(map(&x.view(), |z| z.conj())),
        _ => arr,
    };
    out(py, r)
}

trait SignBit: Copy {
    fn signbit_(self) -> bool;
}
macro_rules! signbit_signed {
    ($($t:ty),*) => {$(impl SignBit for $t { fn signbit_(self) -> bool { self < 0 } })*};
}
signbit_signed!(i8, i16, i32, i64);
macro_rules! signbit_unsigned {
    ($($t:ty),*) => {$(impl SignBit for $t { fn signbit_(self) -> bool { false } })*};
}
signbit_unsigned!(u8, u16, u32, u64);
macro_rules! signbit_float {
    ($($t:ty),*) => {$(impl SignBit for $t { fn signbit_(self) -> bool { Float::is_sign_negative(self) } })*};
}
signbit_float!(f16, f32, f64);

#[pyfunction]
pub fn signbit(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = Arr::from_object(py, a)?;
    let r = match &arr {
        Arr::Bool(x) => Arr::from(map_to(&x.view(), |_| false)),
        Arr::C64(_) | Arr::C128(_) => return Err(not_supported("signbit", arr.dtype_name())),
        other => real_arms!(other, x, T => Arr::from(map_to(&x.view(), <T as SignBit>::signbit_)), else unreachable!("bool and complex handled")),
    };
    out(py, r)
}

macro_rules! class_fn {
    ($name:ident, $core:path) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let arr = Arr::from_object(py, a)?;
            out(py, Arr::from(crate::with_arr!(&arr, x => $core(&x.view()))))
        }
    };
}
class_fn!(isnan, logic::isnan);
class_fn!(isinf, logic::isinf);
class_fn!(isfinite, logic::isfinite);

#[pyfunction]
pub fn logical_not(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = Arr::from_object(py, a)?;
    out(py, Arr::from(crate::with_arr!(&arr, x => logic::logical_not(&x.view()))))
}

#[pyfunction]
pub fn invert(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = Arr::from_object(py, a)?;
    let r = match &arr {
        Arr::Bool(x) => Arr::from(logic::invert(&x.view())),
        other => int_arms1(other)?,
    };
    out(py, r)
}

fn int_arms1(arr: &Arr) -> PyResult<Arr> {
    Ok(match arr {
        Arr::I8(x) => Arr::from(logic::invert(&x.view())),
        Arr::I16(x) => Arr::from(logic::invert(&x.view())),
        Arr::I32(x) => Arr::from(logic::invert(&x.view())),
        Arr::I64(x) => Arr::from(logic::invert(&x.view())),
        Arr::U8(x) => Arr::from(logic::invert(&x.view())),
        Arr::U16(x) => Arr::from(logic::invert(&x.view())),
        Arr::U32(x) => Arr::from(logic::invert(&x.view())),
        Arr::U64(x) => Arr::from(logic::invert(&x.view())),
        other => return Err(not_supported("invert", other.dtype_name())),
    })
}

fn float_pair(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, name: &str) -> PyResult<(Arr, Arr)> {
    resolve_loop(py, a, b, |common, strong| {
        let names = ["float16", "float32", "float64", "complex64", "complex128"];
        let is_float = |d: &str| d.starts_with("float") || d.starts_with("complex");
        let strong: Vec<&str> = strong.into_iter().flatten().collect();
        let rank = if strong.len() == 2 { strong.iter().map(|d| float_rank(d)).max().unwrap_or(0) } else { float_rank(common) };
        let _ = is_float;
        if rank >= 3 {
            return Err(not_supported(name, names[usize::from(rank)]));
        }
        Ok(names[usize::from(rank)])
    })
}

fn u_atan2<T: Float>(y: T, x: T) -> T {
    y.atan2(x)
}
fn u_hypot<T: Float>(a: T, b: T) -> T {
    a.hypot(b)
}
fn u_copysign<T: Float>(a: T, b: T) -> T {
    a.copysign(b)
}
fn u_logaddexp<T: Float>(x: T, y: T) -> T {
    if x == y {
        return x + ln2::<T>();
    }
    let tmp = x - y;
    if tmp > T::zero() {
        x + (-tmp).exp().ln_1p()
    } else if tmp <= T::zero() {
        y + tmp.exp().ln_1p()
    } else {
        tmp
    }
}
fn u_logaddexp2<T: Float>(x: T, y: T) -> T {
    let log2e = T::from(std::f64::consts::LOG2_E).expect("log2(e) fits every float");
    if x == y {
        return x + T::one();
    }
    let tmp = x - y;
    if tmp > T::zero() {
        x + log2e * (-tmp).exp2().ln_1p() * T::one()
    } else if tmp <= T::zero() {
        y + log2e * tmp.exp2().ln_1p()
    } else {
        tmp
    }
}
fn u_heaviside<T: Float>(x: T, h: T) -> T {
    if x.is_nan() {
        x
    } else if x == T::zero() {
        h
    } else if x < T::zero() {
        T::zero()
    } else {
        T::one()
    }
}

macro_rules! binary_float {
    ($name:ident, $f:path) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let (x, y) = float_pair(py, a, b, stringify!($name))?;
            let r = float_arms2(&x, &y, |p, q| zip_with(p, q, $f), |p, q| zip_with(p, q, $f), |p, q| zip_with(p, q, $f))?;
            out(py, r)
        }
    };
}

fn float_arms2<A, B, C>(x: &Arr, y: &Arr, f16f: A, f32f: B, f64f: C) -> PyResult<Arr>
where
    A: Fn(&rustnumpy::ArrayView<f16>, &rustnumpy::ArrayView<f16>) -> Result<NdArray<f16>, ShapeError>,
    B: Fn(&rustnumpy::ArrayView<f32>, &rustnumpy::ArrayView<f32>) -> Result<NdArray<f32>, ShapeError>,
    C: Fn(&rustnumpy::ArrayView<f64>, &rustnumpy::ArrayView<f64>) -> Result<NdArray<f64>, ShapeError>,
{
    Ok(match (x, y) {
        (Arr::F16(p), Arr::F16(q)) => Arr::from(f16f(&p.view(), &q.view()).map_err(shape_err)?),
        (Arr::F32(p), Arr::F32(q)) => Arr::from(f32f(&p.view(), &q.view()).map_err(shape_err)?),
        (Arr::F64(p), Arr::F64(q)) => Arr::from(f64f(&p.view(), &q.view()).map_err(shape_err)?),
        _ => return Err(unsupported("float pair expected")),
    })
}

binary_float!(arctan2, u_atan2);
binary_float!(hypot, u_hypot);
binary_float!(copysign, u_copysign);
binary_float!(logaddexp, u_logaddexp);
binary_float!(logaddexp2, u_logaddexp2);
binary_float!(heaviside, u_heaviside);

trait Ieee: Copy + PartialOrd {
    fn next_after_(self, toward: Self) -> Self;
    fn spacing_(self) -> Self;
}

macro_rules! ieee_impl {
    ($t:ty, $sign:expr) => {
        impl Ieee for $t {
            fn next_after_(self, y: Self) -> Self {
                if self.is_nan() || y.is_nan() {
                    return self + y;
                }
                if self == y {
                    return y;
                }
                let bits = self.to_bits();
                if bits << 1 == 0 {
                    let tiny = <$t>::from_bits(1);
                    return if y > self { tiny } else { -tiny };
                }
                let away = (y > self) == !self.is_sign_negative();
                <$t>::from_bits(if away { bits + 1 } else { bits - 1 })
            }
            fn spacing_(self) -> Self {
                if self.is_infinite() || self.is_nan() {
                    return <$t>::NAN;
                }
                if self.to_bits() << 1 == 0 {
                    return <$t>::from_bits(1);
                }
                let toward = if self.is_sign_negative() { <$t>::NEG_INFINITY } else { <$t>::INFINITY };
                self.next_after_(toward) - self
            }
        }
    };
}
ieee_impl!(f16, 0);
ieee_impl!(f32, 0);
ieee_impl!(f64, 0);

fn spacing_f16(x: f16) -> f16 {
    if x.is_infinite() || x.is_nan() {
        return f16::NAN;
    }
    if x.to_bits() << 1 == 0 {
        return f16::from_bits(1);
    }
    x.next_after_(f16::INFINITY) - x
}

#[pyfunction]
pub fn nextafter(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (x, y) = float_pair(py, a, b, "nextafter")?;
    let r = float_arms2(&x, &y, |p, q| zip_with(p, q, <f16 as Ieee>::next_after_), |p, q| zip_with(p, q, <f32 as Ieee>::next_after_), |p, q| zip_with(p, q, <f64 as Ieee>::next_after_))?;
    out(py, r)
}

#[pyfunction]
pub fn spacing(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = float_input(py, a)?;
    if arr.is_complex() {
        return Err(not_supported("spacing", arr.dtype_name()));
    }
    let r = match &arr {
        Arr::F16(x) => Arr::from(map(&x.view(), spacing_f16)),
        Arr::F32(x) => Arr::from(map(&x.view(), <f32 as Ieee>::spacing_)),
        Arr::F64(x) => Arr::from(map(&x.view(), <f64 as Ieee>::spacing_)),
        _ => unreachable!("float loop input"),
    };
    out(py, r)
}

fn frexp64(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    if exp == 0 {
        let (m, e) = frexp64(x * 18014398509481984.0);
        return (m, e - 54);
    }
    (f64::from_bits((bits & !(0x7ffu64 << 52)) | (1022u64 << 52)), exp - 1022)
}

fn ldexp64(x: f64, n: i64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let mut n = n.clamp(-3000, 3000);
    let mut y = x;
    let big = f64::from_bits(0x7fe0_0000_0000_0000);
    let small = f64::from_bits(0x0010_0000_0000_0000) * f64::from_bits(0x4340_0000_0000_0000);
    if n > 1023 {
        y *= big;
        n -= 1023;
        if n > 1023 {
            y *= big;
            n -= 1023;
            n = n.min(1023);
        }
    } else if n < -1022 {
        y *= small;
        n += 1022 - 53;
        if n < -1022 {
            y *= small;
            n += 1022 - 53;
            n = n.max(-1022);
        }
    }
    y * f64::from_bits(((0x3ff + n) as u64) << 52)
}

#[pyfunction]
pub fn frexp(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let arr = float_input(py, a)?;
    if arr.is_complex() {
        return Err(not_supported("frexp", arr.dtype_name()));
    }
    let exps: NdArray<i32>;
    let mant = match &arr {
        Arr::F16(x) => {
            exps = map_to(&x.view(), |v| frexp64(f64::from(v)).1);
            Arr::from(map(&x.view(), |v| f16::from_f64(frexp64(f64::from(v)).0)))
        }
        Arr::F32(x) => {
            exps = map_to(&x.view(), |v| frexp64(f64::from(v)).1);
            Arr::from(map(&x.view(), |v| frexp64(f64::from(v)).0 as f32))
        }
        Arr::F64(x) => {
            exps = map_to(&x.view(), |v| frexp64(v).1);
            Arr::from(map(&x.view(), |v| frexp64(v).0))
        }
        _ => unreachable!("float loop input"),
    };
    Ok((out(py, mant)?, out(py, Arr::from(exps))?))
}

#[pyfunction]
pub fn ldexp(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let x = float_input(py, a)?;
    let e = Arr::from_object(py, b)?;
    if !(e.is_int() || e.is_bool()) || matches!(e, Arr::U64(_)) {
        return Err(not_supported("ldexp", e.dtype_name()));
    }
    let Arr::I64(e) = astype(&e, "int64")? else { unreachable!("cast to int64") };
    let r = match &x {
        Arr::F16(p) => Arr::from(zip_pair(p, &e, |v, n| f16::from_f64(ldexp64(f64::from(v), n))).map_err(shape_err)?),
        Arr::F32(p) => Arr::from(zip_pair(p, &e, |v, n| ldexp64(f64::from(v), n) as f32).map_err(shape_err)?),
        Arr::F64(p) => Arr::from(zip_pair(p, &e, ldexp64).map_err(shape_err)?),
        _ => return Err(not_supported("ldexp", x.dtype_name())),
    };
    out(py, r)
}

fn zip_pair<T: Copy>(p: &NdArray<T>, e: &NdArray<i64>, f: impl Fn(T, i64) -> T) -> Result<NdArray<T>, ShapeError> {
    let shape = rustnumpy::shape::broadcast_shapes(p.shape(), e.shape())
        .ok_or_else(|| ShapeError::NotBroadcastable { lhs: p.shape().to_vec(), rhs: e.shape().to_vec() })?;
    let (pv, ev) = (p.view().broadcast_to(&shape)?, e.view().broadcast_to(&shape)?);
    let data: Vec<T> = pv.iter().zip(ev.iter()).map(|(x, n)| f(x, n)).collect();
    NdArray::from_vec(data, &shape)
}

#[pyfunction]
pub fn modf(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let arr = float_input(py, a)?;
    if arr.is_complex() {
        return Err(not_supported("modf", arr.dtype_name()));
    }
    fn parts<T: Float>(x: T) -> (T, T) {
        let i = x.trunc();
        let f = if x.is_infinite() { T::zero().copysign(x) } else { (x - i).copysign(x) };
        (f, i)
    }
    let (frac, int) = float_arms!(&arr, x, T => {
        let f = Arr::from(map(&x.view(), |v: T| parts(v).0));
        let i = Arr::from(map(&x.view(), |v: T| parts(v).1));
        (f, i)
    }, else unreachable!("float loop input"));
    Ok((out(py, frac)?, out(py, int)?))
}

macro_rules! arith_binary {
    ($name:ident, $core:path) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let (x, y) = resolve_binary(py, a, b)?;
            let (x, y) = (bool_as_i8(x)?, bool_as_i8(y)?);
            let r = real_arms2!(&x, &y, p, q, T => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?), else return Err(not_supported(stringify!($name), x.dtype_name())));
            out(py, r)
        }
    };
}

fn floor_divide_g<T: Arith>(a: &rustnumpy::ArrayView<T>, b: &rustnumpy::ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    mathfunc::floor_divide(a, b)
}
fn remainder_g<T: Arith>(a: &rustnumpy::ArrayView<T>, b: &rustnumpy::ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    mathfunc::remainder(a, b)
}

arith_binary!(floor_divide, floor_divide_g);
arith_binary!(remainder, remainder_g);

trait CFmod: Copy {
    fn fmod_(self, rhs: Self) -> Self;
}
macro_rules! fmod_int {
    ($($t:ty),*) => {$(impl CFmod for $t {
        fn fmod_(self, rhs: Self) -> Self { if rhs == 0 { 0 } else { self.wrapping_rem(rhs) } }
    })*};
}
fmod_int!(i8, i16, i32, i64, u8, u16, u32, u64);
macro_rules! fmod_float {
    ($($t:ty),*) => {$(impl CFmod for $t { fn fmod_(self, rhs: Self) -> Self { self % rhs } })*};
}
fmod_float!(f16, f32, f64);

#[pyfunction]
pub fn fmod(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (x, y) = resolve_binary(py, a, b)?;
    let (x, y) = (bool_as_i8(x)?, bool_as_i8(y)?);
    let r = real_arms2!(&x, &y, p, q, T => Arr::from(zip_with(&p.view(), &q.view(), <T as CFmod>::fmod_).map_err(shape_err)?), else return Err(not_supported("fmod", x.dtype_name())));
    out(py, r)
}

#[pyfunction]
pub fn divmod(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    Ok((floor_divide(py, a, b)?, remainder(py, a, b)?))
}

fn c_pow<T: Float>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    let wide = |z: Complex<T>| Complex::new(z.re.to_f64().expect("float converts"), z.im.to_f64().expect("float converts"));
    let r = c_pow64(wide(a), wide(b));
    Complex::new(T::from(r.re).expect("float converts"), T::from(r.im).expect("float converts"))
}

fn c_pow64(a: Complex<f64>, b: Complex<f64>) -> Complex<f64> {
    if b.re == 0.0 && b.im == 0.0 {
        return Complex::new(1.0, 0.0);
    }
    if a.re == 0.0 && a.im == 0.0 {
        return if b.re > 0.0 { Complex::new(0.0, 0.0) } else { Complex::new(f64::NAN, f64::NAN) };
    }
    if b.im == 0.0 {
        let n = b.re as i64;
        if n as f64 == b.re {
            match n {
                1 => return a,
                2 => return a * a,
                3 => return a * a * a,
                _ if n > -100 && n < 100 => {
                    let mut acc = Complex::new(1.0, 0.0);
                    let mut base = a;
                    let mut p = n.abs();
                    while p > 0 {
                        if p & 1 == 1 {
                            acc *= base;
                        }
                        p >>= 1;
                        if p > 0 {
                            base = base * base;
                        }
                    }
                    return if n > 0 { acc } else { Complex::new(1.0, 0.0).divide_generic(acc) };
                }
                _ => {}
            }
        }
    }
    a.powc(b)
}

trait DivideGeneric<T> {
    fn divide_generic(self, rhs: Complex<T>) -> Complex<T>;
}
impl<T: Float> DivideGeneric<T> for Complex<T> {
    fn divide_generic(self, b: Complex<T>) -> Complex<T> {
        let (ar, ai, br, bi) = (self.re, self.im, b.re, b.im);
        let (abs_br, abs_bi) = (br.abs(), bi.abs());
        if abs_br >= abs_bi {
            if abs_br == T::zero() && abs_bi == T::zero() {
                return Complex::new(ar / abs_br, ai / abs_bi);
            }
            let rat = bi / br;
            let scl = T::one() / (br + bi * rat);
            Complex::new((ar + ai * rat) * scl, (ai - ar * rat) * scl)
        } else {
            let rat = br / bi;
            let scl = T::one() / (bi + br * rat);
            Complex::new((ar * rat + ai) * scl, (ai * rat - ar) * scl)
        }
    }
}

#[pyfunction]
pub fn power(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (x, y) = resolve_binary(py, a, b)?;
    out(py, power_arrs(x, y)?)
}

fn power_arrs(x: Arr, y: Arr) -> PyResult<Arr> {
    let (x, y) = (bool_as_i8(x)?, bool_as_i8(y)?);
    Ok(match (&x, &y) {
        (Arr::C64(p), Arr::C64(q)) => Arr::from(zip_with(&p.view(), &q.view(), c_pow).map_err(shape_err)?),
        (Arr::C128(p), Arr::C128(q)) => Arr::from(zip_with(&p.view(), &q.view(), c_pow).map_err(shape_err)?),
        _ => real_arms2!(&x, &y, p, q, T => Arr::from(mathfunc::power(&p.view(), &q.view()).map_err(shape_err)?), else return Err(not_supported("power", x.dtype_name()))),
    })
}

#[pyfunction]
pub fn float_power(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (x, y) = resolve_loop(py, a, b, |common, _| Ok(if common.starts_with("complex") { "complex128" } else { "float64" }))?;
    out(py, power_arrs(x, y)?)
}

macro_rules! extreme {
    ($name:ident, $keep_first:expr, $bool_op:expr) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let (x, y) = resolve_binary(py, a, b)?;
            let r = match (&x, &y) {
                (Arr::Bool(p), Arr::Bool(q)) => Arr::from(zip_with(&p.view(), &q.view(), $bool_op).map_err(shape_err)?),
                (Arr::C64(p), Arr::C64(q)) => Arr::from(zip_with(&p.view(), &q.view(), |m, n| $keep_first(m, n)).map_err(shape_err)?),
                (Arr::C128(p), Arr::C128(q)) => Arr::from(zip_with(&p.view(), &q.view(), |m, n| $keep_first(m, n)).map_err(shape_err)?),
                _ => real_arms2!(&x, &y, p, q, T => Arr::from(mathfunc::$name(&p.view(), &q.view()).map_err(shape_err)?), else return Err(not_supported(stringify!($name), x.dtype_name()))),
            };
            out(py, r)
        }
    };
}

fn c_isnan<T: Float>(z: Complex<T>) -> bool {
    z.re.is_nan() || z.im.is_nan()
}
fn c_ge<T: Float>(a: Complex<T>, b: Complex<T>) -> bool {
    (a.re > b.re && !a.im.is_nan() && !b.im.is_nan()) || (a.re == b.re && a.im >= b.im)
}
fn c_le<T: Float>(a: Complex<T>, b: Complex<T>) -> bool {
    (a.re < b.re && !a.im.is_nan() && !b.im.is_nan()) || (a.re == b.re && a.im <= b.im)
}
pub(crate) fn c_maximum<T: Float>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    if c_isnan(a) || c_ge(a, b) { a } else { b }
}
pub(crate) fn c_minimum<T: Float>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    if c_isnan(a) || c_le(a, b) { a } else { b }
}
fn c_fmax<T: Float>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    if c_isnan(b) || c_ge(a, b) { a } else { b }
}
fn c_fmin<T: Float>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    if c_isnan(b) || c_le(a, b) { a } else { b }
}

extreme!(maximum, c_maximum, |m: bool, n: bool| m || n);
extreme!(minimum, c_minimum, |m: bool, n: bool| m && n);
extreme!(fmax, c_fmax, |m: bool, n: bool| m || n);
extreme!(fmin, c_fmin, |m: bool, n: bool| m && n);

#[pyfunction]
pub fn divide(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (x, y) = resolve_loop(py, a, b, |common, _| {
        Ok(if common.starts_with("float") || common.starts_with("complex") { common_static(common) } else { "float64" })
    })?;
    let r = match (&x, &y) {
        (Arr::F16(p), Arr::F16(q)) => Arr::from(mathfunc::divide(&p.view(), &q.view()).map_err(shape_err)?),
        (Arr::F32(p), Arr::F32(q)) => Arr::from(mathfunc::divide(&p.view(), &q.view()).map_err(shape_err)?),
        (Arr::F64(p), Arr::F64(q)) => Arr::from(mathfunc::divide(&p.view(), &q.view()).map_err(shape_err)?),
        (Arr::C64(p), Arr::C64(q)) => Arr::from(mathfunc::divide(&p.view(), &q.view()).map_err(shape_err)?),
        (Arr::C128(p), Arr::C128(q)) => Arr::from(mathfunc::divide(&p.view(), &q.view()).map_err(shape_err)?),
        _ => return Err(unsupported("unreachable divide dtype")),
    };
    out(py, r)
}

trait Gcd: Copy {
    fn gcd_(self, rhs: Self) -> Self;
    fn lcm_(self, rhs: Self) -> Self;
}
macro_rules! gcd_signed {
    ($($t:ty, $u:ty);*) => {$(impl Gcd for $t {
        fn gcd_(self, rhs: Self) -> Self {
            let (mut a, mut b) = (self.unsigned_abs(), rhs.unsigned_abs());
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a as $t
        }
        fn lcm_(self, rhs: Self) -> Self {
            let (ua, ub) = (self.unsigned_abs() as u64, rhs.unsigned_abs() as u64);
            let g = ua.gcd_(ub);
            if g == 0 { 0 } else { (ua / g).wrapping_mul(ub) as $t }
        }
    })*};
}
gcd_signed!(i8, u8; i16, u16; i32, u32; i64, u64);
macro_rules! gcd_unsigned {
    ($($t:ty),*) => {$(impl Gcd for $t {
        fn gcd_(self, rhs: Self) -> Self {
            let (mut a, mut b) = (self, rhs);
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        }
        fn lcm_(self, rhs: Self) -> Self {
            let g = self.gcd_(rhs);
            if g == 0 { 0 } else { ((self as u64 / g as u64).wrapping_mul(rhs as u64)) as $t }
        }
    })*};
}
gcd_unsigned!(u8, u16, u32, u64);

macro_rules! gcd_fn {
    ($name:ident, $m:ident) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let (x, y) = resolve_binary(py, a, b)?;
            let r = int_arms2!(&x, &y, p, q, T => Arr::from(zip_with(&p.view(), &q.view(), <T as Gcd>::$m).map_err(shape_err)?), else return Err(not_supported(stringify!($name), x.dtype_name())));
            out(py, r)
        }
    };
}
gcd_fn!(gcd, gcd_);
gcd_fn!(lcm, lcm_);

trait CmpOps: Copy {
    fn eq_(self, o: Self) -> bool;
    fn lt_(self, o: Self) -> bool;
    fn le_(self, o: Self) -> bool;
}
macro_rules! cmp_plain {
    ($($t:ty),*) => {$(impl CmpOps for $t {
        fn eq_(self, o: Self) -> bool { self == o }
        fn lt_(self, o: Self) -> bool { self < o }
        fn le_(self, o: Self) -> bool { self <= o }
    })*};
}
cmp_plain!(bool, i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64);
macro_rules! cmp_complex {
    ($($t:ty),*) => {$(impl CmpOps for Complex<$t> {
        fn eq_(self, o: Self) -> bool { self == o }
        fn lt_(self, o: Self) -> bool { self.re < o.re || (self.re == o.re && self.im < o.im) }
        fn le_(self, o: Self) -> bool { self.re < o.re || (self.re == o.re && self.im <= o.im) }
    })*};
}
cmp_complex!(f32, f64);

fn int_bounds(name: &str) -> Option<(i128, i128)> {
    Some(match name {
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

fn out_of_range_constant(strong: &Arr, weak: i64, weak_on_left: bool, op: u8) -> Option<NdArray<bool>> {
    let (lo, hi) = int_bounds(strong.dtype_name())?;
    let v = weak as i128;
    if (lo..=hi).contains(&v) {
        return None;
    }
    let below = v < lo;
    let truth = match op {
        0 => false,
        1 => true,
        2 | 3 => if weak_on_left { below } else { !below },
        _ => if weak_on_left { !below } else { below },
    };
    let n: usize = strong.shape().iter().product();
    Some(NdArray::from_vec(vec![truth; n], &strong.shape()).expect("shape matches count"))
}

macro_rules! compare_fn {
    ($name:ident, $op:expr, $f:expr) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let (oa, ob) = (Operand::parse(py, a)?, Operand::parse(py, b)?);
            let constant = match (&oa, &ob) {
                (Operand::Arr(s), Operand::WeakInt(w)) => out_of_range_constant(s, *w, false, $op),
                (Operand::WeakInt(w), Operand::Arr(s)) => out_of_range_constant(s, *w, true, $op),
                _ => None,
            };
            if let Some(c) = constant {
                return out(py, Arr::from(c));
            }
            let name = crate::ops::common_name(&oa, &ob);
            let (x, y) = (crate::ops::materialize(oa, name)?, crate::ops::materialize(ob, name)?);
            let r = all_arms2!(&x, &y, p, q, T => zip_map(&p.view(), &q.view(), $f).map_err(shape_err)?, else return Err(unsupported("unreachable comparison dtype")));
            out(py, Arr::from(r))
        }
    };
}
compare_fn!(equal, 0, CmpOps::eq_);
compare_fn!(not_equal, 1, |m, n| !CmpOps::eq_(m, n));
compare_fn!(less, 2, CmpOps::lt_);
compare_fn!(less_equal, 3, CmpOps::le_);
compare_fn!(greater, 4, |m, n| CmpOps::lt_(n, m));
compare_fn!(greater_equal, 5, |m, n| CmpOps::le_(n, m));

fn truth_operand(py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<rustnumpy::NdArray<bool>> {
    match astype(&Operand::parse(py, o)?.into_arr(py)?, "bool")? {
        Arr::Bool(m) => Ok(m),
        _ => unreachable!("cast to bool"),
    }
}

macro_rules! truthy_binary {
    ($name:ident, $core:path) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let (x, y) = (truth_operand(py, a)?, truth_operand(py, b)?);
            out(py, Arr::from($core(&x.view(), &y.view()).map_err(shape_err)?))
        }
    };
}
truthy_binary!(logical_and, logic::logical_and);
truthy_binary!(logical_or, logic::logical_or);
truthy_binary!(logical_xor, logic::logical_xor);

macro_rules! bit_binary {
    ($name:ident, $core:path) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let (x, y) = resolve_binary(py, a, b)?;
            let r = match (&x, &y) {
                (Arr::Bool(p), Arr::Bool(q)) => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?),
                _ => int_arms2!(&x, &y, p, q, T => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?), else return Err(not_supported(stringify!($name), x.dtype_name()))),
            };
            out(py, r)
        }
    };
}
bit_binary!(bitwise_and, logic::bitwise_and);
bit_binary!(bitwise_or, logic::bitwise_or);
bit_binary!(bitwise_xor, logic::bitwise_xor);

macro_rules! shift_binary {
    ($name:ident, $core:path) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            let (x, y) = resolve_binary(py, a, b)?;
            let (x, y) = (bool_as_i8(x)?, bool_as_i8(y)?);
            let r = int_arms2!(&x, &y, p, q, T => Arr::from($core(&p.view(), &q.view()).map_err(shape_err)?), else return Err(not_supported(stringify!($name), x.dtype_name())));
            out(py, r)
        }
    };
}
shift_binary!(left_shift, logic::left_shift);
shift_binary!(right_shift, logic::right_shift);

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! reg {
        ($($f:ident),* $(,)?) => {$( m.add_function(wrap_pyfunction!($f, m)?)?; )*};
    }
    reg!(
        sqrt, cbrt, exp, exp2, expm1, log, log2, log10, log1p, sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh,
        arcsinh, arccosh, arctanh, degrees, radians, fabs, rint, floor, ceil, trunc, reciprocal, square, negative, positive,
        sign, absolute, conjugate, signbit, isnan, isinf, isfinite, logical_not, invert, arctan2, hypot, copysign,
        logaddexp, logaddexp2, heaviside, nextafter, spacing, frexp, ldexp, modf, floor_divide, remainder, fmod, divmod,
        power, float_power, maximum, minimum, fmax, fmin, divide, gcd, lcm, equal, not_equal, less, less_equal, greater,
        greater_equal, logical_and, logical_or, logical_xor, bitwise_and, bitwise_or, bitwise_xor, left_shift, right_shift
    );
    m.add("true_divide", wrap_pyfunction!(divide, m)?)?;
    Ok(())
}
