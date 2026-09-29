use crate::error::{OpError, ShapeError};
use crate::ndarray::NdArray;
use crate::reductions::FloatIsh;
use crate::ufunc::{map, zip_with};
use crate::view::ArrayView;
use num_traits::Float;
use std::cell::Cell;

pub trait Arith: FloatIsh {
    fn abs_(self) -> Self;
    fn neg_(self) -> Self;
    fn square_(self) -> Self;
    fn sign_(self) -> Self;
    fn floor_div_(self, rhs: Self) -> Self;
    fn mod_(self, rhs: Self) -> Self;
    fn pow_(self, rhs: Self) -> Option<Self>;
}

macro_rules! impl_arith_signed {
    ($($t:ty),*) => {$(
        impl Arith for $t {
            fn abs_(self) -> Self { self.wrapping_abs() }
            fn neg_(self) -> Self { self.wrapping_neg() }
            fn square_(self) -> Self { self.wrapping_mul(self) }
            fn sign_(self) -> Self { self.signum() }
            fn floor_div_(self, rhs: Self) -> Self {
                if rhs == 0 {
                    return 0;
                }
                let q = self.wrapping_div(rhs);
                if self.wrapping_rem(rhs) != 0 && ((self < 0) != (rhs < 0)) { q.wrapping_sub(1) } else { q }
            }
            fn mod_(self, rhs: Self) -> Self {
                if rhs == 0 {
                    return 0;
                }
                let r = self.wrapping_rem(rhs);
                if r != 0 && ((r < 0) != (rhs < 0)) { r.wrapping_add(rhs) } else { r }
            }
            fn pow_(self, rhs: Self) -> Option<Self> {
                if rhs < 0 {
                    return None;
                }
                let (mut base, mut exp, mut acc) = (self, rhs as u64, 1 as $t);
                while exp > 0 {
                    if exp & 1 == 1 {
                        acc = acc.wrapping_mul(base);
                    }
                    base = base.wrapping_mul(base);
                    exp >>= 1;
                }
                Some(acc)
            }
        }
    )*};
}
impl_arith_signed!(i8, i16, i32, i64);

macro_rules! impl_arith_unsigned {
    ($($t:ty),*) => {$(
        impl Arith for $t {
            fn abs_(self) -> Self { self }
            fn neg_(self) -> Self { self.wrapping_neg() }
            fn square_(self) -> Self { self.wrapping_mul(self) }
            fn sign_(self) -> Self { if self == 0 { 0 } else { 1 } }
            fn floor_div_(self, rhs: Self) -> Self { if rhs == 0 { 0 } else { self / rhs } }
            fn mod_(self, rhs: Self) -> Self { if rhs == 0 { 0 } else { self % rhs } }
            fn pow_(self, rhs: Self) -> Option<Self> {
                let (mut base, mut exp, mut acc) = (self, rhs as u64, 1 as $t);
                while exp > 0 {
                    if exp & 1 == 1 {
                        acc = acc.wrapping_mul(base);
                    }
                    base = base.wrapping_mul(base);
                    exp >>= 1;
                }
                Some(acc)
            }
        }
    )*};
}
impl_arith_unsigned!(u8, u16, u32, u64);

macro_rules! impl_arith_float {
    ($($t:ty),*) => {$(
        impl Arith for $t {
            fn abs_(self) -> Self { self.abs() }
            fn neg_(self) -> Self { -self }
            fn square_(self) -> Self { self * self }
            fn sign_(self) -> Self {
                if self.is_nan() { self } else if self > 0.0 { 1.0 } else if self < 0.0 { -1.0 } else { 0.0 }
            }
            fn floor_div_(self, rhs: Self) -> Self {
                if rhs == 0.0 {
                    return self / rhs;
                }
                let m = self % rhs;
                let mut div = (self - m) / rhs;
                if m != 0.0 && ((rhs < 0.0) != (m < 0.0)) {
                    div -= 1.0;
                }
                if div != 0.0 {
                    let mut f = div.floor();
                    if div - f > 0.5 {
                        f += 1.0;
                    }
                    f
                } else {
                    (0.0 as $t).copysign(self / rhs)
                }
            }
            fn mod_(self, rhs: Self) -> Self {
                if rhs == 0.0 {
                    return Self::NAN;
                }
                let m = self % rhs;
                if m != 0.0 {
                    if (rhs < 0.0) != (m < 0.0) { m + rhs } else { m }
                } else {
                    (0.0 as $t).copysign(rhs)
                }
            }
            fn pow_(self, rhs: Self) -> Option<Self> { Some(self.powf(rhs)) }
        }
    )*};
}
impl_arith_float!(f32, f64);

impl Arith for half::f16 {
    fn abs_(self) -> Self { half::f16::from_f32(f32::from(self).abs_()) }
    fn neg_(self) -> Self { -self }
    fn square_(self) -> Self { half::f16::from_f32(f32::from(self).square_()) }
    fn sign_(self) -> Self { half::f16::from_f32(f32::from(self).sign_()) }
    fn floor_div_(self, rhs: Self) -> Self { half::f16::from_f32(f32::from(self).floor_div_(f32::from(rhs))) }
    fn mod_(self, rhs: Self) -> Self { half::f16::from_f32(f32::from(self).mod_(f32::from(rhs))) }
    fn pow_(self, rhs: Self) -> Option<Self> { Some(half::f16::from_f32(f32::from(self).powf(f32::from(rhs)))) }
}

macro_rules! float_unary {
    ($($name:ident => $f:expr),* $(,)?) => {$(
        pub fn $name<T: Float>(a: &ArrayView<T>) -> NdArray<T> {
            map(a, $f)
        }
    )*};
}

float_unary! {
    sqrt => |x: T| x.sqrt(),
    cbrt => |x: T| x.cbrt(),
    exp => |x: T| x.exp(),
    exp2 => |x: T| x.exp2(),
    expm1 => |x: T| x.exp_m1(),
    log => |x: T| x.ln(),
    log2 => |x: T| x.log2(),
    log10 => |x: T| x.log10(),
    log1p => |x: T| x.ln_1p(),
    sin => |x: T| x.sin(),
    cos => |x: T| x.cos(),
    tan => |x: T| x.tan(),
    arcsin => |x: T| x.asin(),
    arccos => |x: T| x.acos(),
    arctan => |x: T| x.atan(),
    sinh => |x: T| x.sinh(),
    cosh => |x: T| x.cosh(),
    tanh => |x: T| x.tanh(),
    arcsinh => |x: T| x.asinh(),
    arccosh => |x: T| x.acosh(),
    arctanh => |x: T| x.atanh(),
    floor => |x: T| x.floor(),
    ceil => |x: T| x.ceil(),
    trunc => |x: T| x.trunc(),
    reciprocal => |x: T| x.recip(),
    degrees => |x: T| x.to_degrees(),
    radians => |x: T| x.to_radians(),
}

pub fn rint<T: Float>(a: &ArrayView<T>) -> NdArray<T> {
    map(a, |x| {
        let r = x.round();
        let two = T::one() + T::one();
        if (x - x.trunc()).abs() == T::one() / two && (r / two).fract() != T::zero() {
            r - x.signum()
        } else {
            r
        }
    })
}

pub fn abs<T: Arith>(a: &ArrayView<T>) -> NdArray<T> {
    map(a, T::abs_)
}

pub fn negative<T: Arith>(a: &ArrayView<T>) -> NdArray<T> {
    map(a, T::neg_)
}

pub fn square<T: Arith>(a: &ArrayView<T>) -> NdArray<T> {
    map(a, T::square_)
}

pub fn sign<T: Arith>(a: &ArrayView<T>) -> NdArray<T> {
    map(a, T::sign_)
}

pub trait Divide: Copy {
    fn divide(self, rhs: Self) -> Self;
}

macro_rules! divide_float {
    ($($t:ty),*) => {$(
        impl Divide for $t {
            fn divide(self, rhs: Self) -> Self { self / rhs }
        }
        impl Divide for num_complex::Complex<$t> {
            fn divide(self, b: Self) -> Self {
                let (ar, ai, br, bi) = (self.re, self.im, b.re, b.im);
                let (abs_br, abs_bi) = (br.abs(), bi.abs());
                if abs_br >= abs_bi {
                    if abs_br == 0.0 && abs_bi == 0.0 {
                        return num_complex::Complex::new(ar / abs_br, ai / abs_bi);
                    }
                    let rat = bi / br;
                    let scl = 1.0 / (br + bi * rat);
                    num_complex::Complex::new((ar + ai * rat) * scl, (ai - ar * rat) * scl)
                } else {
                    let rat = br / bi;
                    let scl = 1.0 / (bi + br * rat);
                    num_complex::Complex::new((ar * rat + ai) * scl, (ai * rat - ar) * scl)
                }
            }
        }
    )*};
}
divide_float!(f32, f64);

impl Divide for half::f16 {
    fn divide(self, rhs: Self) -> Self {
        self / rhs
    }
}

pub fn divide<T: Divide>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    zip_with(a, b, T::divide)
}

pub fn arctan2<T: Float>(y: &ArrayView<T>, x: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    zip_with(y, x, |a, b| a.atan2(b))
}

pub fn hypot<T: Float>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    zip_with(a, b, |x, y| x.hypot(y))
}

pub fn copysign<T: Float>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    zip_with(a, b, |x, y| x.copysign(y))
}

pub fn fmod<T: Float>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    zip_with(a, b, |x, y| x % y)
}

pub fn maximum<T: FloatIsh>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    zip_with(a, b, |x, y| if x.is_nan_ish() || x >= y { x } else { y })
}

pub fn minimum<T: FloatIsh>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    zip_with(a, b, |x, y| if x.is_nan_ish() || x <= y { x } else { y })
}

pub fn fmax<T: FloatIsh>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    zip_with(a, b, |x, y| if y.is_nan_ish() || x >= y { x } else { y })
}

pub fn fmin<T: FloatIsh>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    zip_with(a, b, |x, y| if y.is_nan_ish() || x <= y { x } else { y })
}

pub fn floor_divide<T: Arith>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    zip_with(a, b, T::floor_div_)
}

pub fn remainder<T: Arith>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    zip_with(a, b, T::mod_)
}

pub fn power<T: Arith>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, OpError> {
    let failed = Cell::new(false);
    let out = zip_with(a, b, |x, y| {
        x.pow_(y).unwrap_or_else(|| {
            failed.set(true);
            x
        })
    })?;
    if failed.get() {
        return Err(OpError::NegativeIntegerPower);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected_pow(mut base: u64, mut exp: u64) -> u64 {
        let mut acc = 1u64;
        while exp > 0 {
            if exp & 1 == 1 {
                acc = acc.wrapping_mul(base);
            }
            base = base.wrapping_mul(base);
            exp >>= 1;
        }
        acc
    }

    fn arr<T>(data: Vec<T>) -> NdArray<T> {
        let n = data.len();
        NdArray::from_vec(data, &[n]).unwrap()
    }

    fn same(got: &NdArray<f64>, want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.as_slice().iter().zip(want) {
            assert!(
                (g.is_nan() && w.is_nan()) || (g == w && g.is_sign_negative() == w.is_sign_negative()) || (g - w).abs() < 1e-12,
                "got {g}, want {w}"
            );
        }
    }

    #[test]
    fn rounding_family_matches_numpy_including_negative_zero_and_ties() {
        let f = arr(vec![-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, f64::NAN, f64::INFINITY, -0.0]);
        same(&rint(&f.view()), &[-2.0, -2.0, -0.0, 0.0, 2.0, 2.0, f64::NAN, f64::INFINITY, -0.0]);
        same(&floor(&f.view()), &[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, f64::NAN, f64::INFINITY, -0.0]);
        same(&ceil(&f.view()), &[-2.0, -1.0, -0.0, 1.0, 2.0, 3.0, f64::NAN, f64::INFINITY, -0.0]);
        same(&trunc(&f.view()), &[-2.0, -1.0, -0.0, 0.0, 1.0, 2.0, f64::NAN, f64::INFINITY, -0.0]);
        same(&sign(&f.view()), &[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, f64::NAN, 1.0, 0.0]);
    }

    #[test]
    fn float16_math_rounds_through_f32_like_numpy() {
        use half::f16;
        let x = arr(vec![f16::from_f32(2.0), f16::from_f32(-1.5), f16::from_f32(0.1)]);
        let s = sqrt(&arr(vec![f16::from_f32(4.0), f16::from_f32(2.0)]).view());
        assert_eq!(s.as_slice()[0], f16::from_f32(2.0));
        assert_eq!(s.as_slice()[1], f16::from_f32(2.0_f32.sqrt()));
        assert_eq!(abs(&x.view()).as_slice()[1], f16::from_f32(1.5));
        assert_eq!(floor_divide(&x.view(), &arr(vec![f16::from_f32(0.75); 3]).view()).unwrap().as_slice()[0], f16::from_f32(2.0));
        assert_eq!(rint(&arr(vec![f16::from_f32(2.5), f16::from_f32(3.5)]).view()).as_slice(), &[f16::from_f32(2.0), f16::from_f32(4.0)]);
    }

    #[test]
    fn integer_power_wraps_with_a_full_width_exponent() {
        let base = arr(vec![3u64, 2]);
        let exp = arr(vec![u64::MAX, 64]);
        let r = power(&base.view(), &exp.view()).unwrap();
        assert_eq!(r.as_slice(), &[expected_pow(3, u64::MAX), 0]);
    }

    #[test]
    fn transcendental_functions_match_numpy() {
        same(&sqrt(&arr(vec![-1.0, 4.0, 0.0]).view()), &[f64::NAN, 2.0, 0.0]);
        same(&log(&arr(vec![0.0, -1.0, 1.0]).view()), &[f64::NEG_INFINITY, f64::NAN, 0.0]);
        same(&log2(&arr(vec![8.0]).view()), &[3.0]);
        same(&cbrt(&arr(vec![-8.0]).view()), &[-2.0]);
        same(&exp2(&arr(vec![3.0]).view()), &[8.0]);
        same(&expm1(&arr(vec![1e-10]).view()), &[1e-10]);
        same(&log1p(&arr(vec![1e-10]).view()), &[1e-10]);
        same(&sinh(&arr(vec![1.0]).view()), &[1.1752011936438014]);
        same(&arcsinh(&arr(vec![1.0]).view()), &[0.881373587019543]);
        same(&arccosh(&arr(vec![2.0]).view()), &[1.3169578969248168]);
        same(&arctanh(&arr(vec![0.5]).view()), &[0.5493061443340549]);
        same(&arcsin(&arr(vec![2.0]).view()), &[f64::NAN]);
        same(&tan(&arr(vec![1.0]).view()), &[1.5574077246549023]);
        same(&degrees(&arr(vec![std::f64::consts::PI]).view()), &[180.0]);
        same(&radians(&arr(vec![180.0]).view()), &[std::f64::consts::PI]);
        same(&reciprocal(&arr(vec![2.0, 0.0]).view()), &[0.5, f64::INFINITY]);
    }

    #[test]
    fn integer_abs_negative_square_wrap_like_numpy() {
        assert_eq!(abs(&arr(vec![-128i8]).view()).as_slice(), &[-128]);
        assert_eq!(negative(&arr(vec![-128i8]).view()).as_slice(), &[-128]);
        assert_eq!(abs(&arr(vec![3i64, -4]).view()).as_slice(), &[3, 4]);
        assert_eq!(square(&arr(vec![-106i8]).view()).as_slice(), &[-28]);
        assert_eq!(negative(&arr(vec![1u8]).view()).as_slice(), &[255]);
        assert_eq!(sign(&arr(vec![-5i32, 0, 9]).view()).as_slice(), &[-1, 0, 1]);
    }

    #[test]
    fn binary_float_functions_match_numpy() {
        let y = arr(vec![1.0, -1.0, 0.0, 0.0]);
        let x = arr(vec![1.0, -1.0, -1.0, 0.0]);
        same(&arctan2(&y.view(), &x.view()).unwrap(), &[std::f64::consts::FRAC_PI_4, -2.356194490192345, std::f64::consts::PI, 0.0]);
        same(&hypot(&arr(vec![3.0]).view(), &arr(vec![4.0]).view()).unwrap(), &[5.0]);
        same(&copysign(&arr(vec![3.0]).view(), &arr(vec![-0.0]).view()).unwrap(), &[-3.0]);
    }

    #[test]
    fn copysign_keeps_the_sign_of_nan_and_zero() {
        let nan = arr(vec![f64::NAN, f64::NAN, 0.0, 0.0]);
        let sgn = arr(vec![-1.0, 1.0, -1.0, 1.0]);
        let r = copysign(&nan.view(), &sgn.view()).unwrap();
        let bits: Vec<bool> = r.as_slice().iter().map(|x| x.is_sign_negative()).collect();
        assert_eq!(bits, vec![true, false, true, false]);
    }

    #[test]
    fn maximum_propagates_nan_but_fmax_ignores_it() {
        let a = arr(vec![1.0, f64::NAN, 3.0]);
        let b = arr(vec![f64::NAN, 2.0, 1.0]);
        same(&maximum(&a.view(), &b.view()).unwrap(), &[f64::NAN, f64::NAN, 3.0]);
        same(&fmax(&a.view(), &b.view()).unwrap(), &[1.0, 2.0, 3.0]);
        let c = arr(vec![1.0, f64::NAN]);
        let d = arr(vec![0.0, 0.0]);
        same(&minimum(&c.view(), &d.view()).unwrap(), &[0.0, f64::NAN]);
        let e = arr(vec![f64::NAN, f64::NAN]);
        let g = arr(vec![f64::NAN, 1.0]);
        same(&fmin(&e.view(), &g.view()).unwrap(), &[f64::NAN, 1.0]);
    }

    #[test]
    fn floor_divide_and_remainder_follow_the_divisor_sign() {
        let a = arr(vec![-7i32, 7, -7, 7]);
        let b = arr(vec![3i32, -3, -3, 3]);
        assert_eq!(remainder(&a.view(), &b.view()).unwrap().as_slice(), &[2, -2, -1, 1]);
        assert_eq!(
            floor_divide(&arr(vec![-7i32, 7, -7]).view(), &arr(vec![2, -2, 0]).view()).unwrap().as_slice(),
            &[-4, -4, 0]
        );
        same(
            &remainder(&arr(vec![-7.5, 7.5]).view(), &arr(vec![2.0, -2.0]).view()).unwrap(),
            &[0.5, -0.5],
        );
        same(
            &floor_divide(&arr(vec![-7.5, 7.5]).view(), &arr(vec![2.0, 0.0]).view()).unwrap(),
            &[-4.0, f64::INFINITY],
        );
        same(&fmod(&arr(vec![-7.5]).view(), &arr(vec![2.0]).view()).unwrap(), &[-1.5]);
    }

    #[test]
    fn power_wraps_for_ints_and_rejects_negative_integer_exponents() {
        assert_eq!(power(&arr(vec![2i64, 3]).view(), &arr(vec![3, 4]).view()).unwrap().as_slice(), &[8, 81]);
        assert_eq!(
            power(&arr(vec![2i64]).view(), &arr(vec![-1]).view()).unwrap_err(),
            OpError::NegativeIntegerPower
        );
        same(
            &power(&arr(vec![2.0, 0.0, -8.0]).view(), &arr(vec![3.0, 0.0, 1.0 / 3.0]).view()).unwrap(),
            &[8.0, 1.0, f64::NAN],
        );
    }

    #[test]
    fn integer_power_handles_zero_exponent_and_narrow_types_like_numpy() {
        assert_eq!(power(&arr(vec![0i8, 7, -128, 2]).view(), &arr(vec![0i8, 0, 0, 7]).view()).unwrap().as_slice(), &[1, 1, 1, -128]);
        assert_eq!(power(&arr(vec![3i8]).view(), &arr(vec![5i8]).view()).unwrap().as_slice(), &[-13]);
        assert_eq!(power(&arr(vec![0i16, 5]).view(), &arr(vec![0i16, 3]).view()).unwrap().as_slice(), &[1, 125]);
        assert_eq!(power(&arr(vec![0i32, 2]).view(), &arr(vec![0i32, 31]).view()).unwrap().as_slice(), &[1, i32::MIN]);
        assert_eq!(power(&arr(vec![0u8, 2]).view(), &arr(vec![0u8, 9]).view()).unwrap().as_slice(), &[1, 0]);
    }

    #[test]
    fn complex_division_uses_smiths_algorithm_like_numpy() {
        use num_complex::Complex;
        let a = arr(vec![Complex::new(1.0, 1.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(f64::INFINITY, 1.0)]);
        let b = arr(vec![Complex::new(1.0, -1.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(2.0, 0.0)]);
        let r = divide(&a.view(), &b.view()).unwrap();
        assert_eq!(r.as_slice()[0], Complex::new(0.0, 1.0));
        assert!(r.as_slice()[1].re == f64::INFINITY && r.as_slice()[1].im.is_nan());
        assert!(r.as_slice()[2].re.is_nan() && r.as_slice()[2].im.is_nan());
        assert_eq!(r.as_slice()[3].re, f64::INFINITY);
        assert_eq!(divide(&arr(vec![1.0f32]).view(), &arr(vec![4.0]).view()).unwrap().as_slice(), &[0.25f32]);
    }

    #[test]
    fn math_functions_work_on_f32_too() {
        let a = arr(vec![4.0f32, 9.0]);
        assert_eq!(sqrt(&a.view()).as_slice(), &[2.0f32, 3.0]);
    }
}
