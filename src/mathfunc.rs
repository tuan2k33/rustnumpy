use crate::error::ShapeError;
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
                Some(self.wrapping_pow(rhs.min(u32::MAX as $t) as u32))
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
                Some(self.wrapping_pow(rhs.min(u32::MAX as $t) as u32))
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

pub fn arctan2<T: Float>(y: &ArrayView<T>, x: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    zip_with(y, x, |a, b| a.atan2(b))
}

pub fn hypot<T: Float>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    zip_with(a, b, |x, y| x.hypot(y))
}

pub fn copysign<T: Float>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    zip_with(a, b, |x, y| x.abs() * if y.is_sign_negative() { -T::one() } else { T::one() })
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

pub fn power<T: Arith>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    let failed = Cell::new(false);
    let out = zip_with(a, b, |x, y| {
        x.pow_(y).unwrap_or_else(|| {
            failed.set(true);
            x
        })
    })?;
    if failed.get() {
        return Err(ShapeError::NegativeIntegerPower);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

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
            ShapeError::NegativeIntegerPower
        );
        same(
            &power(&arr(vec![2.0, 0.0, -8.0]).view(), &arr(vec![3.0, 0.0, 1.0 / 3.0]).view()).unwrap(),
            &[8.0, 1.0, f64::NAN],
        );
    }

    #[test]
    fn math_functions_work_on_f32_too() {
        let a = arr(vec![4.0f32, 9.0]);
        assert_eq!(sqrt(&a.view()).as_slice(), &[2.0f32, 3.0]);
    }
}
