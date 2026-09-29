use crate::dispatch::{accumulate, WrapAdd, WrapMul};
use crate::error::ShapeError;
use crate::ndarray::NdArray;
use crate::reductions::FloatIsh;
use crate::shape::broadcast_shapes;
use crate::view::ArrayView;
use num_complex::Complex;

pub fn zip_map<T: Copy, O>(a: &ArrayView<T>, b: &ArrayView<T>, f: impl Fn(T, T) -> O) -> Result<NdArray<O>, ShapeError> {
    let shape = broadcast_shapes(a.shape(), b.shape())
        .ok_or_else(|| ShapeError::NotBroadcastable { lhs: a.shape().to_vec(), rhs: b.shape().to_vec() })?;
    let (a_b, b_b) = (a.broadcast_to(&shape)?, b.broadcast_to(&shape)?);
    let data: Vec<O> = a_b.iter().zip(b_b.iter()).map(|(x, y)| f(x, y)).collect();
    NdArray::from_vec(data, &shape)
}

pub fn map_to<T: Copy, O>(a: &ArrayView<T>, f: impl Fn(T) -> O) -> NdArray<O> {
    let data: Vec<O> = a.iter().map(f).collect();
    NdArray::from_vec(data, a.shape()).expect("same element count")
}

pub fn equal<T: Copy + PartialEq>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<bool>, ShapeError> {
    zip_map(a, b, |x, y| x == y)
}

pub fn not_equal<T: Copy + PartialEq>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<bool>, ShapeError> {
    zip_map(a, b, |x, y| x != y)
}

pub fn less<T: Copy + PartialOrd>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<bool>, ShapeError> {
    zip_map(a, b, |x, y| x < y)
}

pub fn less_equal<T: Copy + PartialOrd>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<bool>, ShapeError> {
    zip_map(a, b, |x, y| x <= y)
}

pub fn greater<T: Copy + PartialOrd>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<bool>, ShapeError> {
    zip_map(a, b, |x, y| x > y)
}

pub fn greater_equal<T: Copy + PartialOrd>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<bool>, ShapeError> {
    zip_map(a, b, |x, y| x >= y)
}

pub trait Truthy: Copy {
    fn truthy(self) -> bool;
}

macro_rules! truthy_num {
    ($($t:ty),*) => {$(impl Truthy for $t { fn truthy(self) -> bool { self != (0 as $t) } })*};
}
truthy_num!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

impl Truthy for half::f16 {
    fn truthy(self) -> bool {
        self != half::f16::ZERO
    }
}

impl Truthy for bool {
    fn truthy(self) -> bool {
        self
    }
}

impl<F: Copy + PartialEq + Default> Truthy for Complex<F> {
    fn truthy(self) -> bool {
        self.re != F::default() || self.im != F::default()
    }
}

pub fn logical_and<T: Truthy>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<bool>, ShapeError> {
    zip_map(a, b, |x, y| x.truthy() && y.truthy())
}

pub fn logical_or<T: Truthy>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<bool>, ShapeError> {
    zip_map(a, b, |x, y| x.truthy() || y.truthy())
}

pub fn logical_xor<T: Truthy>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<bool>, ShapeError> {
    zip_map(a, b, |x, y| x.truthy() != y.truthy())
}

pub fn logical_not<T: Truthy>(a: &ArrayView<T>) -> NdArray<bool> {
    map_to(a, |x| !x.truthy())
}

pub fn any<T: Truthy>(a: &ArrayView<T>) -> bool {
    a.iter().any(Truthy::truthy)
}

pub fn all<T: Truthy>(a: &ArrayView<T>) -> bool {
    a.iter().all(Truthy::truthy)
}

pub fn count_nonzero<T: Truthy>(a: &ArrayView<T>) -> usize {
    a.iter().filter(|x| x.truthy()).count()
}

pub trait BitOps: Copy {
    fn bit_and(self, rhs: Self) -> Self;
    fn bit_or(self, rhs: Self) -> Self;
    fn bit_xor(self, rhs: Self) -> Self;
    fn bit_not(self) -> Self;
}

macro_rules! bitops_int {
    ($($t:ty),*) => {$(impl BitOps for $t {
        fn bit_and(self, r: Self) -> Self { self & r }
        fn bit_or(self, r: Self) -> Self { self | r }
        fn bit_xor(self, r: Self) -> Self { self ^ r }
        fn bit_not(self) -> Self { !self }
    })*};
}
bitops_int!(bool, i8, i16, i32, i64, u8, u16, u32, u64);

pub trait Shiftable: BitOps {
    fn shift_left(self, by: Self) -> Self;
    fn shift_right(self, by: Self) -> Self;
}

macro_rules! shift_signed {
    ($($t:ty),*) => {$(impl Shiftable for $t {
        fn shift_left(self, by: Self) -> Self {
            if by < 0 || by >= <$t>::BITS as $t { 0 } else { self << by }
        }
        fn shift_right(self, by: Self) -> Self {
            if by < 0 || by >= <$t>::BITS as $t { if self < 0 { -1 } else { 0 } } else { self >> by }
        }
    })*};
}
shift_signed!(i8, i16, i32, i64);

macro_rules! shift_unsigned {
    ($($t:ty),*) => {$(impl Shiftable for $t {
        fn shift_left(self, by: Self) -> Self {
            if by >= <$t>::BITS as $t { 0 } else { self << by }
        }
        fn shift_right(self, by: Self) -> Self {
            if by >= <$t>::BITS as $t { 0 } else { self >> by }
        }
    })*};
}
shift_unsigned!(u8, u16, u32, u64);

pub fn bitwise_and<T: BitOps>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    crate::ufunc::zip_with(a, b, T::bit_and)
}

pub fn bitwise_or<T: BitOps>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    crate::ufunc::zip_with(a, b, T::bit_or)
}

pub fn bitwise_xor<T: BitOps>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    crate::ufunc::zip_with(a, b, T::bit_xor)
}

pub fn invert<T: BitOps>(a: &ArrayView<T>) -> NdArray<T> {
    crate::ufunc::map(a, T::bit_not)
}

pub fn left_shift<T: Shiftable>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    crate::ufunc::zip_with(a, b, T::shift_left)
}

pub fn right_shift<T: Shiftable>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError> {
    crate::ufunc::zip_with(a, b, T::shift_right)
}

pub trait FloatClass: Copy {
    fn nan(self) -> bool;
    fn inf(self) -> bool;
    fn finite(self) -> bool;
}

macro_rules! float_class {
    ($($t:ty),*) => {$(
        impl FloatClass for $t {
            fn nan(self) -> bool { self.is_nan() }
            fn inf(self) -> bool { self.is_infinite() }
            fn finite(self) -> bool { self.is_finite() }
        }
        impl FloatClass for Complex<$t> {
            fn nan(self) -> bool { self.re.is_nan() || self.im.is_nan() }
            fn inf(self) -> bool { self.re.is_infinite() || self.im.is_infinite() }
            fn finite(self) -> bool { self.re.is_finite() && self.im.is_finite() }
        }
    )*};
}
float_class!(f32, f64);

impl FloatClass for half::f16 {
    fn nan(self) -> bool { self.is_nan() }
    fn inf(self) -> bool { self.is_infinite() }
    fn finite(self) -> bool { self.is_finite() }
}

macro_rules! int_class {
    ($($t:ty),*) => {$(impl FloatClass for $t {
        fn nan(self) -> bool { false }
        fn inf(self) -> bool { false }
        fn finite(self) -> bool { true }
    })*};
}
int_class!(bool, i8, i16, i32, i64, u8, u16, u32, u64);

pub fn isnan<T: FloatClass>(a: &ArrayView<T>) -> NdArray<bool> {
    map_to(a, FloatClass::nan)
}

pub fn isinf<T: FloatClass>(a: &ArrayView<T>) -> NdArray<bool> {
    map_to(a, FloatClass::inf)
}

pub fn isfinite<T: FloatClass>(a: &ArrayView<T>) -> NdArray<bool> {
    map_to(a, FloatClass::finite)
}

pub fn signbit<T: num_traits::Float>(a: &ArrayView<T>) -> NdArray<bool> {
    map_to(a, |x| x.is_sign_negative())
}

fn lane_extreme<T: FloatIsh>(lane: &[T], want_max: bool) -> usize {
    let mut best = 0;
    for (i, &x) in lane.iter().enumerate().skip(1) {
        let b = lane[best];
        if b.is_nan_ish() {
            break;
        }
        if x.is_nan_ish() || (want_max && x > b) || (!want_max && x < b) {
            best = i;
        }
    }
    best
}

pub fn arg_extreme<T: FloatIsh>(a: &ArrayView<T>, axis: Option<usize>, want_max: bool) -> Result<NdArray<usize>, ShapeError> {
    let owned = a.to_owned();
    if owned.is_empty() && axis.is_none() {
        return Err(ShapeError::ReduceNoIdentity);
    }
    match axis {
        None => NdArray::from_vec(vec![lane_extreme(owned.as_slice(), want_max)], &[]),
        Some(ax) => {
            if ax >= owned.ndim() {
                return Err(ShapeError::AxisOutOfBounds { axis: ax, ndim: owned.ndim() });
            }
            let n = owned.shape()[ax];
            if n == 0 {
                return Err(ShapeError::ReduceNoIdentity);
            }
            let outer: usize = owned.shape()[..ax].iter().product();
            let inner: usize = owned.shape()[ax + 1..].iter().product();
            let mut out = Vec::with_capacity(outer * inner);
            let mut lane: Vec<T> = Vec::with_capacity(n);
            for o in 0..outer {
                for i in 0..inner {
                    lane.clear();
                    lane.extend((0..n).map(|k| owned.as_slice()[(o * n + k) * inner + i]));
                    out.push(lane_extreme(&lane, want_max));
                }
            }
            let mut shape = owned.shape().to_vec();
            shape.remove(ax);
            NdArray::from_vec(out, &shape)
        }
    }
}

pub fn argmax<T: FloatIsh>(a: &ArrayView<T>, axis: Option<usize>) -> Result<NdArray<usize>, ShapeError> {
    arg_extreme(a, axis, true)
}

pub fn argmin<T: FloatIsh>(a: &ArrayView<T>, axis: Option<usize>) -> Result<NdArray<usize>, ShapeError> {
    arg_extreme(a, axis, false)
}

pub fn cumsum<T: Copy + WrapAdd>(a: &ArrayView<T>, axis: Option<usize>) -> Result<NdArray<T>, ShapeError> {
    match axis {
        None => {
            let flat = a.to_owned().into_shape(&[-1])?;
            accumulate(&flat.view(), 0, |x, y| x.wrap_add(y))
        }
        Some(ax) => accumulate(a, ax, |x, y| x.wrap_add(y)),
    }
}

pub fn cumprod<T: Copy + WrapMul>(a: &ArrayView<T>, axis: Option<usize>) -> Result<NdArray<T>, ShapeError> {
    match axis {
        None => {
            let flat = a.to_owned().into_shape(&[-1])?;
            accumulate(&flat.view(), 0, |x, y| x.wrap_mul(y))
        }
        Some(ax) => accumulate(a, ax, |x, y| x.wrap_mul(y)),
    }
}

pub fn clip<T: FloatIsh>(a: &ArrayView<T>, low: Option<T>, high: Option<T>) -> NdArray<T> {
    map_to(a, |x| {
        let mut v = x;
        if let Some(lo) = low {
            v = if v.is_nan_ish() || v >= lo { v } else { lo };
            if lo.is_nan_ish() {
                v = lo;
            }
        }
        if let Some(hi) = high {
            v = if v.is_nan_ish() || v <= hi { v } else { hi };
            if hi.is_nan_ish() {
                v = hi;
            }
        }
        v
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr<T>(v: Vec<T>) -> NdArray<T> {
        let n = v.len();
        NdArray::from_vec(v, &[n]).unwrap()
    }

    #[test]
    fn comparisons_follow_ieee_for_nan_and_broadcast() {
        let a = arr(vec![1.0, f64::NAN, 3.0]);
        let b = arr(vec![1.0, f64::NAN, 2.0]);
        assert_eq!(equal(&a.view(), &b.view()).unwrap().as_slice(), &[true, false, false]);
        assert_eq!(not_equal(&a.view(), &b.view()).unwrap().as_slice(), &[false, true, true]);
        assert_eq!(less(&a.view(), &b.view()).unwrap().as_slice(), &[false, false, false]);
        assert_eq!(greater_equal(&a.view(), &b.view()).unwrap().as_slice(), &[true, false, true]);
        let col = NdArray::from_vec(vec![1, 2], &[2, 1]).unwrap();
        let row = arr(vec![1, 2, 3]);
        assert_eq!(greater(&col.view(), &row.view()).unwrap().shape(), &[2, 3]);
        assert!(less(&arr(vec![1]).view(), &arr(vec![1, 2]).view()).is_ok());
    }

    #[test]
    fn logical_and_truthiness_follow_numpy() {
        let a = arr(vec![1, 0, 2]);
        let b = arr(vec![1, 1, 0]);
        assert_eq!(logical_and(&a.view(), &b.view()).unwrap().as_slice(), &[true, false, false]);
        assert_eq!(logical_or(&a.view(), &b.view()).unwrap().as_slice(), &[true, true, true]);
        assert_eq!(logical_xor(&a.view(), &b.view()).unwrap().as_slice(), &[false, true, true]);
        assert_eq!(logical_not(&arr(vec![0.0, f64::NAN]).view()).as_slice(), &[true, false]);
        assert!(any(&arr(vec![0.0, f64::NAN]).view()));
        assert!(all(&arr::<f64>(vec![]).view()));
        assert!(!any(&arr::<bool>(vec![]).view()));
        assert_eq!(count_nonzero(&arr(vec![0, 1, 2, 0]).view()), 2);
        assert!(Complex::new(0.0, 1.0).truthy());
    }

    #[test]
    fn bitwise_and_shift_edge_cases_match_numpy() {
        assert_eq!(left_shift(&arr(vec![1i8, 1]).view(), &arr(vec![3i8, 9]).view()).unwrap().as_slice(), &[8, 0]);
        assert_eq!(right_shift(&arr(vec![-8i64, -8]).view(), &arr(vec![1, 70]).view()).unwrap().as_slice(), &[-4, -1]);
        assert_eq!(invert(&arr(vec![true, false]).view()).as_slice(), &[false, true]);
        assert_eq!(invert(&arr(vec![5u8]).view()).as_slice(), &[250]);
        assert_eq!(bitwise_xor(&arr(vec![6i32]).view(), &arr(vec![3]).view()).unwrap().as_slice(), &[5]);
        assert_eq!(bitwise_and(&arr(vec![true, true]).view(), &arr(vec![true, false]).view()).unwrap().as_slice(), &[true, false]);
    }

    #[test]
    fn classification_predicates() {
        let z = arr(vec![Complex::new(f64::NAN, 1.0), Complex::new(1.0, 0.0), Complex::new(f64::INFINITY, -1.0)]);
        assert_eq!(isnan(&z.view()).as_slice(), &[true, false, false]);
        assert_eq!(isinf(&z.view()).as_slice(), &[false, false, true]);
        assert_eq!(isfinite(&z.view()).as_slice(), &[false, true, false]);
        assert_eq!(signbit(&arr(vec![-0.0, 0.0, -1.0]).view()).as_slice(), &[true, false, true]);
        assert_eq!(isnan(&arr(vec![1, 2]).view()).as_slice(), &[false, false]);
    }

    #[test]
    fn argmax_argmin_pick_first_extreme_and_nan_wins() {
        let a = arr(vec![1.0, f64::NAN, 3.0, f64::NAN]);
        assert_eq!(argmax(&a.view(), None).unwrap().as_slice(), &[1]);
        assert_eq!(argmin(&a.view(), None).unwrap().as_slice(), &[1]);
        assert_eq!(argmin(&arr(vec![2, 1, 1, 0, 0]).view(), None).unwrap().as_slice(), &[3]);
        let m = NdArray::from_vec(vec![1, 5, 7, 2], &[2, 2]).unwrap();
        assert_eq!(argmax(&m.view(), Some(0)).unwrap().as_slice(), &[1, 0]);
        assert_eq!(argmax(&m.view(), Some(1)).unwrap().as_slice(), &[1, 0]);
        assert!(argmax(&arr::<f64>(vec![]).view(), None).is_err());
        assert!(argmax(&m.view(), Some(2)).is_err());
    }

    #[test]
    fn cumulative_ops_and_clip_match_numpy() {
        let m = NdArray::from_vec(vec![1, 2, 3, 4], &[2, 2]).unwrap();
        assert_eq!(cumsum(&m.view(), None).unwrap().as_slice(), &[1, 3, 6, 10]);
        assert_eq!(cumsum(&m.view(), Some(0)).unwrap().as_slice(), &[1, 2, 4, 6]);
        assert_eq!(cumprod(&arr(vec![1, 2, 3, 4]).view(), None).unwrap().as_slice(), &[1, 2, 6, 24]);
        assert_eq!(cumsum(&arr(vec![i8::MAX, 1]).view(), None).unwrap().as_slice(), &[127, -128]);
        let c = clip(&arr(vec![1.0, 5.0, f64::NAN]).view(), Some(2.0), Some(4.0));
        assert_eq!(&c.as_slice()[..2], &[2.0, 4.0]);
        assert!(c.as_slice()[2].is_nan());
        assert_eq!(clip(&arr(vec![1, 5, 9]).view(), Some(6), Some(4)).as_slice(), &[4, 4, 4]);
        assert_eq!(clip(&arr(vec![1, 5, 9]).view(), None, Some(4)).as_slice(), &[1, 4, 4]);
    }
}
