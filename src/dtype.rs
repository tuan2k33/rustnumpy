#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Bool,

    Uint(u8),

    Int(u8),

    Float(u8),

    Complex(u8),
}

impl Kind {

    fn rank(self) -> u8 {
        match self {
            Kind::Bool => 0,
            Kind::Uint(_) | Kind::Int(_) => 1,
            Kind::Float(_) => 2,
            Kind::Complex(_) => 3,
        }
    }
}

fn float_mantissa_bits(float_bits: u8) -> u8 {
    match float_bits {
        16 => 11,
        32 => 24,
        64 => 53,
        _ => 53,
    }
}

const PROMOTION_ORDER: [Kind; 14] = [
    Kind::Bool,
    Kind::Uint(8),
    Kind::Int(8),
    Kind::Uint(16),
    Kind::Int(16),
    Kind::Uint(32),
    Kind::Int(32),
    Kind::Uint(64),
    Kind::Int(64),
    Kind::Float(16),
    Kind::Float(32),
    Kind::Float(64),
    Kind::Complex(32),
    Kind::Complex(64),
];

pub fn common_dtype(a: Kind, b: Kind) -> Kind {
    let safe = |from: Kind, to: Kind| matches!(can_cast(from, to), CastSafety::Equivalent | CastSafety::Safe);
    PROMOTION_ORDER.into_iter().find(|&t| safe(a, t) && safe(b, t)).unwrap_or(if a.rank().max(b.rank()) == 3 {
        Kind::Complex(64)
    } else {
        Kind::Float(64)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Weak {
    Int,
    Float,
    Complex,
}

impl Weak {
    fn rank(self) -> u8 {
        self as u8 + 1
    }
}

pub fn with_weak(k: Kind, w: Weak) -> Kind {
    match (k, w) {
        _ if k.rank() >= w.rank() => k,
        (Kind::Float(bits), Weak::Complex) => Kind::Complex(bits.max(32)),
        (_, Weak::Int) => Kind::Int(64),
        (_, Weak::Float) => Kind::Float(64),
        (_, Weak::Complex) => Kind::Complex(64),
    }
}

pub fn result_type(strong: impl IntoIterator<Item = Kind>, weak: impl IntoIterator<Item = Weak>) -> Option<Kind> {
    let strong = strong.into_iter().reduce(common_dtype);
    match (strong, weak.into_iter().max()) {
        (Some(k), Some(w)) => Some(with_weak(k, w)),
        (Some(k), None) => Some(k),
        (None, Some(w)) => Some(with_weak(Kind::Bool, w)),
        (None, None) => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastSafety {

    Equivalent,

    Safe,

    SameKind,

    Unsafe,
}

pub fn can_cast(from: Kind, to: Kind) -> CastSafety {
    use Kind::*;
    if from == to {
        return CastSafety::Equivalent;
    }
    match (from, to) {
        (Bool, _) => CastSafety::Safe,
        (_, Bool) => CastSafety::Unsafe,

        (Int(a), Int(b)) => width_cast(a, b),
        (Uint(a), Uint(b)) => width_cast(a, b),
        (Float(a), Float(b)) => width_cast(a, b),
        (Complex(a), Complex(b)) => width_cast(a, b),

        (Int(_), Uint(_)) => CastSafety::Unsafe,

        (Uint(u), Int(i)) => {
            if i > u {
                CastSafety::Safe
            } else {
                CastSafety::SameKind
            }
        }

        (Int(i), Float(f)) | (Uint(i), Float(f)) => {
            if i <= float_mantissa_bits(f) {
                CastSafety::Safe
            } else {
                CastSafety::SameKind
            }
        }

        (Float(_), Int(_)) | (Float(_), Uint(_)) => CastSafety::Unsafe,

        (Float(f), Complex(c)) => width_cast(f, c),
        (Complex(_), Float(_)) => CastSafety::Unsafe,

        (Int(i), Complex(c)) | (Uint(i), Complex(c)) => {
            if i <= float_mantissa_bits(c) {
                CastSafety::Safe
            } else {
                CastSafety::SameKind
            }
        }

        (Complex(_), Int(_)) | (Complex(_), Uint(_)) => CastSafety::Unsafe,
    }
}

fn width_cast(from_bits: u8, to_bits: u8) -> CastSafety {
    if to_bits >= from_bits {
        CastSafety::Safe
    } else {
        CastSafety::SameKind
    }
}

pub trait DType: Copy {
    const KIND: Kind;
    fn type_name() -> &'static str;
}

impl DType for bool {
    const KIND: Kind = Kind::Bool;
    fn type_name() -> &'static str {
        "bool"
    }
}

impl DType for i8 {
    const KIND: Kind = Kind::Int(8);
    fn type_name() -> &'static str {
        "int8"
    }
}

impl DType for i16 {
    const KIND: Kind = Kind::Int(16);
    fn type_name() -> &'static str {
        "int16"
    }
}

impl DType for i32 {
    const KIND: Kind = Kind::Int(32);
    fn type_name() -> &'static str {
        "int32"
    }
}

impl DType for i64 {
    const KIND: Kind = Kind::Int(64);
    fn type_name() -> &'static str {
        "int64"
    }
}

impl DType for u8 {
    const KIND: Kind = Kind::Uint(8);
    fn type_name() -> &'static str {
        "uint8"
    }
}

impl DType for u16 {
    const KIND: Kind = Kind::Uint(16);
    fn type_name() -> &'static str {
        "uint16"
    }
}

impl DType for u32 {
    const KIND: Kind = Kind::Uint(32);
    fn type_name() -> &'static str {
        "uint32"
    }
}

impl DType for u64 {
    const KIND: Kind = Kind::Uint(64);
    fn type_name() -> &'static str {
        "uint64"
    }
}

impl DType for half::f16 {
    const KIND: Kind = Kind::Float(16);
    fn type_name() -> &'static str {
        "float16"
    }
}

impl DType for f32 {
    const KIND: Kind = Kind::Float(32);
    fn type_name() -> &'static str {
        "float32"
    }
}

impl DType for f64 {
    const KIND: Kind = Kind::Float(64);
    fn type_name() -> &'static str {
        "float64"
    }
}

impl DType for num_complex::Complex<f32> {
    const KIND: Kind = Kind::Complex(32);
    fn type_name() -> &'static str {
        "complex64"
    }
}

impl DType for num_complex::Complex<f64> {
    const KIND: Kind = Kind::Complex(64);
    fn type_name() -> &'static str {
        "complex128"
    }
}

use half::f16;
use num_complex::Complex;

pub trait Cast<U>: Copy {
    fn cast(self) -> U;
}

macro_rules! cast_row {
    ($s:ty; $($t:ty),*) => {$(
        impl Cast<$t> for $s {
            fn cast(self) -> $t { self as $t }
        }
    )*};
}

macro_rules! cast_table {
    ($($s:ty),*) => {$( cast_row!($s; i8, i16, i32, i64, u8, u16, u32, u64, f32, f64); )*};
}
cast_table!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

macro_rules! cast_bool_and_complex {
    ($($t:ty),*) => {$(
        impl Cast<$t> for bool {
            fn cast(self) -> $t { self as u8 as $t }
        }
        impl Cast<bool> for $t {
            fn cast(self) -> bool { self != (0 as $t) }
        }
        impl Cast<Complex<f32>> for $t {
            fn cast(self) -> Complex<f32> { Complex::new(self as f32, 0.0) }
        }
        impl Cast<Complex<f64>> for $t {
            fn cast(self) -> Complex<f64> { Complex::new(self as f64, 0.0) }
        }
        impl Cast<$t> for Complex<f32> {
            fn cast(self) -> $t { self.re as $t }
        }
        impl Cast<$t> for Complex<f64> {
            fn cast(self) -> $t { self.re as $t }
        }
    )*};
}
cast_bool_and_complex!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

macro_rules! cast_f16_ints_floats {
    ($($t:ty),*) => {$(
        impl Cast<$t> for f16 {
            fn cast(self) -> $t { f32::from(self) as $t }
        }
    )*};
}
cast_f16_ints_floats!(i8, i16, i32, i64, u8, u16, u32, u64, f32);

macro_rules! cast_to_f16_via_f32 {
    ($($t:ty),*) => {$(
        impl Cast<f16> for $t {
            fn cast(self) -> f16 { f16::from_f32(self as f32) }
        }
    )*};
}
cast_to_f16_via_f32!(i8, i16, i32, i64, u8, u16, u32, u64, f32);

impl Cast<f64> for f16 {
    fn cast(self) -> f64 {
        f64::from(self)
    }
}
impl Cast<f16> for f64 {
    fn cast(self) -> f16 {
        f16::from_f64(self)
    }
}
impl Cast<f16> for f16 {
    fn cast(self) -> f16 {
        self
    }
}
impl Cast<bool> for f16 {
    fn cast(self) -> bool {
        self != f16::ZERO
    }
}
impl Cast<f16> for bool {
    fn cast(self) -> f16 {
        f16::from_f32(f32::from(u8::from(self)))
    }
}
impl Cast<Complex<f32>> for f16 {
    fn cast(self) -> Complex<f32> {
        Complex::new(f32::from(self), 0.0)
    }
}
impl Cast<Complex<f64>> for f16 {
    fn cast(self) -> Complex<f64> {
        Complex::new(f64::from(self), 0.0)
    }
}
impl Cast<f16> for Complex<f32> {
    fn cast(self) -> f16 {
        f16::from_f32(self.re)
    }
}
impl Cast<f16> for Complex<f64> {
    fn cast(self) -> f16 {
        f16::from_f64(self.re)
    }
}

impl Cast<bool> for bool {
    fn cast(self) -> bool {
        self
    }
}
impl Cast<Complex<f32>> for bool {
    fn cast(self) -> Complex<f32> {
        Complex::new(f32::from(u8::from(self)), 0.0)
    }
}
impl Cast<Complex<f64>> for bool {
    fn cast(self) -> Complex<f64> {
        Complex::new(f64::from(u8::from(self)), 0.0)
    }
}
impl Cast<bool> for Complex<f32> {
    fn cast(self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
}
impl Cast<bool> for Complex<f64> {
    fn cast(self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
}
impl Cast<Complex<f32>> for Complex<f32> {
    fn cast(self) -> Complex<f32> {
        self
    }
}
impl Cast<Complex<f64>> for Complex<f64> {
    fn cast(self) -> Complex<f64> {
        self
    }
}
impl Cast<Complex<f64>> for Complex<f32> {
    fn cast(self) -> Complex<f64> {
        Complex::new(f64::from(self.re), f64::from(self.im))
    }
}
impl Cast<Complex<f32>> for Complex<f64> {
    fn cast(self) -> Complex<f32> {
        Complex::new(self.re as f32, self.im as f32)
    }
}

impl<T: Copy> crate::ndarray::NdArray<T> {
    pub fn astype<U: Copy>(&self) -> crate::ndarray::NdArray<U>
    where
        T: Cast<U>,
    {
        let data: Vec<U> = self.view().iter().map(Cast::cast).collect();
        crate::ndarray::NdArray::from_vec(data, self.shape()).expect("same element count")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float16_promotion_matches_numpy() {
        assert_eq!(common_dtype(Kind::Int(8), Kind::Float(16)), Kind::Float(16));
        assert_eq!(common_dtype(Kind::Uint(8), Kind::Float(16)), Kind::Float(16));
        assert_eq!(common_dtype(Kind::Int(16), Kind::Float(16)), Kind::Float(32));
        assert_eq!(common_dtype(Kind::Int(32), Kind::Float(16)), Kind::Float(64));
        assert_eq!(common_dtype(Kind::Bool, Kind::Float(16)), Kind::Float(16));
        assert_eq!(common_dtype(Kind::Float(16), Kind::Float(32)), Kind::Float(32));
        assert_eq!(common_dtype(Kind::Float(16), Kind::Complex(32)), Kind::Complex(32));
        assert_eq!(can_cast(Kind::Int(8), Kind::Float(16)), CastSafety::Safe);
        assert_eq!(can_cast(Kind::Int(16), Kind::Float(16)), CastSafety::SameKind);
        assert_eq!(can_cast(Kind::Float(16), Kind::Float(32)), CastSafety::Safe);
        assert_eq!(can_cast(Kind::Float(32), Kind::Float(16)), CastSafety::SameKind);
    }

    #[test]
    fn kind_ordering_matches_nep50_rank() {
        assert!(Kind::Bool < Kind::Int(8));
        assert!(Kind::Int(64) < Kind::Float(32));
        assert!(Kind::Int(8) < Kind::Int(64));
        assert!(Kind::Float(32) < Kind::Float(64));
    }

    #[test]
    fn common_dtype_same_kind_picks_wider() {
        assert_eq!(common_dtype(Kind::Int(8), Kind::Int(16)), Kind::Int(16));
        assert_eq!(common_dtype(Kind::Float(64), Kind::Float(32)), Kind::Float(64));
    }

    #[test]
    fn common_dtype_bool_is_absorbed() {

        assert_eq!(common_dtype(Kind::Bool, Kind::Int(32)), Kind::Int(32));

        assert_eq!(common_dtype(Kind::Bool, Kind::Float(64)), Kind::Float(64));
    }

    #[test]
    fn common_dtype_int_float_matches_real_numpy_result_type() {

        assert_eq!(common_dtype(Kind::Int(16), Kind::Float(32)), Kind::Float(32));

        assert_eq!(common_dtype(Kind::Int(32), Kind::Float(32)), Kind::Float(64));

        assert_eq!(common_dtype(Kind::Int(64), Kind::Float(32)), Kind::Float(64));
    }

    const NUMPY_CHARS: &str = "?bhilBHILefdFD";

    fn kind_of_char(c: char) -> Kind {
        [
            Kind::Bool,
            Kind::Int(8),
            Kind::Int(16),
            Kind::Int(32),
            Kind::Int(64),
            Kind::Uint(8),
            Kind::Uint(16),
            Kind::Uint(32),
            Kind::Uint(64),
            Kind::Float(16),
            Kind::Float(32),
            Kind::Float(64),
            Kind::Complex(32),
            Kind::Complex(64),
        ][NUMPY_CHARS.find(c).unwrap()]
    }

    const NUMPY_RESULT_TYPE: [&str; 14] = [
        "?bhilBHILefdFD",
        "bbhilhildefdFD",
        "hhhilhildffdFD",
        "iiiiliilddddDD",
        "llllllllddddDD",
        "BhhilBHILefdFD",
        "HiiilHHILffdFD",
        "IllllIIILdddDD",
        "LddddLLLLdddDD",
        "eefddefddefdFD",
        "fffddffddffdFD",
        "ddddddddddddDD",
        "FFFDDFFDDFFDFD",
        "DDDDDDDDDDDDDD",
    ];

    const NUMPY_WEAK: [(Weak, &str); 3] = [
        (Weak::Int, "lbhilBHILefdFD"),
        (Weak::Float, "dddddddddefdFD"),
        (Weak::Complex, "DDDDDDDDDFFDFD"),
    ];

    #[test]
    fn common_dtype_matches_numpy_result_type_for_all_196_pairs() {
        for (a, row) in NUMPY_CHARS.chars().zip(NUMPY_RESULT_TYPE) {
            for (b, want) in NUMPY_CHARS.chars().zip(row.chars()) {
                assert_eq!(common_dtype(kind_of_char(a), kind_of_char(b)), kind_of_char(want), "{a} + {b}");
            }
        }
    }

    #[test]
    fn weak_python_scalars_match_numpy_for_every_dtype() {
        for (w, row) in NUMPY_WEAK {
            for (a, want) in NUMPY_CHARS.chars().zip(row.chars()) {
                assert_eq!(with_weak(kind_of_char(a), w), kind_of_char(want), "{a} + {w:?}");
            }
        }
    }

    #[test]
    fn result_type_combines_strong_dtypes_then_the_widest_weak_scalar() {
        assert_eq!(result_type([Kind::Int(8), Kind::Uint(8)], [Weak::Int]), Some(Kind::Int(16)));
        assert_eq!(result_type([Kind::Float(32)], [Weak::Int, Weak::Complex]), Some(Kind::Complex(32)));
        assert_eq!(result_type([], [Weak::Float, Weak::Int]), Some(Kind::Float(64)));
        assert_eq!(result_type([], []), None);
    }

    #[test]
    fn astype_casts_elementwise_like_numpy() {
        let a = crate::ndarray::NdArray::from_vec(vec![-1.5f64, 0.0, 2.7], &[3]).unwrap();
        assert_eq!(a.astype::<i32>().as_slice(), &[-1, 0, 2]);
        assert_eq!(a.astype::<bool>().as_slice(), &[true, false, true]);
        assert_eq!(a.astype::<Complex<f32>>().as_slice()[2], Complex::new(2.7f32, 0.0));
        let m = crate::ndarray::NdArray::from_vec(vec![1u8, 2, 3, 4], &[2, 2]).unwrap();
        let t = m.view().matrix_transpose().unwrap().to_owned();
        assert_eq!(t.astype::<f16>().as_slice(), &[1.0, 3.0, 2.0, 4.0].map(f16::from_f32));
    }

    #[test]
    fn can_cast_levels() {
        assert_eq!(can_cast(Kind::Int(32), Kind::Int(32)), CastSafety::Equivalent);
        assert_eq!(can_cast(Kind::Int(16), Kind::Int(32)), CastSafety::Safe);
        assert_eq!(can_cast(Kind::Int(32), Kind::Int(16)), CastSafety::SameKind);
        assert_eq!(can_cast(Kind::Float(64), Kind::Float(32)), CastSafety::SameKind);
        assert_eq!(can_cast(Kind::Bool, Kind::Int(8)), CastSafety::Safe);
        assert_eq!(can_cast(Kind::Int(8), Kind::Bool), CastSafety::Unsafe);
        assert_eq!(can_cast(Kind::Float(32), Kind::Int(32)), CastSafety::Unsafe);
        assert_eq!(can_cast(Kind::Int(16), Kind::Float(32)), CastSafety::Safe);
        assert_eq!(can_cast(Kind::Int(32), Kind::Float(32)), CastSafety::SameKind);
    }

    #[test]
    fn can_cast_int64_to_float64_uses_the_consistent_mantissa_rule_not_numpy() {

        assert_eq!(can_cast(Kind::Int(64), Kind::Float(64)), CastSafety::SameKind);
        assert_eq!(can_cast(Kind::Int(32), Kind::Float(32)), CastSafety::SameKind);

        assert_eq!(can_cast(Kind::Int(64), Kind::Float(32)), CastSafety::SameKind);
        assert_eq!(can_cast(Kind::Int(32), Kind::Float(64)), CastSafety::Safe);
    }

    #[test]
    fn common_dtype_int_uint_matches_real_numpy_result_type() {

        assert_eq!(common_dtype(Kind::Int(8), Kind::Uint(8)), Kind::Int(16));

        assert_eq!(common_dtype(Kind::Uint(16), Kind::Int(8)), Kind::Int(32));

        assert_eq!(common_dtype(Kind::Uint(8), Kind::Int(16)), Kind::Int(16));

        assert_eq!(common_dtype(Kind::Int(32), Kind::Uint(32)), Kind::Int(64));

        assert_eq!(common_dtype(Kind::Int(64), Kind::Uint(64)), Kind::Float(64));
    }

    #[test]
    fn can_cast_int_to_uint_is_always_unsafe() {

        assert_eq!(can_cast(Kind::Int(8), Kind::Uint(64)), CastSafety::Unsafe);
        assert_eq!(can_cast(Kind::Int(64), Kind::Uint(8)), CastSafety::Unsafe);
    }

    #[test]
    fn can_cast_uint_to_int_depends_on_width_like_int_to_int() {

        assert_eq!(can_cast(Kind::Uint(8), Kind::Int(8)), CastSafety::SameKind);

        assert_eq!(can_cast(Kind::Uint(8), Kind::Int(16)), CastSafety::Safe);

        assert_eq!(can_cast(Kind::Uint(64), Kind::Int(64)), CastSafety::SameKind);
    }

    #[test]
    fn can_cast_uint64_to_float64_also_uses_the_consistent_rule() {

        assert_eq!(can_cast(Kind::Uint(64), Kind::Float(64)), CastSafety::SameKind);
        assert_eq!(can_cast(Kind::Uint(32), Kind::Float(32)), CastSafety::SameKind);
        assert_eq!(can_cast(Kind::Uint(32), Kind::Float(64)), CastSafety::Safe);
    }

    #[test]
    fn can_cast_float_to_complex_matches_numpy() {

        assert_eq!(can_cast(Kind::Float(32), Kind::Complex(32)), CastSafety::Safe);

        assert_eq!(can_cast(Kind::Float(64), Kind::Complex(32)), CastSafety::SameKind);

        assert_eq!(can_cast(Kind::Float(32), Kind::Complex(64)), CastSafety::Safe);
    }

    #[test]
    fn can_cast_complex_to_float_is_always_unsafe() {

        assert_eq!(can_cast(Kind::Complex(32), Kind::Float(32)), CastSafety::Unsafe);
        assert_eq!(can_cast(Kind::Complex(64), Kind::Float(64)), CastSafety::Unsafe);
    }

    #[test]
    fn can_cast_complex_to_complex_matches_numpy() {

        assert_eq!(can_cast(Kind::Complex(32), Kind::Complex(64)), CastSafety::Safe);

        assert_eq!(can_cast(Kind::Complex(64), Kind::Complex(32)), CastSafety::SameKind);
    }

    #[test]
    fn can_cast_int_to_complex_uses_the_consistent_mantissa_rule() {

        assert_eq!(can_cast(Kind::Int(32), Kind::Complex(32)), CastSafety::SameKind);

        assert_eq!(can_cast(Kind::Int(64), Kind::Complex(32)), CastSafety::SameKind);

        assert_eq!(can_cast(Kind::Int(64), Kind::Complex(64)), CastSafety::SameKind);

        assert_eq!(can_cast(Kind::Int(32), Kind::Complex(64)), CastSafety::Safe);
    }

    #[test]
    fn can_cast_complex_to_int_is_always_unsafe() {

        assert_eq!(can_cast(Kind::Complex(32), Kind::Int(32)), CastSafety::Unsafe);
    }

    #[test]
    fn bool_complex_uint_interactions_match_numpy() {

        assert_eq!(can_cast(Kind::Bool, Kind::Uint(8)), CastSafety::Safe);
        assert_eq!(can_cast(Kind::Uint(8), Kind::Bool), CastSafety::Unsafe);

        assert_eq!(can_cast(Kind::Bool, Kind::Complex(32)), CastSafety::Safe);
        assert_eq!(can_cast(Kind::Complex(32), Kind::Bool), CastSafety::Unsafe);
    }

    #[test]
    fn dtype_trait_on_primitives() {
        assert_eq!(bool::KIND, Kind::Bool);
        assert_eq!(i8::KIND, Kind::Int(8));
        assert_eq!(i16::KIND, Kind::Int(16));
        assert_eq!(i32::KIND, Kind::Int(32));
        assert_eq!(i64::KIND, Kind::Int(64));
        assert_eq!(u8::KIND, Kind::Uint(8));
        assert_eq!(u16::KIND, Kind::Uint(16));
        assert_eq!(u32::KIND, Kind::Uint(32));
        assert_eq!(u64::KIND, Kind::Uint(64));
        assert_eq!(f32::KIND, Kind::Float(32));
        assert_eq!(f64::KIND, Kind::Float(64));
        assert_eq!(<num_complex::Complex<f32> as DType>::KIND, Kind::Complex(32));
        assert_eq!(<num_complex::Complex<f64> as DType>::KIND, Kind::Complex(64));
        assert_eq!(i32::type_name(), "int32");
        assert_eq!(u32::type_name(), "uint32");
        assert_eq!(i64::type_name(), "int64");
        assert_eq!(u64::type_name(), "uint64");
        assert_eq!(<num_complex::Complex<f32> as DType>::type_name(), "complex64");
    }
}
