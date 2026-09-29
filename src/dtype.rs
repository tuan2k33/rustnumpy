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

fn smallest_exact_float_for_int(int_bits: u8) -> u8 {
    for float_bits in [16u8, 32, 64] {
        if int_bits <= float_mantissa_bits(float_bits) {
            return float_bits;
        }
    }
    64
}

pub fn common_dtype(a: Kind, b: Kind) -> Kind {
    use Kind::*;
    match (a, b) {
        (Bool, Bool) => Bool,
        (Bool, other) | (other, Bool) => other,

        (Int(x), Int(y)) => Int(x.max(y)),
        (Uint(x), Uint(y)) => Uint(x.max(y)),
        (Int(i), Uint(u)) | (Uint(u), Int(i)) => int_uint_common(i, u),

        (Float(x), Float(y)) => Float(x.max(y)),
        (Int(i), Float(f)) | (Float(f), Int(i)) => Float(smallest_exact_float_for_int(i).max(f)),
        (Uint(u), Float(f)) | (Float(f), Uint(u)) => Float(smallest_exact_float_for_int(u).max(f)),

        (Complex(x), Complex(y)) => Complex(x.max(y)),
        (Float(f), Complex(c)) | (Complex(c), Float(f)) => Complex(f.max(c)),
        (Int(i), Complex(c)) | (Complex(c), Int(i)) => Complex(smallest_exact_float_for_int(i).max(c)),
        (Uint(u), Complex(c)) | (Complex(c), Uint(u)) => Complex(smallest_exact_float_for_int(u).max(c)),
    }
}

fn int_uint_common(signed_bits: u8, unsigned_bits: u8) -> Kind {
    if signed_bits > unsigned_bits {
        return Kind::Int(signed_bits);
    }
    for candidate in [16u8, 32, 64] {
        if candidate > unsigned_bits {
            return Kind::Int(candidate);
        }
    }
    Kind::Float(64)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeakScalar {
    Int(i64),
    Float(f64),
}

impl WeakScalar {
    fn rank(self) -> u8 {
        match self {
            WeakScalar::Int(_) => 1,
            WeakScalar::Float(_) => 2,
        }
    }
}

pub fn promote_array_with_weak_scalar(array_kind: Kind, scalar: WeakScalar) -> Kind {
    if scalar.rank() <= array_kind.rank() {
        array_kind
    } else {
        match scalar {
            WeakScalar::Int(_) => Kind::Int(64),
            WeakScalar::Float(_) => Kind::Float(64),
        }
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

pub fn common_dtype_of<A: DType, B: DType>() -> Kind {
    common_dtype(A::KIND, B::KIND)
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

    #[test]
    fn weak_scalar_never_downgrades_or_narrows_the_array_dtype() {

        assert_eq!(
            promote_array_with_weak_scalar(Kind::Int(32), WeakScalar::Int(1)),
            Kind::Int(32)
        );

        assert_eq!(
            promote_array_with_weak_scalar(Kind::Float(32), WeakScalar::Float(1.0)),
            Kind::Float(32)
        );

        assert_eq!(
            promote_array_with_weak_scalar(Kind::Float(32), WeakScalar::Int(1)),
            Kind::Float(32)
        );
    }

    #[test]
    fn weak_scalar_promotes_to_the_default_dtype_of_its_own_kind() {

        assert_eq!(
            promote_array_with_weak_scalar(Kind::Bool, WeakScalar::Int(1)),
            Kind::Int(64)
        );

        assert_eq!(
            promote_array_with_weak_scalar(Kind::Bool, WeakScalar::Float(1.0)),
            Kind::Float(64)
        );

        assert_eq!(
            promote_array_with_weak_scalar(Kind::Int(32), WeakScalar::Float(1.0)),
            Kind::Float(64)
        );
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

    #[test]
    fn common_dtype_of_is_resolved_at_compile_time() {

        assert_eq!(common_dtype_of::<i32, f32>(), Kind::Float(64));
        assert_eq!(common_dtype_of::<bool, i32>(), Kind::Int(32));
        assert_eq!(common_dtype_of::<f32, f32>(), Kind::Float(32));
    }
}
