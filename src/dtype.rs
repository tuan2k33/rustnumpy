//! Step 3: a basic DType system via a trait, plus the NEP 50 promotion
//! algorithm (see NumPy.md, "The new DType system" and "Promotion rules").
//!
//! Design choice made here, straight from NumPy.md's own mapping table:
//! **hybrid** dispatch — a closed `Kind` enum for the built-in numeric
//! kinds (fast, exhaustive `match`, what most code will use) plus a
//! `DType` trait implemented directly on Rust's own primitive types
//! (`bool`, `i32`, `f32`, `f64`), so a generic function like
//! `common_dtype_of::<i32, f32>()` gets monomorphized at compile time
//! instead of paying for dynamic dispatch. A real "third parties can add
//! dtypes" story (NEP 41/42's actual point) would need `dyn DType`
//! instead — deliberately out of scope for this step.
//!
//! **Known, documented gaps against real NumPy's full casting model**
//! (found while cross-checking [`can_cast`] against `np.can_cast(...,
//! casting=...)` directly):
//! - `Kind` has no separate unsigned-integer variant (`Int(u8)` only,
//!   signed) and no `Complex` variant at all, so casting rules that only
//!   make sense between those (`int32 -> uint32` is `unsafe`, `uint32 ->
//!   int32` is `same_kind`, `float32 -> complex64` is `safe`, `complex64
//!   -> float32` is `unsafe`) simply can't be expressed here yet — there's
//!   no `Kind` value to represent "unsigned" or "complex" with.
//! - [`CastSafety`] has 4 levels, not NumPy's full 5 (`no`, `equiv`,
//!   `safe`, `same_kind`, `unsafe`): `no` and `equiv` are collapsed into
//!   one `Equivalent` here, since the distinction between them is entirely
//!   about byte-order (`equiv` allows a byte-swap, `no` doesn't), and this
//!   project has no byte-order/endianness model at all (`npy.rs` only
//!   reads/writes native-order files, explicitly refusing big-endian).
//! - This module only classifies *whether a cast is allowed* — there's no
//!   function anywhere in this crate that actually performs one
//!   (`NdArray<T>` has no `astype::<U>()` yet, even though `NdArray<T>`
//!   becoming generic makes one straightforward to add). Nothing here
//!   converts real data.
//! - Real NumPy's weak-scalar-overflow behavior (`np.int8(1) + 1000`
//!   raising `OverflowError` instead of silently upcasting) and its
//!   reduction-specific default-dtype rule (`sum()`/`prod()` on a narrow
//!   int always accumulates in at least `int64`) both depend on actually
//!   *running* int arithmetic — out of scope until `ufunc`/`reductions`
//!   grow int support (currently both are `f64`-only, see their own doc
//!   comments), not something `dtype.rs`'s cast-safety table alone could
//!   express.

/// A dtype's "kind" plus bit width, ordered exactly per NEP 50:
/// `boolean < integral < inexact (float)`, and within the same kind, wider
/// wins. Deriving `PartialOrd`/`Ord` here isn't just convenient — the
/// derive compares the variant's declaration order first (`Bool` before
/// `Int` before `Float`), which happens to be *exactly* NEP 50's kind
/// ranking, and only falls back to comparing the payload (bit width) when
/// the variant matches. So `Kind::Int(16) < Kind::Float(32)` and
/// `Kind::Int(16) < Kind::Int(32)` both fall out of one `derive` for free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Bool,
    /// Bit width: 8, 16, 32, or 64 in this exercise.
    Int(u8),
    /// Bit width: 32 or 64 in this exercise (no float16/float128 here).
    Float(u8),
}

impl Kind {
    /// NEP 50's "kind rank" as a plain integer — used where we need to
    /// compare *only* the kind and ignore bit width (the weak-scalar rule
    /// below needs exactly this, since a Python scalar has no width).
    fn rank(self) -> u8 {
        match self {
            Kind::Bool => 0,
            Kind::Int(_) => 1,
            Kind::Float(_) => 2,
        }
    }
}

/// How many bits of an integer a float's mantissa can represent *exactly*
/// (sign + all bit patterns up to `2^mantissa_bits` round-trip losslessly
/// through the float). This is the real reason `int32 + float32 ->
/// float64` in NumPy: a 32-bit int can hold values float32's 24-bit
/// mantissa (23 stored + 1 implicit) can't represent exactly, so NumPy
/// bumps to the next float width that can, capping at 64 bits since this
/// exercise has no float128.
fn float_mantissa_bits(float_bits: u8) -> u8 {
    match float_bits {
        32 => 24,
        64 => 53,
        _ => 53,
    }
}

/// Pick the narrowest available float width whose mantissa can exactly
/// represent an integer of `int_bits` bits, falling back to the widest
/// float we support (64) if none can — matching NumPy's own fallback
/// (there's no float128 to reach for either).
fn smallest_exact_float_for_int(int_bits: u8) -> u8 {
    for float_bits in [32u8, 64] {
        if int_bits <= float_mantissa_bits(float_bits) {
            return float_bits;
        }
    }
    64
}

/// NEP 50's `common_dtype` for two *concrete* array dtypes: no Python
/// scalars involved here, both sides are already real array dtypes.
///
/// Cross-checked against real NumPy's `np.result_type`:
/// `int16+float32->float32`, `int32+float32->float64`,
/// `int64+float32->float64`, `bool+int32->int32`.
pub fn common_dtype(a: Kind, b: Kind) -> Kind {
    use Kind::*;
    match (a, b) {
        (Bool, Bool) => Bool,
        (Bool, other) | (other, Bool) => other, // bool is the additive identity of kind ranking
        (Int(x), Int(y)) => Int(x.max(y)),
        (Float(x), Float(y)) => Float(x.max(y)),
        (Int(i), Float(f)) | (Float(f), Int(i)) => {
            Float(smallest_exact_float_for_int(i).max(f))
        }
    }
}

/// A Python `int`/`float` literal as NEP 50 sees it: a "weak" scalar with
/// a kind but **no fixed width** — it only becomes a concrete dtype once
/// it collides with a real array.
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

/// NEP 50's headline rule, stated precisely: combining an array with a
/// weak Python scalar **never changes the array's concrete dtype** as
/// long as the scalar's kind is the same rank or lower. Only when the
/// scalar's kind outranks the array's does the result change — and even
/// then it becomes the *default* dtype for that kind (`int64`/`float64`),
/// completely ignoring the array's own bit width, because the weak scalar
/// never had a width to contribute in the first place.
///
/// This is the exact behavior NEP 50 was written to make predictable: old
/// NumPy would sometimes look at the scalar's *value* to decide; NEP 50
/// looks at its *kind* only, so the result never depends on what number
/// you happened to write.
///
/// Cross-checked against real NumPy (`arr + scalar`):
/// `bool_array + 1 -> int64`, `bool_array + 1.0 -> float64`,
/// `int32_array + 1 -> int32` (unchanged), `int32_array + 1.0 -> float64`
/// (not float32!), `float32_array + 1 -> float32`,
/// `float32_array + 1.0 -> float32` (unchanged).
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

/// The 4 casting safety levels NumPy tracks between dtypes (see
/// NumPy.md, "Casting rules"). Ordered loosest-last so `derive(Ord)`
/// gives a sensible "is at least this safe" comparison if ever needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastSafety {
    /// Bit-preserving (e.g. same dtype).
    Equivalent,
    /// No information can be lost.
    Safe,
    /// Same kind, but precision may be lost (e.g. float64 -> float32).
    SameKind,
    /// May lose information or change kind entirely (e.g. float -> int).
    Unsafe,
}

/// How safe is it to cast a value of kind `from` into kind `to`?
///
/// This is deliberately a *simplified* table (real NumPy's is a fixed,
/// exhaustively-enumerated matrix per concrete dtype pair) — but the
/// shape of the reasoning is the same: same-kind-and-wider is `Safe`,
/// same-kind-and-narrower is `SameKind`, going to a strictly higher kind
/// that can hold the value exactly is `Safe`, and anything that can
/// silently truncate or change meaning (float -> int, int -> bool where
/// the value isn't 0/1) is `Unsafe`.
pub fn can_cast(from: Kind, to: Kind) -> CastSafety {
    use Kind::*;
    if from == to {
        return CastSafety::Equivalent;
    }
    match (from, to) {
        (Bool, _) => CastSafety::Safe, // 0/1 always fits anywhere
        (_, Bool) => CastSafety::Unsafe, // collapses every other value down to true/false
        (Int(a), Int(b)) => {
            if b >= a {
                CastSafety::Safe
            } else {
                CastSafety::SameKind
            }
        }
        (Float(a), Float(b)) => {
            if b >= a {
                CastSafety::Safe
            } else {
                CastSafety::SameKind
            }
        }
        // Special case, verified against real NumPy's own can_cast table
        // rather than derived from the mantissa-bits reasoning below:
        // int64 -> float64 is classified `safe` even though float64's
        // 53-bit mantissa can't represent every int64 value exactly — a
        // historical NumPy quirk (same total bit width is treated as
        // "close enough"), not something the general rule predicts. Every
        // other same-or-different-width int/float pair *does* follow the
        // general mantissa rule (verified: int32->float64 safe,
        // int32->float32 and int64->float32 both same_kind).
        (Int(64), Float(64)) => CastSafety::Safe,
        (Int(i), Float(f)) => {
            if i <= float_mantissa_bits(f) {
                CastSafety::Safe
            } else {
                CastSafety::SameKind
            }
        }
        (Float(_), Int(_)) => CastSafety::Unsafe, // drops the fractional part
    }
}

/// A dtype as a Rust *type*, not just a runtime value — implemented
/// directly on the primitive types themselves rather than on wrapper
/// structs, so `i32::KIND` and `f64::KIND` just work. This is the
/// "generic, monomorphized" half of the hybrid design: a function generic
/// over `D: DType` gets a fully specialized, branch-free version per
/// concrete type at compile time, the same way NumPy's C core has to
/// dispatch at runtime because Python types aren't known until the
/// interpreter runs.
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

impl DType for i32 {
    const KIND: Kind = Kind::Int(32);
    fn type_name() -> &'static str {
        "int32"
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

/// Compile-time promotion between two concrete Rust dtypes — no runtime
/// branching at all, `A::KIND`/`B::KIND` are `const`s the compiler folds
/// away. This is the generic-dispatch counterpart to the runtime
/// `common_dtype(Kind, Kind)` above; both exist on purpose, so the design
/// question from NumPy.md ("`dyn DType` vs. generic") is visible in code
/// rather than just in prose.
pub fn common_dtype_of<A: DType, B: DType>() -> Kind {
    common_dtype(A::KIND, B::KIND)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // np.result_type('bool', 'int32') == int32
        assert_eq!(common_dtype(Kind::Bool, Kind::Int(32)), Kind::Int(32));
        // np.result_type('bool', 'float64') == float64
        assert_eq!(common_dtype(Kind::Bool, Kind::Float(64)), Kind::Float(64));
    }

    #[test]
    fn common_dtype_int_float_matches_real_numpy_result_type() {
        // np.result_type('int16', 'float32') == float32
        assert_eq!(common_dtype(Kind::Int(16), Kind::Float(32)), Kind::Float(32));
        // np.result_type('int32', 'float32') == float64 (float32 can't
        // exactly represent every int32 value)
        assert_eq!(common_dtype(Kind::Int(32), Kind::Float(32)), Kind::Float(64));
        // np.result_type('int64', 'float32') == float64 (best available,
        // even though float64 can't exactly represent every int64 either)
        assert_eq!(common_dtype(Kind::Int(64), Kind::Float(32)), Kind::Float(64));
    }

    #[test]
    fn weak_scalar_never_downgrades_or_narrows_the_array_dtype() {
        // int32_array + python int -> int32 (unchanged)
        assert_eq!(
            promote_array_with_weak_scalar(Kind::Int(32), WeakScalar::Int(1)),
            Kind::Int(32)
        );
        // float32_array + python float -> float32 (unchanged, NOT float64)
        assert_eq!(
            promote_array_with_weak_scalar(Kind::Float(32), WeakScalar::Float(1.0)),
            Kind::Float(32)
        );
        // float32_array + python int -> float32 (int scalar doesn't raise the kind)
        assert_eq!(
            promote_array_with_weak_scalar(Kind::Float(32), WeakScalar::Int(1)),
            Kind::Float(32)
        );
    }

    #[test]
    fn weak_scalar_promotes_to_the_default_dtype_of_its_own_kind() {
        // bool_array + python int -> int64 (the *default* int, not "bool widened")
        assert_eq!(
            promote_array_with_weak_scalar(Kind::Bool, WeakScalar::Int(1)),
            Kind::Int(64)
        );
        // bool_array + python float -> float64
        assert_eq!(
            promote_array_with_weak_scalar(Kind::Bool, WeakScalar::Float(1.0)),
            Kind::Float(64)
        );
        // int32_array + python float -> float64 (NOT float32 — the array's
        // width is irrelevant once the scalar's kind outranks it)
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
    fn can_cast_int64_to_float64_is_safe_despite_the_mantissa_shortfall() {
        // np.can_cast(np.int64, np.float64, casting='safe') -> True, even
        // though float64's 53-bit mantissa can't hold every int64 value
        // exactly. Verified directly against real NumPy 2.5.3 -- see this
        // function's own comment for why the general mantissa rule
        // (correctly used for every other int/float pair) doesn't predict
        // this one.
        assert_eq!(can_cast(Kind::Int(64), Kind::Float(64)), CastSafety::Safe);
        // The general rule still applies to every other pairing, including
        // same-width-looking ones that aren't 64/64:
        assert_eq!(can_cast(Kind::Int(64), Kind::Float(32)), CastSafety::SameKind);
    }

    #[test]
    fn dtype_trait_on_primitives() {
        assert_eq!(bool::KIND, Kind::Bool);
        assert_eq!(i32::KIND, Kind::Int(32));
        assert_eq!(f32::KIND, Kind::Float(32));
        assert_eq!(f64::KIND, Kind::Float(64));
        assert_eq!(i32::type_name(), "int32");
    }

    #[test]
    fn common_dtype_of_is_resolved_at_compile_time() {
        // Same underlying rule as common_dtype(), but driven entirely by
        // Rust's own types instead of runtime Kind values.
        assert_eq!(common_dtype_of::<i32, f32>(), Kind::Float(64));
        assert_eq!(common_dtype_of::<bool, i32>(), Kind::Int(32));
        assert_eq!(common_dtype_of::<f32, f32>(), Kind::Float(32));
    }
}
