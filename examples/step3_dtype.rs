//! Run: `cargo run --example step3_dtype`
//!
//! Demonstrates the NEP 50 promotion algorithm two ways: the runtime
//! `Kind` enum (what a real interpreter has to use, since it only learns
//! dtypes at runtime) and the compile-time `DType` trait (what Rust can
//! do instead, since the types are known statically). Every result here
//! is cross-checked against real NumPy — see the comments.

use rustnumpy::{can_cast, common_dtype, common_dtype_of, CastSafety, DType, Kind, WeakScalar};

fn show_common(a: Kind, b: Kind) {
    println!("  common_dtype({a:?}, {b:?}) = {:?}", common_dtype(a, b));
}

fn show_weak(array: Kind, scalar: WeakScalar) {
    let result = rustnumpy::dtype::promote_array_with_weak_scalar(array, scalar);
    println!("  array({array:?}) + weak_scalar({scalar:?}) -> {result:?}");
}

fn main() {
    println!("-- concrete dtype x concrete dtype (like np.result_type) --");
    show_common(Kind::Int(16), Kind::Float(32)); // -> Float(32)
    show_common(Kind::Int(32), Kind::Float(32)); // -> Float(64), not Float(32)!
    show_common(Kind::Bool, Kind::Int(32)); // -> Int(32)
    show_common(Kind::Int(8), Kind::Int(64)); // -> Int(64)

    println!("\n-- array dtype x Python weak scalar (NEP 50) --");
    show_weak(Kind::Bool, WeakScalar::Int(1)); // -> Int(64), the *default* int
    show_weak(Kind::Int(32), WeakScalar::Int(1)); // -> Int(32), unchanged
    show_weak(Kind::Int(32), WeakScalar::Float(1.0)); // -> Float(64), NOT Float(32)
    show_weak(Kind::Float(32), WeakScalar::Float(1.0)); // -> Float(32), unchanged

    println!("\n-- compile-time promotion via the DType trait (no runtime branching) --");
    println!("  common_dtype_of::<i32, f32>() = {:?}", common_dtype_of::<i32, f32>());
    println!("  common_dtype_of::<bool, i32>() = {:?}", common_dtype_of::<bool, i32>());
    println!(
        "  {} vs {}: common kind = {:?}",
        i32::type_name(),
        f64::type_name(),
        common_dtype_of::<i32, f64>()
    );

    println!("\n-- casting safety levels --");
    for (from, to) in [
        (Kind::Int(16), Kind::Int(32)),
        (Kind::Int(32), Kind::Int(16)),
        (Kind::Float(64), Kind::Float(32)),
        (Kind::Int(8), Kind::Bool),
        (Kind::Bool, Kind::Int(8)),
    ] {
        let safety: CastSafety = can_cast(from, to);
        println!("  can_cast({from:?} -> {to:?}) = {safety:?}");
    }
}
