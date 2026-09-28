//! Run: `cargo run --example step20_generic_numeric`
//!
//! Step 20: `ufunc`/`reductions` genericized over `T`. Same engine as
//! every earlier `f64` example -- just called with `NdArray<i32>`,
//! `NdArray<u64>`, and `NdArray<Complex64>` instead.

use rustnumpy::{add, max, mean, min, sub, sum, Complex64, NdArray};

fn main() {
    let a = NdArray::from_vec(vec![1i32, 2, 3, 4, 5], &[5]).unwrap();
    let b = NdArray::from_vec(vec![10i32, 20, 30, 40, 50], &[5]).unwrap();
    println!("i32 add(a, b) -> {:?}", add(&a.view(), &b.view()).unwrap().as_slice());
    println!("i32 sub(b, a) -> {:?}", sub(&b.view(), &a.view()).unwrap().as_slice());

    // sum/min/max preserve the input's own type...
    let s: i32 = sum(&a.view());
    println!("\ni32 sum(a) -> {s} (still i32, not upcast to i64 like real NumPy -- see reductions.rs's `sum` doc comment)");
    println!("i32 min(a) -> {}", min(&a.view()).unwrap());
    println!("i32 max(a) -> {}", max(&a.view()).unwrap());
    // ...but mean always promotes to f64, matching real NumPy exactly.
    println!("i32 mean(a) -> {} (f64, not truncated back to i32)", mean(&a.view()));

    let ua = NdArray::from_vec(vec![100u64, 200, 300], &[3]).unwrap();
    let ub = NdArray::from_vec(vec![1u64, 2, 3], &[3]).unwrap();
    println!("\nu64 add(ua, ub) -> {:?}", add(&ua.view(), &ub.view()).unwrap().as_slice());

    let ca = NdArray::from_vec(vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)], &[2]).unwrap();
    let cb = NdArray::from_vec(vec![Complex64::new(0.0, 1.0), Complex64::new(1.0, 1.0)], &[2]).unwrap();
    println!("\ncomplex128 add(ca, cb) -> {:?}", add(&ca.view(), &cb.view()).unwrap().as_slice());
}
