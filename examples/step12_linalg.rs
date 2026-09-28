//! Run: `cargo run --example step12_linalg`
//!
//! Step 12: `linalg` via `faer` (pure Rust, no LAPACK/FFI). Every value
//! below was checked against real NumPy 2.5.3 first (see `linalg.rs`'s
//! doc comment for the tolerance-not-exact-equality policy this follows).

use rustnumpy::{cholesky, det, eigh, eigvals, frobenius_norm, inv, matrix_power, qr, solve, svd, NdArray};

fn main() {
    let a = NdArray::from_vec(vec![4.0, 3.0, 6.0, 3.0], &[2, 2]).unwrap();
    let b = NdArray::from_vec(vec![1.0, 2.0], &[2]).unwrap();

    println!("solve(A, b) -> {:?}", solve(&a, &b).unwrap().as_slice());
    println!("inv(A) -> {:?}", inv(&a).unwrap().as_slice());
    println!("det(A) -> {}", det(&a).unwrap());

    let (q, r) = qr(&a).unwrap();
    println!("\nqr(A).Q -> {:?}", q.as_slice());
    println!("qr(A).R -> {:?}", r.as_slice());

    let s = NdArray::from_vec(vec![4.0, 2.0, 2.0, 3.0], &[2, 2]).unwrap();
    println!("\ncholesky(S) -> {:?}", cholesky(&s).unwrap().as_slice());

    let (values, _vectors) = eigh(&s).unwrap();
    println!("eigh(S).values -> {values:?}");

    let rot = NdArray::from_vec(vec![0.0, -1.0, 1.0, 0.0], &[2, 2]).unwrap();
    println!("eigvals(rotation matrix) -> {:?}", eigvals(&rot).unwrap());

    let (_u, singular_values, _vt) = svd(&a).unwrap();
    println!("\nsvd(A).singular_values -> {singular_values:?}");
    println!("frobenius_norm(A) -> {}", frobenius_norm(&a));

    println!("\nmatrix_power(A, 3) -> {:?}", matrix_power(&a, 3).unwrap().as_slice());
    println!("matrix_power(A, 0) -> {:?}", matrix_power(&a, 0).unwrap().as_slice());
    println!("matrix_power(A, -1) -> {:?}", matrix_power(&a, -1).unwrap().as_slice());
}
