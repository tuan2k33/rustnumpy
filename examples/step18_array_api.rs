//! Run: `cargo run --example step18_array_api`
//!
//! Step 18: audit this crate's public function names against the Python
//! Array API standard (v2022.12, NEP 56). See `NumPy.md`'s "Step 18
//! audit" section for the full comparison table. This example just shows
//! the standard-aligned names added here are thin wrappers -- same
//! result as the original, NumPy-flavored name each one replaces/joins.

use rustnumpy::{concat, concatenate, matrix_norm, multiply, subtract, NdArray};

fn main() {
    let a = NdArray::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    let b = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    // `subtract`/`multiply`: the Array API standard's own elementwise
    // names, wrapping this crate's original `sub`/`mul` from step 4.
    println!("subtract(a, b) -> {:?}", subtract(&a.view(), &b.view()).unwrap().as_slice());
    println!("multiply(a, b) -> {:?}", multiply(&a.view(), &b.view()).unwrap().as_slice());

    // `concat`: the standard's own name for step 13's `concatenate`
    // (NumPy itself keeps both names too, even after adopting NEP 56).
    let concatenated = concatenate(&[&a, &b], 0).unwrap();
    let via_concat = concat(&[&a, &b], 0).unwrap();
    println!("\nconcat(a, b, axis=0) -> {:?}", via_concat.as_slice());
    assert_eq!(concatenated, via_concat);

    // `matrix_norm`: the standard's own name for step 14's
    // `frobenius_norm` (narrower than the standard: Frobenius only, no
    // `ord=` parameter -- see `matrix_norm`'s own doc comment).
    println!("\nmatrix_norm(a) -> {}", matrix_norm(&a));
}
