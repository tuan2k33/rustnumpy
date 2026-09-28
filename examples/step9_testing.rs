//! Run: `cargo run --example step9_testing`
//!
//! Step 9: a `numpy.testing` equivalent. Every default/boundary here was
//! checked against real NumPy 2.5.3 first (see `testing.rs`'s doc
//! comment) -- including the two easy-to-get-wrong details demonstrated
//! below: `assert_array_equal` treats `NaN == NaN` as equal, and
//! `assert_allclose`'s default tolerance is tighter than `np.isclose`'s.

use rustnumpy::{assert_allclose, assert_allclose_default, assert_array_equal, NdArray};

fn arr(values: &[f64]) -> NdArray {
    NdArray::from_vec(values.to_vec(), &[values.len()]).unwrap()
}

fn main() {
    // np.testing.assert_array_equal([1,NaN,3],[1,NaN,3]) passes -- NaN
    // counts as equal here, unlike plain `==`.
    let a = arr(&[1.0, f64::NAN, 3.0]);
    let b = arr(&[1.0, f64::NAN, 3.0]);
    println!("assert_array_equal(NaN, NaN) -> {:?}", assert_array_equal(&a.view(), &b.view()));

    // np.testing.assert_allclose(1.0, 1.0 + 1e-6) fails: default rtol=1e-7
    // is tighter than most people expect.
    let x = arr(&[1.0]);
    let y = arr(&[1.0 + 1e-6]);
    match assert_allclose_default(&x.view(), &y.view()) {
        Ok(()) => println!("assert_allclose_default: unexpectedly passed"),
        Err(e) => println!("assert_allclose_default(1.0, 1.000001) -> Err:\n{e}\n"),
    }

    // Loosen the tolerance explicitly and it passes.
    assert_allclose(&x.view(), &y.view(), 1e-5, 0.0).unwrap();
    println!("assert_allclose(1.0, 1.000001, rtol=1e-5) -> Ok");

    // A real value mismatch, to show the message shape.
    let p = arr(&[1.0, 2.0, 3.0]);
    let q = arr(&[1.0, 5.0, 3.0]);
    match assert_array_equal(&p.view(), &q.view()) {
        Ok(()) => println!("unexpectedly passed"),
        Err(e) => println!("assert_array_equal([1,2,3],[1,5,3]) -> Err:\n{e}"),
    }
}
