//! Run: `cargo run --example step11_utils`
//!
//! Step 11: `lib/`-layer utilities. Every value below was checked against
//! real NumPy 2.5.3 first (see `utils.rs`'s doc comment) -- including
//! `unique`'s NaN-collapsing and `tile`'s bidirectional shape padding.

use rustnumpy::{concatenate, gradient, interp, intersect1d, split, stack, tile, union1d, unique, NdArray};

fn main() {
    let data = [3.0, 1.0, 2.0, 1.0, f64::NAN, f64::NAN, 2.0];
    println!("unique([3,1,2,1,NaN,NaN,2]) -> {:?}", unique(&data));

    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [3.0, 4.0, 5.0, 6.0];
    println!("intersect1d(a, b) -> {:?}", intersect1d(&a, &b));
    println!("union1d(a, b) -> {:?}", union1d(&a, &b));

    let c1 = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let c2 = NdArray::from_vec(vec![5.0, 6.0], &[1, 2]).unwrap();
    let cat = concatenate(&[&c1, &c2], 0).unwrap();
    println!("\nconcatenate axis=0 -> shape={:?} data={:?}", cat.shape(), cat.as_slice());

    let av = NdArray::from_vec(a.to_vec(), &[4]).unwrap();
    let bv = NdArray::from_vec(b.to_vec(), &[4]).unwrap();
    let stacked = stack(&[&av, &bv], 0).unwrap();
    println!("stack axis=0 -> shape={:?} data={:?}", stacked.shape(), stacked.as_slice());

    let d = NdArray::from_vec((0..9).map(|i| i as f64).collect(), &[9]).unwrap();
    let parts = split(&d, 3, 0).unwrap();
    println!("\nsplit into 3 -> {:?}", parts.iter().map(|p| p.as_slice()).collect::<Vec<_>>());

    let e = NdArray::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    println!("tile([1,2], (2,2)) -> shape={:?} data={:?}", tile(&e, &[2, 2]).shape(), tile(&e, &[2, 2]).as_slice());

    let xp = [0.0, 1.0, 2.0, 3.0];
    let fp = [0.0, 10.0, 20.0, 30.0];
    println!("\ninterp([0.5,1.5,2.9]) -> {:?}", interp(&[0.5, 1.5, 2.9], &xp, &fp));
    println!("interp([-1,5]) (extrapolated, clamped) -> {:?}", interp(&[-1.0, 5.0], &xp, &fp));

    println!("\ngradient([1,2,4,7,11]) -> {:?}", gradient(&[1.0, 2.0, 4.0, 7.0, 11.0], 1.0));
}
