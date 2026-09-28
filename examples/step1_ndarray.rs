//! Run: `cargo run --example step1_ndarray`
//!
//! A visual demo of step 1's 3 main ideas: creating an array, slicing out
//! a view (no copy), and broadcasting when adding two arrays of different
//! shapes — compared directly against the equivalent NumPy result noted
//! in the comments.

use rustnumpy::{add_broadcast, NdArray};

fn main() {
    // np.arange(12).reshape(3, 4)
    let a = NdArray::from_vec((0..12).map(|i| i as f64).collect(), &[3, 4]).unwrap();
    println!("a.shape = {:?}, a.strides = {:?}", a.shape(), a.strides());

    // a[1:3, 1:3] -> a view, no copy
    let v = a.slice(&[1..3, 1..3]).unwrap();
    println!("a[1:3, 1:3] = view shape {:?}, strides {:?}", v.shape(), v.strides());
    for row in 0..v.shape()[0] {
        let vals: Vec<f64> = (0..v.shape()[1]).map(|col| v.get(&[row, col]).unwrap()).collect();
        println!("  row {row}: {vals:?}");
    }
    // Expected: [[5, 6], [9, 10]] — same as np.arange(12).reshape(3,4)[1:3,1:3]

    // (3,4) + (4,) -> broadcast a row vector across every row of the matrix
    let row_vec = NdArray::from_vec(vec![100.0, 200.0, 300.0, 400.0], &[4]).unwrap();
    let summed = add_broadcast(&a.view(), &row_vec.view()).unwrap();
    println!("\na + row_vec (broadcast (3,4)+(4,)):");
    for row in 0..summed.shape()[0] {
        let vals: Vec<f64> = (0..summed.shape()[1])
            .map(|col| summed.get(&[row, col]).unwrap())
            .collect();
        println!("  row {row}: {vals:?}");
    }

    // Materialize a broadcast view into an array that owns its own data
    let col = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
    let stretched = col.view().broadcast_to(&[3, 4]).unwrap().to_owned();
    println!("\ncol (3,1) broadcast to (3,4), then to_owned():");
    println!("  shape {:?}, is_c_contiguous = {}", stretched.shape(), stretched.is_c_contiguous());
}
