//! Run: `cargo run --example step4_ufunc`
//!
//! Shows the same broadcasting engine from step 1, now generalized behind
//! a closure (`zip_with`) so `add`/`sub`/`mul` are one-liners, plus the
//! `out=` pattern (`zip_with_into`) and a unary ufunc (`map`). Every
//! result is cross-checked against real NumPy — see the comments.

use rustnumpy::{add, map, mul, zip_with_into, NdArray};

fn print_matrix(label: &str, arr: &NdArray) {
    println!("{label}: shape {:?}", arr.shape());
    for row in 0..arr.shape()[0] {
        let vals: Vec<f64> = (0..arr.shape()[1]).map(|c| arr.get(&[row, c]).unwrap()).collect();
        println!("  {vals:?}");
    }
}

fn main() {
    let a = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = NdArray::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();

    // np.array([[1,2],[3,4]]) + np.array([[10,20],[30,40]]) -> [[11,22],[33,44]]
    print_matrix("a + b", &add(&a.view(), &b.view()).unwrap());
    // -> [[10,40],[90,160]]
    print_matrix("a * b", &mul(&a.view(), &b.view()).unwrap());

    // Broadcasting still works exactly like step 1, just through mul() now.
    let col = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
    let ones = NdArray::from_vec(vec![1.0; 12], &[3, 4]).unwrap();
    print_matrix("col (3,1) * ones (3,4)", &mul(&col.view(), &ones.view()).unwrap());

    // The `out=` pattern: write into a buffer we already own instead of
    // allocating a new NdArray. Matches `np.add(a, b, out=out)`.
    let mut out = NdArray::zeros(&[2, 2]);
    zip_with_into(&mut out.view_mut(), &a.view(), &b.view(), |x, y| x + y).unwrap();
    print_matrix("out= (pre-allocated buffer, filled by zip_with_into)", &out);

    // A unary ufunc: no broadcasting needed, just map() over every element.
    let squares = NdArray::from_vec(vec![1.0, 4.0, 9.0, 16.0], &[4]).unwrap();
    let roots = map(&squares.view(), f64::sqrt);
    println!("sqrt({:?}) = {:?}", squares.as_slice(), roots.as_slice());
}
