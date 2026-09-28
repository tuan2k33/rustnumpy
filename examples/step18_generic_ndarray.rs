//! Run: `cargo run --example step18_generic_ndarray`
//!
//! `NdArray<T = f64>` is genuinely generic over its element type -- one
//! struct definition, one set of method bodies, monomorphized per `T` at
//! compile time. This is the same idea real NumPy's `.c.src` templates
//! express at build time (one template, many generated C functions);
//! Rust's compiler does the equivalent expansion itself, so there's no
//! separate templating tool and no `NdArray<f64>`/`IntNdArray`/
//! `ComplexNdArray` triplication.
//!
//! Every other module in this crate (`ufunc`, `reductions`, `linalg`,
//! `fft`, `random`, `npy`, `testing`) still only writes the *bare* name
//! `NdArray`/`ArrayView` in its own signatures -- thanks to the struct's
//! default type parameter (`NdArray<T = f64>`), those bare references
//! keep meaning exactly `NdArray<f64>`, unchanged, with zero edits to any
//! of those files.

use rustnumpy::{AxisIndex, Complex64, NdArray};

fn main() {
    // The default: bare `NdArray` still means `NdArray<f64>`, exactly as
    // it always did -- everything from steps 1-17 keeps working.
    let floats: NdArray = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    println!("NdArray<f64> (default): {:?}", floats.as_slice());

    // Same struct, same methods, a different concrete T -- monomorphized
    // as its own specialized copy at compile time, not a runtime branch.
    let ints: NdArray<i32> = NdArray::from_vec(vec![1, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    println!("NdArray<i32>: {:?}, shape {:?}", ints.as_slice(), ints.shape());
    println!("ints.get([1, 2]) -> {:?}", ints.get(&[1, 2]));

    // Fancy indexing (index.rs's oindex/vindex) is generic too -- it only
    // ever copies T values around, no f64-specific arithmetic involved.
    let picked = ints.oindex(&[AxisIndex::Fancy(vec![0, 1]), AxisIndex::Single(2)]).unwrap();
    println!("ints.oindex(rows 0&1, col 2) -> {:?}", picked.as_slice());

    // A complex-valued array: the exact gap fft.rs's own ComplexArray
    // worked around (see its doc comment) now has a real, N-D, indexable
    // NdArray<Complex64> answer for future steps to build on.
    let complex: NdArray<Complex64> = NdArray::from_vec(
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)],
        &[2],
    )
    .unwrap();
    println!("NdArray<Complex64>: {:?}", complex.as_slice());

    // zeros() works for any T: Default + Clone -- 0 for i32, 0+0i for
    // Complex64, without writing a separate zero-fill per type.
    let zero_ints: NdArray<i32> = NdArray::zeros(&[3]);
    let zero_complex: NdArray<Complex64> = NdArray::zeros(&[2]);
    println!("\nNdArray::<i32>::zeros([3]) -> {:?}", zero_ints.as_slice());
    println!("NdArray::<Complex64>::zeros([2]) -> {:?}", zero_complex.as_slice());
}
