use rustnumpy::{assert_allclose, assert_allclose_default, assert_array_equal, NdArray};

fn arr(values: &[f64]) -> NdArray {
    NdArray::from_vec(values.to_vec(), &[values.len()]).unwrap()
}

fn main() {

    let a = arr(&[1.0, f64::NAN, 3.0]);
    let b = arr(&[1.0, f64::NAN, 3.0]);
    println!("assert_array_equal(NaN, NaN) -> {:?}", assert_array_equal(&a.view(), &b.view()));

    let x = arr(&[1.0]);
    let y = arr(&[1.0 + 1e-6]);
    match assert_allclose_default(&x.view(), &y.view()) {
        Ok(()) => println!("assert_allclose_default: unexpectedly passed"),
        Err(e) => println!("assert_allclose_default(1.0, 1.000001) -> Err:\n{e}\n"),
    }

    assert_allclose(&x.view(), &y.view(), 1e-5, 0.0).unwrap();
    println!("assert_allclose(1.0, 1.000001, rtol=1e-5) -> Ok");

    let p = arr(&[1.0, 2.0, 3.0]);
    let q = arr(&[1.0, 5.0, 3.0]);
    match assert_array_equal(&p.view(), &q.view()) {
        Ok(()) => println!("unexpectedly passed"),
        Err(e) => println!("assert_array_equal([1,2,3],[1,5,3]) -> Err:\n{e}"),
    }
}
