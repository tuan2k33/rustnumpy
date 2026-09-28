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

    print_matrix("a + b", &add(&a.view(), &b.view()).unwrap());

    print_matrix("a * b", &mul(&a.view(), &b.view()).unwrap());

    let col = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
    let ones = NdArray::from_vec(vec![1.0; 12], &[3, 4]).unwrap();
    print_matrix("col (3,1) * ones (3,4)", &mul(&col.view(), &ones.view()).unwrap());

    let mut out = NdArray::zeros(&[2, 2]);
    zip_with_into(&mut out.view_mut(), &a.view(), &b.view(), |x, y| x + y).unwrap();
    print_matrix("out= (pre-allocated buffer, filled by zip_with_into)", &out);

    let squares = NdArray::from_vec(vec![1.0, 4.0, 9.0, 16.0], &[4]).unwrap();
    let roots = map(&squares.view(), f64::sqrt);
    println!("sqrt({:?}) = {:?}", squares.as_slice(), roots.as_slice());
}
