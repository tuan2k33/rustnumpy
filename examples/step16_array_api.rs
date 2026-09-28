use rustnumpy::{concat, concatenate, matrix_norm, multiply, subtract, NdArray};

fn main() {
    let a = NdArray::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    let b = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    println!("subtract(a, b) -> {:?}", subtract(&a.view(), &b.view()).unwrap().as_slice());
    println!("multiply(a, b) -> {:?}", multiply(&a.view(), &b.view()).unwrap().as_slice());

    let concatenated = concatenate(&[&a, &b], 0).unwrap();
    let via_concat = concat(&[&a, &b], 0).unwrap();
    println!("\nconcat(a, b, axis=0) -> {:?}", via_concat.as_slice());
    assert_eq!(concatenated, via_concat);

    println!("\nmatrix_norm(a) -> {}", matrix_norm(&a));
}
