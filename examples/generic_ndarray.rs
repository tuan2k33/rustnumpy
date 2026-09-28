use rustnumpy::{AxisIndex, Complex64, NdArray};

fn main() {

    let floats: NdArray = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    println!("NdArray<f64> (default): {:?}", floats.as_slice());

    let ints: NdArray<i32> = NdArray::from_vec(vec![1, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    println!("NdArray<i32>: {:?}, shape {:?}", ints.as_slice(), ints.shape());
    println!("ints.get([1, 2]) -> {:?}", ints.get(&[1, 2]));

    let picked = ints.oindex(&[AxisIndex::Fancy(vec![0, 1]), AxisIndex::Single(2)]).unwrap();
    println!("ints.oindex(rows 0&1, col 2) -> {:?}", picked.as_slice());

    let complex: NdArray<Complex64> = NdArray::from_vec(
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)],
        &[2],
    )
    .unwrap();
    println!("NdArray<Complex64>: {:?}", complex.as_slice());

    let zero_ints: NdArray<i32> = NdArray::zeros(&[3]);
    let zero_complex: NdArray<Complex64> = NdArray::zeros(&[2]);
    println!("\nNdArray::<i32>::zeros([3]) -> {:?}", zero_ints.as_slice());
    println!("NdArray::<Complex64>::zeros([2]) -> {:?}", zero_complex.as_slice());
}
