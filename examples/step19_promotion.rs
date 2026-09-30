use rustnumpy::dispatch::{accumulate, add_assign, outer_with, reduce, ReduceOptions};
use rustnumpy::*;

fn main() {
    let ints = NdArray::from_vec(vec![1i32, 2, 3], &[3]).unwrap();
    let floats = NdArray::from_vec(vec![0.5f64, 0.5, 0.5], &[3]).unwrap();
    let kind = result_type([i32::KIND, f64::KIND], []).unwrap();
    let mixed = add(&ints.astype::<f64>().view(), &floats.view()).unwrap();
    println!("int32 + float64 -> {kind:?} {:?}", mixed.as_slice());

    println!("int8 array + 3 -> {:?}, + 3.0 -> {:?}", with_weak(i8::KIND, Weak::Int), with_weak(i8::KIND, Weak::Float));

    let mut acc = NdArray::from_vec(vec![1i32, 2, 3], &[3]).unwrap();
    add_assign(&mut acc.view_mut(), &NdArray::from_vec(vec![10i64, 10, 10], &[3]).unwrap().astype::<i32>().view()).unwrap();
    println!("int32 += int64 (same_kind cast first) -> {:?}", acc.as_slice());

    let m = NdArray::from_vec((0..6).collect::<Vec<i64>>(), &[2, 3]).unwrap();
    let sums = reduce(&m.view(), 1, ReduceOptions::default(), |a, b| a + b).unwrap();
    let running = accumulate(&m.view(), 1, |a, b| a + b).unwrap();
    let table = outer_with(&NdArray::from_vec(vec![1i64, 2], &[2]).unwrap().view(), &NdArray::from_vec(vec![10i64, 20], &[2]).unwrap().view(), |a, b| a * b).unwrap();
    println!("add.reduce -> {:?}, add.accumulate -> {:?}, multiply.outer -> {:?}", sums.as_slice(), running.as_slice(), table.as_slice());
}
