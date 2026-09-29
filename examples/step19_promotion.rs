use rustnumpy::dispatch::{accumulate, add_assign, add_weak_float, add_weak_int, outer_with, reduce, ReduceOptions};
use rustnumpy::*;

fn main() {
    let ints = NdArray::from_vec(vec![1i32, 2, 3], &[3]).unwrap();
    let floats = NdArray::from_vec(vec![0.5f64, 0.5, 0.5], &[3]).unwrap();
    let mixed: NdArray<f64> = add(&ints.view(), &floats.view()).unwrap();
    println!("int32 + float64 -> {:?}", mixed.as_slice());

    let small = NdArray::from_vec(vec![1i8], &[1]).unwrap();
    let kept: NdArray<i8> = add_weak_int(&small.view(), 3).unwrap();
    let widened: NdArray<f64> = add_weak_float(&small.view(), 3.0).unwrap();
    println!("int8 + 3 -> {:?}, int8 + 3.0 -> {:?}", kept.as_slice(), widened.as_slice());
    println!("int8 + 300 -> {:?}", add_weak_int(&small.view(), 300).unwrap_err());

    let mut acc = NdArray::from_vec(vec![1i32, 2, 3], &[3]).unwrap();
    add_assign(&mut acc.view_mut(), &NdArray::from_vec(vec![10i64, 10, 10], &[3]).unwrap().view()).unwrap();
    println!("int32 += int64 -> {:?}", acc.as_slice());

    let m = NdArray::from_vec((0..6).collect::<Vec<i64>>(), &[2, 3]).unwrap();
    let sums = reduce(&m.view(), 1, ReduceOptions::default(), |a, b| a + b).unwrap();
    let running = accumulate(&m.view(), 1, |a, b| a + b).unwrap();
    let table = outer_with(&NdArray::from_vec(vec![1i64, 2], &[2]).unwrap().view(), &NdArray::from_vec(vec![10i64, 20], &[2]).unwrap().view(), |a, b| a * b).unwrap();
    println!("add.reduce -> {:?}, add.accumulate -> {:?}, multiply.outer -> {:?}", sums.as_slice(), running.as_slice(), table.as_slice());
}
