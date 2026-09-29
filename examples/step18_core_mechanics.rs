use rustnumpy::gufunc::gufunc;
use rustnumpy::mathfunc;
use rustnumpy::*;

fn main() {
    let a = NdArray::from_vec((0..12).map(f64::from).collect(), &[12]).unwrap();
    let m = a.reshape(&[3, -1]).unwrap();
    println!("reshape(3, -1) -> {:?}", m.shape());

    let shuffled = NdArray::from_vec(vec![3.0, f64::NAN, 1.0, 2.0], &[4]).unwrap();
    println!("sort (NaN last) -> {:?}", sort(&shuffled.view(), 0).unwrap().as_slice());
    println!("argsort -> {:?}", argsort(&shuffled.view(), 0).unwrap().as_slice());
    println!("searchsorted -> {:?}", searchsorted(&[1, 2, 2, 5], &[2, 3], Side::Right));

    println!("sqrt -> {:?}", mathfunc::sqrt(&m.slice(&[0..1, 0..4]).unwrap()).as_slice());

    let mask = NdArray::from_vec(vec![true, false, true, false], &[4]).unwrap();
    let x = NdArray::from_vec(vec![1, 2, 3, 4], &[4]).unwrap();
    let y = NdArray::from_vec(vec![-1, -2, -3, -4], &[4]).unwrap();
    println!("where -> {:?}", where_cond(&mask.view(), &x.view(), &y.view()).unwrap().as_slice());

    let parts = array_split(&NdArray::from_vec((0..10).collect::<Vec<i32>>(), &[10]).unwrap(), 3, 0).unwrap();
    println!("array_split(10, 3) -> {:?}", parts.iter().map(|p| p.len()).collect::<Vec<_>>());

    let (p, q) = (m.to_owned(), a.reshape(&[4, 3]).unwrap().to_owned());
    println!("matmul -> {:?}", matmul(&p.view(), &q.view()).unwrap().as_slice());
    println!("einsum ij,ij->i -> {:?}", einsum("ij,ij->i", &[&p.view(), &p.view()]).unwrap().as_slice());

    let v = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let norms = gufunc("(n)->()", &[&NdArray::from_vec(vec![3.0, 4.0, 6.0, 8.0], &[2, 2]).unwrap().view()], |i, o, _| {
        o[0][0] = i[0].iter().map(|x| x * x).sum::<f64>().sqrt();
    })
    .unwrap();
    println!("gufunc (n)->() -> {:?}", norms[0].as_slice());
    println!("vecdot -> {:?}", vecdot(&v.view(), &v.view()).unwrap().as_slice());
}
