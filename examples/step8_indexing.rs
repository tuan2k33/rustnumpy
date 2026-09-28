use rustnumpy::{AxisIndex, NdArray};

fn arange(shape: &[usize]) -> NdArray {
    let len: usize = shape.iter().product();
    NdArray::from_vec((0..len).map(|i| i as f64).collect(), shape).unwrap()
}

fn main() {
    let a = arange(&[4, 6]);
    println!("a = np.arange(24).reshape(4, 6)\n");

    let out = a.vindex(&[AxisIndex::Fancy(vec![0, 2]), AxisIndex::Slice(1..4)]).unwrap();
    println!("a.vindex([Fancy([0,2]), Slice(1..4)])  (== a[[0,2], 1:4])");
    println!("  shape={:?}  data={:?}\n", out.shape(), out.as_slice());

    let out = a.vindex(&[AxisIndex::Fancy(vec![0, 1, 2]), AxisIndex::Fancy(vec![0, 1, 2])]).unwrap();
    println!("a.vindex([Fancy([0,1,2]), Fancy([0,1,2])])  (== a[[0,1,2],[0,1,2]])");
    println!("  shape={:?}  data={:?}\n", out.shape(), out.as_slice());

    let out = a.oindex(&[AxisIndex::Fancy(vec![0, 2]), AxisIndex::Fancy(vec![1, 3, 4])]).unwrap();
    println!("a.oindex([Fancy([0,2]), Fancy([1,3,4])])  (== a[np.ix_([0,2],[1,3,4])])");
    println!("  shape={:?}  data={:?}\n", out.shape(), out.as_slice());

    let mask: Vec<bool> = a.as_slice().iter().map(|&x| x as i64 % 2 == 0).collect();
    let out = a.boolean_index(&mask).unwrap();
    println!("a.boolean_index(a % 2 == 0)  (== a[a % 2 == 0])");
    println!("  shape={:?}  data={:?}\n", out.shape(), out.as_slice());

    let b = arange(&[2, 3, 4]);
    let err = b
        .vindex(&[AxisIndex::Fancy(vec![0, 1]), AxisIndex::Full, AxisIndex::Fancy(vec![0, 1])])
        .unwrap_err();
    println!("b.vindex with non-adjacent fancy axes -> Err (real NumPy's axis-jump rule is not implemented):");
    println!("  {err}");
}
