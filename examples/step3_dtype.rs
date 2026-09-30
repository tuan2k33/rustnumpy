use rustnumpy::{can_cast, common_dtype, with_weak, CastSafety, DType, Kind, NdArray, Weak};

fn show_common(a: Kind, b: Kind) {
    println!("  common_dtype({a:?}, {b:?}) = {:?}", common_dtype(a, b));
}

fn show_weak(array: Kind, scalar: Weak) {
    let result = with_weak(array, scalar);
    println!("  array({array:?}) + weak_scalar({scalar:?}) -> {result:?}");
}

fn main() {
    println!("-- concrete dtype x concrete dtype (like np.result_type) --");
    show_common(Kind::Int(16), Kind::Float(32));
    show_common(Kind::Int(32), Kind::Float(32));
    show_common(Kind::Bool, Kind::Int(32));
    show_common(Kind::Int(8), Kind::Int(64));

    println!("\n-- array dtype x Python weak scalar (NEP 50) --");
    show_weak(Kind::Bool, Weak::Int);
    show_weak(Kind::Int(32), Weak::Int);
    show_weak(Kind::Int(32), Weak::Float);
    show_weak(Kind::Float(32), Weak::Float);
    show_weak(Kind::Float(32), Weak::Complex);

    println!("\n-- mixed dtypes: resolve the common kind, then cast explicitly --");
    let ints = NdArray::from_vec(vec![1i32, 2, 3], &[3]).unwrap();
    let floats = NdArray::from_vec(vec![0.5f32, 0.5, 0.5], &[3]).unwrap();
    println!("  common_dtype({}, {}) = {:?}", i32::type_name(), f32::type_name(), common_dtype(i32::KIND, f32::KIND));
    let sum = rustnumpy::add(&ints.astype::<f64>().view(), &floats.astype::<f64>().view()).unwrap();
    println!("  ints.astype::<f64>() + floats.astype::<f64>() = {:?}", sum.as_slice());

    println!("\n-- casting safety levels --");
    for (from, to) in [
        (Kind::Int(16), Kind::Int(32)),
        (Kind::Int(32), Kind::Int(16)),
        (Kind::Float(64), Kind::Float(32)),
        (Kind::Int(8), Kind::Bool),
        (Kind::Bool, Kind::Int(8)),
    ] {
        let safety: CastSafety = can_cast(from, to);
        println!("  can_cast({from:?} -> {to:?}) = {safety:?}");
    }
}
