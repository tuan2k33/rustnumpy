//! Chạy: `cargo run --example step1_ndarray`
//!
//! Minh họa trực quan 3 ý chính của bước 1: tạo mảng, slice ra view
//! (không copy), và broadcasting khi cộng hai mảng khác shape — so sánh
//! trực tiếp với kết quả NumPy tương ứng ghi trong comment.

use rustnumpy::{add_broadcast, NdArray};

fn main() {
    // np.arange(12).reshape(3, 4)
    let a = NdArray::from_vec((0..12).map(|i| i as f64).collect(), &[3, 4]).unwrap();
    println!("a.shape = {:?}, a.strides = {:?}", a.shape(), a.strides());

    // a[1:3, 1:3] -> view, không copy
    let v = a.slice(&[1..3, 1..3]).unwrap();
    println!("a[1:3, 1:3] = view shape {:?}, strides {:?}", v.shape(), v.strides());
    for row in 0..v.shape()[0] {
        let vals: Vec<f64> = (0..v.shape()[1]).map(|col| v.get(&[row, col]).unwrap()).collect();
        println!("  row {row}: {vals:?}");
    }
    // Kỳ vọng: [[5, 6], [9, 10]] — giống np.arange(12).reshape(3,4)[1:3,1:3]

    // (3,4) + (4,) -> broadcast hàng vector qua từng dòng ma trận
    let row_vec = NdArray::from_vec(vec![100.0, 200.0, 300.0, 400.0], &[4]).unwrap();
    let summed = add_broadcast(&a.view(), &row_vec.view()).unwrap();
    println!("\na + row_vec (broadcast (3,4)+(4,)):");
    for row in 0..summed.shape()[0] {
        let vals: Vec<f64> = (0..summed.shape()[1])
            .map(|col| summed.get(&[row, col]).unwrap())
            .collect();
        println!("  row {row}: {vals:?}");
    }

    // Vật chất hóa một view broadcast thành mảng sở hữu dữ liệu riêng
    let col = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
    let stretched = col.view().broadcast_to(&[3, 4]).unwrap().to_owned();
    println!("\ncol (3,1) broadcast to (3,4), then to_owned():");
    println!("  shape {:?}, is_c_contiguous = {}", stretched.shape(), stretched.is_c_contiguous());
}
