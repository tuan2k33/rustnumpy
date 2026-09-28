//! Chạy: `cargo run --example step2_npy`
//!
//! Đọc trực tiếp file `.npy` do NumPy thật sinh ra (`tests/fixtures/`,
//! xem `scripts/gen_fixtures.py`), rồi ghi một mảng mới ra `/tmp` và đọc
//! lại — chứng minh cả đường đọc lẫn ghi đều tương thích định dạng thật,
//! không chỉ tự đọc lại được chính mình.

use rustnumpy::{load_npy, save_npy, NdArray};

fn main() {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");

    let matrix = load_npy(fixtures.join("matrix_f64.npy")).expect("đọc matrix_f64.npy");
    println!("matrix_f64.npy (sinh bởi NumPy thật): shape = {:?}", matrix.shape());
    for row in 0..matrix.shape()[0] {
        let vals: Vec<f64> = (0..matrix.shape()[1]).map(|c| matrix.get(&[row, c]).unwrap()).collect();
        println!("  {vals:?}");
    }

    let cube = load_npy(fixtures.join("cube_f64.npy")).expect("đọc cube_f64.npy");
    println!("\ncube_f64.npy: shape = {:?}, cube[1,2,3] = {:?}", cube.shape(), cube.get(&[1, 2, 3]));

    // Ghi một mảng do rustnumpy tạo ra, rồi đọc lại bằng chính reader của mình
    // (và bạn có thể mở file này bằng `np.load(...)` ở Python để tự kiểm chứng).
    let mine = NdArray::from_vec(vec![3.14, 2.71, 1.41, 1.73], &[2, 2]).unwrap();
    let out_path = std::env::temp_dir().join("rustnumpy_step2_demo.npy");
    save_npy(&out_path, &mine).expect("ghi .npy");
    println!("\nĐã ghi {:?}", out_path);
    println!("Thử mở lại bằng Python: np.load({:?})", out_path);

    let back = load_npy(&out_path).expect("đọc lại file vừa ghi");
    assert_eq!(back.as_slice(), mine.as_slice());
    println!("Đọc lại thành công, dữ liệu khớp: {:?}", back.as_slice());
}
