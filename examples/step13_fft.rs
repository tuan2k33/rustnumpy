//! Run: `cargo run --example step13_fft`
//!
//! Step 13: `fft` via `rustfft` (pure Rust, no FFI). Every value below was
//! checked against real NumPy 2.5.3 first (see `fft.rs`'s doc comment for
//! the tolerance-not-exact-equality policy this follows).

use rustnumpy::{fft, fft2, fftfreq, fftn, fftshift, ifft, ifftshift, irfft, rfft, rfftfreq, Complex64, ComplexArray};

fn main() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let complex_x: Vec<_> = x.iter().map(|&r| rustnumpy::Complex64::new(r, 0.0)).collect();

    let spectrum = fft(&complex_x).unwrap();
    println!("fft([1,2,3,4]) -> {spectrum:?}");
    println!("ifft(fft(x)) -> {:?}", ifft(&spectrum).unwrap());

    let half_spectrum = rfft(&x).unwrap();
    println!("\nrfft([1,2,3,4]) -> {half_spectrum:?}");
    println!("irfft(rfft(x), n=4) -> {:?}", irfft(&half_spectrum, 4).unwrap());

    println!("\nfftfreq(4) -> {:?}", fftfreq(4, 1.0).unwrap());
    println!("rfftfreq(4) -> {:?}", rfftfreq(4, 1.0).unwrap());

    let a = [0, 1, 2, 3];
    let shifted = fftshift(&a);
    println!("\nfftshift([0,1,2,3]) -> {shifted:?}");
    println!("ifftshift(fftshift(a)) -> {:?}", ifftshift(&shifted));

    // fftn/fft2: auto-detects ndim, loops a 1-D fft over every axis.
    let matrix = ComplexArray::from_vec(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(|r| Complex64::new(r, 0.0)).to_vec(),
        &[2, 3],
    )
    .unwrap();
    println!("\nfft2([[1,2,3],[4,5,6]]) -> {:?}", fft2(&matrix).unwrap().as_slice());

    let cube_data: Vec<Complex64> = (0..24).map(|i| Complex64::new(i as f64, 0.0)).collect();
    let cube = ComplexArray::from_vec(cube_data, &[2, 3, 4]).unwrap();
    let spectrum_3d = fftn(&cube).unwrap();
    println!("fftn(shape [2,3,4])[0] -> {:?}", spectrum_3d.as_slice()[0]);
}
