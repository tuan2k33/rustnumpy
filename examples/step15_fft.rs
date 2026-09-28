//! Run: `cargo run --example step15_fft`
//!
//! Step 15: `fft` via `rustfft` (pure Rust, no FFI). Every value below was
//! checked against real NumPy 2.5.3 first (see `fft.rs`'s doc comment for
//! the tolerance-not-exact-equality policy this follows).

use rustnumpy::{fft, fftfreq, fftshift, ifft, ifftshift, irfft, rfft, rfftfreq};

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
}
