//! `numpy.fft` — `fft`/`ifft` (complex), `rfft`/`irfft` (real input),
//! `fftfreq`/`rfftfreq`, `fftshift`/`ifftshift`.
//!
//! Delegates the actual transform to the `rustfft` crate (pure Rust, no
//! FFI) per the same "depend on a crate, don't hand-convert" decision
//! `linalg.rs` follows for `faer`/LAPACK: `pocketfft` is a numerical
//! *tool* NumPy vendors, not one of NumPy's own design ideas.
//!
//! **Scope note**: these functions operate on plain `Vec<Complex64>`/
//! `Vec<f64>`, not `NdArray` — `NdArray` is f64-only (no complex dtype
//! yet, see `lib.rs`'s doc comment), and a spectrum is inherently complex,
//! so there's no lossless way to hand it back as an `NdArray` today. Only
//! 1-D transforms are implemented (no `fft2`/`fftn`).
//!
//! **Known, accepted deviation from NumPy**: `rustfft` and `pocketfft` are
//! both mathematically correct but not bit-for-bit identical (different
//! floating-point summation order -> ULP-level error) — compare with a
//! tolerance, never exact equality, matching `NumPy.md`'s project-wide
//! testing policy.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// A complex sample/spectrum value — `f64` real and imaginary parts.
pub type Complex64 = Complex<f64>;

/// Any error from an `fft` operation.
#[derive(Debug, Clone, PartialEq)]
pub enum FftError {
    /// `fft`/`ifft`/`rfft`/`irfft` were given a zero-length input.
    EmptyInput,
    /// `irfft`'s input doesn't have the `n / 2 + 1` entries a real
    /// spectrum of output length `n` requires.
    LengthMismatch { expected: usize, got: usize },
}

/// `np.fft.fft(a)`: the discrete Fourier transform of a complex sequence.
/// Unnormalized (matches NumPy: forward has no `1/n` scaling, `ifft` does).
pub fn fft(input: &[Complex64]) -> Result<Vec<Complex64>, FftError> {
    if input.is_empty() {
        return Err(FftError::EmptyInput);
    }
    let mut buffer = input.to_vec();
    let mut planner = FftPlanner::new();
    planner.plan_fft_forward(buffer.len()).process(&mut buffer);
    Ok(buffer)
}

/// `np.fft.ifft(a)`: the inverse discrete Fourier transform. Normalized by
/// `1/n` (`rustfft`'s own inverse transform, like `pocketfft`'s, is
/// unnormalized -- this divides by `n` itself to match NumPy's `ifft`).
pub fn ifft(input: &[Complex64]) -> Result<Vec<Complex64>, FftError> {
    if input.is_empty() {
        return Err(FftError::EmptyInput);
    }
    let n = input.len();
    let mut buffer = input.to_vec();
    let mut planner = FftPlanner::new();
    planner.plan_fft_inverse(n).process(&mut buffer);
    let scale = 1.0 / n as f64;
    buffer.iter_mut().for_each(|c| *c *= scale);
    Ok(buffer)
}

/// `np.fft.rfft(a)`: the FFT of a real sequence, returning only the
/// non-redundant half (indices `0..=n/2`) since a real input's spectrum is
/// Hermitian-symmetric (`X[n-k] == conj(X[k])`) -- the other half carries
/// no extra information, exactly as real NumPy documents.
pub fn rfft(input: &[f64]) -> Result<Vec<Complex64>, FftError> {
    if input.is_empty() {
        return Err(FftError::EmptyInput);
    }
    let complex_input: Vec<Complex64> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let full = fft(&complex_input)?;
    Ok(full[..input.len() / 2 + 1].to_vec())
}

/// `np.fft.irfft(a, n)`: the inverse of [`rfft`], reconstructing a
/// length-`n` real sequence from its non-redundant half-spectrum (`input`
/// must have exactly `n / 2 + 1` entries -- the caller passes `n`
/// explicitly because that length can't be recovered from `input.len()`
/// alone, matching real NumPy's own `n=` parameter for the same reason).
pub fn irfft(input: &[Complex64], n: usize) -> Result<Vec<f64>, FftError> {
    if n == 0 {
        return Err(FftError::EmptyInput);
    }
    let expected = n / 2 + 1;
    if input.len() != expected {
        return Err(FftError::LengthMismatch { expected, got: input.len() });
    }
    let full: Vec<Complex64> =
        (0..n).map(|k| if k <= n / 2 { input[k] } else { input[n - k].conj() }).collect();
    Ok(ifft(&full)?.iter().map(|c| c.re).collect())
}

/// `np.fft.fftfreq(n, d)`: the sample frequencies for an `n`-point `fft`
/// output, in cycles per unit of `d` (the sample spacing) -- `[0, 1, ...,
/// n/2-1, -n/2, ..., -1] / (n*d)` for even `n`, `[0, 1, ..., (n-1)/2,
/// -(n-1)/2, ..., -1] / (n*d)` for odd `n`.
pub fn fftfreq(n: usize, d: f64) -> Result<Vec<f64>, FftError> {
    if n == 0 {
        return Err(FftError::EmptyInput);
    }
    let positive_count = n.div_ceil(2);
    let scale = 1.0 / (n as f64 * d);
    Ok((0..n)
        .map(|i| {
            let unscaled =
                if i < positive_count { i as f64 } else { i as f64 - n as f64 };
            unscaled * scale
        })
        .collect())
}

/// `np.fft.rfftfreq(n, d)`: the sample frequencies for an `n`-point
/// [`rfft`] output (`0..=n/2`), matching [`fftfreq`]'s scaling.
pub fn rfftfreq(n: usize, d: f64) -> Result<Vec<f64>, FftError> {
    if n == 0 {
        return Err(FftError::EmptyInput);
    }
    let scale = 1.0 / (n as f64 * d);
    Ok((0..=n / 2).map(|i| i as f64 * scale).collect())
}

/// `np.fft.fftshift(x)`: rotates the zero-frequency component to the
/// center of the spectrum.
pub fn fftshift<T: Clone>(a: &[T]) -> Vec<T> {
    rotate_left_copy(a, a.len().div_ceil(2))
}

/// `np.fft.ifftshift(x)`: the exact inverse of [`fftshift`] (identical to
/// it for even-length input; for odd-length input the rotation amount is
/// `n/2` instead of `n.div_ceil(2)`).
pub fn ifftshift<T: Clone>(a: &[T]) -> Vec<T> {
    rotate_left_copy(a, a.len() / 2)
}

fn rotate_left_copy<T: Clone>(a: &[T], mid: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(a.len());
    out.extend_from_slice(&a[mid..]);
    out.extend_from_slice(&a[..mid]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_complex_close(actual: &[Complex64], expected: &[Complex64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a.re - e.re).abs() < 1e-9, "re: {a:?} vs {e:?}");
            assert!((a.im - e.im).abs() < 1e-9, "im: {a:?} vs {e:?}");
        }
    }

    fn assert_reals_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-9, "{a} vs {e}");
        }
    }

    // Every expected value below was computed by real NumPy 2.5.3 first
    // (see this module's doc comment for the tolerance-not-exact-equality
    // policy).

    #[test]
    fn fft_matches_numpy() {
        let x: Vec<Complex64> = [1.0, 2.0, 3.0, 4.0].iter().map(|&r| Complex::new(r, 0.0)).collect();
        let expected = [(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)]
            .map(|(re, im)| Complex::new(re, im));
        assert_complex_close(&fft(&x).unwrap(), &expected);
    }

    #[test]
    fn fft_of_complex_input_matches_numpy() {
        // np.fft.fft([1+1j, 2-1j, 0+0j, -1+2j]) -> [2+2j, -2-2j, 0+0j, 4+4j]
        let x = [(1.0, 1.0), (2.0, -1.0), (0.0, 0.0), (-1.0, 2.0)]
            .map(|(re, im)| Complex::new(re, im));
        let expected =
            [(2.0, 2.0), (-2.0, -2.0), (0.0, 0.0), (4.0, 4.0)].map(|(re, im)| Complex::new(re, im));
        assert_complex_close(&fft(&x).unwrap(), &expected);
    }

    #[test]
    fn ifft_of_fft_roundtrips() {
        let x: Vec<Complex64> = [1.0, 2.0, 3.0, 4.0].iter().map(|&r| Complex::new(r, 0.0)).collect();
        let spectrum = fft(&x).unwrap();
        let back = ifft(&spectrum).unwrap();
        let expected: Vec<Complex64> = x;
        assert_complex_close(&back, &expected);
    }

    #[test]
    fn rfft_matches_numpy() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let expected =
            [(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0)].map(|(re, im)| Complex::new(re, im));
        assert_complex_close(&rfft(&x).unwrap(), &expected);
    }

    #[test]
    fn rfft_matches_numpy_for_odd_length() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let expected = [(15.0, 0.0), (-2.5, 3.440_954_801_177_934), (-2.5, 0.812_299_240_582_266)]
            .map(|(re, im)| Complex::new(re, im));
        assert_complex_close(&rfft(&x).unwrap(), &expected);
    }

    #[test]
    fn irfft_roundtrips_even_and_odd_length() {
        let even = [1.0, 2.0, 3.0, 4.0];
        assert_reals_close(&irfft(&rfft(&even).unwrap(), even.len()).unwrap(), &even);

        let odd = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_reals_close(&irfft(&rfft(&odd).unwrap(), odd.len()).unwrap(), &odd);
    }

    #[test]
    fn irfft_rejects_wrong_length() {
        // rfft of a length-4 input has 3 entries; n=6 expects 6/2+1 = 4.
        let spectrum = rfft(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(
            irfft(&spectrum, 6).unwrap_err(),
            FftError::LengthMismatch { expected: 4, got: 3 }
        );
    }

    #[test]
    fn fftfreq_matches_numpy() {
        assert_reals_close(&fftfreq(4, 1.0).unwrap(), &[0.0, 0.25, -0.5, -0.25]);
        assert_reals_close(&fftfreq(5, 1.0).unwrap(), &[0.0, 0.2, 0.4, -0.4, -0.2]);
        assert_reals_close(&fftfreq(4, 0.5).unwrap(), &[0.0, 0.5, -1.0, -0.5]);
    }

    #[test]
    fn rfftfreq_matches_numpy() {
        assert_reals_close(&rfftfreq(4, 1.0).unwrap(), &[0.0, 0.25, 0.5]);
    }

    #[test]
    fn fftshift_matches_numpy_even_and_odd() {
        assert_eq!(fftshift(&[0, 1, 2, 3]), vec![2, 3, 0, 1]);
        assert_eq!(fftshift(&[0, 1, 2, 3, 4]), vec![3, 4, 0, 1, 2]);
    }

    #[test]
    fn ifftshift_undoes_fftshift() {
        let x = [0, 1, 2, 3, 4];
        assert_eq!(ifftshift(&fftshift(&x)), x.to_vec());
        let y = [0, 1, 2, 3];
        assert_eq!(ifftshift(&fftshift(&y)), y.to_vec());
    }

    #[test]
    fn empty_input_errs() {
        assert_eq!(fft(&[]).unwrap_err(), FftError::EmptyInput);
        assert_eq!(rfft(&[]).unwrap_err(), FftError::EmptyInput);
        assert_eq!(fftfreq(0, 1.0).unwrap_err(), FftError::EmptyInput);
    }
}
