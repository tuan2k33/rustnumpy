//! `numpy.fft` — `fft`/`ifft` (complex), `rfft`/`irfft` (real input),
//! `fftfreq`/`rfftfreq`, `fftshift`/`ifftshift`.
//!
//! Delegates the actual transform to the `rustfft` crate (pure Rust, no
//! FFI) per the same "depend on a crate, don't hand-convert" decision
//! `linalg.rs` follows for `faer`/LAPACK: `pocketfft` is a numerical
//! *tool* NumPy vendors, not one of NumPy's own design ideas.
//!
//! **Scope note**: the 1-D functions (`fft`/`ifft`/`rfft`/`irfft`) operate
//! on plain `Vec<Complex64>`/`Vec<f64>`; the N-D functions (`fftn`/`ifftn`/
//! `fft2`/`ifft2`) operate on this module's own [`ComplexArray`] (shape +
//! flat `Vec<Complex64>`) rather than `NdArray` -- `NdArray` is f64-only
//! (no complex dtype yet, see `lib.rs`'s doc comment), and a spectrum is
//! inherently complex, so there's no lossless way to hand it back as an
//! `NdArray` today. `fftn` auto-detects the input's dimensionality and
//! transforms every axis (NumPy's own `fftn`/`pocketfft` do the same
//! thing under the hood: a separable N-D DFT is just a 1-D FFT applied
//! along each axis in turn).
//!
//! **Known, accepted deviation from NumPy**: `rustfft` and `pocketfft` are
//! both mathematically correct but not bit-for-bit identical (different
//! floating-point summation order -> ULP-level error) — compare with a
//! tolerance, never exact equality, matching `NumPy.md`'s project-wide
//! testing policy.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::ndarray::NdArray;
use crate::shape::{c_contiguous_strides, offset_of, IndexIter};

/// A complex sample/spectrum value — `f64` real and imaginary parts.
pub type Complex64 = Complex<f64>;

/// Any error from an `fft` operation.
#[derive(Debug, Clone, PartialEq)]
pub enum FftError {
    /// `fft`/`ifft`/`rfft`/`irfft`/`fftn`/`ifftn` were given a zero-length
    /// input, or a shape with a zero-length axis.
    EmptyInput,
    /// `irfft`'s input doesn't have the `n / 2 + 1` entries a real
    /// spectrum of output length `n` requires.
    LengthMismatch { expected: usize, got: usize },
    /// [`ComplexArray::from_vec`]'s `data` doesn't have `shape.iter().product()` elements.
    ShapeMismatch { data_len: usize, shape: Vec<usize> },
    /// [`fft2`]/[`ifft2`] require exactly 2 dimensions.
    Not2D { ndim: usize },
}

/// A dense N-D array of complex values, row-major (C-contiguous) layout --
/// a separate type from [`NdArray`] because a spectrum is inherently
/// complex and `NdArray` has no complex dtype yet (see this module's own
/// and `lib.rs`'s doc comments).
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexArray {
    data: Vec<Complex64>,
    shape: Vec<usize>,
}

impl ComplexArray {
    /// Build from flat (row-major) data plus a declared shape.
    pub fn from_vec(data: Vec<Complex64>, shape: &[usize]) -> Result<Self, FftError> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(FftError::ShapeMismatch { data_len: data.len(), shape: shape.to_vec() });
        }
        Ok(Self { data, shape: shape.to_vec() })
    }

    /// Lift a real [`NdArray`] into a [`ComplexArray`] with the same shape
    /// (each value's imaginary part is `0`).
    pub fn from_real(a: &NdArray) -> Self {
        Self {
            data: a.as_slice().iter().map(|&x| Complex64::new(x, 0.0)).collect(),
            shape: a.shape().to_vec(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn as_slice(&self) -> &[Complex64] {
        &self.data
    }
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

/// `np.fft.fftn(a)`: the N-D discrete Fourier transform, computed the same
/// way NumPy's own `fftn`/`pocketfft` do it -- a separable transform is
/// just a 1-D [`fft`] applied along every axis in turn (the order doesn't
/// matter: each axis's transform is independent of the others, verified
/// against real NumPy for both a 2-D and a 3-D input). Auto-detects
/// `input.ndim()` and loops over every axis; there's no separate "which
/// axes" parameter to get wrong.
pub fn fftn(input: &ComplexArray) -> Result<ComplexArray, FftError> {
    transform_every_axis(input, fft)
}

/// `np.fft.ifftn(a)`: the inverse of [`fftn`], one [`ifft`] per axis.
pub fn ifftn(input: &ComplexArray) -> Result<ComplexArray, FftError> {
    transform_every_axis(input, ifft)
}

/// `np.fft.fft2(a)`: [`fftn`] restricted to exactly 2 dimensions (matching
/// NumPy's own `fft2`, which is really just `fftn` over the last two axes
/// -- since this only supports whole-array transforms, that's every axis
/// for a 2-D input).
pub fn fft2(input: &ComplexArray) -> Result<ComplexArray, FftError> {
    require_2d(input)?;
    fftn(input)
}

/// `np.fft.ifft2(a)`: the inverse of [`fft2`].
pub fn ifft2(input: &ComplexArray) -> Result<ComplexArray, FftError> {
    require_2d(input)?;
    ifftn(input)
}

fn require_2d(input: &ComplexArray) -> Result<(), FftError> {
    if input.ndim() != 2 {
        return Err(FftError::Not2D { ndim: input.ndim() });
    }
    Ok(())
}

fn transform_every_axis(
    input: &ComplexArray,
    per_line: impl Fn(&[Complex64]) -> Result<Vec<Complex64>, FftError>,
) -> Result<ComplexArray, FftError> {
    if input.shape.is_empty() || input.shape.contains(&0) {
        return Err(FftError::EmptyInput);
    }
    let mut data = input.data.clone();
    for axis in 0..input.ndim() {
        transform_axis_in_place(&mut data, &input.shape, axis, &per_line)?;
    }
    Ok(ComplexArray { data, shape: input.shape.clone() })
}

/// Applies `per_line` to every 1-D "line" of `data` running along `axis`,
/// in place. Walks the line-start positions with [`IndexIter`] over a copy
/// of `shape` with `axis` pinned to size 1 -- exactly the set of flat
/// offsets whose `axis` coordinate is `0`, one per line.
fn transform_axis_in_place(
    data: &mut [Complex64],
    shape: &[usize],
    axis: usize,
    per_line: &impl Fn(&[Complex64]) -> Result<Vec<Complex64>, FftError>,
) -> Result<(), FftError> {
    let strides = c_contiguous_strides(shape);
    let axis_stride = strides[axis] as usize;
    let axis_len = shape[axis];

    let mut line_start_shape = shape.to_vec();
    line_start_shape[axis] = 1;

    for start_index in IndexIter::new(&line_start_shape) {
        let start = offset_of(&start_index, &strides) as usize;
        let line: Vec<Complex64> = (0..axis_len).map(|k| data[start + k * axis_stride]).collect();
        let transformed = per_line(&line)?;
        for (k, value) in transformed.into_iter().enumerate() {
            data[start + k * axis_stride] = value;
        }
    }
    Ok(())
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
    fn fft2_matches_numpy() {
        // np.fft.fft2([[1,2,3],[4,5,6]])
        let a = ComplexArray::from_vec(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(|r| Complex64::new(r, 0.0)).to_vec(),
            &[2, 3],
        )
        .unwrap();
        let expected = [
            (21.0, 0.0),
            (-3.0, 1.732_050_807_568_877),
            (-3.0, -1.732_050_807_568_877),
            (-9.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        ]
        .map(|(re, im)| Complex64::new(re, im));
        assert_complex_close(fft2(&a).unwrap().as_slice(), &expected);
    }

    #[test]
    fn fft2_rejects_non_2d() {
        let a = ComplexArray::from_vec(vec![Complex64::new(1.0, 0.0)], &[1, 1, 1]).unwrap();
        assert_eq!(fft2(&a).unwrap_err(), FftError::Not2D { ndim: 3 });
    }

    #[test]
    fn ifft2_of_fft2_roundtrips() {
        let a = ComplexArray::from_vec(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(|r| Complex64::new(r, 0.0)).to_vec(),
            &[2, 3],
        )
        .unwrap();
        let back = ifft2(&fft2(&a).unwrap()).unwrap();
        assert_complex_close(back.as_slice(), a.as_slice());
    }

    #[test]
    fn fftn_matches_numpy_for_3d_input() {
        // np.fft.fftn(np.arange(24.0).reshape(2, 3, 4))
        let data: Vec<Complex64> = (0..24).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let a = ComplexArray::from_vec(data, &[2, 3, 4]).unwrap();
        let result = fftn(&a).unwrap();
        assert_eq!(result.shape(), &[2, 3, 4]);
        // Spot-check a handful of entries against real NumPy's output.
        assert_complex_close(&result.as_slice()[0..1], &[Complex64::new(276.0, 0.0)]);
        assert_complex_close(&result.as_slice()[1..2], &[Complex64::new(-12.0, 12.0)]);
        assert_complex_close(&result.as_slice()[4..5], &[Complex64::new(-48.0, 27.712_812_921_102_04)]);
        assert_complex_close(&result.as_slice()[12..13], &[Complex64::new(-144.0, 0.0)]);
    }

    #[test]
    fn ifftn_of_fftn_roundtrips_for_3d_input() {
        let data: Vec<Complex64> = (0..24).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let a = ComplexArray::from_vec(data, &[2, 3, 4]).unwrap();
        let back = ifftn(&fftn(&a).unwrap()).unwrap();
        assert_complex_close(back.as_slice(), a.as_slice());
    }

    #[test]
    fn complex_array_from_vec_rejects_mismatched_shape() {
        assert_eq!(
            ComplexArray::from_vec(vec![Complex64::new(1.0, 0.0)], &[2, 2]).unwrap_err(),
            FftError::ShapeMismatch { data_len: 1, shape: vec![2, 2] }
        );
    }

    #[test]
    fn empty_input_errs() {
        assert_eq!(fft(&[]).unwrap_err(), FftError::EmptyInput);
        assert_eq!(rfft(&[]).unwrap_err(), FftError::EmptyInput);
        assert_eq!(fftfreq(0, 1.0).unwrap_err(), FftError::EmptyInput);
    }
}
