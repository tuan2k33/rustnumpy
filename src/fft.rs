use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::ndarray::NdArray;
use crate::shape::{c_contiguous_strides, offset_of, IndexIter};

pub type Complex64 = Complex<f64>;

#[derive(Debug, Clone, PartialEq)]
pub enum FftError {

    EmptyInput,

    LengthMismatch { expected: usize, got: usize },

    ShapeMismatch { data_len: usize, shape: Vec<usize> },

    Not2D { ndim: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComplexArray {
    data: Vec<Complex64>,
    shape: Vec<usize>,
}

impl ComplexArray {

    pub fn from_vec(data: Vec<Complex64>, shape: &[usize]) -> Result<Self, FftError> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(FftError::ShapeMismatch { data_len: data.len(), shape: shape.to_vec() });
        }
        Ok(Self { data, shape: shape.to_vec() })
    }

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

pub fn fft(input: &[Complex64]) -> Result<Vec<Complex64>, FftError> {
    if input.is_empty() {
        return Err(FftError::EmptyInput);
    }
    let mut buffer = input.to_vec();
    let mut planner = FftPlanner::new();
    planner.plan_fft_forward(buffer.len()).process(&mut buffer);
    Ok(buffer)
}

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

pub fn rfft(input: &[f64]) -> Result<Vec<Complex64>, FftError> {
    if input.is_empty() {
        return Err(FftError::EmptyInput);
    }
    let complex_input: Vec<Complex64> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let full = fft(&complex_input)?;
    Ok(full[..input.len() / 2 + 1].to_vec())
}

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

pub fn fftn(input: &ComplexArray) -> Result<ComplexArray, FftError> {
    transform_every_axis(input, fft)
}

pub fn ifftn(input: &ComplexArray) -> Result<ComplexArray, FftError> {
    transform_every_axis(input, ifft)
}

pub fn fft2(input: &ComplexArray) -> Result<ComplexArray, FftError> {
    require_2d(input)?;
    fftn(input)
}

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

pub fn rfftfreq(n: usize, d: f64) -> Result<Vec<f64>, FftError> {
    if n == 0 {
        return Err(FftError::EmptyInput);
    }
    let scale = 1.0 / (n as f64 * d);
    Ok((0..=n / 2).map(|i| i as f64 * scale).collect())
}

pub fn fftshift<T: Clone>(a: &[T]) -> Vec<T> {
    rotate_left_copy(a, a.len().div_ceil(2))
}

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

    #[test]
    fn fft_matches_numpy() {
        let x: Vec<Complex64> = [1.0, 2.0, 3.0, 4.0].iter().map(|&r| Complex::new(r, 0.0)).collect();
        let expected = [(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)]
            .map(|(re, im)| Complex::new(re, im));
        assert_complex_close(&fft(&x).unwrap(), &expected);
    }

    #[test]
    fn fft_of_complex_input_matches_numpy() {

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

        let data: Vec<Complex64> = (0..24).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let a = ComplexArray::from_vec(data, &[2, 3, 4]).unwrap();
        let result = fftn(&a).unwrap();
        assert_eq!(result.shape(), &[2, 3, 4]);

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
