use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::ndarray::NdArray;
use crate::shape::{c_contiguous_strides, offset_of, IndexIter};

pub type Complex64 = Complex<f64>;

pub trait FftFloat: rustfft::FftNum + num_traits::Float + Default {}
impl FftFloat for f32 {}
impl FftFloat for f64 {}

#[derive(Debug, Clone, PartialEq)]
pub enum FftError {

    EmptyInput,

    LengthMismatch { expected: usize, got: usize },

    Not2D { ndim: usize },
}

impl std::fmt::Display for FftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FftError::EmptyInput => write!(f, "input array is empty"),
            FftError::LengthMismatch { expected, got } => {
                write!(f, "expected length {expected}, got {got}")
            }
            FftError::Not2D { ndim } => write!(f, "expected a 2-D array, got {ndim} dimensions"),
        }
    }
}

impl std::error::Error for FftError {}

pub fn to_complex<T: FftFloat>(a: &NdArray<T>) -> NdArray<Complex<T>> {
    let data: Vec<Complex<T>> = a.as_slice().iter().map(|&x| Complex::new(x, T::zero())).collect();
    NdArray::from_vec(data, a.shape()).expect("same element count as the input")
}

pub fn fft<T: FftFloat>(input: &[Complex<T>]) -> Result<Vec<Complex<T>>, FftError> {
    if input.is_empty() {
        return Err(FftError::EmptyInput);
    }
    let mut buffer = input.to_vec();
    let mut planner = FftPlanner::new();
    planner.plan_fft_forward(buffer.len()).process(&mut buffer);
    Ok(buffer)
}

pub fn ifft<T: FftFloat>(input: &[Complex<T>]) -> Result<Vec<Complex<T>>, FftError> {
    if input.is_empty() {
        return Err(FftError::EmptyInput);
    }
    let n = input.len();
    let mut buffer = input.to_vec();
    let mut planner = FftPlanner::new();
    planner.plan_fft_inverse(n).process(&mut buffer);
    let scale = T::one() / T::from(n).expect("a length converts to a float");
    buffer.iter_mut().for_each(|c| *c = *c * scale);
    Ok(buffer)
}

pub fn rfft<T: FftFloat>(input: &[T]) -> Result<Vec<Complex<T>>, FftError> {
    if input.is_empty() {
        return Err(FftError::EmptyInput);
    }
    let complex_input: Vec<Complex<T>> = input.iter().map(|&x| Complex::new(x, T::zero())).collect();
    let full = fft(&complex_input)?;
    Ok(full[..input.len() / 2 + 1].to_vec())
}

fn fit_to<T: Copy + Default>(input: &[T], len: usize) -> Vec<T> {
    let mut v: Vec<T> = input.iter().copied().take(len).collect();
    v.resize(len, T::default());
    v
}

pub fn irfft<T: FftFloat>(input: &[Complex<T>], n: usize) -> Result<Vec<T>, FftError> {
    if n == 0 {
        return Err(FftError::EmptyInput);
    }
    let half = fit_to(input, n / 2 + 1);
    let full: Vec<Complex<T>> = (0..n).map(|k| if k <= n / 2 { half[k] } else { half[n - k].conj() }).collect();
    Ok(ifft(&full)?.iter().map(|c| c.re).collect())
}

pub fn hfft<T: FftFloat>(input: &[Complex<T>], n: Option<usize>) -> Result<Vec<T>, FftError> {
    if input.is_empty() {
        return Err(FftError::EmptyInput);
    }
    let n = n.unwrap_or(2 * (input.len() - 1));
    let conj: Vec<Complex<T>> = input.iter().map(|c| c.conj()).collect();
    let scale = T::from(n).expect("a length converts to a float");
    Ok(irfft(&conj, n)?.into_iter().map(|x| x * scale).collect())
}

pub fn ihfft<T: FftFloat>(input: &[T], n: Option<usize>) -> Result<Vec<Complex<T>>, FftError> {
    if input.is_empty() {
        return Err(FftError::EmptyInput);
    }
    let n = n.unwrap_or(input.len());
    if n == 0 {
        return Err(FftError::EmptyInput);
    }
    let spectrum = rfft(&fit_to(input, n))?;
    let scale = T::from(n).expect("a length converts to a float");
    Ok(spectrum.into_iter().map(|c| c.conj() / scale).collect())
}

pub fn fftn<T: FftFloat>(input: &NdArray<Complex<T>>) -> Result<NdArray<Complex<T>>, FftError> {
    transform_every_axis(input, fft)
}

pub fn ifftn<T: FftFloat>(input: &NdArray<Complex<T>>) -> Result<NdArray<Complex<T>>, FftError> {
    transform_every_axis(input, ifft)
}

pub fn rfftn<T: FftFloat>(input: &NdArray<T>) -> Result<NdArray<Complex<T>>, FftError> {
    if input.ndim() == 0 || input.shape().contains(&0) {
        return Err(FftError::EmptyInput);
    }
    let last = input.ndim() - 1;
    let n = input.shape()[last];
    let mut data: Vec<Complex<T>> = Vec::with_capacity(input.len() / n * (n / 2 + 1));
    for line in input.as_slice().chunks(n) {
        data.extend(rfft(line)?);
    }
    let mut shape = input.shape().to_vec();
    shape[last] = n / 2 + 1;
    for axis in 0..last {
        transform_axis_in_place(&mut data, &shape, axis, &fft)?;
    }
    Ok(NdArray::from_vec(data, &shape).expect("half-spectrum element count"))
}

pub fn irfftn<T: FftFloat>(input: &NdArray<Complex<T>>, s: Option<&[usize]>) -> Result<NdArray<T>, FftError> {
    if input.ndim() == 0 || input.shape().contains(&0) {
        return Err(FftError::EmptyInput);
    }
    let last = input.ndim() - 1;
    let default_shape: Vec<usize> = input
        .shape()
        .iter()
        .enumerate()
        .map(|(axis, &d)| if axis == last { 2 * (d - 1) } else { d })
        .collect();
    let out_shape = s.map_or(default_shape, <[usize]>::to_vec);
    if out_shape.len() != input.ndim() {
        return Err(FftError::LengthMismatch { expected: input.ndim(), got: out_shape.len() });
    }
    for (&want, &got) in input.shape()[..last].iter().zip(&out_shape[..last]) {
        if got != want {
            return Err(FftError::LengthMismatch { expected: want, got });
        }
    }
    let mut data = input.as_slice().to_vec();
    for axis in 0..last {
        transform_axis_in_place(&mut data, input.shape(), axis, &ifft)?;
    }
    let width = input.shape()[last];
    let mut out: Vec<T> = Vec::with_capacity(out_shape.iter().product());
    for line in data.chunks(width) {
        out.extend(irfft(line, out_shape[last])?);
    }
    Ok(NdArray::from_vec(out, &out_shape).expect("real output element count"))
}

pub fn rfft2<T: FftFloat>(input: &NdArray<T>) -> Result<NdArray<Complex<T>>, FftError> {
    if input.ndim() != 2 {
        return Err(FftError::Not2D { ndim: input.ndim() });
    }
    rfftn(input)
}

pub fn irfft2<T: FftFloat>(input: &NdArray<Complex<T>>, s: Option<&[usize]>) -> Result<NdArray<T>, FftError> {
    if input.ndim() != 2 {
        return Err(FftError::Not2D { ndim: input.ndim() });
    }
    irfftn(input, s)
}

pub fn fft2<T: FftFloat>(input: &NdArray<Complex<T>>) -> Result<NdArray<Complex<T>>, FftError> {
    require_2d(input)?;
    fftn(input)
}

pub fn ifft2<T: FftFloat>(input: &NdArray<Complex<T>>) -> Result<NdArray<Complex<T>>, FftError> {
    require_2d(input)?;
    ifftn(input)
}

fn require_2d<T>(input: &NdArray<T>) -> Result<(), FftError> {
    if input.ndim() != 2 {
        return Err(FftError::Not2D { ndim: input.ndim() });
    }
    Ok(())
}

fn transform_every_axis<T: FftFloat>(
    input: &NdArray<Complex<T>>,
    per_line: impl Fn(&[Complex<T>]) -> Result<Vec<Complex<T>>, FftError>,
) -> Result<NdArray<Complex<T>>, FftError> {
    if input.ndim() == 0 || input.shape().contains(&0) {
        return Err(FftError::EmptyInput);
    }
    let mut data = input.as_slice().to_vec();
    for axis in 0..input.ndim() {
        transform_axis_in_place(&mut data, input.shape(), axis, &per_line)?;
    }
    Ok(NdArray::from_vec(data, input.shape()).expect("transform keeps the element count"))
}

fn transform_axis_in_place<T: FftFloat>(
    data: &mut [Complex<T>],
    shape: &[usize],
    axis: usize,
    per_line: &impl Fn(&[Complex<T>]) -> Result<Vec<Complex<T>>, FftError>,
) -> Result<(), FftError> {
    let strides = c_contiguous_strides(shape);
    let axis_stride = strides[axis] as usize;
    let axis_len = shape[axis];

    let mut line_start_shape = shape.to_vec();
    line_start_shape[axis] = 1;

    for start_index in IndexIter::new(&line_start_shape) {
        let start = offset_of(&start_index, &strides) as usize;
        let line: Vec<Complex<T>> = (0..axis_len).map(|k| data[start + k * axis_stride]).collect();
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
    fn irfft_crops_or_zero_pads_the_spectrum_like_numpy() {
        let five: Vec<Complex64> = [1.0, 2.0, 3.0, 4.0, 5.0].iter().map(|&r| Complex64::new(r, 0.0)).collect();
        assert_reals_close(&irfft(&five, 4).unwrap(), &[2.0, -0.5, 0.0, -0.5]);
        let two: Vec<Complex64> = [1.0, 2.0].iter().map(|&r| Complex64::new(r, 0.0)).collect();
        assert_reals_close(
            &irfft(&two, 6).unwrap(),
            &[0.8333333333333333, 0.5, -0.16666666666666666, -0.4999999999999999, -0.16666666666666666, 0.4999999999999999],
        );
        assert_eq!(irfft(&two, 0).unwrap_err(), FftError::EmptyInput);
    }

    #[test]
    fn hfft_and_ihfft_match_numpy() {
        let c = |v: &[f64]| -> Vec<Complex64> { v.iter().map(|&r| Complex64::new(r, 0.0)).collect() };
        assert_reals_close(&hfft(&c(&[1.0, 2.0, 3.0]), None).unwrap(), &[8.0, -2.0, 0.0, -2.0]);
        assert_reals_close(
            &hfft(&c(&[1.0, 2.0, 3.0]), Some(5)).unwrap(),
            &[11.0, -2.618033988749895, -0.3819660112501051, -0.3819660112501051, -2.618033988749895],
        );
        assert_reals_close(&hfft(&c(&[1.0, 2.0, 3.0, 4.0, 5.0]), Some(4)).unwrap(), &[8.0, -2.0, 0.0, -2.0]);
        assert_reals_close(
            &hfft(&c(&[1.0, 2.0]), Some(6)).unwrap(),
            &[5.0, 3.0, -1.0, -3.0, -1.0, 3.0],
        );
        let mixed = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0), Complex64::new(3.0, -1.0), Complex64::new(4.0, 2.0)];
        assert_reals_close(&hfft(&mixed, None).unwrap(), &[15.0, -4.0, 3.4641016151377544, -1.0000000000000002, -3.4641016151377544, -4.0]);
        assert_complex_close(
            &ihfft(&[1.0, 2.0, 3.0, 4.0], None).unwrap(),
            &[Complex64::new(2.5, 0.0), Complex64::new(-0.5, -0.5), Complex64::new(-0.5, 0.0)],
        );
        assert_complex_close(
            &ihfft(&[1.0, 2.0, 3.0, 4.0, 5.0], None).unwrap(),
            &[Complex64::new(3.0, 0.0), Complex64::new(-0.5, -0.6881909602355867), Complex64::new(-0.5, -0.1624598481164532)],
        );
        assert_complex_close(
            &ihfft(&[1.0, 2.0, 3.0, 4.0], Some(6)).unwrap(),
            &[
                Complex64::new(1.6666666666666665, 0.0),
                Complex64::new(-0.5833333333333333, 0.721687836487032),
                Complex64::new(0.4166666666666666, -0.14433756729740646),
                Complex64::new(-0.3333333333333333, 0.0),
            ],
        );
        assert_complex_close(&ihfft(&[1.0, 2.0, 3.0, 4.0], Some(2)).unwrap(), &[Complex64::new(1.5, 0.0), Complex64::new(-0.5, 0.0)]);
        assert_reals_close(&hfft(&ihfft(&[1.0, 2.0, 3.0, 4.0, 5.0], None).unwrap(), Some(5)).unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(hfft::<f64>(&[], None).unwrap_err(), FftError::EmptyInput);
        assert_eq!(ihfft::<f64>(&[], None).unwrap_err(), FftError::EmptyInput);
    }

    #[test]
    fn hfft_ignores_the_imaginary_part_of_the_dc_term() {
        let with_imag_dc = [Complex64::new(1.0, 5.0), Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)];
        assert_reals_close(&hfft(&with_imag_dc, Some(4)).unwrap(), &[8.0, -2.0, 0.0, -2.0]);
    }

    #[test]
    fn rfftn_matches_numpy_2d_and_3d() {
        let x = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let expected = [Complex64::new(21.0, 0.0), Complex64::new(-3.0, 1.7320508075688772), Complex64::new(-9.0, 0.0), Complex64::new(0.0, 0.0)];
        let r = rfftn(&x).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        assert_complex_close(r.as_slice(), &expected);
        assert_complex_close(rfft2(&x).unwrap().as_slice(), &expected);
        assert_eq!(irfft2(&rfft2(&x).unwrap(), Some(&[2, 3])).unwrap().as_slice(), irfft2(&r, Some(&[2, 3])).unwrap().as_slice());
        assert_reals_close(irfft2(&r, Some(&[2, 3])).unwrap().as_slice(), x.as_slice());

        let cube = NdArray::from_vec((0..24).map(f64::from).collect(), &[2, 3, 4]).unwrap();
        let r3 = rfftn(&cube).unwrap();
        assert_eq!(r3.shape(), &[2, 3, 3]);
        for (idx, want) in [
            ([0, 0, 0], Complex64::new(276.0, 0.0)),
            ([0, 0, 1], Complex64::new(-12.0, 12.0)),
            ([0, 0, 2], Complex64::new(-12.0, 0.0)),
            ([0, 1, 0], Complex64::new(-48.0, 27.712812921102035)),
            ([0, 1, 1], Complex64::new(0.0, 0.0)),
            ([1, 0, 0], Complex64::new(-144.0, 0.0)),
            ([1, 2, 1], Complex64::new(0.0, 0.0)),
        ] {
            assert_complex_close(&[r3.get(&idx).unwrap()], &[want]);
        }
        assert_reals_close(irfftn(&r3, Some(&[2, 3, 4])).unwrap().as_slice(), cube.as_slice());
        assert_eq!(irfftn(&r3, None).unwrap().shape(), &[2, 3, 4]);
    }

    #[test]
    fn rfftn_roundtrips_odd_last_axis_and_irfftn_matches_numpy_on_complex_input() {
        let odd = NdArray::from_vec((0..30).map(|i| (i as f64).powf(1.5)).collect(), &[2, 3, 5]).unwrap();
        let r = rfftn(&odd).unwrap();
        assert_eq!(r.shape(), &[2, 3, 3]);
        assert_complex_close(&[r.get(&[1, 1, 2]).unwrap()], &[Complex64::new(3.1862317226683015, -2.004341290755435)]);
        assert_complex_close(&[r.get(&[0, 2, 1]).unwrap()], &[Complex64::new(16.09341902929033, -7.829419978640091)]);
        assert_complex_close(&[r.get(&[0, 0, 0]).unwrap()], &[Complex64::new(1890.3019945571912, 0.0)]);
        assert_reals_close(irfftn(&r, Some(&[2, 3, 5])).unwrap().as_slice(), odd.as_slice());

        let c = NdArray::from_vec(
            vec![
                Complex64::new(1.0, 2.0),
                Complex64::new(3.0, -1.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 0.5),
                Complex64::new(1.0, 0.0),
                Complex64::new(4.0, 4.0),
            ],
            &[2, 3],
        )
        .unwrap();
        let out = irfftn(&c, Some(&[2, 4])).unwrap();
        assert_reals_close(out.as_slice(), &[1.875, -0.375, -0.125, -0.875, 0.375, 0.625, -0.625, 0.125]);
        assert_eq!(irfftn(&c, None).unwrap().shape(), &[2, 4]);
    }

    #[test]
    fn rfftn_family_rejects_bad_input() {
        assert_eq!(rfftn(&NdArray::<f64>::zeros(&[0, 3])).unwrap_err(), FftError::EmptyInput);
        assert_eq!(rfft2(&NdArray::<f64>::zeros(&[2, 2, 2])).unwrap_err(), FftError::Not2D { ndim: 3 });
        let r = rfftn(&NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap()).unwrap();
        assert!(matches!(irfftn(&r, Some(&[3, 2])), Err(FftError::LengthMismatch { .. })));
        assert!(matches!(irfftn(&r, Some(&[2])), Err(FftError::LengthMismatch { .. })));
        let one_wide: NdArray<Complex64> = NdArray::zeros(&[2, 1]);
        assert_eq!(irfftn(&one_wide, None).unwrap_err(), FftError::EmptyInput);
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

        let a = NdArray::from_vec(
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
        let a = NdArray::from_vec(vec![Complex64::new(1.0, 0.0)], &[1, 1, 1]).unwrap();
        assert_eq!(fft2(&a).unwrap_err(), FftError::Not2D { ndim: 3 });
    }

    #[test]
    fn ifft2_of_fft2_roundtrips() {
        let a = NdArray::from_vec(
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
        let a = NdArray::from_vec(data, &[2, 3, 4]).unwrap();
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
        let a = NdArray::from_vec(data, &[2, 3, 4]).unwrap();
        let back = ifftn(&fftn(&a).unwrap()).unwrap();
        assert_complex_close(back.as_slice(), a.as_slice());
    }

    #[test]
    fn empty_input_errs() {
        assert_eq!(fft::<f64>(&[]).unwrap_err(), FftError::EmptyInput);
        assert_eq!(rfft::<f64>(&[]).unwrap_err(), FftError::EmptyInput);
        assert_eq!(fftfreq(0, 1.0).unwrap_err(), FftError::EmptyInput);
    }

    #[test]
    fn single_precision_transforms_agree_with_double_and_round_trip() {
        let x64: Vec<Complex<f64>> = [1.0, 2.0, 3.0, 4.0, 0.5].iter().map(|&r| Complex::new(r, -r / 2.0)).collect();
        let x32: Vec<Complex<f32>> = x64.iter().map(|c| Complex::new(c.re as f32, c.im as f32)).collect();
        for (a, e) in fft(&x32).unwrap().iter().zip(fft(&x64).unwrap()) {
            assert!((f64::from(a.re) - e.re).abs() < 1e-5 && (f64::from(a.im) - e.im).abs() < 1e-5);
        }
        for (a, e) in ifft(&fft(&x32).unwrap()).unwrap().iter().zip(&x32) {
            assert!((a.re - e.re).abs() < 1e-5 && (a.im - e.im).abs() < 1e-5);
        }
        let real: Vec<f32> = vec![1.0, -2.0, 0.25, 3.0, 5.0, 1.5];
        let back = irfft(&rfft(&real).unwrap(), real.len()).unwrap();
        assert!(back.iter().zip(&real).all(|(a, b)| (a - b).abs() < 1e-5));
        let grid = NdArray::from_vec(real.clone(), &[2, 3]).unwrap();
        assert_eq!(rfftn(&grid).unwrap().shape(), &[2, 2]);
        assert_eq!(irfftn(&rfftn(&grid).unwrap(), Some(&[2, 3])).unwrap().shape(), &[2, 3]);
    }
}
