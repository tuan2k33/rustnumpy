use crate::error::ShapeError;
use crate::ndarray::NdArray;
use crate::shape::IndexIter;

pub fn unique(data: &[f64]) -> Vec<f64> {
    let mut v = data.to_vec();
    v.sort_unstable_by(|a, b| a.total_cmp(b));
    v.dedup_by(|a, b| a == b || (a.is_nan() && b.is_nan()));
    v
}

pub fn intersect1d(a: &[f64], b: &[f64]) -> Vec<f64> {
    let ua = unique(a);
    let ub = unique(b);
    ua.into_iter().filter(|x| ub.iter().any(|y| x == y || (x.is_nan() && y.is_nan()))).collect()
}

pub fn union1d(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut combined = a.to_vec();
    combined.extend_from_slice(b);
    unique(&combined)
}

fn check_axis(axis: usize, ndim: usize) -> Result<(), ShapeError> {
    if axis >= ndim {
        Err(ShapeError::AxisOutOfBounds { axis, ndim })
    } else {
        Ok(())
    }
}

pub fn concatenate<T: Copy>(arrays: &[&NdArray<T>], axis: usize) -> Result<NdArray<T>, ShapeError> {
    let first = arrays.first().ok_or(ShapeError::EmptyArrayList)?;
    check_axis(axis, first.ndim())?;
    for a in arrays {
        if a.ndim() != first.ndim()
            || a.shape().iter().enumerate().any(|(i, &d)| i != axis && d != first.shape()[i])
        {
            return Err(ShapeError::ConcatShapeMismatch {
                axis,
                shapes: arrays.iter().map(|a| a.shape().to_vec()).collect(),
            });
        }
    }

    let mut out_shape = first.shape().to_vec();
    out_shape[axis] = arrays.iter().map(|a| a.shape()[axis]).sum();

    let mut offsets = Vec::with_capacity(arrays.len());
    let mut acc = 0;
    for a in arrays {
        offsets.push(acc);
        acc += a.shape()[axis];
    }

    let data: Vec<T> = IndexIter::new(&out_shape)
        .map(|mut idx| {
            let target = idx[axis];
            let (array_index, offset) = offsets
                .iter()
                .enumerate()
                .rev()
                .find(|&(_, &start)| start <= target)
                .expect("target is within [0, out_shape[axis]) by construction");
            idx[axis] = target - offset;
            arrays[array_index].get(&idx).expect("adjusted index is in-bounds for its source array")
        })
        .collect();
    Ok(NdArray::from_vec(data, &out_shape).expect("data.len() == out_shape.iter().product() by construction"))
}

pub fn concat<T: Copy>(arrays: &[&NdArray<T>], axis: usize) -> Result<NdArray<T>, ShapeError> {
    concatenate(arrays, axis)
}

pub fn stack<T: Copy>(arrays: &[&NdArray<T>], axis: usize) -> Result<NdArray<T>, ShapeError> {
    let first = arrays.first().ok_or(ShapeError::EmptyArrayList)?;
    let out_ndim = first.ndim() + 1;
    check_axis(axis, out_ndim)?;
    for a in arrays {
        if a.shape() != first.shape() {
            return Err(ShapeError::StackShapeMismatch { shapes: arrays.iter().map(|a| a.shape().to_vec()).collect() });
        }
    }

    let mut out_shape = first.shape().to_vec();
    out_shape.insert(axis, arrays.len());

    let data: Vec<T> = IndexIter::new(&out_shape)
        .map(|idx| {
            let array_index = idx[axis];
            let mut source_idx = idx.clone();
            source_idx.remove(axis);
            arrays[array_index].get(&source_idx).expect("removing the stacked axis gives a valid source index")
        })
        .collect();
    Ok(NdArray::from_vec(data, &out_shape).expect("data.len() == out_shape.iter().product() by construction"))
}

pub fn split(arr: &NdArray, sections: usize, axis: usize) -> Result<Vec<NdArray>, ShapeError> {
    check_axis(axis, arr.ndim())?;
    if sections == 0 {
        return Err(ShapeError::EmptyArrayList);
    }
    let axis_len = arr.shape()[axis];
    if !axis_len.is_multiple_of(sections) {
        return Err(ShapeError::NotEvenlyDivisible { axis_len, sections });
    }
    let piece = axis_len / sections;
    (0..sections)
        .map(|i| {
            let ranges: Vec<std::ops::Range<usize>> = arr
                .shape()
                .iter()
                .enumerate()
                .map(|(a, &d)| if a == axis { i * piece..(i + 1) * piece } else { 0..d })
                .collect();
            Ok(arr.slice(&ranges)?.to_owned())
        })
        .collect()
}

fn slice_axis<T: Copy>(arr: &NdArray<T>, axis: usize, start: usize, end: usize) -> Result<NdArray<T>, ShapeError> {
    let ranges: Vec<std::ops::Range<usize>> = arr
        .shape()
        .iter()
        .enumerate()
        .map(|(a, &d)| if a == axis { start..end } else { 0..d })
        .collect();
    Ok(arr.slice(&ranges)?.to_owned())
}

pub fn array_split<T: Copy>(arr: &NdArray<T>, sections: usize, axis: usize) -> Result<Vec<NdArray<T>>, ShapeError> {
    check_axis(axis, arr.ndim())?;
    if sections == 0 {
        return Err(ShapeError::ZeroSections);
    }
    let axis_len = arr.shape()[axis];
    let (each, extras) = (axis_len / sections, axis_len % sections);
    let mut start = 0;
    (0..sections)
        .map(|i| {
            let end = start + each + usize::from(i < extras);
            let piece = slice_axis(arr, axis, start, end);
            start = end;
            piece
        })
        .collect()
}

pub fn array_split_at<T: Copy>(arr: &NdArray<T>, indices: &[usize], axis: usize) -> Result<Vec<NdArray<T>>, ShapeError> {
    check_axis(axis, arr.ndim())?;
    let axis_len = arr.shape()[axis];
    let mut bounds = vec![0usize];
    bounds.extend(indices.iter().map(|&i| i.min(axis_len)));
    bounds.push(axis_len);
    bounds
        .windows(2)
        .map(|w| slice_axis(arr, axis, w[0], w[1].max(w[0])))
        .collect()
}

pub fn tile<T: Copy>(arr: &NdArray<T>, reps: &[usize]) -> NdArray<T> {

    let out_ndim = arr.ndim().max(reps.len());
    let mut padded_shape = vec![1usize; out_ndim - arr.ndim()];
    padded_shape.extend_from_slice(arr.shape());
    let mut padded_reps = vec![1usize; out_ndim - reps.len()];
    padded_reps.extend_from_slice(reps);

    let out_shape: Vec<usize> =
        padded_shape.iter().zip(padded_reps.iter()).map(|(&d, &r)| d * r).collect();
    let data: Vec<T> = IndexIter::new(&out_shape)
        .map(|idx| {

            let padded_source_idx: Vec<usize> =
                idx.iter().zip(padded_shape.iter()).map(|(&i, &d)| i % d).collect();
            let source_idx = &padded_source_idx[out_ndim - arr.ndim()..];
            arr.get(source_idx).expect("modulo keeps the index within the source array's bounds")
        })
        .collect();
    NdArray::from_vec(data, &out_shape).expect("data.len() == out_shape.iter().product() by construction")
}

pub fn interp(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&xi| {
            if xi <= xp[0] {
                return fp[0];
            }
            if xi >= xp[xp.len() - 1] {
                return fp[fp.len() - 1];
            }
            let i = xp.partition_point(|&p| p <= xi) - 1;
            let frac = (xi - xp[i]) / (xp[i + 1] - xp[i]);
            fp[i] + frac * (fp[i + 1] - fp[i])
        })
        .collect()
}

pub fn gradient(f: &[f64], dx: f64) -> Vec<f64> {
    let n = f.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let mut out = vec![0.0; n];
    out[0] = (f[1] - f[0]) / dx;
    out[n - 1] = (f[n - 1] - f[n - 2]) / dx;
    for i in 1..n - 1 {
        out[i] = (f[i + 1] - f[i - 1]) / (2.0 * dx);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unique_sorts_dedups_and_collapses_nan_to_one() {

        let data = [3.0, 1.0, 2.0, 1.0, f64::NAN, f64::NAN, 2.0];
        let u = unique(&data);
        assert_eq!(&u[..3], &[1.0, 2.0, 3.0]);
        assert_eq!(u.len(), 4);
        assert!(u[3].is_nan());
    }

    #[test]
    fn intersect1d_union1d_match_real_numpy() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [3.0, 4.0, 5.0, 6.0];
        assert_eq!(intersect1d(&a, &b), vec![3.0, 4.0]);
        assert_eq!(union1d(&a, &b), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn concatenate_axis0_and_axis1_match_real_numpy() {

        let c1 = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let c2 = NdArray::from_vec(vec![5.0, 6.0], &[1, 2]).unwrap();
        let out = concatenate(&[&c1, &c2], 0).unwrap();
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let c3 = NdArray::from_vec(vec![5.0, 6.0], &[2, 1]).unwrap();
        let out = concatenate(&[&c1, &c3], 1).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn concat_is_the_array_api_standard_name_for_concatenate() {
        let c1 = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let c2 = NdArray::from_vec(vec![5.0, 6.0], &[1, 2]).unwrap();
        assert_eq!(concat(&[&c1, &c2], 0).unwrap(), concatenate(&[&c1, &c2], 0).unwrap());
    }

    #[test]
    fn concatenate_shape_mismatch_errs() {
        let a = NdArray::from_vec(vec![1.0, 2.0], &[1, 2]).unwrap();
        let b = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
        assert!(concatenate(&[&a, &b], 0).is_err());
    }

    #[test]
    fn stack_axis0_and_axis1_match_real_numpy() {

        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = NdArray::from_vec(vec![3.0, 4.0, 5.0, 6.0], &[4]).unwrap();
        let out = stack(&[&a, &b], 0).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0]);

        let out = stack(&[&a, &b], 1).unwrap();
        assert_eq!(out.shape(), &[4, 2]);
        assert_eq!(out.as_slice(), &[1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0]);
    }

    #[test]
    fn split_into_equal_sections_matches_real_numpy() {

        let d = NdArray::from_vec((0..9).map(|i| i as f64).collect(), &[9]).unwrap();
        let parts = split(&d, 3, 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].as_slice(), &[0.0, 1.0, 2.0]);
        assert_eq!(parts[1].as_slice(), &[3.0, 4.0, 5.0]);
        assert_eq!(parts[2].as_slice(), &[6.0, 7.0, 8.0]);
    }

    #[test]
    fn split_not_evenly_divisible_errs() {
        let d = NdArray::from_vec((0..9).map(|i| i as f64).collect(), &[9]).unwrap();
        assert_eq!(split(&d, 4, 0), Err(ShapeError::NotEvenlyDivisible { axis_len: 9, sections: 4 }));
    }

    #[test]
    fn tile_1d_and_2d_reps_match_real_numpy() {

        let e = NdArray::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        assert_eq!(tile(&e, &[3]).as_slice(), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

        let out = tile(&e, &[2, 2]);
        assert_eq!(out.shape(), &[2, 4]);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn tile_matrix_with_shorter_reps_pads_leading_axes() {

        let m = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let out = tile(&m, &[2]);
        assert_eq!(out.shape(), &[2, 4]);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn array_split_gives_the_first_remainder_sections_one_extra_element() {
        let r = NdArray::from_vec((0..10).collect::<Vec<i32>>(), &[10]).unwrap();
        let three: Vec<Vec<i32>> = array_split(&r, 3, 0).unwrap().iter().map(|p| p.as_slice().to_vec()).collect();
        assert_eq!(three, vec![vec![0, 1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);
        let four: Vec<Vec<i32>> = array_split(&r, 4, 0).unwrap().iter().map(|p| p.as_slice().to_vec()).collect();
        assert_eq!(four, vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7], vec![8, 9]]);
        let tiny = NdArray::from_vec(vec![1, 2, 3], &[3]).unwrap();
        let shapes: Vec<Vec<usize>> = array_split(&tiny, 5, 0).unwrap().iter().map(|p| p.shape().to_vec()).collect();
        assert_eq!(shapes, vec![vec![1], vec![1], vec![1], vec![0], vec![0]]);
        assert_eq!(array_split(&r, 0, 0).unwrap_err(), ShapeError::ZeroSections);
    }

    #[test]
    fn array_split_along_a_non_leading_axis_and_by_indices() {
        let m = NdArray::from_vec((0..12).collect::<Vec<i32>>(), &[3, 4]).unwrap();
        let parts = array_split(&m, 2, 1).unwrap();
        assert_eq!(parts.iter().map(|p| p.shape().to_vec()).collect::<Vec<_>>(), vec![vec![3, 2], vec![3, 2]]);
        assert_eq!(parts[1].as_slice(), &[2, 3, 6, 7, 10, 11]);

        let r = NdArray::from_vec((0..10).collect::<Vec<i32>>(), &[10]).unwrap();
        let by_idx: Vec<Vec<i32>> = array_split_at(&r, &[2, 5, 5, 20], 0)
            .unwrap()
            .iter()
            .map(|p| p.as_slice().to_vec())
            .collect();
        assert_eq!(by_idx, vec![vec![0, 1], vec![2, 3, 4], vec![], vec![5, 6, 7, 8, 9], vec![]]);
    }

    #[test]
    fn interp_matches_real_numpy_inside_and_extrapolated() {
        let xp = [0.0, 1.0, 2.0, 3.0];
        let fp = [0.0, 10.0, 20.0, 30.0];

        let inside = interp(&[0.5, 1.5, 2.9], &xp, &fp);
        for (got, want) in inside.iter().zip([5.0, 15.0, 29.0]) {
            assert!((got - want).abs() < 1e-9);
        }

        let outside = interp(&[-1.0, 5.0], &xp, &fp);
        assert_eq!(outside, vec![0.0, 30.0]);
    }

    #[test]
    fn gradient_matches_real_numpy_edges_and_interior() {

        let g = gradient(&[1.0, 2.0, 4.0, 7.0, 11.0], 1.0);
        assert_eq!(g, vec![1.0, 1.5, 2.5, 3.5, 4.0]);

        let g2 = gradient(&[1.0, 2.0, 4.0, 7.0, 11.0], 2.0);
        assert_eq!(g2, vec![0.5, 0.75, 1.25, 1.75, 2.0]);
    }
}
