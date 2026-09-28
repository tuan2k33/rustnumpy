//! Step 11: `lib/`-layer utility functions — set operations (`unique`,
//! `intersect1d`, `union1d`), shape ops (`concatenate`, `stack`, `split`,
//! `tile`), `interp`, and `gradient`.
//!
//! Scope: a representative subset of what NumPy's `lib/` layer offers
//! (per `NumPy.md`, this is "the largest volume of functions, but built
//! on top of the core that's already there" — not meant to be
//! exhaustive). `split` here matches real NumPy's plain `split` (requires
//! an exact even division) rather than `array_split` (handles uneven
//! splits) — not implemented, since `split` alone already exercises the
//! same shape logic. `interp`/`gradient` assume their inputs are already
//! sorted/uniformly spaced, matching what real NumPy itself requires
//! (undefined behavior otherwise, in both).
//!
//! Every formula and edge case — including `unique`'s `NaN`-deduplication
//! (`NaN`s collapse to one, even though `NaN != NaN`), `tile`'s
//! shorter-`reps`-gets-padded-with-leading-1s rule, and `gradient`'s exact
//! edge-vs-interior formula — was checked against real NumPy 2.5.3 first.

use crate::error::ShapeError;
use crate::ndarray::NdArray;
use crate::shape::IndexIter;

/// `np.unique(data)`: sorted, deduplicated. Real NumPy collapses every
/// `NaN` into a single trailing entry (verified: even though `NaN !=
/// NaN`, `unique` treats them as one group and sorts them to the end,
/// the same way `sort_unstable_by(total_cmp)` orders `NaN` last).
pub fn unique(data: &[f64]) -> Vec<f64> {
    let mut v = data.to_vec();
    v.sort_unstable_by(|a, b| a.total_cmp(b));
    v.dedup_by(|a, b| a == b || (a.is_nan() && b.is_nan()));
    v
}

/// `np.intersect1d(a, b)`: sorted values present in both.
pub fn intersect1d(a: &[f64], b: &[f64]) -> Vec<f64> {
    let ua = unique(a);
    let ub = unique(b);
    ua.into_iter().filter(|x| ub.iter().any(|y| x == y || (x.is_nan() && y.is_nan()))).collect()
}

/// `np.union1d(a, b)`: sorted values present in either.
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

/// `np.concatenate([arrays...], axis)`: join arrays end-to-end along
/// `axis`. Every array must have the same `ndim` and the same size on
/// every axis *except* `axis` itself.
pub fn concatenate(arrays: &[&NdArray], axis: usize) -> Result<NdArray, ShapeError> {
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

    // Precompute, for each array, the running offset of its slice along
    // `axis` in the output — turns "which source array does output index
    // N belong to" into a linear scan over a handful of boundaries.
    let mut offsets = Vec::with_capacity(arrays.len());
    let mut acc = 0;
    for a in arrays {
        offsets.push(acc);
        acc += a.shape()[axis];
    }

    let data: Vec<f64> = IndexIter::new(&out_shape)
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

/// Step 16 (NEP 56 / Array API standard v2022.12) audit: the standard's
/// own manipulation function is named `concat`, not `concatenate` —
/// NumPy itself keeps `concatenate` as the primary name even post-NEP 56,
/// so both stay valid entry points here too, the same "old name still
/// works, new name is what the standard calls it" precedent
/// [`crate::ufunc::subtract`] follows.
pub fn concat(arrays: &[&NdArray], axis: usize) -> Result<NdArray, ShapeError> {
    concatenate(arrays, axis)
}

/// `np.stack([arrays...], axis)`: like [`concatenate`], but inserts a
/// **new** axis (of length `arrays.len()`) at position `axis` instead of
/// joining along an existing one. Every array must have exactly the same
/// shape.
pub fn stack(arrays: &[&NdArray], axis: usize) -> Result<NdArray, ShapeError> {
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

    let data: Vec<f64> = IndexIter::new(&out_shape)
        .map(|idx| {
            let array_index = idx[axis];
            let mut source_idx = idx.clone();
            source_idx.remove(axis);
            arrays[array_index].get(&source_idx).expect("removing the stacked axis gives a valid source index")
        })
        .collect();
    Ok(NdArray::from_vec(data, &out_shape).expect("data.len() == out_shape.iter().product() by construction"))
}

/// `np.split(arr, sections, axis)`: cut `arr` into `sections` equal
/// pieces along `axis`. Errs if the axis length isn't evenly divisible by
/// `sections` (matches real NumPy's plain `split`; see the module doc
/// comment for the `array_split` distinction this doesn't implement).
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

/// `np.tile(arr, reps)`: repeat `arr`'s content `reps[i]` times along
/// axis `i`. Whichever of `reps`/`arr.ndim()` is shorter gets padded with
/// leading `1`s to match the other, both directions verified against real
/// NumPy: `tile(2x2 matrix, 2)` pads `reps` to `(1, 2)` (only the *last*
/// axis repeats, the matrix's row count is untouched); `tile(2-vector,
/// (2, 2))` pads the vector's own shape to `(1, 2)` (a new leading axis
/// appears) before applying `reps`.
pub fn tile(arr: &NdArray, reps: &[usize]) -> NdArray {
    // Whichever of `arr`'s own shape / `reps` is shorter gets padded with
    // leading size-1 / 1-rep entries, matching real NumPy exactly:
    // `tile((2,2)-matrix, 2)` pads `reps` to `(1,2)` (repeats only the
    // last axis); `tile((2,)-vector, (2,2))` pads the vector's *shape* to
    // `(1,2)` (adds a new leading axis) before applying `reps=(2,2)`.
    let out_ndim = arr.ndim().max(reps.len());
    let mut padded_shape = vec![1usize; out_ndim - arr.ndim()];
    padded_shape.extend_from_slice(arr.shape());
    let mut padded_reps = vec![1usize; out_ndim - reps.len()];
    padded_reps.extend_from_slice(reps);

    let out_shape: Vec<usize> =
        padded_shape.iter().zip(padded_reps.iter()).map(|(&d, &r)| d * r).collect();
    let data: Vec<f64> = IndexIter::new(&out_shape)
        .map(|idx| {
            // Modulo against the padded shape, then drop the leading
            // padded axes (each always size 1, so index 0) to get back to
            // `arr`'s own real dimensionality.
            let padded_source_idx: Vec<usize> =
                idx.iter().zip(padded_shape.iter()).map(|(&i, &d)| i % d).collect();
            let source_idx = &padded_source_idx[out_ndim - arr.ndim()..];
            arr.get(source_idx).expect("modulo keeps the index within the source array's bounds")
        })
        .collect();
    NdArray::from_vec(data, &out_shape).expect("data.len() == out_shape.iter().product() by construction")
}

/// `np.interp(x, xp, fp)`: 1-D linear interpolation. `xp` must already be
/// sorted ascending (real NumPy's own requirement — results are
/// unspecified otherwise, not checked here). Outside `[xp[0], xp[-1]]`,
/// clamps flat to `fp[0]`/`fp[-1]` (real NumPy's default; it also accepts
/// explicit `left`/`right` override values, not implemented here).
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

/// `np.gradient(f, dx)`: the numerical derivative of a 1-D sequence with
/// uniform spacing `dx`. Interior points use the central difference
/// `(f[i+1] - f[i-1]) / (2*dx)`; the two edges use a one-sided difference
/// `(f[1]-f[0])/dx` / `(f[n-1]-f[n-2])/dx` (real NumPy's default
/// `edge_order=1`) — verified against real NumPy at every point, not just
/// the formula in the abstract.
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
        // np.unique([3,1,2,1,NaN,NaN,2]) -> [1,2,3,NaN]
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
        // np.concatenate([[[1,2],[3,4]], [[5,6]]], axis=0)
        let c1 = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let c2 = NdArray::from_vec(vec![5.0, 6.0], &[1, 2]).unwrap();
        let out = concatenate(&[&c1, &c2], 0).unwrap();
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // np.concatenate([[[1,2],[3,4]], [[5],[6]]], axis=1)
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
        // np.stack([[1,2,3,4],[3,4,5,6]], axis=0) -> [[1,2,3,4],[3,4,5,6]]
        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = NdArray::from_vec(vec![3.0, 4.0, 5.0, 6.0], &[4]).unwrap();
        let out = stack(&[&a, &b], 0).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0]);

        // np.stack([...], axis=1) -> [[1,3],[2,4],[3,5],[4,6]]
        let out = stack(&[&a, &b], 1).unwrap();
        assert_eq!(out.shape(), &[4, 2]);
        assert_eq!(out.as_slice(), &[1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0]);
    }

    #[test]
    fn split_into_equal_sections_matches_real_numpy() {
        // np.split(np.arange(9.), 3) -> [[0,1,2],[3,4,5],[6,7,8]]
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
        // np.tile([1,2], 3) -> [1,2,1,2,1,2]
        let e = NdArray::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        assert_eq!(tile(&e, &[3]).as_slice(), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

        // np.tile([1,2], (2,2)) -> [[1,2,1,2],[1,2,1,2]]
        let out = tile(&e, &[2, 2]);
        assert_eq!(out.shape(), &[2, 4]);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn tile_matrix_with_shorter_reps_pads_leading_axes() {
        // np.tile([[1,2],[3,4]], 2) -> [[1,2,1,2],[3,4,3,4]] (only the
        // last axis repeats; reps=(2,) is padded to (1,2))
        let m = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let out = tile(&m, &[2]);
        assert_eq!(out.shape(), &[2, 4]);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn interp_matches_real_numpy_inside_and_extrapolated() {
        let xp = [0.0, 1.0, 2.0, 3.0];
        let fp = [0.0, 10.0, 20.0, 30.0];
        // np.interp([0.5,1.5,2.9], xp, fp) -> [5., 15., 29.]
        let inside = interp(&[0.5, 1.5, 2.9], &xp, &fp);
        for (got, want) in inside.iter().zip([5.0, 15.0, 29.0]) {
            assert!((got - want).abs() < 1e-9);
        }
        // np.interp([-1,5], xp, fp) -> [0., 30.] (clamped flat)
        let outside = interp(&[-1.0, 5.0], &xp, &fp);
        assert_eq!(outside, vec![0.0, 30.0]);
    }

    #[test]
    fn gradient_matches_real_numpy_edges_and_interior() {
        // np.gradient([1,2,4,7,11]) -> [1., 1.5, 2.5, 3.5, 4.]
        let g = gradient(&[1.0, 2.0, 4.0, 7.0, 11.0], 1.0);
        assert_eq!(g, vec![1.0, 1.5, 2.5, 3.5, 4.0]);
        // np.gradient([1,2,4,7,11], 2.0) -> [0.5, 0.75, 1.25, 1.75, 2.]
        let g2 = gradient(&[1.0, 2.0, 4.0, 7.0, 11.0], 2.0);
        assert_eq!(g2, vec![0.5, 0.75, 1.25, 1.75, 2.0]);
    }
}
