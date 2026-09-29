use crate::error::ShapeError;
use crate::ndarray::NdArray;
use crate::reductions::FloatIsh;
use crate::view::ArrayView;
use std::cmp::Ordering;

fn total_cmp<T: FloatIsh>(a: &T, b: &T) -> Ordering {
    match (a.is_nan_ish(), b.is_nan_ish()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
    }
}

fn lanes<T: Copy>(view: &ArrayView<T>, axis: usize) -> Result<(NdArray<T>, usize, usize, usize), ShapeError> {
    if axis >= view.ndim() {
        return Err(ShapeError::AxisOutOfBounds { axis, ndim: view.ndim() });
    }
    let owned = view.to_owned();
    let n = owned.shape()[axis];
    let outer: usize = owned.shape()[..axis].iter().product();
    let inner: usize = owned.shape()[axis + 1..].iter().product();
    Ok((owned, outer, n, inner))
}

pub fn sort<T: FloatIsh>(view: &ArrayView<T>, axis: usize) -> Result<NdArray<T>, ShapeError> {
    let (owned, outer, n, inner) = lanes(view, axis)?;
    let src = owned.as_slice();
    let mut out = src.to_vec();
    let mut lane: Vec<T> = Vec::with_capacity(n);
    for o in 0..outer {
        for i in 0..inner {
            lane.clear();
            lane.extend((0..n).map(|k| src[(o * n + k) * inner + i]));
            lane.sort_by(total_cmp);
            for (k, &v) in lane.iter().enumerate() {
                out[(o * n + k) * inner + i] = v;
            }
        }
    }
    NdArray::from_vec(out, owned.shape())
}

pub fn argsort<T: FloatIsh>(view: &ArrayView<T>, axis: usize) -> Result<NdArray<usize>, ShapeError> {
    let (owned, outer, n, inner) = lanes(view, axis)?;
    let src = owned.as_slice();
    let mut out = vec![0usize; src.len()];
    let mut order: Vec<usize> = Vec::with_capacity(n);
    for o in 0..outer {
        for i in 0..inner {
            order.clear();
            order.extend(0..n);
            order.sort_by(|&x, &y| total_cmp(&src[(o * n + x) * inner + i], &src[(o * n + y) * inner + i]));
            for (k, &idx) in order.iter().enumerate() {
                out[(o * n + k) * inner + i] = idx;
            }
        }
    }
    NdArray::from_vec(out, owned.shape())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

pub fn searchsorted<T: FloatIsh>(sorted: &[T], values: &[T], side: Side) -> Vec<usize> {
    values
        .iter()
        .map(|v| match side {
            Side::Left => sorted.partition_point(|x| total_cmp(x, v) == Ordering::Less),
            Side::Right => sorted.partition_point(|x| total_cmp(x, v) != Ordering::Greater),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr<T>(data: Vec<T>, shape: &[usize]) -> NdArray<T> {
        NdArray::from_vec(data, shape).unwrap()
    }

    #[test]
    fn sort_puts_nan_last_like_numpy() {
        let x = arr(vec![3.0, f64::NAN, 1.0, f64::NEG_INFINITY, f64::NAN, 2.0, f64::INFINITY], &[7]);
        let s = sort(&x.view(), 0).unwrap();
        let s = s.as_slice();
        assert_eq!(&s[..5], &[f64::NEG_INFINITY, 1.0, 2.0, 3.0, f64::INFINITY]);
        assert!(s[5].is_nan() && s[6].is_nan());
        let idx = argsort(&x.view(), 0).unwrap();
        assert_eq!(idx.as_slice(), &[3, 2, 5, 0, 6, 1, 4]);
    }

    #[test]
    fn sort_and_argsort_along_each_axis() {
        let m = arr(vec![3, 1, 2, 9, 7, 8], &[2, 3]);
        assert_eq!(sort(&m.view(), 0).unwrap().as_slice(), &[3, 1, 2, 9, 7, 8]);
        assert_eq!(sort(&m.view(), 1).unwrap().as_slice(), &[1, 2, 3, 7, 8, 9]);
        assert_eq!(argsort(&m.view(), 0).unwrap().as_slice(), &[0, 0, 0, 1, 1, 1]);
        assert_eq!(argsort(&m.view(), 1).unwrap().as_slice(), &[1, 2, 0, 1, 2, 0]);
        assert_eq!(sort(&m.view(), 2).unwrap_err(), ShapeError::AxisOutOfBounds { axis: 2, ndim: 2 });
    }

    #[test]
    fn sort_accepts_non_contiguous_views() {
        let m = arr(vec![1, 2, 0, 5], &[2, 2]);
        let t = m.view().slice(&[0..2, 0..2]).unwrap();
        assert_eq!(sort(&t, 1).unwrap().as_slice(), &[1, 2, 0, 5]);
        let col = m.view().slice(&[0..2, 1..2]).unwrap();
        assert_eq!(sort(&col, 0).unwrap().as_slice(), &[2, 5]);
    }

    #[test]
    fn argsort_is_stable_on_ties() {
        let t = arr(vec![2, 1, 2, 1, 2, 1], &[6]);
        assert_eq!(argsort(&t.view(), 0).unwrap().as_slice(), &[1, 3, 5, 0, 2, 4]);
    }

    #[test]
    fn searchsorted_matches_numpy_for_both_sides() {
        let s = [1, 2, 2, 2, 5];
        let v = [0, 1, 2, 3, 5, 6];
        assert_eq!(searchsorted(&s, &v, Side::Left), vec![0, 0, 1, 4, 4, 5]);
        assert_eq!(searchsorted(&s, &v, Side::Right), vec![0, 1, 4, 4, 5, 5]);
    }

    #[test]
    fn searchsorted_treats_nan_as_largest() {
        let s = [1.0, 2.0, f64::NAN, f64::NAN];
        assert_eq!(searchsorted(&s, &[f64::NAN, 2.0, 3.0], Side::Left), vec![2, 1, 2]);
        assert_eq!(searchsorted(&s, &[f64::NAN], Side::Right), vec![4]);
    }
}
