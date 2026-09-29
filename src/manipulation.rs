use crate::error::ShapeError;
use crate::ndarray::NdArray;
use crate::reductions::FloatIsh;
use crate::shape::broadcast_shapes;
use crate::view::ArrayView;
use std::cmp::Ordering;

fn norm_axis(axis: isize, ndim: usize) -> Result<usize, ShapeError> {
    let n = ndim as isize;
    if axis < -n || axis >= n {
        return Err(ShapeError::InvalidAxis { axis, ndim });
    }
    Ok(if axis < 0 { axis + n } else { axis } as usize)
}

fn norm_axes(axes: &[isize], ndim: usize) -> Result<Vec<usize>, ShapeError> {
    let out: Vec<usize> = axes.iter().map(|&a| norm_axis(a, ndim)).collect::<Result<_, _>>()?;
    for (i, a) in out.iter().enumerate() {
        if out[..i].contains(a) {
            return Err(ShapeError::RepeatedAxis);
        }
    }
    Ok(out)
}

impl<'a, T> ArrayView<'a, T> {
    fn base_offset(&self) -> isize {
        self.raw().1 as isize
    }

    pub fn permute_dims(&self, axes: &[isize]) -> Result<ArrayView<'a, T>, ShapeError> {
        if axes.len() != self.ndim() {
            return Err(ShapeError::PermutationMismatch { axes: axes.len(), ndim: self.ndim() });
        }
        let order = norm_axes(axes, self.ndim())?;
        let shape = order.iter().map(|&a| self.shape()[a]).collect();
        let strides = order.iter().map(|&a| self.strides()[a]).collect();
        Ok(self.relayout(shape, strides, self.base_offset()))
    }

    pub fn transpose(&self) -> ArrayView<'a, T> {
        let axes: Vec<isize> = (0..self.ndim() as isize).rev().collect();
        self.permute_dims(&axes).expect("reversing every axis is always a valid permutation")
    }

    pub fn moveaxis(&self, source: &[isize], destination: &[isize]) -> Result<ArrayView<'a, T>, ShapeError> {
        let nd = self.ndim();
        let src = norm_axes(source, nd)?;
        let dst = norm_axes(destination, nd)?;
        if src.len() != dst.len() {
            return Err(ShapeError::AxisCountMismatch { source: src.len(), destination: dst.len() });
        }
        let mut order: Vec<usize> = (0..nd).filter(|n| !src.contains(n)).collect();
        let mut pairs: Vec<(usize, usize)> = dst.into_iter().zip(src).collect();
        pairs.sort_unstable();
        for (d, s) in pairs {
            order.insert(d, s);
        }
        let axes: Vec<isize> = order.into_iter().map(|a| a as isize).collect();
        self.permute_dims(&axes)
    }

    pub fn expand_dims(&self, axes: &[isize]) -> Result<ArrayView<'a, T>, ShapeError> {
        let out_nd = self.ndim() + axes.len();
        let new_axes = norm_axes(axes, out_nd)?;
        let (mut shape, mut strides) = (Vec::with_capacity(out_nd), Vec::with_capacity(out_nd));
        let mut old = 0;
        for pos in 0..out_nd {
            if new_axes.contains(&pos) {
                shape.push(1);
                strides.push(0);
            } else {
                shape.push(self.shape()[old]);
                strides.push(self.strides()[old]);
                old += 1;
            }
        }
        Ok(self.relayout(shape, strides, self.base_offset()))
    }

    pub fn squeeze(&self, axes: Option<&[isize]>) -> Result<ArrayView<'a, T>, ShapeError> {
        let drop: Vec<usize> = match axes {
            None => (0..self.ndim()).filter(|&a| self.shape()[a] == 1).collect(),
            Some(list) => {
                let norm = norm_axes(list, self.ndim())?;
                for &a in &norm {
                    if self.shape()[a] != 1 {
                        return Err(ShapeError::SqueezeNotOne { axis: a, size: self.shape()[a] });
                    }
                }
                norm
            }
        };
        let keep: Vec<usize> = (0..self.ndim()).filter(|a| !drop.contains(a)).collect();
        let shape = keep.iter().map(|&a| self.shape()[a]).collect();
        let strides = keep.iter().map(|&a| self.strides()[a]).collect();
        Ok(self.relayout(shape, strides, self.base_offset()))
    }

    pub fn flip(&self, axes: Option<&[isize]>) -> Result<ArrayView<'a, T>, ShapeError> {
        let flipped: Vec<usize> = match axes {
            None => (0..self.ndim()).collect(),
            Some(list) => norm_axes(list, self.ndim())?,
        };
        let (mut strides, mut offset) = (self.strides().to_vec(), self.base_offset());
        for &a in &flipped {
            let n = self.shape()[a];
            if n > 0 {
                offset += (n as isize - 1) * strides[a];
            }
            strides[a] = -strides[a];
        }
        Ok(self.relayout(self.shape().to_vec(), strides, offset))
    }

    pub fn index_axis(&self, axis: isize, index: usize) -> Result<ArrayView<'a, T>, ShapeError> {
        let a = norm_axis(axis, self.ndim())?;
        if index >= self.shape()[a] {
            return Err(ShapeError::FancyIndexOutOfBounds { axis: a, index, dim: self.shape()[a] });
        }
        let offset = self.base_offset() + index as isize * self.strides()[a];
        let (mut shape, mut strides) = (self.shape().to_vec(), self.strides().to_vec());
        shape.remove(a);
        strides.remove(a);
        Ok(self.relayout(shape, strides, offset))
    }
}

pub fn unstack<'a, T>(view: &ArrayView<'a, T>, axis: isize) -> Result<Vec<ArrayView<'a, T>>, ShapeError> {
    if view.ndim() == 0 {
        return Err(ShapeError::ZeroDimOperand);
    }
    let a = norm_axis(axis, view.ndim())?;
    (0..view.shape()[a]).map(|i| view.index_axis(a as isize, i)).collect()
}

pub fn broadcast_arrays<'a, T>(views: &[&ArrayView<'a, T>]) -> Result<Vec<ArrayView<'a, T>>, ShapeError> {
    let mut shape: Vec<usize> = Vec::new();
    for v in views {
        shape = broadcast_shapes(&shape, v.shape()).ok_or_else(|| ShapeError::ChoiceShapeMismatch {
            shapes: views.iter().map(|v| v.shape().to_vec()).collect(),
        })?;
    }
    views.iter().map(|v| v.broadcast_to(&shape)).collect()
}

pub fn expand_dims<T: Copy>(a: &NdArray<T>, axes: &[isize]) -> Result<NdArray<T>, ShapeError> {
    Ok(a.view().expand_dims(axes)?.to_owned())
}

pub fn repeat<T: Copy>(view: &ArrayView<T>, repeats: &[usize], axis: Option<isize>) -> Result<NdArray<T>, ShapeError> {
    let owned = match axis {
        None => {
            let flat = view.to_owned();
            let n = flat.len() as isize;
            flat.into_shape(&[n])?
        }
        Some(_) => view.to_owned(),
    };
    let ax = match axis {
        None => 0,
        Some(a) => norm_axis(a, owned.ndim())?,
    };
    let len = owned.shape()[ax];
    if repeats.len() != 1 && repeats.len() != len {
        return Err(ShapeError::RepeatLengthMismatch { repeats: repeats.len(), len });
    }
    let per = |i: usize| if repeats.len() == 1 { repeats[0] } else { repeats[i] };
    let outer: usize = owned.shape()[..ax].iter().product();
    let inner: usize = owned.shape()[ax + 1..].iter().product();
    let new_len: usize = (0..len).map(per).sum();
    let mut data = Vec::with_capacity(outer * new_len * inner);
    for o in 0..outer {
        for i in 0..len {
            let block = &owned.as_slice()[(o * len + i) * inner..(o * len + i + 1) * inner];
            for _ in 0..per(i) {
                data.extend_from_slice(block);
            }
        }
    }
    let mut shape = owned.shape().to_vec();
    shape[ax] = new_len;
    NdArray::from_vec(data, &shape)
}

pub fn roll<T: Copy>(view: &ArrayView<T>, shifts: &[isize], axes: Option<&[isize]>) -> Result<NdArray<T>, ShapeError> {
    let mut current = view.to_owned();
    match axes {
        None => {
            let shape = current.shape().to_vec();
            let n = current.len();
            let mut flat = current.into_vec();
            for &shift in shifts {
                flat = roll_axis(&flat, &[n], 0, shift);
            }
            NdArray::from_vec(flat, &shape)
        }
        Some(list) => {
            let norm = norm_axes_allow_repeat(list, current.ndim())?;
            if shifts.len() != norm.len() && shifts.len() != 1 && norm.len() != 1 {
                return Err(ShapeError::RollShiftAxisMismatch { shifts: shifts.len(), axes: norm.len() });
            }
            let count = shifts.len().max(norm.len());
            for k in 0..count {
                let shift = if shifts.len() == 1 { shifts[0] } else { shifts[k] };
                let ax = if norm.len() == 1 { norm[0] } else { norm[k] };
                let shape = current.shape().to_vec();
                let rolled = roll_axis(current.as_slice(), &shape, ax, shift);
                current = NdArray::from_vec(rolled, &shape)?;
            }
            Ok(current)
        }
    }
}

fn norm_axes_allow_repeat(axes: &[isize], ndim: usize) -> Result<Vec<usize>, ShapeError> {
    axes.iter().map(|&a| norm_axis(a, ndim)).collect()
}

fn roll_axis<T: Copy>(data: &[T], shape: &[usize], axis: usize, shift: isize) -> Vec<T> {
    let n = shape[axis];
    if n == 0 {
        return data.to_vec();
    }
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let s = shift.rem_euclid(n as isize) as usize;
    let mut out = data.to_vec();
    for o in 0..outer {
        for i in 0..n {
            let dst = (i + s) % n;
            let src = (o * n + i) * inner;
            let dst = (o * n + dst) * inner;
            out[dst..dst + inner].copy_from_slice(&data[src..src + inner]);
        }
    }
    out
}

#[derive(Debug, Clone, PartialEq)]
pub struct UniqueAll<T> {
    pub values: Vec<T>,
    pub indices: Vec<usize>,
    pub inverse_indices: NdArray<usize>,
    pub counts: Vec<usize>,
}

fn cmp_nan_last<T: FloatIsh>(a: &T, b: &T) -> Ordering {
    match (a.is_nan_ish(), b.is_nan_ish()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
    }
}

pub fn unique_all<T: FloatIsh>(view: &ArrayView<T>) -> UniqueAll<T> {
    let flat: Vec<T> = view.iter().collect();
    let mut order: Vec<usize> = (0..flat.len()).collect();
    order.sort_by(|&x, &y| cmp_nan_last(&flat[x], &flat[y]));
    let mut values: Vec<T> = Vec::new();
    let mut indices: Vec<usize> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    let mut inverse = vec![0usize; flat.len()];
    for &i in &order {
        let starts_new_group = match values.last() {
            None => true,
            Some(last) => last.is_nan_ish() || flat[i].is_nan_ish() || *last != flat[i],
        };
        if starts_new_group {
            values.push(flat[i]);
            indices.push(i);
            counts.push(0);
        }
        *counts.last_mut().unwrap() += 1;
        inverse[i] = values.len() - 1;
    }
    let inverse_indices = NdArray::from_vec(inverse, view.shape()).expect("one inverse index per element");
    UniqueAll { values, indices, inverse_indices, counts }
}

pub fn unique_values<T: FloatIsh>(view: &ArrayView<T>) -> Vec<T> {
    unique_all(view).values
}

pub fn unique_counts<T: FloatIsh>(view: &ArrayView<T>) -> (Vec<T>, Vec<usize>) {
    let u = unique_all(view);
    (u.values, u.counts)
}

pub fn unique_inverse<T: FloatIsh>(view: &ArrayView<T>) -> (Vec<T>, NdArray<usize>) {
    let u = unique_all(view);
    (u.values, u.inverse_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ar(shape: &[usize]) -> NdArray<i64> {
        let n: usize = shape.iter().product();
        NdArray::from_vec((0..n as i64).collect(), shape).unwrap()
    }

    fn vals(v: &ArrayView<i64>) -> Vec<i64> {
        v.iter().collect()
    }

    #[test]
    fn unique_family_matches_numpy_including_nan_and_signed_zero() {
        let x = NdArray::from_vec(vec![3.0, 1.0, 2.0, 1.0, 3.0, 3.0, f64::NAN, f64::NAN, 0.0, -0.0], &[10]).unwrap();
        let u = unique_all(&x.view());
        assert_eq!(&u.values[..4], &[0.0, 1.0, 2.0, 3.0]);
        assert!(u.values[4].is_nan() && u.values[5].is_nan());
        assert_eq!(u.values.len(), 6);
        assert_eq!(u.indices, vec![8, 1, 2, 0, 6, 7]);
        assert_eq!(u.inverse_indices.as_slice(), &[3, 1, 2, 1, 3, 3, 4, 5, 0, 0]);
        assert_eq!(u.counts, vec![2, 2, 1, 3, 1, 1]);
        assert_eq!(unique_counts(&x.view()).1, u.counts);
        assert_eq!(unique_inverse(&x.view()).1, u.inverse_indices);
        assert_eq!(unique_values(&x.view()).len(), 6);
    }

    #[test]
    fn unique_inverse_keeps_the_input_shape_for_n_d_input() {
        let m = NdArray::from_vec(vec![3i64, 1, 1, 2], &[2, 2]).unwrap();
        let u = unique_all(&m.view());
        assert_eq!(u.values, vec![1, 2, 3]);
        assert_eq!(u.indices, vec![1, 3, 0]);
        assert_eq!(u.inverse_indices.shape(), &[2, 2]);
        assert_eq!(u.inverse_indices.as_slice(), &[2, 0, 0, 1]);
        assert_eq!(u.counts, vec![2, 1, 1]);
        let rebuilt: Vec<i64> = u.inverse_indices.as_slice().iter().map(|&i| u.values[i]).collect();
        assert_eq!(rebuilt, m.as_slice());
        let empty: NdArray<f64> = NdArray::zeros(&[0]);
        assert!(unique_all(&empty.view()).values.is_empty());
        assert_eq!(unique_counts(&NdArray::from_vec(vec![5i64, 5], &[2]).unwrap().view()).1, vec![2]);
    }

    #[test]
    fn broadcast_arrays_returns_zero_copy_views_of_the_common_shape() {
        let (a, b, c) = (ar(&[3, 1]), ar(&[4]), ar(&[2, 1, 1]));
        let out = broadcast_arrays(&[&a.view(), &b.view(), &c.view()]).unwrap();
        assert!(out.iter().all(|v| v.shape() == [2, 3, 4]));
        assert_eq!(out[0].strides()[2], 0);
        assert_eq!(broadcast_arrays(&[&ar(&[2]).view()]).unwrap()[0].shape(), &[2]);
        assert!(matches!(
            broadcast_arrays(&[&ar(&[3]).view(), &ar(&[4]).view()]),
            Err(ShapeError::ChoiceShapeMismatch { .. })
        ));
    }

    #[test]
    fn expand_dims_inserts_size_one_axes_and_validates_like_numpy() {
        let a = ar(&[2, 3]);
        let v = a.view();
        for (axes, shape) in [(vec![0], vec![1, 2, 3]), (vec![1], vec![2, 1, 3]), (vec![-1], vec![2, 3, 1]), (vec![0, 3], vec![1, 2, 3, 1])] {
            assert_eq!(v.expand_dims(&axes).unwrap().shape(), shape.as_slice());
        }
        assert_eq!(v.expand_dims(&[4]).unwrap_err(), ShapeError::InvalidAxis { axis: 4, ndim: 3 });
        assert_eq!(v.expand_dims(&[-5]).unwrap_err(), ShapeError::InvalidAxis { axis: -5, ndim: 3 });
        assert_eq!(v.expand_dims(&[0, 0]).unwrap_err(), ShapeError::RepeatedAxis);
        assert_eq!(v.expand_dims(&[1, -3]).unwrap_err(), ShapeError::RepeatedAxis);
        assert_eq!(expand_dims(&a, &[0]).unwrap().shape(), &[1, 2, 3]);
        assert_eq!(vals(&v.expand_dims(&[1]).unwrap()), vals(&v));
    }

    #[test]
    fn flip_is_a_view_with_negative_strides() {
        let a = ar(&[2, 3]);
        let v = a.view();
        assert_eq!(vals(&v.flip(None).unwrap()), vec![5, 4, 3, 2, 1, 0]);
        assert_eq!(vals(&v.flip(Some(&[0])).unwrap()), vec![3, 4, 5, 0, 1, 2]);
        assert_eq!(vals(&v.flip(Some(&[1])).unwrap()), vec![2, 1, 0, 5, 4, 3]);
        assert_eq!(vals(&v.flip(Some(&[0, 1])).unwrap()), vec![5, 4, 3, 2, 1, 0]);
        assert_eq!(vals(&v.flip(Some(&[-1])).unwrap()), vec![2, 1, 0, 5, 4, 3]);
        assert_eq!(v.flip(Some(&[1])).unwrap().strides(), &[3, -1]);
        assert_eq!(vals(&v.flip(Some(&[1])).unwrap().flip(Some(&[1])).unwrap()), vals(&v));
        assert_eq!(v.flip(Some(&[2])).unwrap_err(), ShapeError::InvalidAxis { axis: 2, ndim: 2 });
        let sliced = v.slice(&[0..2, 1..3]).unwrap();
        assert_eq!(vals(&sliced.flip(None).unwrap()), vec![5, 4, 2, 1]);
        assert_eq!(vals(&sliced.flip(None).unwrap().to_owned().view()), vec![5, 4, 2, 1]);
    }

    #[test]
    fn moveaxis_and_permute_dims_follow_numpy() {
        let b = ar(&[2, 3, 4]);
        let v = b.view();
        assert_eq!(v.moveaxis(&[0], &[-1]).unwrap().shape(), &[3, 4, 2]);
        let m = v.moveaxis(&[0, 1], &[-1, -2]).unwrap();
        assert_eq!(m.shape(), &[4, 3, 2]);
        assert_eq!(m.get(&[1, 2, 0]), Some(9));
        assert_eq!(v.moveaxis(&[2], &[0]).unwrap().shape(), &[4, 2, 3]);
        assert_eq!(v.moveaxis(&[0], &[0]).unwrap().shape(), &[2, 3, 4]);
        assert_eq!(v.moveaxis(&[5], &[0]).unwrap_err(), ShapeError::InvalidAxis { axis: 5, ndim: 3 });
        assert_eq!(v.moveaxis(&[0, 0], &[1, 2]).unwrap_err(), ShapeError::RepeatedAxis);
        assert_eq!(
            v.moveaxis(&[0, 1], &[2]).unwrap_err(),
            ShapeError::AxisCountMismatch { source: 2, destination: 1 }
        );

        let p = v.permute_dims(&[2, 0, 1]).unwrap();
        assert_eq!(p.shape(), &[4, 2, 3]);
        assert_eq!(p.get(&[3, 1, 2]), Some(23));
        assert_eq!(v.permute_dims(&[0, 1]).unwrap_err(), ShapeError::PermutationMismatch { axes: 2, ndim: 3 });
        assert_eq!(v.permute_dims(&[0, 0, 1]).unwrap_err(), ShapeError::RepeatedAxis);
        assert_eq!(v.permute_dims(&[0, 1, 3]).unwrap_err(), ShapeError::InvalidAxis { axis: 3, ndim: 3 });
        assert_eq!(v.transpose().shape(), &[4, 3, 2]);
        assert_eq!(v.transpose().get(&[3, 2, 1]), Some(23));
    }

    #[test]
    fn repeat_matches_numpy_for_scalar_and_per_element_counts() {
        let a = ar(&[2, 3]);
        let v = a.view();
        assert_eq!(repeat(&v, &[2], None).unwrap().as_slice(), &[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
        assert_eq!(repeat(&v, &[2], Some(0)).unwrap().as_slice(), &[0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5]);
        assert_eq!(repeat(&v, &[1, 2, 3], Some(1)).unwrap().as_slice(), &[0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 5]);
        let r = repeat(&v, &[1, 2], Some(0)).unwrap();
        assert_eq!(r.shape(), &[3, 3]);
        assert_eq!(r.as_slice(), &[0, 1, 2, 3, 4, 5, 3, 4, 5]);
        assert_eq!(repeat(&v, &[0], Some(1)).unwrap().shape(), &[2, 0]);
        assert_eq!(repeat(&v, &[2], Some(-1)).unwrap().shape(), &[2, 6]);
        assert_eq!(repeat(&v, &[1, 2], Some(1)).unwrap_err(), ShapeError::RepeatLengthMismatch { repeats: 2, len: 3 });
        assert_eq!(repeat(&v, &[1, 2, 3], None).unwrap_err(), ShapeError::RepeatLengthMismatch { repeats: 3, len: 6 });
        assert_eq!(repeat(&v, &[1], Some(2)).unwrap_err(), ShapeError::InvalidAxis { axis: 2, ndim: 2 });
    }

    #[test]
    fn roll_matches_numpy_for_flat_axis_tuple_and_wraparound_shifts() {
        let a = ar(&[2, 3]);
        let v = a.view();
        let flat = roll(&v, &[1], None).unwrap();
        assert_eq!(flat.shape(), &[2, 3]);
        assert_eq!(flat.as_slice(), &[5, 0, 1, 2, 3, 4]);
        assert_eq!(roll(&v, &[1, 2], None).unwrap().as_slice(), &[3, 4, 5, 0, 1, 2]);
        assert_eq!(roll(&v, &[1], Some(&[0])).unwrap().as_slice(), &[3, 4, 5, 0, 1, 2]);
        assert_eq!(roll(&v, &[-1], Some(&[1])).unwrap().as_slice(), &[1, 2, 0, 4, 5, 3]);
        assert_eq!(roll(&v, &[1, 1], Some(&[0, 1])).unwrap().as_slice(), &[5, 3, 4, 2, 0, 1]);
        assert_eq!(roll(&v, &[7], Some(&[1])).unwrap().as_slice(), &[2, 0, 1, 5, 3, 4]);
        assert_eq!(roll(&v, &[1], Some(&[0, 1])).unwrap().as_slice(), &[5, 3, 4, 2, 0, 1]);
        let empty: NdArray<i64> = NdArray::zeros(&[0]);
        assert!(roll(&empty.view(), &[2], None).unwrap().is_empty());
        assert!(matches!(roll(&v, &[1, 2, 3], Some(&[0, 1])), Err(ShapeError::RollShiftAxisMismatch { .. })));
        assert_eq!(roll(&v, &[1], Some(&[2])).unwrap_err(), ShapeError::InvalidAxis { axis: 2, ndim: 2 });
    }

    #[test]
    fn squeeze_and_unstack_match_numpy() {
        let a = NdArray::from_vec(vec![1i64, 2, 3], &[1, 3, 1]).unwrap();
        let v = a.view();
        assert_eq!(v.squeeze(None).unwrap().shape(), &[3]);
        assert_eq!(v.squeeze(Some(&[0])).unwrap().shape(), &[3, 1]);
        assert_eq!(v.squeeze(Some(&[0, 2])).unwrap().shape(), &[3]);
        assert_eq!(v.squeeze(Some(&[-1])).unwrap().shape(), &[1, 3]);
        assert_eq!(v.squeeze(Some(&[1])).unwrap_err(), ShapeError::SqueezeNotOne { axis: 1, size: 3 });
        let ones = NdArray::from_vec(vec![1i64], &[1, 1]).unwrap();
        assert_eq!(ones.view().squeeze(None).unwrap().shape(), &[] as &[usize]);

        let m = ar(&[2, 3]);
        let rows: Vec<Vec<i64>> = unstack(&m.view(), 0).unwrap().iter().map(vals).collect();
        assert_eq!(rows, vec![vec![0, 1, 2], vec![3, 4, 5]]);
        let cols: Vec<Vec<i64>> = unstack(&m.view(), 1).unwrap().iter().map(vals).collect();
        assert_eq!(cols, vec![vec![0, 3], vec![1, 4], vec![2, 5]]);
        assert_eq!(unstack(&ar(&[2, 3, 4]).view(), -1).unwrap()[0].shape(), &[2, 3]);
        let scalar = NdArray::from_vec(vec![3i64], &[]).unwrap();
        assert_eq!(unstack(&scalar.view(), 0).unwrap_err(), ShapeError::ZeroDimOperand);
        assert!(m.view().index_axis(0, 2).is_err());
    }

    #[test]
    fn views_compose_without_copying() {
        let a = ar(&[2, 3, 4]);
        let v = a.view().moveaxis(&[0], &[-1]).unwrap().flip(Some(&[0])).unwrap().expand_dims(&[0]).unwrap();
        assert_eq!(v.shape(), &[1, 3, 4, 2]);
        assert_eq!(v.get(&[0, 0, 0, 1]), Some(a.get(&[1, 2, 0]).unwrap()));
        let owned = v.to_owned();
        assert_eq!(owned.shape(), &[1, 3, 4, 2]);
        assert_eq!(owned.get(&[0, 0, 0, 1]), v.get(&[0, 0, 0, 1]));
    }
}
