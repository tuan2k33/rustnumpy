use crate::reductions::ReductionError;
use crate::error::{OpError, ShapeError};
use crate::ndarray::NdArray;
use crate::shape::{broadcast_shapes, IndexIter};
use crate::view::{ArrayView, ArrayViewMut};
use num_complex::Complex;

pub trait WrapAdd: Copy {
    fn wrap_add(self, rhs: Self) -> Self;
}

pub trait WrapSub: Copy {
    fn wrap_sub(self, rhs: Self) -> Self;
}

pub trait WrapMul: Copy {
    fn wrap_mul(self, rhs: Self) -> Self;
}

macro_rules! wrap_ints {
    ($($t:ty),*) => {$(
        impl WrapAdd for $t { fn wrap_add(self, rhs: Self) -> Self { self.wrapping_add(rhs) } }
        impl WrapSub for $t { fn wrap_sub(self, rhs: Self) -> Self { self.wrapping_sub(rhs) } }
        impl WrapMul for $t { fn wrap_mul(self, rhs: Self) -> Self { self.wrapping_mul(rhs) } }
    )*};
}
wrap_ints!(i8, i16, i32, i64, u8, u16, u32, u64);

macro_rules! wrap_plain {
    ($($t:ty),*) => {$(
        impl WrapAdd for $t { fn wrap_add(self, rhs: Self) -> Self { self + rhs } }
        impl WrapSub for $t { fn wrap_sub(self, rhs: Self) -> Self { self - rhs } }
        impl WrapMul for $t { fn wrap_mul(self, rhs: Self) -> Self { self * rhs } }
    )*};
}
wrap_plain!(half::f16, f32, f64, Complex<f32>, Complex<f64>);

impl WrapAdd for bool {
    fn wrap_add(self, rhs: Self) -> Self {
        self | rhs
    }
}

impl WrapMul for bool {
    fn wrap_mul(self, rhs: Self) -> Self {
        self & rhs
    }
}

fn out_shape<A, B>(a: &ArrayView<A>, b: &ArrayView<B>) -> Result<Vec<usize>, ShapeError> {
    broadcast_shapes(a.shape(), b.shape()).ok_or_else(|| ShapeError::NotBroadcastable {
        lhs: a.shape().to_vec(),
        rhs: b.shape().to_vec(),
    })
}

pub fn zip_assign<T: Copy>(
    out: &mut ArrayViewMut<T>,
    b: &ArrayView<T>,
    mask: Option<&ArrayView<bool>>,
    f: impl Fn(T, T) -> T,
) -> Result<(), ShapeError> {
    let shape = out.shape().to_vec();
    let b_b = b.broadcast_to(&shape)?;
    let mask_b = mask.map(|m| m.broadcast_to(&shape)).transpose()?;
    for idx in IndexIter::new(&shape) {
        if mask_b.as_ref().is_some_and(|m| !m.get(&idx).unwrap()) {
            continue;
        }
        let value = f(out.get(&idx).unwrap(), b_b.get(&idx).unwrap());
        out.set(&idx, value)?;
    }
    Ok(())
}

pub fn add_assign<T: WrapAdd>(out: &mut ArrayViewMut<T>, b: &ArrayView<T>) -> Result<(), ShapeError> {
    zip_assign(out, b, None, T::wrap_add)
}

pub fn sub_assign<T: WrapSub>(out: &mut ArrayViewMut<T>, b: &ArrayView<T>) -> Result<(), ShapeError> {
    zip_assign(out, b, None, T::wrap_sub)
}

pub fn mul_assign<T: WrapMul>(out: &mut ArrayViewMut<T>, b: &ArrayView<T>) -> Result<(), ShapeError> {
    zip_assign(out, b, None, T::wrap_mul)
}

pub fn zip_into_where<T: Copy>(
    out: &mut ArrayViewMut<T>,
    a: &ArrayView<T>,
    b: &ArrayView<T>,
    mask: Option<&ArrayView<bool>>,
    f: impl Fn(T, T) -> T,
) -> Result<(), ShapeError> {
    let shape = out_shape(a, b)?;
    if out.shape() != shape.as_slice() {
        return Err(ShapeError::DataShapeMismatch { data_len: out.shape().iter().product(), shape });
    }
    let (a_b, b_b) = (a.broadcast_to(&shape)?, b.broadcast_to(&shape)?);
    let mask_b = mask.map(|m| m.broadcast_to(&shape)).transpose()?;
    for idx in IndexIter::new(&shape) {
        if mask_b.as_ref().is_some_and(|m| !m.get(&idx).unwrap()) {
            continue;
        }
        out.set(&idx, f(a_b.get(&idx).unwrap(), b_b.get(&idx).unwrap()))?;
    }
    Ok(())
}

pub struct ReduceOptions<'a, T> {
    pub initial: Option<T>,
    pub where_: Option<&'a ArrayView<'a, bool>>,
    pub keepdims: bool,
}

impl<T> Default for ReduceOptions<'_, T> {
    fn default() -> Self {
        Self { initial: None, where_: None, keepdims: false }
    }
}

fn check_axis(axis: usize, ndim: usize) -> Result<(), ShapeError> {
    if axis >= ndim {
        Err(ShapeError::AxisOutOfBounds { axis, ndim })
    } else {
        Ok(())
    }
}

pub fn reduce<T: Copy>(
    view: &ArrayView<T>,
    axis: usize,
    opts: ReduceOptions<T>,
    f: impl Fn(T, T) -> T,
) -> Result<NdArray<T>, OpError> {
    check_axis(axis, view.ndim())?;
    if opts.where_.is_some() && opts.initial.is_none() {
        return Err(ReductionError::WhereNeedsInitial.into());
    }
    let owned = view.to_owned();
    let mask = opts.where_.map(|m| m.broadcast_to(view.shape()).map(|m| m.to_owned())).transpose()?;
    let n = owned.shape()[axis];
    let outer: usize = owned.shape()[..axis].iter().product();
    let inner: usize = owned.shape()[axis + 1..].iter().product();
    if n == 0 && opts.initial.is_none() {
        return Err(ReductionError::EmptyInput.into());
    }
    let src = owned.as_slice();
    let mut data = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = opts.initial;
            for k in 0..n {
                let at = (o * n + k) * inner + i;
                if mask.as_ref().is_some_and(|m| !m.as_slice()[at]) {
                    continue;
                }
                acc = Some(match acc {
                    Some(a) => f(a, src[at]),
                    None => src[at],
                });
            }
            data.push(acc.ok_or(ReductionError::EmptyInput)?);
        }
    }
    let mut shape: Vec<usize> = owned.shape().to_vec();
    if opts.keepdims {
        shape[axis] = 1;
    } else {
        shape.remove(axis);
    }
    NdArray::from_vec(data, &shape).map_err(OpError::from)
}

pub fn accumulate<T: Copy>(view: &ArrayView<T>, axis: usize, f: impl Fn(T, T) -> T) -> Result<NdArray<T>, ShapeError> {
    check_axis(axis, view.ndim())?;
    let owned = view.to_owned();
    let n = owned.shape()[axis];
    let inner: usize = owned.shape()[axis + 1..].iter().product();
    let outer: usize = owned.shape()[..axis].iter().product();
    let mut data = owned.as_slice().to_vec();
    for o in 0..outer {
        for i in 0..inner {
            for k in 1..n {
                let at = (o * n + k) * inner + i;
                data[at] = f(data[at - inner], data[at]);
            }
        }
    }
    NdArray::from_vec(data, owned.shape())
}

pub fn outer_with<T: Copy>(a: &ArrayView<T>, b: &ArrayView<T>, f: impl Fn(T, T) -> T) -> Result<NdArray<T>, ShapeError> {
    let bv: Vec<T> = b.iter().collect();
    let mut data = Vec::with_capacity(a.len() * bv.len());
    for x in a.iter() {
        data.extend(bv.iter().map(|&y| f(x, y)));
    }
    let shape: Vec<usize> = a.shape().iter().chain(b.shape()).copied().collect();
    NdArray::from_vec(data, &shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::ufunc::{add, mul, sub};
    use num_complex::Complex;

    fn arr<T>(data: Vec<T>) -> NdArray<T> {
        let n = data.len();
        NdArray::from_vec(data, &[n]).unwrap()
    }

    #[test]
    fn mixed_dtypes_are_cast_explicitly_to_the_common_dtype() {
        let i = arr(vec![1i32, 2, 3]);
        let f = arr(vec![0.5f64, 0.5, 0.5]);
        assert_eq!(crate::dtype::common_dtype(i32::KIND, f64::KIND), f64::KIND);
        assert_eq!(add(&i.astype::<f64>().view(), &f.view()).unwrap().as_slice(), &[1.5, 2.5, 3.5]);
        let (u, s8) = (arr(vec![200u8]), arr(vec![-100i8]));
        assert_eq!(sub(&u.astype::<i16>().view(), &s8.astype::<i16>().view()).unwrap().as_slice(), &[300]);
        let c = arr(vec![Complex::new(1.0f32, 1.0)]);
        let cd = add(&c.astype::<Complex<f64>>().view(), &arr(vec![2.0f64]).astype::<Complex<f64>>().view()).unwrap();
        assert_eq!(cd.as_slice(), &[Complex::new(3.0, 1.0)]);
        let (b, bi) = (arr(vec![true, false]), arr(vec![10i8, 20]));
        assert_eq!(mul(&b.astype::<i8>().view(), &bi.view()).unwrap().as_slice(), &[10, 0]);
    }

    #[test]
    fn int_overflow_wraps_in_the_operand_type() {
        let a = arr(vec![100i8]);
        assert_eq!(add(&a.view(), &a.view()).unwrap().as_slice(), &[-56]);
        assert_eq!(add(&a.astype::<i16>().view(), &a.astype::<i16>().view()).unwrap().as_slice(), &[200i16]);
    }

    #[test]
    fn integer_arithmetic_wraps_in_every_entry_point_like_numpy() {
        let a = arr(vec![i8::MAX]);
        assert_eq!(crate::reductions::sum(&arr(vec![i8::MAX, 1]).view()), i8::MIN);
        assert_eq!(crate::contraction::dot(&a.view(), &arr(vec![2i8]).view()).unwrap().as_slice(), &[-2]);
        let mut m = arr(vec![i8::MAX]);
        add_assign(&mut m.view_mut(), &arr(vec![1i8]).view()).unwrap();
        assert_eq!(m.as_slice(), &[i8::MIN]);
    }

    #[test]
    fn in_place_ops_broadcast_the_operand_into_the_output() {
        let mut a = arr(vec![1i32, 2, 3]);
        add_assign(&mut a.view_mut(), &arr(vec![1i64, 1, 1]).astype::<i32>().view()).unwrap();
        assert_eq!(a.as_slice(), &[2, 3, 4]);
        mul_assign(&mut a.view_mut(), &arr(vec![2i32]).view()).unwrap();
        assert_eq!(a.as_slice(), &[4, 6, 8]);
        let mut m = NdArray::from_vec(vec![1, 2, 3, 4], &[2, 2]).unwrap();
        sub_assign(&mut m.view_mut(), &arr(vec![10, 20]).view()).unwrap();
        assert_eq!(m.as_slice(), &[-9, -18, -7, -16]);
        let mut too_small = arr(vec![1, 2, 3]);
        let err = add_assign(&mut too_small.view_mut(), &arr(vec![1, 2]).view()).unwrap_err();
        assert!(matches!(err, ShapeError::NotBroadcastable { .. }));
    }

    #[test]
    fn where_mask_leaves_masked_out_slots_untouched() {
        let x = arr(vec![1i64, 2, 3, 4]);
        let y = arr(vec![10i64, 20, 30, 40]);
        let mut out = arr(vec![-1i64; 4]);
        let mask = arr(vec![true, false, true, false]);
        zip_into_where(&mut out.view_mut(), &x.view(), &y.view(), Some(&mask.view()), |a, b| a + b).unwrap();
        assert_eq!(out.as_slice(), &[11, -1, 33, -1]);

        let mut z = arr(vec![1i64, 2, 3, 4]);
        zip_assign(&mut z.view_mut(), &y.view(), Some(&mask.view()), |a, b| a + b).unwrap();
        assert_eq!(z.as_slice(), &[11, 2, 33, 4]);

        let mut small = arr(vec![0i64; 3]);
        assert!(zip_into_where(&mut small.view_mut(), &x.view(), &y.view(), None, |a, b| a + b).is_err());
    }

    #[test]
    fn reduce_matches_numpy_add_multiply_maximum_reduce() {
        let m = NdArray::from_vec((0..6).collect::<Vec<i64>>(), &[2, 3]).unwrap();
        let sum = |x: i64, y: i64| x + y;
        assert_eq!(reduce(&m.view(), 0, ReduceOptions::default(), sum).unwrap().as_slice(), &[3, 5, 7]);
        assert_eq!(reduce(&m.view(), 1, ReduceOptions::default(), sum).unwrap().as_slice(), &[3, 12]);
        assert_eq!(reduce(&m.view(), 1, ReduceOptions::default(), |x, y| x * y).unwrap().as_slice(), &[0, 60]);
        let keep = reduce(&m.view(), 1, ReduceOptions { keepdims: true, ..Default::default() }, sum).unwrap();
        assert_eq!(keep.shape(), &[2, 1]);
        let seeded = reduce(&m.view(), 1, ReduceOptions { initial: Some(10), ..Default::default() }, sum).unwrap();
        assert_eq!(seeded.as_slice(), &[13, 22]);
        assert_eq!(reduce(&m.view(), 2, ReduceOptions::default(), sum).unwrap_err(), OpError::Shape(ShapeError::AxisOutOfBounds { axis: 2, ndim: 2 }));
    }

    #[test]
    fn reduce_on_empty_axes_needs_an_identity_or_initial() {
        let empty: NdArray<f64> = NdArray::zeros(&[0, 3]);
        let max = |x: f64, y: f64| if x >= y { x } else { y };
        assert_eq!(reduce(&empty.view(), 0, ReduceOptions::default(), max).unwrap_err(), OpError::Reduction(ReductionError::EmptyInput));
        let with_initial = reduce(&empty.view(), 0, ReduceOptions { initial: Some(-1.0), ..Default::default() }, max).unwrap();
        assert_eq!(with_initial.as_slice(), &[-1.0, -1.0, -1.0]);
        let sum = reduce(&empty.view(), 0, ReduceOptions { initial: Some(0.0), ..Default::default() }, |x, y| x + y).unwrap();
        assert_eq!(sum.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn reduce_with_where_mask_requires_initial_like_numpy() {
        let x = arr(vec![1i64, 2, 3, 4]);
        let mask = arr(vec![true, false, true, false]);
        let mask_view = mask.view();
        let opts = |initial| ReduceOptions { initial, where_: Some(&mask_view), keepdims: false };
        let masked = reduce(&x.view(), 0, opts(Some(0)), |a, b| a + b);
        assert_eq!(masked.unwrap().as_slice(), &[4]);
        assert_eq!(reduce(&x.view(), 0, opts(None), |a, b| a + b).unwrap_err(), OpError::Reduction(ReductionError::WhereNeedsInitial));

        let m = NdArray::from_vec((0..6).collect::<Vec<i64>>(), &[2, 3]).unwrap();
        let row_mask = NdArray::from_vec(vec![true, false, true], &[1, 3]).unwrap();
        let row_view = row_mask.view();
        let row_opts = ReduceOptions { initial: Some(0), where_: Some(&row_view), keepdims: false };
        assert_eq!(reduce(&m.view(), 1, row_opts, |a, b| a + b).unwrap().as_slice(), &[2, 8]);
    }

    #[test]
    fn accumulate_and_outer_match_numpy() {
        let m = NdArray::from_vec((0..6).collect::<Vec<i64>>(), &[2, 3]).unwrap();
        assert_eq!(accumulate(&m.view(), 1, |a, b| a + b).unwrap().as_slice(), &[0, 1, 3, 3, 7, 12]);
        assert_eq!(accumulate(&m.view(), 0, |a, b| a + b).unwrap().as_slice(), &[0, 1, 2, 3, 5, 7]);
        let p = arr(vec![1i64, 2, 3, 4]);
        assert_eq!(accumulate(&p.view(), 0, |a, b| a * b).unwrap().as_slice(), &[1, 2, 6, 24]);
        let q = arr(vec![1i64, 3, 2, 5, 4]);
        assert_eq!(accumulate(&q.view(), 0, |a, b| a.max(b)).unwrap().as_slice(), &[1, 3, 3, 5, 5]);

        let (u, v) = (arr(vec![0i64, 1, 2]), arr(vec![0i64, 1]));
        assert_eq!(outer_with(&u.view(), &v.view(), |a, b| a + b).unwrap().as_slice(), &[0, 1, 1, 2, 2, 3]);
        let flat = outer_with(&m.view(), &v.view(), |a, b| a * b).unwrap();
        assert_eq!(flat.shape(), &[2, 3, 2]);
        let mixed = outer_with(&arr(vec![1i32, 2]).astype::<f64>().view(), &arr(vec![0.5f64]).view(), |a, b| a * b).unwrap();
        assert_eq!(mixed.as_slice(), &[0.5, 1.0]);
    }
}
