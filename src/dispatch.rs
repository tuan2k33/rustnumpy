use crate::reductions::ReductionError;
use crate::dtype::DType;
use crate::error::{OpError, ShapeError};
use crate::ndarray::NdArray;
use crate::promote::{Promote, PromoteWeak, Widen};
use crate::shape::{broadcast_shapes, IndexIter};
use crate::view::{ArrayView, ArrayViewMut};
use num_complex::Complex;
use rayon::prelude::*;

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeakInt(pub i64);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeakFloat(pub f64);

pub trait FromWeak: Copy {
    fn from_weak_int(v: i64) -> Option<Self>;
    fn from_weak_float(v: f64) -> Self;
}

macro_rules! from_weak_int_type {
    ($($t:ty),*) => {$(
        impl FromWeak for $t {
            fn from_weak_int(v: i64) -> Option<Self> { <$t>::try_from(v).ok() }
            fn from_weak_float(v: f64) -> Self { v as $t }
        }
    )*};
}
from_weak_int_type!(i8, i16, i32, i64, u8, u16, u32, u64);

macro_rules! from_weak_float_type {
    ($($t:ty),*) => {$(
        impl FromWeak for $t {
            fn from_weak_int(v: i64) -> Option<Self> { Some(v as $t) }
            fn from_weak_float(v: f64) -> Self { v as $t }
        }
        impl FromWeak for Complex<$t> {
            fn from_weak_int(v: i64) -> Option<Self> { Some(Complex::new(v as $t, 0.0)) }
            fn from_weak_float(v: f64) -> Self { Complex::new(v as $t, 0.0) }
        }
    )*};
}
from_weak_float_type!(f32, f64);

impl FromWeak for half::f16 {
    fn from_weak_int(v: i64) -> Option<Self> {
        Some(half::f16::from_f64(v as f64))
    }
    fn from_weak_float(v: f64) -> Self {
        half::f16::from_f64(v)
    }
}

impl FromWeak for bool {
    fn from_weak_int(v: i64) -> Option<Self> {
        Some(v != 0)
    }
    fn from_weak_float(v: f64) -> Self {
        v != 0.0
    }
}

pub trait Common<B>: Copy {
    type Out: DType;
    fn lhs(self) -> Self::Out;
    fn rhs(b: B) -> Self::Out;
}

impl<A, B> Common<B> for A
where
    A: Promote<B> + Widen<<A as Promote<B>>::Output>,
    B: Widen<<A as Promote<B>>::Output>,
{
    type Out = <A as Promote<B>>::Output;

    fn lhs(self) -> Self::Out {
        self.widen()
    }

    fn rhs(b: B) -> Self::Out {
        b.widen()
    }
}

pub type Out<A, B> = <A as Common<B>>::Out;

fn out_shape<A, B>(a: &ArrayView<A>, b: &ArrayView<B>) -> Result<Vec<usize>, ShapeError> {
    broadcast_shapes(a.shape(), b.shape()).ok_or_else(|| ShapeError::NotBroadcastable {
        lhs: a.shape().to_vec(),
        rhs: b.shape().to_vec(),
    })
}

pub fn zip_with_promoted<A, B>(
    a: &ArrayView<A>,
    b: &ArrayView<B>,
    f: impl Fn(Out<A, B>, Out<A, B>) -> Out<A, B>,
) -> Result<NdArray<Out<A, B>>, ShapeError>
where
    A: Common<B>,
    B: Copy,
{
    let shape = out_shape(a, b)?;
    let (a_b, b_b) = (a.broadcast_to(&shape)?, b.broadcast_to(&shape)?);
    let data: Vec<Out<A, B>> = a_b
        .iter()
        .zip(b_b.iter())
        .map(|(x, y)| f(Common::lhs(x), A::rhs(y)))
        .collect();
    NdArray::from_vec(data, &shape)
}

const PARALLEL_CHUNK: usize = 4096;

pub fn zip_with_promoted_parallel<A, B>(
    a: &ArrayView<A>,
    b: &ArrayView<B>,
    f: impl Fn(Out<A, B>, Out<A, B>) -> Out<A, B> + Sync,
) -> Result<NdArray<Out<A, B>>, ShapeError>
where
    A: Common<B> + Send + Sync,
    B: Copy + Send + Sync,
    Out<A, B>: Send,
{
    let shape = out_shape(a, b)?;
    let (a_b, b_b) = (a.broadcast_to(&shape)?, b.broadcast_to(&shape)?);
    let len: usize = shape.iter().product();
    let data: Vec<Out<A, B>> = (0..len.div_ceil(PARALLEL_CHUNK))
        .into_par_iter()
        .flat_map_iter(|chunk| {
            let (start, n) = (chunk * PARALLEL_CHUNK, PARALLEL_CHUNK);
            let f = &f;
            a_b.iter_range(start, n)
                .zip(b_b.iter_range(start, n))
                .map(move |(x, y)| f(Common::lhs(x), A::rhs(y)))
        })
        .collect();
    NdArray::from_vec(data, &shape)
}

pub fn map_weak_int<A>(
    a: &ArrayView<A>,
    s: i64,
    f: impl Fn(<A as PromoteWeak<WeakInt>>::Output, <A as PromoteWeak<WeakInt>>::Output) -> <A as PromoteWeak<WeakInt>>::Output,
) -> Result<NdArray<<A as PromoteWeak<WeakInt>>::Output>, OpError>
where
    A: PromoteWeak<WeakInt> + Widen<<A as PromoteWeak<WeakInt>>::Output>,
    <A as PromoteWeak<WeakInt>>::Output: FromWeak,
{
    type O<A> = <A as PromoteWeak<WeakInt>>::Output;
    let scalar = <O<A> as FromWeak>::from_weak_int(s)
        .ok_or(OpError::WeakScalarOverflow { value: s, dtype: <O<A> as DType>::type_name() })?;
    let data: Vec<O<A>> = a.iter().map(|x| f(x.widen(), scalar)).collect();
    NdArray::from_vec(data, a.shape()).map_err(OpError::from)
}

pub fn map_weak_float<A>(
    a: &ArrayView<A>,
    s: f64,
    f: impl Fn(<A as PromoteWeak<WeakFloat>>::Output, <A as PromoteWeak<WeakFloat>>::Output) -> <A as PromoteWeak<WeakFloat>>::Output,
) -> Result<NdArray<<A as PromoteWeak<WeakFloat>>::Output>, ShapeError>
where
    A: PromoteWeak<WeakFloat> + Widen<<A as PromoteWeak<WeakFloat>>::Output>,
    <A as PromoteWeak<WeakFloat>>::Output: FromWeak,
{
    type O<A> = <A as PromoteWeak<WeakFloat>>::Output;
    let scalar = <O<A> as FromWeak>::from_weak_float(s);
    let data: Vec<O<A>> = a.iter().map(|x| f(x.widen(), scalar)).collect();
    NdArray::from_vec(data, a.shape())
}

macro_rules! weak_ops {
    ($($int:ident, $float:ident => $op:ident, $bound:ident);* $(;)?) => {$(
        pub fn $int<A>(a: &ArrayView<A>, s: i64) -> Result<NdArray<<A as PromoteWeak<WeakInt>>::Output>, OpError>
        where
            A: PromoteWeak<WeakInt> + Widen<<A as PromoteWeak<WeakInt>>::Output>,
            <A as PromoteWeak<WeakInt>>::Output: FromWeak + $bound,
        {
            map_weak_int(a, s, |x, y| x.$op(y))
        }

        pub fn $float<A>(a: &ArrayView<A>, s: f64) -> Result<NdArray<<A as PromoteWeak<WeakFloat>>::Output>, ShapeError>
        where
            A: PromoteWeak<WeakFloat> + Widen<<A as PromoteWeak<WeakFloat>>::Output>,
            <A as PromoteWeak<WeakFloat>>::Output: FromWeak + $bound,
        {
            map_weak_float(a, s, |x, y| x.$op(y))
        }
    )*};
}

weak_ops! {
    add_weak_int, add_weak_float => wrap_add, WrapAdd;
    sub_weak_int, sub_weak_float => wrap_sub, WrapSub;
    mul_weak_int, mul_weak_float => wrap_mul, WrapMul;
}

pub fn zip_assign<A, B>(
    out: &mut ArrayViewMut<A>,
    b: &ArrayView<B>,
    mask: Option<&ArrayView<bool>>,
    f: impl Fn(Out<A, B>, Out<A, B>) -> Out<A, B>,
) -> Result<(), ShapeError>
where
    A: Common<B>,
    B: Copy,
    Out<A, B>: Widen<A>,
{
    let shape = out.shape().to_vec();
    let b_b = b.broadcast_to(&shape)?;
    let mask_b = mask.map(|m| m.broadcast_to(&shape)).transpose()?;
    for idx in IndexIter::new(&shape) {
        if mask_b.as_ref().is_some_and(|m| !m.get(&idx).unwrap()) {
            continue;
        }
        let current = out.get(&idx).unwrap();
        let value = f(Common::lhs(current), A::rhs(b_b.get(&idx).unwrap()));
        out.set(&idx, value.widen())?;
    }
    Ok(())
}

pub fn add_assign<A, B>(out: &mut ArrayViewMut<A>, b: &ArrayView<B>) -> Result<(), ShapeError>
where
    A: Common<B>,
    B: Copy,
    Out<A, B>: Widen<A> + WrapAdd,
{
    zip_assign(out, b, None, |x, y| x.wrap_add(y))
}

pub fn sub_assign<A, B>(out: &mut ArrayViewMut<A>, b: &ArrayView<B>) -> Result<(), ShapeError>
where
    A: Common<B>,
    B: Copy,
    Out<A, B>: Widen<A> + WrapSub,
{
    zip_assign(out, b, None, |x, y| x.wrap_sub(y))
}

pub fn mul_assign<A, B>(out: &mut ArrayViewMut<A>, b: &ArrayView<B>) -> Result<(), ShapeError>
where
    A: Common<B>,
    B: Copy,
    Out<A, B>: Widen<A> + WrapMul,
{
    zip_assign(out, b, None, |x, y| x.wrap_mul(y))
}

pub fn zip_promoted_into<A, B, C>(
    out: &mut ArrayViewMut<C>,
    a: &ArrayView<A>,
    b: &ArrayView<B>,
    mask: Option<&ArrayView<bool>>,
    f: impl Fn(Out<A, B>, Out<A, B>) -> Out<A, B>,
) -> Result<(), ShapeError>
where
    A: Common<B>,
    B: Copy,
    Out<A, B>: Widen<C>,
{
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
        let value = f(Common::lhs(a_b.get(&idx).unwrap()), A::rhs(b_b.get(&idx).unwrap()));
        out.set(&idx, value.widen())?;
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

pub fn outer_with<A, B>(
    a: &ArrayView<A>,
    b: &ArrayView<B>,
    f: impl Fn(Out<A, B>, Out<A, B>) -> Out<A, B>,
) -> Result<NdArray<Out<A, B>>, ShapeError>
where
    A: Common<B>,
    B: Copy,
{
    let bv: Vec<B> = b.iter().collect();
    let mut data = Vec::with_capacity(a.len() * bv.len());
    for x in a.iter() {
        data.extend(bv.iter().map(|&y| f(Common::lhs(x), A::rhs(y))));
    }
    let shape: Vec<usize> = a.shape().iter().chain(b.shape()).copied().collect();
    NdArray::from_vec(data, &shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::{common_dtype_of, Kind};
    use crate::ufunc::{add, mul, sub};

    fn arr<T>(data: Vec<T>) -> NdArray<T> {
        let n = data.len();
        NdArray::from_vec(data, &[n]).unwrap()
    }

    fn kind_of_out<A: Common<B>, B>() -> Kind {
        <Out<A, B> as DType>::KIND
    }

    macro_rules! check_pairs {
        ($($a:ty),* ; $bs:tt) => {
            $( check_pairs!(@row $a; $bs); )*
        };
        (@row $a:ty; ($($b:ty),*)) => {
            $( assert_eq!(kind_of_out::<$a, $b>(), common_dtype_of::<$a, $b>(), "{} + {}", <$a as DType>::type_name(), <$b as DType>::type_name()); )*
        };
    }

    #[test]
    fn generated_promotion_table_agrees_with_dtype_rs_for_every_pair() {
        use num_complex::Complex;
        check_pairs!(
            bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, Complex<f32>, Complex<f64>;
            (bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, Complex<f32>, Complex<f64>)
        );
    }

    #[test]
    fn mixed_dtype_add_promotes_like_numpy() {
        let i = arr(vec![1i32, 2, 3]);
        let f = arr(vec![0.5f64, 0.5, 0.5]);
        let sum = add(&i.view(), &f.view()).unwrap();
        assert_eq!(sum.as_slice(), &[1.5, 2.5, 3.5]);
        let f32s = arr(vec![1.5f32, 2.5]);
        let i16s = arr(vec![1i16, 2]);
        let s: NdArray<f32> = add(&i16s.view(), &f32s.view()).unwrap();
        assert_eq!(s.as_slice(), &[2.5f32, 4.5]);
        let i32s = arr(vec![1i32, 2]);
        let widened: NdArray<f64> = mul(&i32s.view(), &f32s.view()).unwrap();
        assert_eq!(widened.as_slice(), &[1.5, 5.0]);
        let u = arr(vec![200u8]);
        let s8 = arr(vec![-100i8]);
        let mixed: NdArray<i16> = sub(&u.view(), &s8.view()).unwrap();
        assert_eq!(mixed.as_slice(), &[300]);
        let u64s = arr(vec![1u64]);
        let i64s = arr(vec![-1i64]);
        let to_float: NdArray<f64> = add(&u64s.view(), &i64s.view()).unwrap();
        assert_eq!(to_float.as_slice(), &[0.0]);
        let b = arr(vec![true, false]);
        let bi = arr(vec![10i8, 20]);
        let out: NdArray<i8> = add(&b.view(), &bi.view()).unwrap();
        assert_eq!(out.as_slice(), &[11, 20]);
        let c = arr(vec![Complex::new(1.0f32, 1.0)]);
        let d = arr(vec![2.0f64]);
        let cd: NdArray<Complex<f64>> = add(&c.view(), &d.view()).unwrap();
        assert_eq!(cd.as_slice(), &[Complex::new(3.0, 1.0)]);
    }

    #[test]
    fn int_overflow_wraps_in_the_promoted_type() {
        let a = arr(vec![100i8]);
        let b = arr(vec![100i8]);
        assert_eq!(add(&a.view(), &b.view()).unwrap().as_slice(), &[-56]);
        let wide = arr(vec![100i16]);
        assert_eq!(add(&a.view(), &wide.view()).unwrap().as_slice(), &[200i16]);
    }

    #[test]
    fn weak_python_scalars_follow_nep_50() {
        let i8s = arr(vec![1i8]);
        let r: NdArray<i8> = add_weak_int(&i8s.view(), 3).unwrap();
        assert_eq!(r.as_slice(), &[4]);
        assert_eq!(
            add_weak_int(&i8s.view(), 300).unwrap_err(),
            OpError::WeakScalarOverflow { value: 300, dtype: "int8" }
        );
        let u8s = arr(vec![1u8]);
        assert_eq!(
            add_weak_int(&u8s.view(), -1).unwrap_err(),
            OpError::WeakScalarOverflow { value: -1, dtype: "uint8" }
        );
        assert_eq!(add_weak_int(&u8s.view(), 255).unwrap().as_slice(), &[0u8]);
        assert!(add_weak_int(&u8s.view(), 256).is_err());

        let f: NdArray<f64> = add_weak_float(&i8s.view(), 3.0).unwrap();
        assert_eq!(f.as_slice(), &[4.0]);
        let f32s = arr(vec![1.0f32]);
        let kept: NdArray<f32> = add_weak_float(&f32s.view(), 3.0).unwrap();
        assert_eq!(kept.as_slice(), &[4.0f32]);
        let kept_int: NdArray<f32> = add_weak_int(&f32s.view(), 3).unwrap();
        assert_eq!(kept_int.as_slice(), &[4.0f32]);
        assert_eq!(add_weak_float(&f32s.view(), 1e300).unwrap().as_slice(), &[f32::INFINITY]);
        let bools = arr(vec![true]);
        let promoted: NdArray<i64> = add_weak_int(&bools.view(), 3).unwrap();
        assert_eq!(promoted.as_slice(), &[4]);
        let bf: NdArray<f64> = add_weak_float(&bools.view(), 3.0).unwrap();
        assert_eq!(bf.as_slice(), &[4.0]);
        let c64 = arr(vec![Complex::new(1.0f32, 0.0)]);
        let cs: NdArray<Complex<f32>> = add_weak_float(&c64.view(), 3.0).unwrap();
        assert_eq!(cs.as_slice(), &[Complex::new(4.0, 0.0)]);
        let i32s = arr(vec![10i32]);
        assert_eq!(sub_weak_int(&i32s.view(), 3).unwrap().as_slice(), &[7]);
        assert_eq!(mul_weak_float(&i32s.view(), 0.5).unwrap().as_slice(), &[5.0]);
    }

    #[test]
    fn integer_arithmetic_wraps_in_every_entry_point_like_numpy() {
        let a = arr(vec![i8::MAX]);
        assert_eq!(crate::reductions::sum(&arr(vec![i8::MAX, 1]).view()), i8::MIN);
        assert_eq!(crate::contraction::dot(&a.view(), &arr(vec![2i8]).view()).unwrap().as_slice(), &[-2]);
        let mut m = arr(vec![i8::MAX]);
        add_assign(&mut m.view_mut(), &arr(vec![1i8]).view()).unwrap();
        assert_eq!(m.as_slice(), &[i8::MIN]);
        assert_eq!(add_weak_int(&a.view(), 1).unwrap().as_slice(), &[i8::MIN]);
    }

    #[test]
    fn in_place_ops_follow_same_kind_casting() {
        let mut a = arr(vec![1i32, 2, 3]);
        let wide = arr(vec![1i64, 1, 1]);
        add_assign(&mut a.view_mut(), &wide.view()).unwrap();
        assert_eq!(a.as_slice(), &[2, 3, 4]);
        let narrow = arr(vec![1i8, 1, 1]);
        mul_assign(&mut a.view_mut(), &narrow.view()).unwrap();
        assert_eq!(a.as_slice(), &[2, 3, 4]);
        let mut f = arr(vec![1.0f32, 2.0]);
        add_assign(&mut f.view_mut(), &arr(vec![0.5f64]).view()).unwrap();
        assert_eq!(f.as_slice(), &[1.5f32, 2.5]);
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
        zip_promoted_into(&mut out.view_mut(), &x.view(), &y.view(), Some(&mask.view()), |a, b| a + b).unwrap();
        assert_eq!(out.as_slice(), &[11, -1, 33, -1]);

        let mut z = arr(vec![1i64, 2, 3, 4]);
        zip_assign(&mut z.view_mut(), &y.view(), Some(&mask.view()), |a, b| a + b).unwrap();
        assert_eq!(z.as_slice(), &[11, 2, 33, 4]);

        let mut small = arr(vec![0i64; 3]);
        assert!(zip_promoted_into(&mut small.view_mut(), &x.view(), &y.view(), None, |a, b| a + b).is_err());

        let mut narrow_out = arr(vec![0i16; 4]);
        zip_promoted_into(&mut narrow_out.view_mut(), &arr(vec![1i8, 2, 3, 4]).view(), &arr(vec![1i8; 4]).view(), None, |a, b| a + b)
            .unwrap();
        assert_eq!(narrow_out.as_slice(), &[2, 3, 4, 5]);
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
        let mixed: NdArray<f64> = outer_with(&arr(vec![1i32, 2]).view(), &arr(vec![0.5f64]).view(), |a, b| a * b).unwrap();
        assert_eq!(mixed.as_slice(), &[0.5, 1.0]);
    }
}
