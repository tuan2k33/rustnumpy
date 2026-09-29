use crate::dynarray::Arr;
use num_complex::Complex;
use num_traits::Float;
use rustnumpy::logic::map_to;
use rustnumpy::reductions::FloatIsh;
use rustnumpy::NdArray;
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug)]
pub struct Cx<T>(pub Complex<T>);

fn lt<T: Float>(a: Complex<T>, b: Complex<T>) -> bool {
    let (an, bn) = (a.im.is_nan(), b.im.is_nan());
    if a.re < b.re {
        !an || bn
    } else if a.re > b.re {
        bn && !an
    } else if a.re == b.re || (a.re.is_nan() && b.re.is_nan()) {
        a.im < b.im || (bn && !an)
    } else {
        b.re.is_nan()
    }
}

impl<T: Float> PartialEq for Cx<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.re == other.0.re && self.0.im == other.0.im
    }
}

impl<T: Float> PartialOrd for Cx<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if lt(self.0, other.0) {
            Some(Ordering::Less)
        } else if lt(other.0, self.0) {
            Some(Ordering::Greater)
        } else {
            Some(Ordering::Equal)
        }
    }
}

impl<T: Float> FloatIsh for Cx<T> {
    fn is_nan_ish(self) -> bool {
        self.0.re.is_nan() || self.0.im.is_nan()
    }
}

pub fn to_keys<T: Float>(x: &NdArray<Complex<T>>) -> NdArray<Cx<T>> {
    map_to(&x.view(), Cx)
}

pub fn from_keys<T: Float>(x: &NdArray<Cx<T>>) -> NdArray<Complex<T>> {
    map_to(&x.view(), |k| k.0)
}

pub fn from_key_result<T: Float>(x: NdArray<Cx<T>>) -> Arr
where
    Arr: From<NdArray<Complex<T>>>,
{
    Arr::from(from_keys(&x))
}
