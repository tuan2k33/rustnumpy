use crate::dispatch::{WrapAdd, WrapMul, WrapSub, zip_with_promoted, zip_with_promoted_parallel, Common, Out};
use crate::error::ShapeError;
use crate::ndarray::NdArray;
use crate::shape::{broadcast_shapes, IndexIter};
use crate::view::{ArrayView, ArrayViewMut};
use rayon::prelude::*;

const PARALLEL_CHUNK: usize = 4096;

pub fn zip_with<T: Copy>(
    a: &ArrayView<T>,
    b: &ArrayView<T>,
    f: impl Fn(T, T) -> T,
) -> Result<NdArray<T>, ShapeError> {
    let out_shape = broadcast_shapes(a.shape(), b.shape()).ok_or_else(|| ShapeError::NotBroadcastable {
        lhs: a.shape().to_vec(),
        rhs: b.shape().to_vec(),
    })?;
    let a_b = a.broadcast_to(&out_shape)?;
    let b_b = b.broadcast_to(&out_shape)?;

    let data: Vec<T> = a_b.iter().zip(b_b.iter()).map(|(x, y)| f(x, y)).collect();
    NdArray::from_vec(data, &out_shape)
}

pub fn zip_with_into<T: Copy>(
    out: &mut ArrayViewMut<T>,
    a: &ArrayView<T>,
    b: &ArrayView<T>,
    f: impl Fn(T, T) -> T,
) -> Result<(), ShapeError> {
    let out_shape = broadcast_shapes(a.shape(), b.shape()).ok_or_else(|| ShapeError::NotBroadcastable {
        lhs: a.shape().to_vec(),
        rhs: b.shape().to_vec(),
    })?;
    if out.shape() != out_shape.as_slice() {
        return Err(ShapeError::DataShapeMismatch {
            data_len: out.shape().iter().product(),
            shape: out_shape,
        });
    }
    let a_b = a.broadcast_to(&out_shape)?;
    let b_b = b.broadcast_to(&out_shape)?;

    for idx in IndexIter::new(&out_shape) {
        let value = f(a_b.get(&idx).unwrap(), b_b.get(&idx).unwrap());
        out.set(&idx, value)?;
    }
    Ok(())
}

pub fn map<T: Copy>(a: &ArrayView<T>, f: impl Fn(T) -> T) -> NdArray<T> {
    let data: Vec<T> = a.iter().map(f).collect();
    NdArray::from_vec(data, a.shape()).expect("data.len() always matches a.shape().iter().product()")
}

pub fn zip_with_parallel<T: Copy + Send + Sync>(
    a: &ArrayView<T>,
    b: &ArrayView<T>,
    f: impl Fn(T, T) -> T + Sync,
) -> Result<NdArray<T>, ShapeError> {
    let out_shape = broadcast_shapes(a.shape(), b.shape()).ok_or_else(|| ShapeError::NotBroadcastable {
        lhs: a.shape().to_vec(),
        rhs: b.shape().to_vec(),
    })?;
    let a_b = a.broadcast_to(&out_shape)?;
    let b_b = b.broadcast_to(&out_shape)?;
    let len: usize = out_shape.iter().product();

    let data: Vec<T> = (0..len.div_ceil(PARALLEL_CHUNK))
        .into_par_iter()
        .flat_map_iter(|chunk| {
            let (start, n) = (chunk * PARALLEL_CHUNK, PARALLEL_CHUNK);
            let f = &f;
            a_b.iter_range(start, n).zip(b_b.iter_range(start, n)).map(move |(x, y)| f(x, y))
        })
        .collect();
    NdArray::from_vec(data, &out_shape)
}

pub fn map_parallel<T: Copy + Send + Sync>(a: &ArrayView<T>, f: impl Fn(T) -> T + Sync) -> NdArray<T> {
    let shape = a.shape();
    let len: usize = shape.iter().product();
    let data: Vec<T> = (0..len.div_ceil(PARALLEL_CHUNK))
        .into_par_iter()
        .flat_map_iter(|chunk| {
            let f = &f;
            a.iter_range(chunk * PARALLEL_CHUNK, PARALLEL_CHUNK).map(f)
        })
        .collect();
    NdArray::from_vec(data, shape).expect("data.len() always matches shape.iter().product()")
}

pub fn add<A, B>(a: &ArrayView<A>, b: &ArrayView<B>) -> Result<NdArray<Out<A, B>>, ShapeError>
where
    A: Common<B>,
    B: Copy,
    Out<A, B>: WrapAdd,
{
    zip_with_promoted(a, b, |x, y| x.wrap_add(y))
}

pub fn sub<A, B>(a: &ArrayView<A>, b: &ArrayView<B>) -> Result<NdArray<Out<A, B>>, ShapeError>
where
    A: Common<B>,
    B: Copy,
    Out<A, B>: WrapSub,
{
    zip_with_promoted(a, b, |x, y| x.wrap_sub(y))
}

pub fn mul<A, B>(a: &ArrayView<A>, b: &ArrayView<B>) -> Result<NdArray<Out<A, B>>, ShapeError>
where
    A: Common<B>,
    B: Copy,
    Out<A, B>: WrapMul,
{
    zip_with_promoted(a, b, |x, y| x.wrap_mul(y))
}

pub fn subtract<A, B>(a: &ArrayView<A>, b: &ArrayView<B>) -> Result<NdArray<Out<A, B>>, ShapeError>
where
    A: Common<B>,
    B: Copy,
    Out<A, B>: WrapSub,
{
    sub(a, b)
}

pub fn multiply<A, B>(a: &ArrayView<A>, b: &ArrayView<B>) -> Result<NdArray<Out<A, B>>, ShapeError>
where
    A: Common<B>,
    B: Copy,
    Out<A, B>: WrapMul,
{
    mul(a, b)
}

pub fn add_parallel<A, B>(a: &ArrayView<A>, b: &ArrayView<B>) -> Result<NdArray<Out<A, B>>, ShapeError>
where
    A: Common<B> + Send + Sync,
    B: Copy + Send + Sync,
    Out<A, B>: WrapAdd + Send,
{
    zip_with_promoted_parallel(a, b, |x, y| x.wrap_add(y))
}

pub fn mul_parallel<A, B>(a: &ArrayView<A>, b: &ArrayView<B>) -> Result<NdArray<Out<A, B>>, ShapeError>
where
    A: Common<B> + Send + Sync,
    B: Copy + Send + Sync,
    Out<A, B>: WrapMul + Send,
{
    zip_with_promoted_parallel(a, b, |x, y| x.wrap_mul(y))
}

pub fn add_broadcast<A, B>(a: &ArrayView<A>, b: &ArrayView<B>) -> Result<NdArray<Out<A, B>>, ShapeError>
where
    A: Common<B>,
    B: Copy,
    Out<A, B>: WrapAdd,
{
    add(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray::NdArray;

    #[test]
    fn parallel_path_is_correct_across_chunk_boundaries_and_broadcasts() {
        let n = PARALLEL_CHUNK * 2 + 37;
        let big = NdArray::from_vec((0..n as i64).collect::<Vec<_>>(), &[n]).unwrap();
        let col = NdArray::from_vec(vec![1i64, 1000], &[2, 1]).unwrap();
        let par = zip_with_parallel(&col.view(), &big.view(), |x, y| x * y + 1).unwrap();
        let seq = zip_with(&col.view(), &big.view(), |x, y| x * y + 1).unwrap();
        assert_eq!(par.shape(), &[2, n]);
        assert_eq!(par, seq);
        assert_eq!(par.get(&[1, n - 1]), Some(1000 * (n as i64 - 1) + 1));
        let range = 3..n - 5;
        let strided = big.slice(std::slice::from_ref(&range)).unwrap();
        assert_eq!(map_parallel(&strided, |x| x + 1), map(&strided, |x| x + 1));
    }

    #[test]
    fn add_same_shape() {
        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = NdArray::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        let out = add(&a.view(), &b.view()).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.get(&[0, 0]), Some(11.0));
        assert_eq!(out.get(&[1, 1]), Some(44.0));
    }

    #[test]
    fn generic_ufunc_engine_works_on_non_f64_numeric_types() {

        let a = NdArray::from_vec(vec![1i32, 2, 3, 4], &[2, 2]).unwrap();
        let b = NdArray::from_vec(vec![10i32, 20, 30, 40], &[2, 2]).unwrap();
        assert_eq!(add(&a.view(), &b.view()).unwrap().as_slice(), &[11, 22, 33, 44]);
        assert_eq!(sub(&b.view(), &a.view()).unwrap().as_slice(), &[9, 18, 27, 36]);
        assert_eq!(mul(&a.view(), &b.view()).unwrap().as_slice(), &[10, 40, 90, 160]);

        let ua = NdArray::from_vec(vec![1u64, 2, 3], &[3]).unwrap();
        let ub = NdArray::from_vec(vec![100u64, 200, 300], &[3]).unwrap();
        assert_eq!(add(&ua.view(), &ub.view()).unwrap().as_slice(), &[101, 202, 303]);

        let ca = NdArray::from_vec(
            vec![crate::fft::Complex64::new(1.0, 1.0), crate::fft::Complex64::new(2.0, 0.0)],
            &[2],
        )
        .unwrap();
        let cb = NdArray::from_vec(
            vec![crate::fft::Complex64::new(0.0, 1.0), crate::fft::Complex64::new(1.0, 1.0)],
            &[2],
        )
        .unwrap();
        assert_eq!(
            add(&ca.view(), &cb.view()).unwrap().as_slice(),
            &[crate::fft::Complex64::new(1.0, 2.0), crate::fft::Complex64::new(3.0, 1.0)]
        );

        assert_eq!(add_parallel(&a.view(), &b.view()).unwrap(), add(&a.view(), &b.view()).unwrap());
    }

    #[test]
    fn add_broadcasts_row_vector_over_matrix() {

        let a = NdArray::from_vec(vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0], &[2, 3]).unwrap();
        let b = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let out = add(&a.view(), &b.view()).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.get(&[0, 0]), Some(1.0));
        assert_eq!(out.get(&[0, 2]), Some(3.0));
        assert_eq!(out.get(&[1, 0]), Some(11.0));
        assert_eq!(out.get(&[1, 2]), Some(13.0));
    }

    #[test]
    fn add_incompatible_shapes_errs() {
        let a: NdArray = NdArray::zeros(&[2, 3]);
        let b: NdArray = NdArray::zeros(&[4]);
        assert!(add(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn mul_broadcasts_column_vector_over_matrix() {

        let col = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
        let ones = NdArray::from_vec(vec![1.0; 12], &[3, 4]).unwrap();
        let out = mul(&col.view(), &ones.view()).unwrap();
        assert_eq!(out.shape(), &[3, 4]);
        for c in 0..4 {
            assert_eq!(out.get(&[0, c]), Some(1.0));
            assert_eq!(out.get(&[1, c]), Some(2.0));
            assert_eq!(out.get(&[2, c]), Some(3.0));
        }
    }

    #[test]
    fn sub_same_shape() {
        let a = NdArray::from_vec(vec![5.0, 5.0], &[2]).unwrap();
        let b = NdArray::from_vec(vec![2.0, 3.0], &[2]).unwrap();
        let out = sub(&a.view(), &b.view()).unwrap();
        assert_eq!(out.as_slice(), &[3.0, 2.0]);
    }

    #[test]
    fn subtract_and_multiply_are_the_array_api_standard_names_for_sub_and_mul() {
        let a = NdArray::from_vec(vec![5.0, 5.0], &[2]).unwrap();
        let b = NdArray::from_vec(vec![2.0, 3.0], &[2]).unwrap();
        assert_eq!(
            subtract(&a.view(), &b.view()).unwrap(),
            sub(&a.view(), &b.view()).unwrap()
        );
        assert_eq!(
            multiply(&a.view(), &b.view()).unwrap(),
            mul(&a.view(), &b.view()).unwrap()
        );
    }

    #[test]
    fn map_applies_a_unary_closure_elementwise() {
        let a = NdArray::from_vec(vec![1.0, 4.0, 9.0, 16.0], &[4]).unwrap();
        let roots = map(&a.view(), f64::sqrt);
        assert_eq!(roots.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn zip_with_into_writes_through_out_param() {

        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = NdArray::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        let mut out = NdArray::zeros(&[2, 2]);
        zip_with_into(&mut out.view_mut(), &a.view(), &b.view(), |x, y| x + y).unwrap();
        assert_eq!(out.as_slice(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn zip_with_into_rejects_wrong_output_shape() {
        let a: NdArray = NdArray::zeros(&[2, 2]);
        let b = NdArray::zeros(&[2, 2]);
        let mut out = NdArray::zeros(&[3, 3]);
        let err = zip_with_into(&mut out.view_mut(), &a.view(), &b.view(), |x, y| x + y).unwrap_err();
        assert!(matches!(err, ShapeError::DataShapeMismatch { .. }));
    }

    #[test]
    fn add_parallel_matches_sequential_add() {

        let data_a: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
        let data_b: Vec<f64> = (0..10_000).map(|i| (i as f64) * 0.5).collect();
        let a = NdArray::from_vec(data_a, &[100, 100]).unwrap();
        let b = NdArray::from_vec(data_b, &[100, 100]).unwrap();

        let sequential = add(&a.view(), &b.view()).unwrap();
        let parallel = add_parallel(&a.view(), &b.view()).unwrap();
        assert_eq!(sequential, parallel);
    }

    #[test]
    fn mul_parallel_matches_sequential_mul_with_broadcasting() {

        let col = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
        let ones = NdArray::from_vec(vec![1.0; 12], &[3, 4]).unwrap();
        let sequential = mul(&col.view(), &ones.view()).unwrap();
        let parallel = mul_parallel(&col.view(), &ones.view()).unwrap();
        assert_eq!(sequential, parallel);
    }

    #[test]
    fn map_parallel_matches_sequential_map() {
        let data: Vec<f64> = (0..5_000).map(|i| i as f64 * 0.1).collect();
        let a = NdArray::from_vec(data, &[5_000]).unwrap();
        let sequential = map(&a.view(), f64::sqrt);
        let parallel = map_parallel(&a.view(), f64::sqrt);
        assert_eq!(sequential, parallel);
    }

    #[test]
    fn zip_with_parallel_propagates_shape_errors() {
        let a: NdArray = NdArray::zeros(&[2, 3]);
        let b: NdArray = NdArray::zeros(&[4]);
        assert!(add_parallel(&a.view(), &b.view()).is_err());
    }
}
