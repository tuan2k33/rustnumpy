use faer::linalg::solvers::{DenseSolveCore, Solve};
use faer::{Mat, MatRef, Side};

use crate::ndarray::NdArray;

#[derive(Debug, Clone, PartialEq)]
pub enum LinalgError {

    NotSquare { shape: Vec<usize> },

    Not2D { shape: Vec<usize> },

    Not1D { shape: Vec<usize> },

    ShapeMismatch { lhs: Vec<usize>, rhs: Vec<usize> },

    Singular,

    NotPositiveDefinite,

    EigenFailed,

    SvdFailed,
}

fn to_mat(a: &NdArray) -> Result<Mat<f64>, LinalgError> {
    if a.ndim() != 2 {
        return Err(LinalgError::Not2D { shape: a.shape().to_vec() });
    }
    let (nrows, ncols) = (a.shape()[0], a.shape()[1]);
    Ok(MatRef::from_row_major_slice(a.as_slice(), nrows, ncols).to_owned())
}

fn to_square_mat(a: &NdArray) -> Result<Mat<f64>, LinalgError> {
    let m = to_mat(a)?;
    if m.nrows() != m.ncols() {
        return Err(LinalgError::NotSquare { shape: a.shape().to_vec() });
    }
    Ok(m)
}

fn from_mat(m: MatRef<f64>) -> NdArray {
    let (nrows, ncols) = (m.nrows(), m.ncols());
    let mut data = Vec::with_capacity(nrows * ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            data.push(m[(i, j)]);
        }
    }
    NdArray::from_vec(data, &[nrows, ncols]).expect("data.len() == nrows * ncols by construction")
}

fn vec_to_col_mat(v: &[f64]) -> Mat<f64> {
    Mat::from_fn(v.len(), 1, |i, _| v[i])
}

pub fn solve(a: &NdArray, b: &NdArray) -> Result<NdArray, LinalgError> {
    let a_mat = to_square_mat(a)?;
    let n = a_mat.nrows();
    let lu = a_mat.as_ref().partial_piv_lu();

    if b.ndim() == 1 {
        if b.shape()[0] != n {
            return Err(LinalgError::ShapeMismatch { lhs: a.shape().to_vec(), rhs: b.shape().to_vec() });
        }
        let rhs = vec_to_col_mat(b.as_slice());
        let x = lu.solve(&rhs);
        Ok(NdArray::from_vec((0..n).map(|i| x[(i, 0)]).collect(), &[n])
            .expect("solve output has exactly n rows"))
    } else {
        let b_mat = to_mat(b)?;
        if b_mat.nrows() != n {
            return Err(LinalgError::ShapeMismatch { lhs: a.shape().to_vec(), rhs: b.shape().to_vec() });
        }
        let x = lu.solve(&b_mat);
        Ok(from_mat(x.as_ref()))
    }
}

pub fn inv(a: &NdArray) -> Result<NdArray, LinalgError> {
    let a_mat = to_square_mat(a)?;
    if det(a)?.abs() < 1e-300 {
        return Err(LinalgError::Singular);
    }
    let lu = a_mat.as_ref().partial_piv_lu();
    Ok(from_mat(lu.inverse().as_ref()))
}

pub fn det(a: &NdArray) -> Result<f64, LinalgError> {
    let a_mat = to_square_mat(a)?;
    Ok(a_mat.as_ref().determinant())
}

pub fn qr(a: &NdArray) -> Result<(NdArray, NdArray), LinalgError> {
    let a_mat = to_mat(a)?;
    let qr = a_mat.as_ref().qr();
    let q = qr.compute_thin_Q();
    let r = qr.thin_R();
    Ok((from_mat(q.as_ref()), from_mat(r)))
}

pub fn cholesky(a: &NdArray) -> Result<NdArray, LinalgError> {
    let a_mat = to_square_mat(a)?;
    let llt = a_mat.as_ref().llt(Side::Lower).map_err(|_| LinalgError::NotPositiveDefinite)?;
    Ok(from_mat(llt.L()))
}

pub fn eigh(a: &NdArray) -> Result<(Vec<f64>, NdArray), LinalgError> {
    let a_mat = to_square_mat(a)?;
    let eig = a_mat.as_ref().self_adjoint_eigen(Side::Lower).map_err(|_| LinalgError::EigenFailed)?;
    let values: Vec<f64> = (0..a_mat.nrows()).map(|i| eig.S()[i]).collect();
    Ok((values, from_mat(eig.U())))
}

pub fn eigvalsh(a: &NdArray) -> Result<Vec<f64>, LinalgError> {
    Ok(eigh(a)?.0)
}

pub fn eigvals(a: &NdArray) -> Result<Vec<(f64, f64)>, LinalgError> {
    let a_mat = to_square_mat(a)?;
    let eig = a_mat.as_ref().eigenvalues().map_err(|_| LinalgError::EigenFailed)?;
    Ok(eig.into_iter().map(|c| (c.re, c.im)).collect())
}

pub fn svd(a: &NdArray) -> Result<(NdArray, Vec<f64>, NdArray), LinalgError> {
    let a_mat = to_mat(a)?;
    let svd = a_mat.as_ref().thin_svd().map_err(|_| LinalgError::SvdFailed)?;
    let k = a_mat.nrows().min(a_mat.ncols());
    let values: Vec<f64> = (0..k).map(|i| svd.S()[i]).collect();
    let vt = svd.V().transpose();
    Ok((from_mat(svd.U()), values, from_mat(vt)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecNormOrd {

    One,

    Two,

    Inf,
}

pub fn vector_norm(a: &[f64], ord: VecNormOrd) -> f64 {
    match ord {
        VecNormOrd::One => a.iter().map(|x| x.abs()).sum(),
        VecNormOrd::Two => a.iter().map(|x| x * x).sum::<f64>().sqrt(),
        VecNormOrd::Inf => a.iter().fold(0.0_f64, |acc, x| acc.max(x.abs())),
    }
}

pub fn frobenius_norm(a: &NdArray) -> f64 {
    a.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt()
}

pub fn matrix_norm(a: &NdArray) -> f64 {
    frobenius_norm(a)
}

pub fn matrix_power(a: &NdArray, n: i32) -> Result<NdArray, LinalgError> {
    let base_mat = to_square_mat(a)?;
    let size = base_mat.nrows();

    let base = if n < 0 { to_square_mat(&inv(a)?)? } else { base_mat };
    let mut exp = n.unsigned_abs();

    let mut result = Mat::<f64>::identity(size, size);
    let mut power = base;
    while exp > 0 {
        if exp & 1 == 1 {
            result = &result * &power;
        }
        power = &power * &power;
        exp >>= 1;
    }
    Ok(from_mat(result.as_ref()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::assert_allclose_default;

    fn arr(data: Vec<f64>, shape: &[usize]) -> NdArray {
        NdArray::from_vec(data, shape).unwrap()
    }

    fn sample_a() -> NdArray {
        arr(vec![4.0, 3.0, 6.0, 3.0], &[2, 2])
    }

    #[test]
    fn solve_matches_numpy() {
        let a = sample_a();
        let b = arr(vec![1.0, 2.0], &[2]);
        let x = solve(&a, &b).unwrap();
        assert_allclose_default(&x.view(), &arr(vec![0.5, -1.0 / 3.0], &[2]).view()).unwrap();
    }

    #[test]
    fn inv_matches_numpy() {
        let a = sample_a();
        let inv_a = inv(&a).unwrap();
        let expected = arr(vec![-0.5, 0.5, 1.0, -2.0 / 3.0], &[2, 2]);
        assert_allclose_default(&inv_a.view(), &expected.view()).unwrap();
    }

    #[test]
    fn det_matches_numpy() {
        assert!((det(&sample_a()).unwrap() - (-6.0)).abs() < 1e-9);
    }

    #[test]
    fn det_rejects_non_square() {
        let a = arr(vec![1.0, 2.0, 3.0], &[1, 3]);
        assert_eq!(det(&a).unwrap_err(), LinalgError::NotSquare { shape: vec![1, 3] });
    }

    #[test]
    fn qr_reconstructs_a() {

        let a = sample_a();
        let (q, r) = qr(&a).unwrap();
        let q_mat = to_mat(&q).unwrap();
        let r_mat = to_mat(&r).unwrap();
        let product = &q_mat * &r_mat;
        assert_allclose_default(&from_mat(product.as_ref()).view(), &a.view()).unwrap();
    }

    #[test]
    fn cholesky_matches_numpy() {

        let s = arr(vec![4.0, 2.0, 2.0, 3.0], &[2, 2]);
        let l = cholesky(&s).unwrap();
        let expected = arr(vec![2.0, 0.0, 1.0, 2.0_f64.sqrt()], &[2, 2]);
        assert_allclose_default(&l.view(), &expected.view()).unwrap();
    }

    #[test]
    fn cholesky_rejects_non_positive_definite() {
        let a = sample_a();
        assert_eq!(cholesky(&a).unwrap_err(), LinalgError::NotPositiveDefinite);
    }

    #[test]
    fn eigh_matches_numpy() {
        let s = arr(vec![4.0, 2.0, 2.0, 3.0], &[2, 2]);
        let (values, _vectors) = eigh(&s).unwrap();

        assert!((values[0] - 1.438_447_19).abs() < 1e-6);
        assert!((values[1] - 5.561_552_81).abs() < 1e-6);
    }

    #[test]
    fn eigvals_matches_numpy_for_rotation_matrix() {

        let a = arr(vec![0.0, -1.0, 1.0, 0.0], &[2, 2]);
        let mut vals = eigvals(&a).unwrap();
        vals.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
        assert!((vals[0].0 - 0.0).abs() < 1e-9 && (vals[0].1 - (-1.0)).abs() < 1e-9);
        assert!((vals[1].0 - 0.0).abs() < 1e-9 && (vals[1].1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn svd_singular_values_match_numpy() {
        let a = sample_a();
        let (_u, s, _vt) = svd(&a).unwrap();

        assert!((s[0] - 8.335_579_12).abs() < 1e-6);
        assert!((s[1] - 0.719_806_02).abs() < 1e-6);
    }

    #[test]
    fn svd_reconstructs_a() {
        let a = sample_a();
        let (u, s, vt) = svd(&a).unwrap();
        let u_mat = to_mat(&u).unwrap();
        let vt_mat = to_mat(&vt).unwrap();
        let s_mat = Mat::from_fn(2, 2, |i, j| if i == j { s[i] } else { 0.0 });
        let product = &(&u_mat * &s_mat) * &vt_mat;
        assert_allclose_default(&from_mat(product.as_ref()).view(), &a.view()).unwrap();
    }

    #[test]
    fn vector_norm_matches_numpy() {
        let v = [3.0, -4.0];
        assert!((vector_norm(&v, VecNormOrd::Two) - 5.0).abs() < 1e-12);
        assert!((vector_norm(&v, VecNormOrd::One) - 7.0).abs() < 1e-12);
        assert!((vector_norm(&v, VecNormOrd::Inf) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn frobenius_norm_matches_numpy() {

        assert!((frobenius_norm(&sample_a()) - 8.366_600_265_340_756).abs() < 1e-9);
    }

    #[test]
    fn matrix_norm_is_the_array_api_standard_name_for_frobenius_norm() {
        assert_eq!(matrix_norm(&sample_a()), frobenius_norm(&sample_a()));
    }

    #[test]
    fn matrix_power_matches_numpy() {
        let a = sample_a();
        let cubed = matrix_power(&a, 3).unwrap();
        assert_allclose_default(&cubed.view(), &arr(vec![262.0, 165.0, 330.0, 207.0], &[2, 2]).view())
            .unwrap();

        let zeroth = matrix_power(&a, 0).unwrap();
        assert_allclose_default(&zeroth.view(), &arr(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).view()).unwrap();

        let inverse = matrix_power(&a, -1).unwrap();
        assert_allclose_default(&inverse.view(), &inv(&a).unwrap().view()).unwrap();
    }
}
