use faer::linalg::solvers::{DenseSolveCore, Solve};
use faer::{Mat, MatRef, Side};

use crate::fft::Complex64;
use crate::ndarray::NdArray;

pub use crate::contraction::{cross, kron, matmul, outer, trace};
pub use crate::gufunc::vecdot;

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

    Empty,

    RankMismatch { a_rows: usize, b_rows: usize },
}

impl std::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinalgError::NotSquare { shape } => write!(f, "expected a square matrix, got shape {shape:?}"),
            LinalgError::Not2D { shape } => write!(f, "expected a 2-D array, got shape {shape:?}"),
            LinalgError::Not1D { shape } => write!(f, "expected a 1-D array, got shape {shape:?}"),
            LinalgError::ShapeMismatch { lhs, rhs } => {
                write!(f, "shapes {lhs:?} and {rhs:?} are not compatible for this operation")
            }
            LinalgError::Singular => write!(f, "matrix is singular"),
            LinalgError::NotPositiveDefinite => write!(f, "matrix is not positive definite"),
            LinalgError::EigenFailed => write!(f, "eigenvalue decomposition failed to converge"),
            LinalgError::SvdFailed => write!(f, "singular value decomposition failed to converge"),
            LinalgError::Empty => write!(f, "operation is not defined on empty arrays"),
            LinalgError::RankMismatch { a_rows, b_rows } => {
                write!(f, "incompatible dimensions: a has {a_rows} rows but b has {b_rows}")
            }
        }
    }
}

impl std::error::Error for LinalgError {}

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
    if a.is_empty() {
        return Err(LinalgError::Empty);
    }
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

fn lu_diagonal(m: &Mat<f64>) -> Option<(f64, Vec<f64>)> {
    let n = m.nrows();
    let mut lu: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| m[(i, j)]).collect()).collect();
    let mut sign = 1.0;
    let mut diag = Vec::with_capacity(n);
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| lu[i][col].abs().total_cmp(&lu[j][col].abs())).unwrap();
        if lu[pivot][col] == 0.0 || lu[pivot][col].is_nan() {
            return None;
        }
        if pivot != col {
            lu.swap(pivot, col);
            sign = -sign;
        }
        let d = lu[col][col];
        diag.push(d);
        let pivot_row = lu[col].clone();
        for row in lu[col + 1..].iter_mut() {
            let f = row[col] / d;
            for (dst, &v) in row[col..].iter_mut().zip(&pivot_row[col..]) {
                *dst -= f * v;
            }
        }
    }
    Some((sign, diag))
}

pub fn inv(a: &NdArray) -> Result<NdArray, LinalgError> {
    let a_mat = to_square_mat(a)?;
    if a_mat.nrows() == 0 {
        return Ok(NdArray::zeros(&[0, 0]));
    }
    if lu_diagonal(&a_mat).is_none() {
        return Err(LinalgError::Singular);
    }
    let lu = a_mat.as_ref().partial_piv_lu();
    Ok(from_mat(lu.inverse().as_ref()))
}

pub fn det(a: &NdArray) -> Result<f64, LinalgError> {
    let a_mat = to_square_mat(a)?;
    if a_mat.nrows() == 0 {
        return Ok(1.0);
    }
    Ok(match lu_diagonal(&a_mat) {
        None => 0.0,
        Some((sign, diag)) => diag.iter().fold(sign, |acc, d| acc * d),
    })
}

pub fn qr(a: &NdArray) -> Result<(NdArray, NdArray), LinalgError> {
    if a.is_empty() {
        return Err(LinalgError::Empty);
    }
    let a_mat = to_mat(a)?;
    let qr = a_mat.as_ref().qr();
    let q = qr.compute_thin_Q();
    let r = qr.thin_R();
    Ok((from_mat(q.as_ref()), from_mat(r)))
}

pub fn cholesky(a: &NdArray) -> Result<NdArray, LinalgError> {
    if a.is_empty() {
        return Err(LinalgError::Empty);
    }
    let a_mat = to_square_mat(a)?;
    let llt = a_mat.as_ref().llt(Side::Lower).map_err(|_| LinalgError::NotPositiveDefinite)?;
    Ok(from_mat(llt.L()))
}

pub fn eigh(a: &NdArray) -> Result<(Vec<f64>, NdArray), LinalgError> {
    if a.is_empty() {
        return Err(LinalgError::Empty);
    }
    let a_mat = to_square_mat(a)?;
    let eig = a_mat.as_ref().self_adjoint_eigen(Side::Lower).map_err(|_| LinalgError::EigenFailed)?;
    let values: Vec<f64> = (0..a_mat.nrows()).map(|i| eig.S()[i]).collect();
    Ok((values, from_mat(eig.U())))
}

pub fn eigvalsh(a: &NdArray) -> Result<Vec<f64>, LinalgError> {
    Ok(eigh(a)?.0)
}

pub fn eigvals(a: &NdArray) -> Result<Vec<(f64, f64)>, LinalgError> {
    if a.is_empty() {
        return Err(LinalgError::Empty);
    }
    let a_mat = to_square_mat(a)?;
    let eig = a_mat.as_ref().eigenvalues().map_err(|_| LinalgError::EigenFailed)?;
    Ok(eig.into_iter().map(|c| (c.re, c.im)).collect())
}

pub(crate) fn extreme_scale(max_abs: f64) -> f64 {
    if max_abs.is_finite() && max_abs > 0.0 && !(1e-100..=1e100).contains(&max_abs) {
        max_abs
    } else {
        1.0
    }
}

fn prescaled(a: &NdArray) -> (NdArray, f64) {
    let scale = extreme_scale(a.as_slice().iter().fold(0.0_f64, |m, x| m.max(x.abs())));
    if scale == 1.0 {
        return (a.clone(), 1.0);
    }
    (NdArray::from_vec(a.as_slice().iter().map(|x| x / scale).collect(), a.shape()).expect("same element count"), scale)
}

pub fn svd(a: &NdArray) -> Result<(NdArray, Vec<f64>, NdArray), LinalgError> {
    if a.is_empty() {
        return Err(LinalgError::Empty);
    }
    let (scaled, scale) = prescaled(a);
    let a_mat = to_mat(&scaled)?;
    let svd = a_mat.as_ref().thin_svd().map_err(|_| LinalgError::SvdFailed)?;
    let k = a_mat.nrows().min(a_mat.ncols());
    let values: Vec<f64> = (0..k).map(|i| svd.S()[i] * scale).collect();
    let vt = svd.V().transpose();
    Ok((from_mat(svd.U()), values, from_mat(vt)))
}

pub fn svd_full(a: &NdArray) -> Result<(NdArray, Vec<f64>, NdArray), LinalgError> {
    if a.is_empty() {
        return Err(LinalgError::Empty);
    }
    let (scaled, scale) = prescaled(a);
    let a_mat = to_mat(&scaled)?;
    let svd = a_mat.as_ref().svd().map_err(|_| LinalgError::SvdFailed)?;
    let k = a_mat.nrows().min(a_mat.ncols());
    let values: Vec<f64> = (0..k).map(|i| svd.S()[i] * scale).collect();
    let vt = svd.V().transpose();
    Ok((from_mat(svd.U()), values, from_mat(vt)))
}

pub fn qr_complete(a: &NdArray) -> Result<(NdArray, NdArray), LinalgError> {
    if a.is_empty() {
        return Err(LinalgError::Empty);
    }
    let a_mat = to_mat(a)?;
    let f = a_mat.as_ref().qr();
    let q = f.compute_Q();
    let thin = f.thin_R();
    let (r, c) = (a_mat.nrows(), a_mat.ncols());
    let full_r = Mat::from_fn(r, c, |i, j| if i < thin.nrows() { thin[(i, j)] } else { 0.0 });
    Ok((from_mat(q.as_ref()), from_mat(full_r.as_ref())))
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
        VecNormOrd::Two => a.iter().fold(0.0, |acc, x| acc + x * x).sqrt(),
        VecNormOrd::Inf => a.iter().fold(0.0_f64, |acc, x| acc.max(x.abs())),
    }
}

pub fn frobenius_norm(a: &NdArray) -> f64 {
    a.as_slice().iter().fold(0.0, |acc, x| acc + x * x).sqrt()
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

pub fn diagonal<'a, T>(a: &crate::view::ArrayView<'a, T>, offset: isize) -> Result<crate::view::ArrayView<'a, T>, LinalgError> {
    a.diagonal(offset).map_err(|_| LinalgError::Not2D { shape: a.shape().to_vec() })
}

pub fn matrix_transpose<'a, T>(a: &crate::view::ArrayView<'a, T>) -> Result<crate::view::ArrayView<'a, T>, LinalgError> {
    a.matrix_transpose().map_err(|_| LinalgError::Not2D { shape: a.shape().to_vec() })
}

pub fn slogdet(a: &NdArray) -> Result<(f64, f64), LinalgError> {
    let m = to_square_mat(a)?;
    if m.nrows() == 0 {
        return Ok((1.0, 0.0));
    }
    Ok(match lu_diagonal(&m) {
        None => (0.0, f64::NEG_INFINITY),
        Some((sign, diag)) => {
            let negatives = diag.iter().filter(|d| **d < 0.0).count();
            let sign = if negatives % 2 == 1 { -sign } else { sign };
            (sign, diag.iter().map(|d| d.abs().ln()).sum())
        }
    })
}

pub fn svdvals(a: &NdArray) -> Result<Vec<f64>, LinalgError> {
    if a.is_empty() {
        return Err(LinalgError::Empty);
    }
    Ok(svd(a)?.1)
}

pub fn matrix_rank(a: &NdArray, tol: Option<f64>) -> Result<usize, LinalgError> {
    if a.ndim() < 2 {
        return Ok(usize::from(a.as_slice().iter().any(|&x| x != 0.0)));
    }
    let s = svdvals(a)?;
    let smax = s.iter().copied().fold(0.0_f64, f64::max);
    let tol = tol.unwrap_or_else(|| smax * a.shape()[0].max(a.shape()[1]) as f64 * f64::EPSILON);
    Ok(s.iter().filter(|&&x| x > tol).count())
}

fn pinv_from_svd(u: &NdArray, s: &[f64], vt: &NdArray, cutoff: f64) -> NdArray {
    let (m, n, k) = (u.shape()[0], vt.shape()[1], s.len());
    let mut out = vec![0.0; n * m];
    for (r, &sv) in s.iter().enumerate().take(k) {
        if sv > cutoff {
            let inv = 1.0 / sv;
            for i in 0..n {
                let v = vt.as_slice()[r * n + i] * inv;
                for j in 0..m {
                    out[i * m + j] += v * u.as_slice()[j * k + r];
                }
            }
        }
    }
    NdArray::from_vec(out, &[n, m]).expect("pinv output is n x m by construction")
}

pub fn pinv(a: &NdArray, rcond: Option<f64>) -> Result<NdArray, LinalgError> {
    if a.ndim() != 2 {
        return Err(LinalgError::Not2D { shape: a.shape().to_vec() });
    }
    if a.is_empty() {
        return Ok(NdArray::zeros(&[a.shape()[1], a.shape()[0]]));
    }
    let (u, s, vt) = svd(a)?;
    let smax = s.iter().copied().fold(0.0_f64, f64::max);
    Ok(pinv_from_svd(&u, &s, &vt, rcond.unwrap_or(1e-15) * smax))
}

#[derive(Debug, Clone, PartialEq)]
pub struct Lstsq {
    pub x: NdArray,
    pub residuals: Vec<f64>,
    pub rank: usize,
    pub singular_values: Vec<f64>,
}

pub fn lstsq(a: &NdArray, b: &NdArray, rcond: Option<f64>) -> Result<Lstsq, LinalgError> {
    if a.ndim() != 2 {
        return Err(LinalgError::Not2D { shape: a.shape().to_vec() });
    }
    let (m, n) = (a.shape()[0], a.shape()[1]);
    if b.ndim() == 0 || b.ndim() > 2 {
        return Err(LinalgError::Not2D { shape: b.shape().to_vec() });
    }
    if b.shape()[0] != m {
        return Err(LinalgError::RankMismatch { a_rows: m, b_rows: b.shape()[0] });
    }
    let nrhs = if b.ndim() == 1 { 1 } else { b.shape()[1] };
    let (u, s, vt) = svd(a)?;
    let smax = s.iter().copied().fold(0.0_f64, f64::max);
    let cutoff = rcond.unwrap_or(f64::EPSILON * m.max(n) as f64) * smax;
    let rank = s.iter().filter(|&&x| x > cutoff).count();
    let p = pinv_from_svd(&u, &s, &vt, cutoff);
    let b2 = b.view().reshape(&[m as isize, nrhs as isize]).expect("b is contiguous").to_owned();
    let x2 = matmul(&p.view(), &b2.view()).expect("pinv is n x m and b is m x nrhs");
    let residuals = if rank == n && m > n {
        let fit = matmul(&a.view(), &x2.view()).expect("a is m x n and x is n x nrhs");
        (0..nrhs)
            .map(|c| (0..m).map(|r| (b2.as_slice()[r * nrhs + c] - fit.as_slice()[r * nrhs + c]).powi(2)).sum())
            .collect()
    } else {
        Vec::new()
    };
    let x = if b.ndim() == 1 { x2.into_shape(&[n as isize]).expect("n x 1 flattens") } else { x2 };
    Ok(Lstsq { x, residuals, rank, singular_values: s })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatNormOrd {
    Fro,
    Nuc,
    Inf,
    NegInf,
    One,
    NegOne,
    Two,
    NegTwo,
}

pub fn matrix_norm_ord(a: &NdArray, ord: MatNormOrd) -> Result<f64, LinalgError> {
    if a.ndim() != 2 {
        return Err(LinalgError::Not2D { shape: a.shape().to_vec() });
    }
    let (rows, cols) = (a.shape()[0], a.shape()[1]);
    let sums = |by_row: bool| -> Vec<f64> {
        let (outer, inner) = if by_row { (rows, cols) } else { (cols, rows) };
        (0..outer)
            .map(|o| {
                (0..inner)
                    .map(|i| a.as_slice()[if by_row { o * cols + i } else { i * cols + o }].abs())
                    .sum()
            })
            .collect()
    };
    let (max, min) = (|v: Vec<f64>| v.iter().copied().fold(f64::NEG_INFINITY, f64::max), |v: Vec<f64>| v.iter().copied().fold(f64::INFINITY, f64::min));
    Ok(match ord {
        MatNormOrd::Fro => frobenius_norm(a),
        MatNormOrd::Inf => max(sums(true)),
        MatNormOrd::NegInf => min(sums(true)),
        MatNormOrd::One => max(sums(false)),
        MatNormOrd::NegOne => min(sums(false)),
        MatNormOrd::Two => svdvals(a)?[0],
        MatNormOrd::NegTwo => *svdvals(a)?.last().unwrap(),
        MatNormOrd::Nuc => svdvals(a)?.iter().sum(),
    })
}

pub fn cond(a: &NdArray, ord: MatNormOrd) -> Result<f64, LinalgError> {
    if a.is_empty() {
        return Err(LinalgError::Empty);
    }
    match ord {
        MatNormOrd::Two | MatNormOrd::NegTwo => {
            let s = svdvals(a)?;
            let (hi, lo) = (s[0], *s.last().unwrap());
            let r = if ord == MatNormOrd::Two { hi / lo } else { lo / hi };
            Ok(if r.is_nan() && !a.as_slice().iter().any(|x| x.is_nan()) { f64::INFINITY } else { r })
        }
        _ => {
            to_square_mat(a)?;
            match inv(a) {
                Ok(inverse) => Ok(matrix_norm_ord(a, ord)? * matrix_norm_ord(&inverse, ord)?),
                Err(LinalgError::Singular) => Ok(f64::INFINITY),
                Err(e) => Err(e),
            }
        }
    }
}

pub fn eig(a: &NdArray) -> Result<(Vec<Complex64>, NdArray<Complex64>), LinalgError> {
    if a.is_empty() {
        return Err(LinalgError::Empty);
    }
    let m = to_square_mat(a)?;
    let n = m.nrows();
    let e = m.as_ref().eigen().map_err(|_| LinalgError::EigenFailed)?;
    let values: Vec<Complex64> = (0..n).map(|i| Complex64::new(e.S()[i].re, e.S()[i].im)).collect();
    let mut vectors = Vec::with_capacity(n * n);
    let cols: Vec<Vec<Complex64>> = (0..n)
        .map(|j| {
            let col: Vec<Complex64> = (0..n).map(|i| Complex64::new(e.U()[(i, j)].re, e.U()[(i, j)].im)).collect();
            let norm = col.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            let big = col.iter().copied().max_by(|x, y| x.norm_sqr().total_cmp(&y.norm_sqr())).unwrap();
            let phase = if big.norm() > 0.0 { big.conj() / big.norm() } else { Complex64::new(1.0, 0.0) };
            col.into_iter().map(|c| c * phase / norm).collect()
        })
        .collect();
    for i in 0..n {
        for col in &cols {
            vectors.push(col[i]);
        }
    }
    Ok((values, NdArray::from_vec(vectors, &[n, n]).expect("n x n eigenvector matrix")))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        close_all(x.as_slice(), arr(vec![0.5, -1.0 / 3.0], &[2]).as_slice());
    }

    #[test]
    fn inv_matches_numpy() {
        let a = sample_a();
        let inv_a = inv(&a).unwrap();
        let expected = arr(vec![-0.5, 0.5, 1.0, -2.0 / 3.0], &[2, 2]);
        close_all(inv_a.as_slice(), expected.as_slice());
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
        close_all(from_mat(product.as_ref()).as_slice(), a.as_slice());
    }

    #[test]
    fn cholesky_matches_numpy() {

        let s = arr(vec![4.0, 2.0, 2.0, 3.0], &[2, 2]);
        let l = cholesky(&s).unwrap();
        let expected = arr(vec![2.0, 0.0, 1.0, 2.0_f64.sqrt()], &[2, 2]);
        close_all(l.as_slice(), expected.as_slice());
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
        close_all(from_mat(product.as_ref()).as_slice(), a.as_slice());
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
        close_all(cubed.as_slice(), arr(vec![262.0, 165.0, 330.0, 207.0], &[2, 2]).as_slice());

        let zeroth = matrix_power(&a, 0).unwrap();
        close_all(zeroth.as_slice(), arr(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).as_slice());

        let inverse = matrix_power(&a, -1).unwrap();
        close_all(inverse.as_slice(), inv(&a).unwrap().as_slice());
    }

    fn close(got: f64, want: f64) {
        assert!((got - want).abs() <= 1e-10 * (1.0 + want.abs()), "got {got}, want {want}");
    }

    fn close_all(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want) {
            close(*g, *w);
        }
    }

    fn tridiag() -> NdArray {
        arr(vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0], &[3, 3])
    }

    fn rank_one() -> NdArray {
        arr(vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0], &[3, 2])
    }

    #[test]
    fn norms_of_empty_input_are_positive_zero_like_numpy() {
        let e: NdArray = NdArray::zeros(&[0]);
        assert!(vector_norm(e.as_slice(), VecNormOrd::Two).is_sign_positive());
        assert!(frobenius_norm(&NdArray::zeros(&[0, 3])).is_sign_positive());
    }

    #[test]
    fn det_and_inv_handle_exactly_singular_and_tiny_scaled_matrices_like_numpy() {
        let zeros: NdArray = NdArray::zeros(&[2, 2]);
        assert_eq!(det(&zeros).unwrap(), 0.0);
        assert_eq!(inv(&zeros), Err(LinalgError::Singular));
        assert_eq!(det(&arr(vec![1.0, 2.0, 2.0, 4.0], &[2, 2])).unwrap(), 0.0);
        assert_eq!(inv(&arr(vec![1.0, 2.0, 2.0, 4.0], &[2, 2])), Err(LinalgError::Singular));
        let tiny = arr(vec![1e-200, 0.0, 0.0, 1e-200], &[2, 2]);
        let t = inv(&tiny).unwrap();
        assert_eq!(t.as_slice(), &[1e200, 0.0, 0.0, 1e200]);
        assert_eq!(cond(&zeros, MatNormOrd::One).unwrap(), f64::INFINITY);
        assert_eq!(cond(&zeros, MatNormOrd::Fro).unwrap(), f64::INFINITY);
        assert_eq!(cond(&zeros, MatNormOrd::Two).unwrap(), f64::INFINITY);
        assert_eq!(cond(&zeros, MatNormOrd::NegTwo).unwrap(), f64::INFINITY);
    }

    #[test]
    fn empty_matrices_never_panic_and_follow_numpy_where_it_defines_a_result() {
        let e: NdArray = NdArray::zeros(&[0, 0]);
        assert_eq!(det(&e).unwrap(), 1.0);
        assert_eq!(slogdet(&e).unwrap(), (1.0, 0.0));
        assert_eq!(inv(&e).unwrap().shape(), &[0, 0]);
        for r in [eigh(&e).map(|_| ()), eig(&e).map(|_| ()), svd(&e).map(|_| ()), qr(&e).map(|_| ()), cholesky(&e).map(|_| ()), solve(&e, &e).map(|_| ())] {
            assert_eq!(r, Err(LinalgError::Empty));
        }
        assert_eq!(eigvals(&e), Err(LinalgError::Empty));
        assert_eq!(eigvalsh(&e), Err(LinalgError::Empty));
    }

    #[test]
    fn slogdet_matches_numpy_including_zero_and_permutation_sign() {
        let (sign, logabs) = slogdet(&tridiag()).unwrap();
        assert_eq!(sign, 1.0);
        close(logabs, 2.8903717578961645);
        let neg = arr(tridiag().as_slice().iter().map(|x| -x).collect(), &[3, 3]);
        let (sign, logabs) = slogdet(&neg).unwrap();
        assert_eq!(sign, -1.0);
        close(logabs, 2.8903717578961645);
        assert_eq!(slogdet(&NdArray::zeros(&[2, 2])).unwrap(), (0.0, f64::NEG_INFINITY));
        assert_eq!(slogdet(&arr(vec![0.0, 1.0, 1.0, 0.0], &[2, 2])).unwrap(), (-1.0, 0.0));
        assert!(matches!(slogdet(&NdArray::zeros(&[2, 3])), Err(LinalgError::NotSquare { .. })));
    }

    #[test]
    fn matrix_rank_and_svdvals_match_numpy() {
        assert_eq!(matrix_rank(&rank_one(), None).unwrap(), 1);
        assert_eq!(matrix_rank(&tridiag(), None).unwrap(), 3);
        assert_eq!(matrix_rank(&NdArray::zeros(&[3, 3]), None).unwrap(), 0);
        assert_eq!(matrix_rank(&arr(vec![1.0, 2.0, 3.0], &[3]), None).unwrap(), 1);
        assert_eq!(matrix_rank(&arr(vec![0.0, 0.0], &[2]), None).unwrap(), 0);
        assert_eq!(matrix_rank(&rank_one(), Some(10.0)).unwrap(), 0);
        let wide = arr((0..12).map(f64::from).collect(), &[3, 4]);
        assert_eq!(matrix_rank(&wide, None).unwrap(), 2);
        let s = svdvals(&rank_one()).unwrap();
        close(s[0], 8.366600265340757);
        assert!(s[1] < 1e-12);
        close_all(&svdvals(&tridiag()).unwrap(), &[4.732050807568878, 3.0, 1.267949192431123]);
        close_all(&svdvals(&wide).unwrap()[..2], &[22.40929816327044, 1.955340336014275]);
        assert_eq!(svdvals(&NdArray::zeros(&[0, 3])), Err(LinalgError::Empty));
    }

    #[test]
    fn pinv_matches_numpy_including_rcond_and_shapes() {
        let p = pinv(&rank_one(), None).unwrap();
        assert_eq!(p.shape(), &[2, 3]);
        close_all(
            p.as_slice(),
            &[0.014285714285714289, 0.028571428571428564, 0.042857142857142844, 0.028571428571428584, 0.05714285714285714, 0.0857142857142857],
        );
        let t = tridiag();
        let id = matmul(&pinv(&t, None).unwrap().view(), &t.view()).unwrap();
        close_all(id.as_slice(), &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let ill = arr(vec![1.0, 0.0, 0.0, 1e-10], &[2, 2]);
        close_all(pinv(&ill, Some(1e-5)).unwrap().as_slice(), &[1.0, 0.0, 0.0, 0.0]);
        close_all(pinv(&ill, None).unwrap().as_slice(), &[1.0, 0.0, 0.0, 1e10]);
        assert_eq!(pinv(&arr((0..12).map(f64::from).collect(), &[3, 4]), None).unwrap().shape(), &[4, 3]);
        assert_eq!(pinv(&NdArray::zeros(&[2, 3]), None).unwrap(), NdArray::zeros(&[3, 2]));
    }

    #[test]
    fn lstsq_matches_numpy_for_over_under_and_rank_deficient_systems() {
        let x = arr(vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0], &[3, 2]);
        let r = lstsq(&x, &arr(vec![1.0, 2.0, 2.0], &[3]), None).unwrap();
        close_all(r.x.as_slice(), &[0.666666666666666, 0.5]);
        close_all(&r.residuals, &[0.166666666666667]);
        assert_eq!(r.rank, 2);
        close_all(&r.singular_values, &[4.079143328941734, 0.600491217213163]);

        let multi = lstsq(&x, &arr(vec![1.0, 0.0, 2.0, 1.0, 2.0, 5.0], &[3, 2]), None).unwrap();
        assert_eq!(multi.x.shape(), &[2, 2]);
        close_all(multi.x.as_slice(), &[0.6666666666666663, -3.0000000000000004, 0.5000000000000002, 2.5]);
        close_all(&multi.residuals, &[0.166666666666667, 1.5]);

        let square = lstsq(&arr(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), &arr(vec![1.0, 2.0], &[2]), None).unwrap();
        close_all(square.x.as_slice(), &[0.0, 0.5]);
        assert!(square.residuals.is_empty());
        assert_eq!(square.rank, 2);

        let deficient = lstsq(&rank_one(), &arr(vec![1.0, 2.0, 3.0], &[3]), None).unwrap();
        close_all(deficient.x.as_slice(), &[0.2, 0.4]);
        assert_eq!(deficient.rank, 1);
        assert!(deficient.residuals.is_empty());

        let wide = lstsq(&arr(vec![1.0, 1.0, 1.0, 1.0, 2.0, 3.0], &[2, 3]), &arr(vec![1.0, 2.0], &[2]), None).unwrap();
        close_all(wide.x.as_slice(), &[0.333333333333332, 0.333333333333333, 0.333333333333334]);
        assert_eq!(wide.rank, 2);
        assert!(wide.residuals.is_empty());

        assert!(matches!(lstsq(&x, &arr(vec![1.0; 4], &[4]), None), Err(LinalgError::RankMismatch { .. })));
    }

    #[test]
    fn cond_and_matrix_norm_ord_match_numpy() {
        let t = tridiag();
        close(cond(&t, MatNormOrd::Two).unwrap(), 3.732050807568877);
        close(cond(&t, MatNormOrd::Fro).unwrap(), 5.066228051190222);
        close(cond(&t, MatNormOrd::One).unwrap(), 4.444444444444445);
        close(cond(&t, MatNormOrd::NegOne).unwrap(), 1.3333333333333335);
        close(cond(&t, MatNormOrd::Inf).unwrap(), 4.444444444444445);
        close(cond(&t, MatNormOrd::NegInf).unwrap(), 1.3333333333333335);
        close(cond(&t, MatNormOrd::NegTwo).unwrap(), 0.2679491924311227);
        let singular = arr(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]);
        assert!(cond(&singular, MatNormOrd::Two).unwrap() > 1e15);
        assert_eq!(cond(&singular, MatNormOrd::Fro).unwrap(), f64::INFINITY);
        assert_eq!(cond(&singular, MatNormOrd::One).unwrap(), f64::INFINITY);
        assert_eq!(cond(&NdArray::zeros(&[0, 0]), MatNormOrd::Two), Err(LinalgError::Empty));

        let m = arr(vec![1.0, -2.0, 3.0, 4.0, 5.0, -6.0, 0.0, 7.0, 8.0], &[3, 3]);
        let want = [
            (MatNormOrd::Fro, 14.2828568570857),
            (MatNormOrd::Nuc, 22.178254010179415),
            (MatNormOrd::Inf, 15.0),
            (MatNormOrd::NegInf, 6.0),
            (MatNormOrd::One, 17.0),
            (MatNormOrd::NegOne, 5.0),
            (MatNormOrd::Two, 10.959993927680461),
            (MatNormOrd::NegTwo, 2.3723219324861042),
        ];
        for (ord, expected) in want {
            close(matrix_norm_ord(&m, ord).unwrap(), expected);
        }
    }

    fn sorted(mut v: Vec<Complex64>) -> Vec<Complex64> {
        v.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
        v
    }

    fn check_eigenpairs(a: &NdArray) {
        let n = a.shape()[0];
        let (w, v) = eig(a).unwrap();
        for (j, lambda) in w.iter().enumerate() {
            let col: Vec<Complex64> = (0..n).map(|i| v.get(&[i, j]).unwrap()).collect();
            let norm: f64 = col.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            close(norm, 1.0);
            for i in 0..n {
                let av: Complex64 = (0..n).map(|k| col[k] * a.get(&[i, k]).unwrap()).sum();
                assert!((av - col[i] * lambda).norm() < 1e-10, "A v != lambda v for eigenpair {j}");
            }
        }
    }

    #[test]
    fn eig_returns_complex_eigenpairs_that_satisfy_a_v_equals_lambda_v() {
        let ns = arr(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let (w, _) = eig(&ns).unwrap();
        let w = sorted(w);
        close(w[0].re, -0.372281323269014);
        close(w[1].re, 5.372281323269014);
        assert!(w.iter().all(|c| c.im.abs() < 1e-12));
        check_eigenpairs(&ns);

        let rot = arr(vec![0.0, -1.0, 1.0, 0.0], &[2, 2]);
        let (w, _) = eig(&rot).unwrap();
        let w = sorted(w);
        assert!(w[0].re.abs() < 1e-12 && (w[0].im + 1.0).abs() < 1e-12);
        assert!((w[1].im - 1.0).abs() < 1e-12);
        check_eigenpairs(&rot);

        check_eigenpairs(&tridiag());
        check_eigenpairs(&arr(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0], &[3, 3]));
        check_eigenpairs(&arr(vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], &[3, 3]));

        let (w, v) = eig(&arr(vec![2.0, 0.0, 0.0, 2.0], &[2, 2])).unwrap();
        assert!(w.iter().all(|c| (c.re - 2.0).abs() < 1e-12));
        assert_eq!(v.shape(), &[2, 2]);
        let (w, _) = eig(&arr(vec![0.0, 1.0, 0.0, 0.0], &[2, 2])).unwrap();
        assert!(w.iter().all(|c| c.norm() < 1e-12));
        assert!(matches!(eig(&NdArray::zeros(&[2, 3])), Err(LinalgError::NotSquare { .. })));
    }

    #[test]
    fn eig_complex_eigenvalues_of_a_real_matrix_come_in_conjugate_pairs() {
        let cyc = arr(vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], &[3, 3]);
        let w = sorted(eig(&cyc).unwrap().0);
        assert!((w[0].re + 0.5).abs() < 1e-12 && (w[0].im + 0.8660254037844386).abs() < 1e-12);
        assert!((w[1].re + 0.5).abs() < 1e-12 && (w[1].im - 0.8660254037844386).abs() < 1e-12);
        assert!((w[2].re - 1.0).abs() < 1e-12 && w[2].im.abs() < 1e-12);
    }

    #[test]
    fn array_api_names_are_reachable_from_linalg() {
        let a = arr((0..6).map(f64::from).collect(), &[2, 3]);
        assert_eq!(matrix_transpose(&a.view()).unwrap().shape(), &[3, 2]);
        assert_eq!(diagonal(&a.view(), 1).unwrap().iter().collect::<Vec<_>>(), vec![1.0, 5.0]);
        assert!(matches!(diagonal(&arr(vec![1.0], &[1]).view(), 0), Err(LinalgError::Not2D { .. })));
        assert_eq!(trace(&a.view(), 0).unwrap(), 4.0);
        assert_eq!(matmul(&a.view(), &matrix_transpose(&a.view()).unwrap()).unwrap().as_slice(), &[5.0, 14.0, 14.0, 50.0]);
        assert_eq!(outer(&arr(vec![1.0, 2.0], &[2]).view(), &arr(vec![3.0, 4.0], &[2]).view()).unwrap().as_slice(), &[3.0, 4.0, 6.0, 8.0]);
        assert_eq!(vecdot(&a.view(), &a.view()).unwrap().as_slice(), &[5.0, 50.0]);
        assert_eq!(kron(&a.view(), &arr(vec![1.0], &[1, 1]).view()).unwrap().shape(), &[2, 3]);
        let e = arr(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
        assert_eq!(cross(&e.view(), &e.view()).unwrap().as_slice(), &[0.0; 6]);
    }
}
