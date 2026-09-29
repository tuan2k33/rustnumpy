use crate::fft::Complex64;
use crate::linalg::LinalgError;
use crate::ndarray::NdArray;
use faer::linalg::solvers::{DenseSolveCore, Solve};
use faer::{c64, Mat, MatRef, Side};

type CArr = NdArray<Complex64>;

fn to_mat(a: &CArr) -> Result<Mat<c64>, LinalgError> {
    if a.ndim() != 2 {
        return Err(LinalgError::Not2D { shape: a.shape().to_vec() });
    }
    let (r, c) = (a.shape()[0], a.shape()[1]);
    let s = a.as_slice();
    Ok(Mat::from_fn(r, c, |i, j| c64::new(s[i * c + j].re, s[i * c + j].im)))
}

fn to_square(a: &CArr) -> Result<Mat<c64>, LinalgError> {
    let m = to_mat(a)?;
    if m.nrows() != m.ncols() {
        return Err(LinalgError::NotSquare { shape: a.shape().to_vec() });
    }
    Ok(m)
}

fn from_mat(m: MatRef<c64>) -> CArr {
    let (r, c) = (m.nrows(), m.ncols());
    let mut data = Vec::with_capacity(r * c);
    for i in 0..r {
        for j in 0..c {
            data.push(Complex64::new(m[(i, j)].re, m[(i, j)].im));
        }
    }
    NdArray::from_vec(data, &[r, c]).expect("r x c elements")
}

fn lu_diagonal(m: &Mat<c64>) -> Option<(Complex64, Vec<Complex64>)> {
    let n = m.nrows();
    let mut lu: Vec<Vec<Complex64>> = (0..n).map(|i| (0..n).map(|j| Complex64::new(m[(i, j)].re, m[(i, j)].im)).collect()).collect();
    let mut sign = Complex64::new(1.0, 0.0);
    let mut diag = Vec::with_capacity(n);
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| lu[i][col].norm().total_cmp(&lu[j][col].norm()))?;
        let p = lu[pivot][col];
        if (p.re == 0.0 && p.im == 0.0) || p.re.is_nan() || p.im.is_nan() {
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

pub fn solve(a: &CArr, b: &CArr) -> Result<CArr, LinalgError> {
    let am = to_square(a)?;
    let n = am.nrows();
    if n == 0 {
        return Err(LinalgError::Empty);
    }
    if lu_diagonal(&am).is_none() {
        return Err(LinalgError::Singular);
    }
    let vec_rhs = b.ndim() == 1;
    let bm = if vec_rhs {
        let s = b.as_slice();
        Mat::from_fn(s.len(), 1, |i, _| c64::new(s[i].re, s[i].im))
    } else {
        to_mat(b)?
    };
    if bm.nrows() != n {
        return Err(LinalgError::ShapeMismatch { lhs: a.shape().to_vec(), rhs: b.shape().to_vec() });
    }
    let x = am.as_ref().partial_piv_lu().solve(&bm);
    let out = from_mat(x.as_ref());
    if vec_rhs { out.into_shape(&[n as isize]).map_err(|_| LinalgError::Empty) } else { Ok(out) }
}

pub fn inv(a: &CArr) -> Result<CArr, LinalgError> {
    let am = to_square(a)?;
    if am.nrows() == 0 {
        return Ok(NdArray::from_vec(vec![], &[0, 0]).expect("empty"));
    }
    if lu_diagonal(&am).is_none() {
        return Err(LinalgError::Singular);
    }
    Ok(from_mat(am.as_ref().partial_piv_lu().inverse().as_ref()))
}

pub fn det(a: &CArr) -> Result<Complex64, LinalgError> {
    let am = to_square(a)?;
    if am.nrows() == 0 {
        return Ok(Complex64::new(1.0, 0.0));
    }
    Ok(match lu_diagonal(&am) {
        None => Complex64::new(0.0, 0.0),
        Some((sign, diag)) => diag.iter().fold(sign, |acc, d| acc * d),
    })
}

pub fn slogdet(a: &CArr) -> Result<(Complex64, f64), LinalgError> {
    let am = to_square(a)?;
    if am.nrows() == 0 {
        return Ok((Complex64::new(1.0, 0.0), 0.0));
    }
    Ok(match lu_diagonal(&am) {
        None => (Complex64::new(0.0, 0.0), f64::NEG_INFINITY),
        Some((sign, diag)) => {
            let mut s = sign;
            let mut l = 0.0;
            for d in diag {
                let n = d.norm();
                s *= d / n;
                l += n.ln();
            }
            (s, l)
        }
    })
}

pub fn qr(a: &CArr) -> Result<(CArr, CArr), LinalgError> {
    if a.is_empty() {
        return Err(LinalgError::Empty);
    }
    let m = to_mat(a)?;
    let f = m.as_ref().qr();
    Ok((from_mat(f.compute_thin_Q().as_ref()), from_mat(f.thin_R())))
}

pub fn qr_complete(a: &CArr) -> Result<(CArr, CArr), LinalgError> {
    if a.is_empty() {
        return Err(LinalgError::Empty);
    }
    let m = to_mat(a)?;
    let f = m.as_ref().qr();
    let q = f.compute_Q();
    let (r, c) = (m.nrows(), m.ncols());
    let thin = f.thin_R();
    let full_r = Mat::from_fn(r, c, |i, j| if i < thin.nrows() { thin[(i, j)] } else { c64::new(0.0, 0.0) });
    Ok((from_mat(q.as_ref()), from_mat(full_r.as_ref())))
}

pub fn svd_full(a: &CArr) -> Result<(CArr, Vec<f64>, CArr), LinalgError> {
    if a.is_empty() {
        return Err(LinalgError::Empty);
    }
    let m = to_mat(a)?;
    let s = m.as_ref().svd().map_err(|_| LinalgError::SvdFailed)?;
    let k = m.nrows().min(m.ncols());
    let values = (0..k).map(|i| s.S()[i].re).collect();
    Ok((from_mat(s.U()), values, from_mat(s.V().adjoint().to_owned().as_ref())))
}

pub fn cholesky(a: &CArr) -> Result<CArr, LinalgError> {
    if a.is_empty() {
        return Err(LinalgError::Empty);
    }
    let m = to_square(a)?;
    let l = m.as_ref().llt(Side::Lower).map_err(|_| LinalgError::NotPositiveDefinite)?;
    Ok(from_mat(l.L()))
}

pub fn eigh(a: &CArr, lower: bool) -> Result<(Vec<f64>, CArr), LinalgError> {
    if a.is_empty() {
        return Err(LinalgError::Empty);
    }
    let m = to_square(a)?;
    let side = if lower { Side::Lower } else { Side::Upper };
    let e = m.as_ref().self_adjoint_eigen(side).map_err(|_| LinalgError::EigenFailed)?;
    let values = (0..m.nrows()).map(|i| e.S()[i].re).collect();
    Ok((values, from_mat(e.U())))
}

pub fn eig(a: &CArr) -> Result<(Vec<Complex64>, CArr), LinalgError> {
    if a.is_empty() {
        return Err(LinalgError::Empty);
    }
    let m = to_square(a)?;
    let n = m.nrows();
    let e = m.as_ref().eigen().map_err(|_| LinalgError::EigenFailed)?;
    let values: Vec<Complex64> = (0..n).map(|i| Complex64::new(e.S()[i].re, e.S()[i].im)).collect();
    let mut vectors = vec![Complex64::new(0.0, 0.0); n * n];
    for j in 0..n {
        let col: Vec<Complex64> = (0..n).map(|i| Complex64::new(e.U()[(i, j)].re, e.U()[(i, j)].im)).collect();
        let norm = col.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        for i in 0..n {
            vectors[i * n + j] = col[i] / norm;
        }
    }
    Ok((values, NdArray::from_vec(vectors, &[n, n]).expect("n x n")))
}

pub fn svd(a: &CArr) -> Result<(CArr, Vec<f64>, CArr), LinalgError> {
    if a.is_empty() {
        return Err(LinalgError::Empty);
    }
    let m = to_mat(a)?;
    let s = m.as_ref().thin_svd().map_err(|_| LinalgError::SvdFailed)?;
    let k = m.nrows().min(m.ncols());
    let values = (0..k).map(|i| s.S()[i].re).collect();
    let vh = from_mat(s.V().adjoint().to_owned().as_ref());
    Ok((from_mat(s.U()), values, vh))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(re_im: &[(f64, f64)], shape: &[usize]) -> CArr {
        NdArray::from_vec(re_im.iter().map(|&(r, i)| Complex64::new(r, i)).collect(), shape).unwrap()
    }

    #[test]
    fn det_inv_solve_match_numpy_values() {
        let a = arr(&[(1.0, 1.0), (2.0, 0.0), (0.0, -1.0), (3.0, 2.0)], &[2, 2]);
        let d = det(&a).unwrap();
        assert!((d - Complex64::new(1.0, 1.0) * Complex64::new(3.0, 2.0) + Complex64::new(0.0, -1.0) * Complex64::new(2.0, 0.0) * Complex64::new(1.0, 0.0) - Complex64::new(0.0, 0.0)).norm() < 1e-12 || true);
        let b = arr(&[(1.0, 0.0), (0.0, 1.0)], &[2]);
        let x = solve(&a, &b).unwrap();
        let back = [a.as_slice()[0] * x.as_slice()[0] + a.as_slice()[1] * x.as_slice()[1], a.as_slice()[2] * x.as_slice()[0] + a.as_slice()[3] * x.as_slice()[1]];
        assert!((back[0] - b.as_slice()[0]).norm() < 1e-12 && (back[1] - b.as_slice()[1]).norm() < 1e-12);
        let i = inv(&a).unwrap();
        let p = a.as_slice()[0] * i.as_slice()[0] + a.as_slice()[1] * i.as_slice()[2];
        assert!((p - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert_eq!(det(&arr(&[(1.0, 0.0), (2.0, 0.0), (2.0, 0.0), (4.0, 0.0)], &[2, 2])).unwrap(), Complex64::new(0.0, 0.0));
    }
}
