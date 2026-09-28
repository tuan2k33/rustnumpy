//! `numpy.linalg` — solve, inverse, determinant, QR, Cholesky, eigenvalues
//! (self-adjoint + general eigenvalues-only), SVD, norms, `matrix_power`.
//!
//! Delegates the actual numerics to the `faer` crate (pure Rust, no
//! FFI/LAPACK) per the "depend on a crate, don't hand-convert" decision in
//! `NumPy.md`: LAPACK is a numerical *tool* NumPy borrows, not one of
//! NumPy's own design ideas, so there's nothing to learn from
//! hand-translating F2C-generated Fortran.
//!
//! **Known, accepted deviations from NumPy** (see `NumPy.md`'s "Known
//! deviations" section): `faer` and LAPACK are both mathematically correct
//! but not bit-for-bit identical — eigenvector/singular-vector signs can
//! flip, and results should always be compared to NumPy with a tolerance
//! (`assert_allclose`-style), never exact equality. This module's own
//! tests do exactly that.
//!
//! **Scope note**: `NdArray` has no complex dtype yet (it's f64-only, see
//! `lib.rs`'s doc comment), so a general (non-symmetric) matrix's complex
//! eigenvectors can't be represented. [`eigvals`] returns the eigenvalues
//! as `(re, im)` pairs (matching `np.linalg.eigvals`'s values exactly,
//! just not its complex *dtype*); full `eig` (eigenvectors too) is only
//! implemented for the self-adjoint case ([`eigh`]/[`eigvalsh`], which is
//! always real), matching `np.linalg.eigh`.

use faer::linalg::solvers::{DenseSolveCore, Solve};
use faer::{Mat, MatRef, Side};

use crate::ndarray::NdArray;

/// Any error from a `linalg` operation. Real NumPy raises `LinAlgError`
/// (a `ValueError` for shape issues); here each case is its own variant.
#[derive(Debug, Clone, PartialEq)]
pub enum LinalgError {
    /// The operation requires a square matrix, but got this shape.
    NotSquare { shape: Vec<usize> },
    /// The operation requires a 2-D array, but got this shape.
    Not2D { shape: Vec<usize> },
    /// The operation requires a 1-D array, but got this shape.
    Not1D { shape: Vec<usize> },
    /// Two shapes are incompatible for the requested operation (e.g. `solve`'s `A`/`b`).
    ShapeMismatch { lhs: Vec<usize>, rhs: Vec<usize> },
    /// The matrix is (numerically) singular — no unique solution/inverse exists.
    Singular,
    /// The matrix isn't positive-definite, so Cholesky can't factor it.
    NotPositiveDefinite,
    /// The self-adjoint eigendecomposition failed to converge.
    EigenFailed,
    /// The SVD failed to converge.
    SvdFailed,
}

/// Convert a 2-D `NdArray` (row-major, as `NdArray` always is) into an
/// owned `faer::Mat<f64>` — `MatRef::from_row_major_slice` reads that same
/// row-major layout directly, no transposing needed.
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

/// Convert a `faer::MatRef<f64>` back into a row-major `NdArray`.
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

/// `np.linalg.solve(a, b)` for a square `a` and a right-hand side `b`
/// (either a 1-D vector or a 2-D matrix of columns). Uses `faer`'s
/// partial-pivoting LU — the same algorithm family LAPACK's `gesv` uses,
/// so results match NumPy to floating-point tolerance.
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

/// `np.linalg.inv(a)`.
pub fn inv(a: &NdArray) -> Result<NdArray, LinalgError> {
    let a_mat = to_square_mat(a)?;
    if det(a)?.abs() < 1e-300 {
        return Err(LinalgError::Singular);
    }
    let lu = a_mat.as_ref().partial_piv_lu();
    Ok(from_mat(lu.inverse().as_ref()))
}

/// `np.linalg.det(a)`.
pub fn det(a: &NdArray) -> Result<f64, LinalgError> {
    let a_mat = to_square_mat(a)?;
    Ok(a_mat.as_ref().determinant())
}

/// `np.linalg.qr(a)` (the default, "reduced"/thin mode): returns `(Q, R)`
/// where `Q` is `(m, k)`, `R` is `(k, n)`, `k = min(m, n)`.
pub fn qr(a: &NdArray) -> Result<(NdArray, NdArray), LinalgError> {
    let a_mat = to_mat(a)?;
    let qr = a_mat.as_ref().qr();
    let q = qr.compute_thin_Q();
    let r = qr.thin_R();
    Ok((from_mat(q.as_ref()), from_mat(r)))
}

/// `np.linalg.cholesky(a)`: the lower-triangular `L` with `a == L @ L.T`.
/// Requires `a` to be symmetric positive-definite (only the lower
/// triangle is read, matching NumPy).
pub fn cholesky(a: &NdArray) -> Result<NdArray, LinalgError> {
    let a_mat = to_square_mat(a)?;
    let llt = a_mat.as_ref().llt(Side::Lower).map_err(|_| LinalgError::NotPositiveDefinite)?;
    Ok(from_mat(llt.L()))
}

/// `np.linalg.eigh(a)` for a symmetric (only the lower triangle is read,
/// matching NumPy's default `UPLO='L'`) real matrix: returns
/// `(eigenvalues, eigenvectors)`, eigenvalues ascending, `eigenvectors`'s
/// columns are the corresponding unit eigenvectors.
pub fn eigh(a: &NdArray) -> Result<(Vec<f64>, NdArray), LinalgError> {
    let a_mat = to_square_mat(a)?;
    let eig = a_mat.as_ref().self_adjoint_eigen(Side::Lower).map_err(|_| LinalgError::EigenFailed)?;
    let values: Vec<f64> = (0..a_mat.nrows()).map(|i| eig.S()[i]).collect();
    Ok((values, from_mat(eig.U())))
}

/// `np.linalg.eigvalsh(a)`: just the eigenvalues from [`eigh`], without
/// paying for the eigenvectors.
pub fn eigvalsh(a: &NdArray) -> Result<Vec<f64>, LinalgError> {
    Ok(eigh(a)?.0)
}

/// `np.linalg.eigvals(a)` for a general (possibly non-symmetric) square
/// matrix: the eigenvalues as `(re, im)` pairs. See this module's doc
/// comment for why eigenvectors aren't returned here (no complex dtype
/// yet to hold them).
pub fn eigvals(a: &NdArray) -> Result<Vec<(f64, f64)>, LinalgError> {
    let a_mat = to_square_mat(a)?;
    let eig = a_mat.as_ref().eigenvalues().map_err(|_| LinalgError::EigenFailed)?;
    Ok(eig.into_iter().map(|c| (c.re, c.im)).collect())
}

/// `np.linalg.svd(a)` (the default, "full_matrices=False" i.e. thin,
/// mode): returns `(U, singular_values, Vt)` with `U` `(m, k)`, `Vt`
/// `(k, n)`, `k = min(m, n)`, singular values descending.
pub fn svd(a: &NdArray) -> Result<(NdArray, Vec<f64>, NdArray), LinalgError> {
    let a_mat = to_mat(a)?;
    let svd = a_mat.as_ref().thin_svd().map_err(|_| LinalgError::SvdFailed)?;
    let k = a_mat.nrows().min(a_mat.ncols());
    let values: Vec<f64> = (0..k).map(|i| svd.S()[i]).collect();
    let vt = svd.V().transpose();
    Ok((from_mat(svd.U()), values, from_mat(vt)))
}

/// Vector norm order, matching `np.linalg.norm`'s `ord=` for a 1-D input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecNormOrd {
    /// `ord=1`: sum of absolute values.
    One,
    /// `ord=2` (the default): Euclidean norm.
    Two,
    /// `ord=np.inf`: maximum absolute value.
    Inf,
}

/// `np.linalg.norm(a, ord=...)` for a 1-D vector.
pub fn vector_norm(a: &[f64], ord: VecNormOrd) -> f64 {
    match ord {
        VecNormOrd::One => a.iter().map(|x| x.abs()).sum(),
        VecNormOrd::Two => a.iter().map(|x| x * x).sum::<f64>().sqrt(),
        VecNormOrd::Inf => a.iter().fold(0.0_f64, |acc, x| acc.max(x.abs())),
    }
}

/// `np.linalg.norm(a)` for a 2-D matrix with no `ord=` given: the
/// Frobenius norm (`sqrt(sum(a**2))`) — NumPy's default for a matrix,
/// distinct from the induced 2-norm (`ord=2`, largest singular value,
/// not implemented here).
pub fn frobenius_norm(a: &NdArray) -> f64 {
    a.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// `np.linalg.matrix_power(a, n)`: `a` raised to the integer power `n` by
/// repeated squaring. `n == 0` returns the identity; `n < 0` inverts `a`
/// first (matching real NumPy).
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

    // A = [[4, 3], [6, 3]] -- every expected value below was computed by
    // real NumPy 2.5.3 first (see linalg.rs's doc comment for the
    // tolerance-not-exact-equality policy this module follows).
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
        // Q @ R should reconstruct A regardless of faer's/NumPy's sign
        // convention differences (see this module's doc comment).
        let a = sample_a();
        let (q, r) = qr(&a).unwrap();
        let q_mat = to_mat(&q).unwrap();
        let r_mat = to_mat(&r).unwrap();
        let product = &q_mat * &r_mat;
        assert_allclose_default(&from_mat(product.as_ref()).view(), &a.view()).unwrap();
    }

    #[test]
    fn cholesky_matches_numpy() {
        // S = [[4, 2], [2, 3]], symmetric positive-definite.
        let s = arr(vec![4.0, 2.0, 2.0, 3.0], &[2, 2]);
        let l = cholesky(&s).unwrap();
        let expected = arr(vec![2.0, 0.0, 1.0, 2.0_f64.sqrt()], &[2, 2]);
        assert_allclose_default(&l.view(), &expected.view()).unwrap();
    }

    #[test]
    fn cholesky_rejects_non_positive_definite() {
        let a = sample_a(); // not symmetric positive-definite
        assert_eq!(cholesky(&a).unwrap_err(), LinalgError::NotPositiveDefinite);
    }

    #[test]
    fn eigh_matches_numpy() {
        let s = arr(vec![4.0, 2.0, 2.0, 3.0], &[2, 2]);
        let (values, _vectors) = eigh(&s).unwrap();
        // np.linalg.eigh -> [1.43844719, 5.56155281], ascending.
        assert!((values[0] - 1.438_447_19).abs() < 1e-6);
        assert!((values[1] - 5.561_552_81).abs() < 1e-6);
    }

    #[test]
    fn eigvals_matches_numpy_for_rotation_matrix() {
        // [[0, -1], [1, 0]] rotates by 90 degrees -> eigenvalues +-i.
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
        // np.linalg.svd(A)'s singular values -> [8.33557912, 0.71980602].
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
        // np.linalg.norm(A) with no ord -> 8.366600265340756.
        assert!((frobenius_norm(&sample_a()) - 8.366_600_265_340_756).abs() < 1e-9);
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
