//! `numpy.polynomial`'s classical orthogonal polynomial families --
//! Chebyshev, Hermite (physicists'), Laguerre, Legendre: evaluation via
//! each family's own three-term recurrence, and `roots()` built on top of
//! the existing [`crate::linalg`] (a polynomial's roots are the
//! eigenvalues of its companion matrix -- exactly how NumPy itself finds
//! them, just via `faer` instead of LAPACK).
//!
//! A `Polynomial` stores its coefficients in the chosen family's own
//! basis (`coeffs[i]` is the coefficient of that family's degree-`i`
//! basis polynomial), matching how `np.polynomial.Chebyshev`/`Hermite`/
//! `Laguerre`/`Legendre` all store coefficients -- *not* in the ordinary
//! power basis (`x^i`).
//!
//! **Scope note**: only evaluation and root-finding are implemented (no
//! derivative/integral/fit) -- the smallest slice that's still genuinely
//! useful and exercises the "roots via companion matrix + `linalg`"
//! connection `NumPy.md` calls out for this step.
//!
//! **Known, accepted deviation from NumPy**: [`Polynomial::roots`]
//! converts to the power basis internally to build a companion matrix,
//! then calls [`crate::linalg::eigvals`] -- mathematically equivalent to
//! NumPy's own basis-specific companion matrices, but less
//! well-conditioned for high degree (NumPy's own companion matrices are
//! specifically scaled to avoid this). Verified against real NumPy for
//! the modest degrees this project tests; per `NumPy.md`'s project-wide
//! tolerance policy, roots are compared numerically (sorted, with
//! tolerance), never bit-for-bit.

use crate::linalg::{eigvals, LinalgError};
use crate::ndarray::NdArray;

/// Which classical orthogonal polynomial family a [`Polynomial`]'s
/// coefficients are expressed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolynomialKind {
    Chebyshev,
    /// The physicists' convention (`H_0 = 1`, `H_1 = 2x`), matching
    /// `np.polynomial.Hermite` (NumPy's `HermiteE` is the probabilists'
    /// convention instead; not implemented here).
    Hermite,
    Laguerre,
    Legendre,
}

/// A polynomial expressed in one of the classical orthogonal bases (see
/// [`PolynomialKind`]), e.g. `Polynomial::new(PolynomialKind::Chebyshev,
/// vec![1.0, 2.0, 3.0])` is `1*T_0(x) + 2*T_1(x) + 3*T_2(x)`.
pub struct Polynomial {
    kind: PolynomialKind,
    coeffs: Vec<f64>,
}

impl Polynomial {
    pub fn new(kind: PolynomialKind, coeffs: Vec<f64>) -> Self {
        Self { kind, coeffs }
    }

    pub fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    /// `p(x)`: evaluates via the family's own three-term recurrence
    /// (matching how NumPy itself evaluates -- not by expanding to the
    /// power basis first, which would be both slower and less stable).
    pub fn evaluate(&self, x: f64) -> f64 {
        basis_values(self.kind, self.degree(), x)
            .iter()
            .zip(&self.coeffs)
            .map(|(&basis, &c)| basis * c)
            .sum()
    }

    /// `p(xs)`, element-wise.
    pub fn evaluate_array(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&x| self.evaluate(x)).collect()
    }

    /// `p.roots()`: the roots of `p`, as `(re, im)` pairs (matching
    /// [`crate::linalg::eigvals`]'s own return type, for the same reason
    /// -- `NdArray` has no complex dtype yet, see `lib.rs`'s doc comment).
    /// Computed as the eigenvalues of `p`'s companion matrix in the power
    /// basis (see this module's doc comment for why that's the accepted,
    /// slightly-less-well-conditioned approach here).
    pub fn roots(&self) -> Result<Vec<(f64, f64)>, LinalgError> {
        let power_coeffs = to_power_basis(self.kind, &self.coeffs);
        let companion = companion_matrix(&power_coeffs);
        eigvals(&companion)
    }
}

/// The values `[basis_0(x), ..., basis_degree(x)]` for the given family,
/// via its own three-term recurrence.
fn basis_values(kind: PolynomialKind, degree: usize, x: f64) -> Vec<f64> {
    let mut values = vec![1.0];
    if degree == 0 {
        return values;
    }
    values.push(match kind {
        PolynomialKind::Chebyshev | PolynomialKind::Legendre => x,
        PolynomialKind::Hermite => 2.0 * x,
        PolynomialKind::Laguerre => 1.0 - x,
    });
    for n in 1..degree {
        let n_f = n as f64;
        let next = match kind {
            PolynomialKind::Chebyshev => 2.0 * x * values[n] - values[n - 1],
            PolynomialKind::Legendre => {
                ((2.0 * n_f + 1.0) * x * values[n] - n_f * values[n - 1]) / (n_f + 1.0)
            }
            PolynomialKind::Hermite => 2.0 * x * values[n] - 2.0 * n_f * values[n - 1],
            PolynomialKind::Laguerre => {
                ((2.0 * n_f + 1.0 - x) * values[n] - n_f * values[n - 1]) / (n_f + 1.0)
            }
        };
        values.push(next);
    }
    values
}

/// Ascending-power-order polynomial arithmetic (`poly[i]` is the
/// coefficient of `x^i`) -- just enough to build each family's basis
/// polynomials symbolically via the same recurrence [`basis_values`] uses
/// numerically.
fn poly_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    let len = a.len().max(b.len());
    (0..len).map(|i| a.get(i).copied().unwrap_or(0.0) + b.get(i).copied().unwrap_or(0.0)).collect()
}

fn poly_scale(a: &[f64], k: f64) -> Vec<f64> {
    a.iter().map(|&v| v * k).collect()
}

/// Multiplies a polynomial by `x` (shifts every coefficient up one degree).
fn poly_mul_x(a: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0];
    out.extend_from_slice(a);
    out
}

/// Converts a coefficient vector in `kind`'s orthogonal basis to the
/// ordinary power basis (ascending, `result[i]` is the coefficient of
/// `x^i`) by building each basis polynomial symbolically via the same
/// recurrence [`basis_values`] evaluates numerically, then summing them
/// weighted by `coeffs`.
fn to_power_basis(kind: PolynomialKind, coeffs: &[f64]) -> Vec<f64> {
    let degree = coeffs.len().saturating_sub(1);
    let mut basis_polys: Vec<Vec<f64>> = vec![vec![1.0]];
    if degree >= 1 {
        basis_polys.push(match kind {
            PolynomialKind::Chebyshev | PolynomialKind::Legendre => vec![0.0, 1.0],
            PolynomialKind::Hermite => vec![0.0, 2.0],
            PolynomialKind::Laguerre => vec![1.0, -1.0],
        });
    }
    for n in 1..degree {
        let n_f = n as f64;
        let prev = &basis_polys[n - 1];
        let curr = &basis_polys[n];
        let next = match kind {
            PolynomialKind::Chebyshev => {
                poly_add(&poly_scale(&poly_mul_x(curr), 2.0), &poly_scale(prev, -1.0))
            }
            PolynomialKind::Legendre => poly_scale(
                &poly_add(
                    &poly_scale(&poly_mul_x(curr), 2.0 * n_f + 1.0),
                    &poly_scale(prev, -n_f),
                ),
                1.0 / (n_f + 1.0),
            ),
            PolynomialKind::Hermite => {
                poly_add(&poly_scale(&poly_mul_x(curr), 2.0), &poly_scale(prev, -2.0 * n_f))
            }
            PolynomialKind::Laguerre => {
                // ((2n+1) - x) * curr - n * prev, all divided by (n+1)
                let shifted = poly_add(&poly_scale(curr, 2.0 * n_f + 1.0), &poly_scale(&poly_mul_x(curr), -1.0));
                poly_scale(&poly_add(&shifted, &poly_scale(prev, -n_f)), 1.0 / (n_f + 1.0))
            }
        };
        basis_polys.push(next);
    }

    let mut result = vec![0.0; degree + 1];
    for (i, &c) in coeffs.iter().enumerate() {
        result = poly_add(&result, &poly_scale(&basis_polys[i], c));
    }
    result
}

/// The standard companion matrix of a monic-normalized power-basis
/// polynomial (`coeffs[i]` is the coefficient of `x^i`, ascending) --
/// its eigenvalues are exactly the polynomial's roots.
fn companion_matrix(coeffs: &[f64]) -> NdArray {
    let degree = coeffs.len() - 1;
    let leading = coeffs[degree];
    let normalized: Vec<f64> = coeffs[..degree].iter().map(|&c| c / leading).collect();

    let mut data = vec![0.0; degree * degree];
    for row in 0..degree {
        data[row * degree + (degree - 1)] = -normalized[row];
        if row + 1 < degree {
            data[(row + 1) * degree + row] = 1.0;
        }
    }
    NdArray::from_vec(data, &[degree, degree]).expect("degree*degree elements by construction")
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every expected value below was computed by real NumPy 2.5.3's
    // np.polynomial.{Chebyshev,Hermite,Laguerre,Legendre} first (see this
    // module's doc comment for the tolerance-not-exact-equality policy).
    const C: [f64; 3] = [1.0, 2.0, 3.0];

    fn sort_roots(mut roots: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
        roots.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        roots
    }

    #[test]
    fn chebyshev_evaluate_matches_numpy() {
        let p = Polynomial::new(PolynomialKind::Chebyshev, C.to_vec());
        assert!((p.evaluate(0.5) - 0.5).abs() < 1e-9);
        assert_eq!(p.evaluate_array(&[0.0, 1.0, 2.0]), vec![-2.0, 6.0, 26.0]);
    }

    #[test]
    fn hermite_evaluate_matches_numpy() {
        let p = Polynomial::new(PolynomialKind::Hermite, C.to_vec());
        assert!((p.evaluate(0.5) - 0.0).abs() < 1e-9);
        assert_eq!(p.evaluate_array(&[0.0, 1.0, 2.0]), vec![-5.0, 11.0, 51.0]);
    }

    #[test]
    fn laguerre_evaluate_matches_numpy() {
        let p = Polynomial::new(PolynomialKind::Laguerre, C.to_vec());
        assert!((p.evaluate(0.5) - 2.375).abs() < 1e-9);
        assert_eq!(p.evaluate_array(&[0.0, 1.0, 2.0]), vec![6.0, -0.5, -4.0]);
    }

    #[test]
    fn legendre_evaluate_matches_numpy() {
        let p = Polynomial::new(PolynomialKind::Legendre, C.to_vec());
        assert!((p.evaluate(0.5) - 1.625).abs() < 1e-9);
        assert_eq!(p.evaluate_array(&[0.0, 1.0, 2.0]), vec![-0.5, 6.0, 21.5]);
    }

    #[test]
    fn chebyshev_roots_match_numpy() {
        // np.polynomial.Chebyshev([1,2,3]).roots() -> [-0.76759188, 0.43425855]
        let p = Polynomial::new(PolynomialKind::Chebyshev, C.to_vec());
        let roots = sort_roots(p.roots().unwrap());
        assert!((roots[0].0 - (-0.767_591_88)).abs() < 1e-6 && roots[0].1.abs() < 1e-9);
        assert!((roots[1].0 - 0.434_258_55).abs() < 1e-6 && roots[1].1.abs() < 1e-9);
    }

    #[test]
    fn hermite_roots_match_numpy() {
        // np.polynomial.Hermite([1,2,3]).roots() -> [-0.83333333, 0.5] (real)
        let p = Polynomial::new(PolynomialKind::Hermite, C.to_vec());
        let roots = sort_roots(p.roots().unwrap());
        assert!((roots[0].0 - (-0.833_333_33)).abs() < 1e-6 && roots[0].1.abs() < 1e-9);
        assert!((roots[1].0 - 0.5).abs() < 1e-6 && roots[1].1.abs() < 1e-9);
    }

    #[test]
    fn laguerre_roots_match_numpy() {
        // np.polynomial.Laguerre([1,2,3]).roots() -> [0.90283246, 4.43050087]
        let p = Polynomial::new(PolynomialKind::Laguerre, C.to_vec());
        let roots = sort_roots(p.roots().unwrap());
        assert!((roots[0].0 - 0.902_832_46).abs() < 1e-6 && roots[0].1.abs() < 1e-9);
        assert!((roots[1].0 - 4.430_500_87).abs() < 1e-6 && roots[1].1.abs() < 1e-9);
    }

    #[test]
    fn legendre_roots_match_numpy() {
        // np.polynomial.Legendre([1,2,3]).roots() -> [-0.62283903, 0.17839459]
        let p = Polynomial::new(PolynomialKind::Legendre, C.to_vec());
        let roots = sort_roots(p.roots().unwrap());
        assert!((roots[0].0 - (-0.622_839_03)).abs() < 1e-6 && roots[0].1.abs() < 1e-9);
        assert!((roots[1].0 - 0.178_394_59).abs() < 1e-6 && roots[1].1.abs() < 1e-9);
    }

    #[test]
    fn degree_zero_polynomial_evaluates_to_constant() {
        let p = Polynomial::new(PolynomialKind::Chebyshev, vec![5.0]);
        assert_eq!(p.evaluate(123.0), 5.0);
        assert_eq!(p.degree(), 0);
    }
}
