use crate::linalg::{eigvals, LinalgError};
use crate::ndarray::NdArray;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolynomialKind {
    Chebyshev,

    Hermite,
    Laguerre,
    Legendre,
}

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

    pub fn evaluate(&self, x: f64) -> f64 {
        basis_values(self.kind, self.degree(), x)
            .iter()
            .zip(&self.coeffs)
            .map(|(&basis, &c)| basis * c)
            .sum()
    }

    pub fn evaluate_array(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&x| self.evaluate(x)).collect()
    }

    pub fn roots(&self) -> Result<Vec<(f64, f64)>, LinalgError> {
        let power_coeffs = to_power_basis(self.kind, &self.coeffs);
        let companion = companion_matrix(&power_coeffs);
        eigvals(&companion)
    }
}

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

fn poly_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    let len = a.len().max(b.len());
    (0..len).map(|i| a.get(i).copied().unwrap_or(0.0) + b.get(i).copied().unwrap_or(0.0)).collect()
}

fn poly_scale(a: &[f64], k: f64) -> Vec<f64> {
    a.iter().map(|&v| v * k).collect()
}

fn poly_mul_x(a: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0];
    out.extend_from_slice(a);
    out
}

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

        let p = Polynomial::new(PolynomialKind::Chebyshev, C.to_vec());
        let roots = sort_roots(p.roots().unwrap());
        assert!((roots[0].0 - (-0.767_591_88)).abs() < 1e-6 && roots[0].1.abs() < 1e-9);
        assert!((roots[1].0 - 0.434_258_55).abs() < 1e-6 && roots[1].1.abs() < 1e-9);
    }

    #[test]
    fn hermite_roots_match_numpy() {

        let p = Polynomial::new(PolynomialKind::Hermite, C.to_vec());
        let roots = sort_roots(p.roots().unwrap());
        assert!((roots[0].0 - (-0.833_333_33)).abs() < 1e-6 && roots[0].1.abs() < 1e-9);
        assert!((roots[1].0 - 0.5).abs() < 1e-6 && roots[1].1.abs() < 1e-9);
    }

    #[test]
    fn laguerre_roots_match_numpy() {

        let p = Polynomial::new(PolynomialKind::Laguerre, C.to_vec());
        let roots = sort_roots(p.roots().unwrap());
        assert!((roots[0].0 - 0.902_832_46).abs() < 1e-6 && roots[0].1.abs() < 1e-9);
        assert!((roots[1].0 - 4.430_500_87).abs() < 1e-6 && roots[1].1.abs() < 1e-9);
    }

    #[test]
    fn legendre_roots_match_numpy() {

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
