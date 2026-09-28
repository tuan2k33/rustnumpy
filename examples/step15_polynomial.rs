use rustnumpy::{Polynomial, PolynomialKind};

fn main() {
    let coeffs = vec![1.0, 2.0, 3.0];

    for (name, kind) in [
        ("Chebyshev", PolynomialKind::Chebyshev),
        ("Hermite", PolynomialKind::Hermite),
        ("Laguerre", PolynomialKind::Laguerre),
        ("Legendre", PolynomialKind::Legendre),
    ] {
        let p = Polynomial::new(kind, coeffs.clone());
        println!("{name}(0.5) -> {}", p.evaluate(0.5));
        println!("{name}([0,1,2]) -> {:?}", p.evaluate_array(&[0.0, 1.0, 2.0]));
        let mut roots = p.roots().unwrap();
        roots.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        println!("{name}.roots() -> {roots:?}\n");
    }
}
