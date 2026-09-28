use rustnumpy::{mean, Generator};

fn main() {
    let mut rng = Generator::seed(42);

    println!("random(5) -> {:?}", rng.random(&[5]).as_slice());
    println!("uniform(2, 10, 5) -> {:?}", rng.uniform(2.0, 10.0, &[5]).unwrap().as_slice());
    println!("integers(0, 6, 5) -> {:?}", rng.integers(0, 6, &[5]).unwrap().as_slice());
    println!("standard_normal(5) -> {:?}", rng.standard_normal(&[5]).as_slice());

    let normal_sample = rng.normal(5.0, 2.0, &[20_000]).unwrap();
    println!("\nnormal(5, 2, 20000).mean() -> {} (expected ~5.0)", mean(&normal_sample.view()));

    let exp_sample = rng.exponential(3.0, &[20_000]).unwrap();
    println!("exponential(3).mean() -> {} (expected ~3.0)", mean(&exp_sample.view()));

    let gamma_sample = rng.gamma(2.0, 3.0, &[20_000]).unwrap();
    println!("gamma(2, 3).mean() -> {} (expected ~6.0)", mean(&gamma_sample.view()));

    let beta_sample = rng.beta(2.0, 3.0, &[20_000]).unwrap();
    println!("beta(2, 3).mean() -> {} (expected ~0.4)", mean(&beta_sample.view()));

    let binomial_sample = rng.binomial(20, 0.3, &[20_000]).unwrap();
    println!("binomial(20, 0.3).mean() -> {} (expected ~6.0)", mean(&binomial_sample.view()));

    let poisson_sample = rng.poisson(4.0, &[20_000]).unwrap();
    println!("poisson(4).mean() -> {} (expected ~4.0)", mean(&poisson_sample.view()));

    let dirichlet_sample = rng.dirichlet(&[1.0, 2.0, 3.0], 3).unwrap();
    println!("\ndirichlet([1,2,3], size=3) -> shape={:?} data={:?}",
        dirichlet_sample.shape(), dirichlet_sample.as_slice());
}
