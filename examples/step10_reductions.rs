//! Run: `cargo run --example step10_reductions`
//!
//! Step 10: whole-array reductions/statistics. Every value below was
//! checked against real NumPy 2.5.3 first (see `reductions.rs`'s doc
//! comment) -- including the two counter-intuitive `nan*` findings:
//! `nansum` of an all-`NaN` array is `0.0`, but `nanmean`/`nanvar`/
//! `nanstd`/`nanmedian`/`nanmin`/`nanmax` of the same array are all `NaN`.

use rustnumpy::{corrcoef, cov_default, histogram, mean, median, nanmean, nansum, percentile, std_default, var_default, NdArray};

fn arr(values: &[f64]) -> NdArray {
    NdArray::from_vec(values.to_vec(), &[values.len()]).unwrap()
}

fn main() {
    let a = arr(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    println!("a = [1, 2, 3, 4, 5]");
    println!("  sum={}  mean={}  var={}  std={:.6}", rustnumpy::sum(&a.view()), mean(&a.view()), var_default(&a.view()), std_default(&a.view()));
    println!(
        "  median={}  p25={}  p75={}",
        median(&a.view()).unwrap(),
        percentile(&a.view(), 25.0).unwrap(),
        percentile(&a.view(), 75.0).unwrap()
    );

    let b = arr(&[1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
    println!("\nb = [1, NaN, 3, NaN, 5]");
    println!("  nansum={}  nanmean={}", nansum(&b.view()), nanmean(&b.view()));

    let all_nan = arr(&[f64::NAN, f64::NAN]);
    println!("\nall_nan = [NaN, NaN]");
    println!(
        "  nansum={}  (0.0, not NaN!)   nanmean={}  (NaN)",
        nansum(&all_nan.view()),
        nanmean(&all_nan.view())
    );

    let data = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0];
    let (counts, edges) = histogram(&data, 4, None).unwrap();
    println!("\nhistogram([1,2,2.5,3,3.5,4], bins=4) -> counts={counts:?}  edges={edges:?}");

    let m = NdArray::from_vec(vec![0.0, 2.0, 1.0, 3.0, 2.0, 1.0, 0.0, 3.0], &[2, 4]).unwrap();
    let c = cov_default(&m.view()).unwrap();
    let r = corrcoef(&m.view()).unwrap();
    println!("\ncov(m) row0 = {:?}", c.as_slice());
    println!("corrcoef(m) row0 = {:?}", r.as_slice());
}
